"""Restricted spherical Poisson correction and labelled volume support."""
from contextlib import ExitStack
from collections.abc import Mapping
from numbers import Integral
import numpy as np
import sympy as sp
from petsc4py import PETSc
import underworld3 as uw
from underworld3.maths.spherical_harmonics import validate_harmonic
from ._spherical_interface_load import _assemble_spherical_harmonic_sheet_load
from ._spherical_solver_contract import require_native_solver_mesh
from .dm_labels import label_stratum_indices

def _volume_region_indicator(mesh, field_name, regions):
    """DG0 support on caller-labelled volume cells, including empty local strata."""
    from mpi4py import MPI
    if not isinstance(regions, Mapping) or not regions:
        raise ValueError('volume_regions must be a nonempty mapping of label names to integer values.')
    if any(not isinstance(name,str) or not name or isinstance(value,bool)
           or not isinstance(value,Integral) for name,value in regions.items()):
        raise TypeError('Volume region labels require nonempty names and integer values.')
    comm=mesh.dm.getComm().tompi4py()
    start,end=mesh.dm.getHeightStratum(0)
    ilocal=None
    if comm.size>1:
        _,ilocal,_=mesh.dm.getPointSF().getGraph()
    ghosts=set() if ilocal is None else set(map(int,ilocal))
    selected=set()
    counts={}
    label_names={mesh.dm.getLabelName(i) for i in range(mesh.dm.getNumLabels())}
    for name,value in regions.items():
        # PETSc may return a null-handle wrapper, rather than None, for an
        # absent label/stratum. Never call methods on that wrapper.
        label=mesh.dm.getLabel(name) if name in label_names else None
        points=[]
        if label:
            values=label.getValueIS()
            try:
                present=bool(values) and int(value) in values.indices
            finally:
                if values:values.destroy()
            if present:
                points = [int(point) for point in label_stratum_indices(label, value)
                          if start <= int(point) < end]
        count=int(comm.allreduce(sum(p not in ghosts for p in points),op=MPI.SUM))
        if count==0:raise ValueError(f'Volume region {name!r}={value} has no labelled cells globally.')
        counts[name]=count
        selected.update(points)
    indicator=uw.discretisation.MeshVariable(str(field_name),mesh,1,degree=0,continuous=False)
    indicator._set_vec(available=True)
    with ExitStack() as cleanup:
        indexset,subdm=mesh.dm.createSubDM(indicator.field_id)
        cleanup.callback(indexset.destroy);cleanup.callback(subdm.destroy)
        section=subdm.getLocalSection()
        values = np.zeros(len(indicator.coords), dtype=float)
        for point in selected:
            if section.getDof(point) != 1:
                raise RuntimeError('Volume region support requires one DG0 scalar DOF per cell.')
            values[section.getOffset(point)] = 1.
        indicator.array[:, 0, 0] = values
    owned_counts=comm.allgather(sum(p not in ghosts for p in selected))
    return indicator,dict(regions={name:int(value) for name,value in regions.items()},
        cells_by_region=counts,owned_cells_per_rank=owned_counts,
        global_owned_cells=sum(owned_counts))


def _spherical_sheet_sources(harmonic_degree, harmonic_order,
        harmonic_component, harmonic_normalization, sheet_densities, sheet_modes):
    """Validate and normalize same-degree sheet specifications before allocation."""
    ell, m, component, normalization = validate_harmonic(
        harmonic_degree, harmonic_order, harmonic_component, harmonic_normalization)
    sources=[(m,component,normalization,dict(sheet_densities or {}))]
    for spec in sheet_modes:
        degree,order,part,norm=validate_harmonic(spec.get('harmonic_degree',ell),spec.get('harmonic_order',0),
            spec.get('harmonic_component','cos'),spec.get('harmonic_normalization',normalization))
        if degree!=ell:
            raise ValueError('sheet_modes must share harmonic_degree; mixed degrees require separate solves.')
        sources.append((order,part,norm,dict(spec['sheet_densities'])))
    if any(not np.isfinite(float(value)) for _,_,_,densities in sources for value in densities.values()):
        raise ValueError('Sheet densities must be finite.')
    return ell, m, component, normalization, sources


def _solve_linear_spherical_sheet_sources(poisson, *, load_variable,
        harmonic_degree, sheet_densities=None, harmonic_order=0,
        harmonic_component="cos", harmonic_normalization="axisymmetric_unnormalized",
        sheet_modes=(), gravitational_constant=1., solver_options=None, tolerance=None):
    """Implementation of Poisson.solve_linear_spherical_sheet_sources."""
    mesh = poisson.mesh
    require_native_solver_mesh(mesh)
    ell, m, component, normalization, sources = _spherical_sheet_sources(
        harmonic_degree, harmonic_order, harmonic_component, harmonic_normalization,
        sheet_densities, sheet_modes)
    load = load_variable
    if mesh.dim != 3 or poisson.u.degree != 2 or not poisson.u.continuous:
        raise ValueError("Spherical sheet correction requires a 3D continuous P2 solution.")
    if (load.mesh is not mesh or load.num_components != 1 or load.degree != 2
            or not load.continuous or load.field_id == poisson.u.field_id):
        raise ValueError("load_variable must be a distinct scalar continuous P2 field on the solver mesh.")
    if type(poisson.constitutive_model) is not uw.constitutive_models.DiffusionModel:
        raise TypeError("Only the linear DiffusionModel gravity operator is supported.")
    diffusivity = uw.function.unwrap(poisson.constitutive_model.Parameters.diffusivity)
    if sp.simplify(diffusivity - 1) != 0:
        raise ValueError("Spherical gravity correction requires unit diffusivity.")
    tolerance = float(poisson.tolerance if tolerance is None else tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Expected a finite positive correction tolerance.")
    if not np.isfinite(gravitational_constant) or gravitational_constant <= 0:
        raise ValueError("Expected finite positive G.")
    options = dict(solver_options or {})
    poisson.solve()
    base_report=poisson.solve_report
    if base_report is None or not base_report.converged:
        raise RuntimeError(f'Spherical potential setup/volume solve failed: {base_report}')
    diagnostics={}
    with ExitStack() as cleanup:
        # SNES keeps a reference to a DM-pool vector. Borrowing further work
        # vectors can return that same storage; preserve the base volume/BC
        # solution before getGlobalVec or the sheet correction can overwrite it
        # and axpy against itself (doubling the MPI solution).
        base_solution=poisson.snes.getSolution().copy()
        cleanup.callback(base_solution.destroy)
        combined=None
        for index,(order,part,norm,densities) in enumerate(sources):
            for label,coefficient in densities.items():
                vector,report=_assemble_spherical_harmonic_sheet_load(mesh=mesh,load_variable=load,
                    interface=label,harmonic_degree=ell,harmonic_order=order,harmonic_component=part,
                    harmonic_normalization=norm,density_coefficient=float(coefficient),
                    gravitational_constant=gravitational_constant)
                cleanup.callback(vector.destroy)
                diagnostics[label if index==0 else f'mode{index}:{label}']=report
                if combined is None:
                    combined=vector.copy();cleanup.callback(combined.destroy)
                else:combined.axpy(1.,vector)
        if combined is not None:
            operator, preconditioner, _ = poisson.snes.getJacobian()
            if poisson.snes.getIterationNumber() == 0:
                # A homogeneous base can converge before SNES evaluates its
                # Jacobian; the sheet correction still needs the linear operator.
                poisson.snes.computeJacobian(base_solution, operator, preconditioner)
            iset,subdm=mesh.dm.createSubDM(load.field_id)
            cleanup.callback(iset.destroy);cleanup.callback(subdm.destroy)
            load._gvec.array[:]=combined.array[:]
            subdm.globalToLocal(load._gvec,load.vec,addv=PETSc.InsertMode.INSERT_VALUES)
            local=poisson.dm.getLocalVec();cleanup.callback(poisson.dm.restoreLocalVec,local)
            rhs=poisson.dm.getGlobalVec();cleanup.callback(poisson.dm.restoreGlobalVec,rhs)
            solution=poisson.dm.getGlobalVec();cleanup.callback(poisson.dm.restoreGlobalVec,solution)
            local.array[:]=load._lvec.array[:]
            rhs.set(0.);poisson.dm.localToGlobal(local,rhs,addv=PETSc.InsertMode.INSERT_VALUES)
            ksp=PETSc.KSP().create(mesh.dm.getComm());cleanup.callback(ksp.destroy)
            ksp.setOperators(operator)
            ksp.setType(options.get('ksp_type','gmres'))
            ksp.getPC().setType(options.get('pc_type','gamg'))
            ksp.setTolerances(rtol=tolerance,atol=1e-12,max_it=1000)
            solution.set(0.);ksp.solve(rhs,solution)
            reason=int(ksp.getConvergedReason())
            if reason<=0:raise RuntimeError(f'Spherical sheet KSP failed: {reason}')
            solution.axpy(1.,base_solution)
            poisson._scatter_global_to_fields(solution)
            poisson.u._sync_lvec_to_gvec()
        else:reason=int(base_report.reason)
    return dict(solver_reason=reason, load_diagnostics=diagnostics)
