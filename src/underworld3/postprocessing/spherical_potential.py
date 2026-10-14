"""Physical modal spherical potential on caller-owned full-ball P2 meshes."""
from contextlib import ExitStack
from collections.abc import Mapping
from numbers import Integral
import numpy as np
import sympy as sp
from petsc4py import PETSc
import underworld3 as uw
from underworld3.maths.spherical_harmonics import real_harmonic_cartesian_sympy, solid_harmonic_sympy, validate_harmonic
from underworld3.analytic.spherical_gravity import spherical_sheet_potential_coefficient, spherical_volume_density_potential_coefficient
from .spherical_interface_load import assemble_spherical_harmonic_sheet_load, project_spherical_harmonic_trace_coefficient


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
                stratum=label.getStratumIS(int(value))
                try:
                    if stratum:
                        points=[int(p) for p in stratum.indices if start<=int(p)<end]
                finally:
                    if stratum:stratum.destroy()
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
        with mesh.access(indicator):
            indicator.data[:,0]=0.
            for point in selected:
                if section.getDof(point)!=1:
                    raise RuntimeError('Volume region support requires one DG0 scalar DOF per cell.')
                indicator.data[section.getOffset(point),0]=1.
    owned_counts=comm.allgather(sum(p not in ghosts for p in selected))
    return indicator,dict(regions={name:int(value) for name,value in regions.items()},
        cells_by_region=counts,owned_cells_per_rank=owned_counts,
        global_owned_cells=sum(owned_counts))


def solve_spherical_potential(*, mesh, field_name, harmonic_degree,
        harmonic_order=0, harmonic_component='cos', harmonic_normalization='axisymmetric_unnormalized',
        sheet_densities=None, sheet_radii=None, sheet_modes=(), volume_density=0., density_power=2,
        volume_regions=None,
        exterior_condition='robin',
        radius_inner, radius_outer, radius_exterior,
        inner_boundary='CMB', outer_boundary='Surface', exterior_boundary='OuterBoundary',
        gravitational_constant, solver_options=None, tolerance=1e-10):
    """Solve -laplacian(Phi)=4*pi*G*rho for a real harmonic degree on a full ball.

    sheet_densities maps caller-specified interface labels to density
    coefficients. Sheets use uniquely owned triangular P2 trace loads.
    sheet_modes adds same-degree angular modes, each a dictionary containing
    sheet_densities and optional harmonic_order/component/normalization. The
    primary mode still selects volume forcing and returned trace coefficients.
    All angular modes of one degree share the exact exterior Robin coefficient;
    mixed-degree Robin continuation requires separate solves, not one local
    Robin coefficient. Solver adapters require the world communicator and
    nonempty mesh charts on every rank.
    Volume density is rho0*(r/r_outer)**k in radius_inner<=r<=radius_outer.
    volume_regions optionally maps caller-supplied volume labels to integer
    values. It confines that radial profile to their union of complete mesh
    cells using DG0 support, consistently with conforming faceted interfaces.
    With no regions supplied, the historical radial Piecewise cutoff is used;
    on straight facets it cuts cells and may need additional integration control.
    Region support approximates the same spherical mantle by its discrete
    labelled geometry; it does not rescale the source to match an oracle.
    The default exterior Robin term (ell+1)*Phi/r_exterior represents exact
    single-mode decaying continuation. exterior_condition=dirichlet_exact is
    an optional comparison using the analytical exterior value; that comparison
    checks the interior solve rather than independently testing the radial kernel. The origin is included, with regularity inherent
    in the continuous Cartesian P2 space. The caller selects solver policy.
    Returned potential coefficients use explicit interface trace integrals.
    """
    ell,m,component,normalization=validate_harmonic(harmonic_degree,harmonic_order,
                                                 harmonic_component,harmonic_normalization)
    from ._spherical_solver_contract import require_native_solver_mesh
    require_native_solver_mesh(mesh)
    sources=[(m,component,normalization,dict(sheet_densities or {}))]
    for spec in sheet_modes:
        degree,order,part,norm=validate_harmonic(spec.get('harmonic_degree',ell),spec.get('harmonic_order',0),
            spec.get('harmonic_component','cos'),spec.get('harmonic_normalization',normalization))
        if degree!=ell:
            raise ValueError('sheet_modes must share harmonic_degree; mixed degrees require separate solves.')
        sources.append((order,part,norm,dict(spec['sheet_densities'])))
    if any(not np.isfinite(float(value)) for _,_,_,densities in sources for value in densities.values()):
        raise ValueError('Sheet densities must be finite.')
    if mesh.dim!=3:raise ValueError('Spherical potential requires a 3D full-ball mesh.')
    if not 0<radius_inner<radius_outer<radius_exterior:raise ValueError('Expected ordered positive radii.')
    if not isinstance(density_power, Integral) or isinstance(density_power, bool):
        raise TypeError("density_power must be an integer.")
    comm=mesh.dm.getComm().tompi4py()
    from mpi4py import MPI
    minimum_radius=comm.allreduce(float(np.min(np.linalg.norm(mesh._coords,axis=1),initial=np.inf)),op=MPI.MIN)
    if minimum_radius>1e-12:
        raise ValueError('This API requires a full ball including the origin.')
    if not np.all(np.isfinite((gravitational_constant,volume_density,tolerance))) or gravitational_constant<=0 or tolerance<=0:
        raise ValueError('Expected finite density, positive G and tolerance.')
    phi=uw.discretisation.MeshVariable(str(field_name),mesh,1,degree=2,continuous=True)
    load=uw.discretisation.MeshVariable(str(field_name)+'Load',mesh,1,degree=2,continuous=True)
    radius=sp.sqrt(sum(x*x for x in mesh.X))
    harmonic=real_harmonic_cartesian_sympy(list(mesh.X),degree=ell,order=m,
                                          component=component,normalization=normalization)
    poisson=uw.systems.Poisson(mesh,u_Field=phi,degree=2)
    poisson.constitutive_model=uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity=1.
    volume_diagnostics={}
    profile=float(volume_density)*(radius/radius_outer)**int(density_power)*harmonic
    if volume_density and volume_regions is not None:
        indicator,volume_diagnostics=_volume_region_indicator(mesh,str(field_name)+'Region',volume_regions)
        # Cellwise support must also guard evaluating singular radial profiles
        # at the origin in excluded core cells (0*NaN is not zero).
        density=sp.Piecewise((profile,sp.Ne(indicator.sym[0],0)),(0.,True))
    else:
        density=sp.Piecewise((profile,sp.And(radius>=radius_inner,radius<=radius_outer)),(0.,True))
    poisson.f=4*sp.pi*gravitational_constant*density if volume_density else 0.
    if exterior_condition=='robin':
        poisson.add_natural_bc(sp.Matrix([(ell+1)*phi.sym[0]/radius_exterior]),exterior_boundary)
    elif exterior_condition=='dirichlet_exact':
        radii=dict(sheet_radii or {})
        boundary_value=0.
        for index,(order,part,norm,densities) in enumerate(sources):
            coefficient=0.
            for label,density in densities.items():
                if label not in radii:raise ValueError(f'Explicit source radius required for {label!r}.')
                coefficient+=spherical_sheet_potential_coefficient(source_radius=radii[label],
                    target_radius=radius_exterior,harmonic_degree=ell,
                    surface_density_coefficient=density,gravitational_constant=gravitational_constant)
            H=solid_harmonic_sympy(list(mesh.X),degree=ell,order=order,component=part,normalization=norm)
            boundary_value+=coefficient*radius_exterior**(ell+1)*H/radius**(2*ell+1)
        if volume_density:
            coefficient=spherical_volume_density_potential_coefficient(radius_inner=radius_inner,
                radius_outer=radius_outer,harmonic_degree=ell,density_power=density_power,
                density_amplitude=volume_density,target_radius=radius_exterior,
                gravitational_constant=gravitational_constant)
            H=solid_harmonic_sympy(list(mesh.X),degree=ell,order=m,component=component,normalization=normalization)
            boundary_value+=coefficient*radius_exterior**(ell+1)*H/radius**(2*ell+1)
        poisson.add_dirichlet_bc([boundary_value],exterior_boundary)
    else:raise ValueError('exterior_condition must be dirichlet_exact or robin.')
    poisson.tolerance=tolerance
    options={'snes_type':'ksponly'} | dict(solver_options or {})
    for key,value in options.items():poisson.petsc_options[key]=value
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
                vector,report=assemble_spherical_harmonic_sheet_load(mesh=mesh,load_variable=load,
                    interface=label,harmonic_degree=ell,harmonic_order=order,harmonic_component=part,
                    harmonic_normalization=norm,density_coefficient=float(coefficient),
                    gravitational_constant=gravitational_constant)
                cleanup.callback(vector.destroy)
                diagnostics[label if index==0 else f'mode{index}:{label}']=report
                if combined is None:
                    combined=vector.copy();cleanup.callback(combined.destroy)
                else:combined.axpy(1.,vector)
        if combined is not None:
            iset,subdm=mesh.dm.createSubDM(load.field_id)
            cleanup.callback(iset.destroy);cleanup.callback(subdm.destroy)
            load._gvec.array[:]=combined.array[:]
            subdm.globalToLocal(load._gvec,load.vec,addv=PETSc.InsertMode.INSERT_VALUES)
            local=poisson.dm.getLocalVec();cleanup.callback(poisson.dm.restoreLocalVec,local)
            rhs=poisson.dm.getGlobalVec();cleanup.callback(poisson.dm.restoreGlobalVec,rhs)
            solution=poisson.dm.getGlobalVec();cleanup.callback(poisson.dm.restoreGlobalVec,solution)
            local.array[:]=load._lvec.array[:]
            rhs.set(0.);poisson.dm.localToGlobal(local,rhs,addv=PETSc.InsertMode.INSERT_VALUES)
            operator=poisson.snes.getJacobian()[0]
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
            phi._sync_lvec_to_gvec()
        else:reason=int(base_report.reason)
    projection=dict(mesh=mesh,variable=phi,harmonic_degree=ell,harmonic_order=m,
                    harmonic_component=component,harmonic_normalization=normalization)
    outer=project_spherical_harmonic_trace_coefficient(interface=outer_boundary,interface_radius=radius_outer,**projection)
    inner=project_spherical_harmonic_trace_coefficient(interface=inner_boundary,interface_radius=radius_inner,**projection)
    return dict(potential_field=phi,poisson_system=poisson,outer_potential=outer,
                inner_potential=inner,solver_reason=reason,load_diagnostics=diagnostics,
                volume_source_support='regions' if volume_density and volume_regions is not None else
                    'radial_cutoff' if volume_density else 'none',
                volume_region_diagnostics=volume_diagnostics)


__all__=['solve_spherical_potential']
