"""Physical modal spherical potential on caller-owned full-ball P2 meshes."""
from contextlib import ExitStack
from collections.abc import Mapping
from numbers import Integral
import numpy as np
import sympy as sp
from petsc4py import PETSc
import underworld3 as uw
from underworld3.maths.spherical_harmonics import real_harmonic_cartesian_sympy, solid_harmonic_sympy, validate_harmonic
from underworld3.analytic.spherical.gravity import spherical_sheet_potential_coefficient, spherical_volume_density_potential_coefficient
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
    r"""Solve physical modal potential on a caller-owned full-ball P2 mesh.

    Parameters
    ----------
    mesh : underworld3.discretisation.Mesh
        Three-dimensional simplicial full-ball mesh including the origin and
        straight conforming triangular sheet facets. Its communicator must be
        congruent to UW3's world communicator, with nonempty charts on every rank.
    field_name : str
        Base name for the continuous scalar P2 potential and weak-load variables.
    harmonic_degree : int
        Spherical-harmonic degree, at least 1.
    harmonic_order : int, default=0
        Order satisfying ``0 <= harmonic_order <= harmonic_degree``.
    harmonic_component : {"cos", "sin"}, default="cos"
        Real angular component. Order zero requires ``"cos"``.
    harmonic_normalization : {"axisymmetric_unnormalized", "real_orthonormal"}, default="axisymmetric_unnormalized"
        Angular normalization used by both sources and projections. The default
        is ``P_l(cos(theta))`` and is restricted to order zero. Use
        ``"real_orthonormal"`` for non-axisymmetric modes.
    sheet_densities : mapping or None, default=None
        Interface label -> finite signed surface-density coefficient for the primary
        mode. Physical coefficients have mass/area units. ``None`` means no sheets.
    sheet_radii : mapping or None, default=None
        Interface label -> physical source radius in mesh units. Required for every
        sheet only with ``exterior_condition="dirichlet_exact"``; Robin assembly
        uses the actual caller mesh facets.
    sheet_modes : sequence of dict, default=()
        Additional same-degree modes. Each requires ``sheet_densities`` and may
        set ``harmonic_degree`` (which must match), ``harmonic_order`` (default 0),
        ``harmonic_component`` (default ``"cos"``) and ``harmonic_normalization``
        (defaulting to the function argument). Volume forcing and returned
        projections use only the primary mode.
    volume_density : float, default=0
        Finite signed volume-density amplitude at ``radius_outer``. Physical units
        are mass/volume; zero disables the volume source.
    density_power : int, default=2
        Integer ``k`` in ``rho(r)=volume_density*(r/radius_outer)**k`` times the
        primary harmonic. The source is confined to the specified mantle support.
    volume_regions : mapping or None, default=None
        Mesh volume-label name -> integer stratum value. Their union selects
        complete DG0-supported cells for nonzero volume density. Every requested
        label/stratum must have cells globally; empty local source strata are allowed.
        ``None`` uses the radial cutoff ``radius_inner <= r <= radius_outer``.
    exterior_condition : {"robin", "dirichlet_exact"}, default="robin"
        Robin uses exact single-degree decaying continuation. ``"dirichlet_exact"``
        supplies the analytical exterior potential as a comparison boundary value.
    radius_inner, radius_outer, radius_exterior : float
        Positive core/mantle, mantle/surface and outer truncation radii in mesh units,
        ordered strictly inner < outer < exterior. They must match interface geometry.
    inner_boundary, outer_boundary, exterior_boundary : str
        Labels for inner/outer projection and exterior boundary treatment;
        defaults are ``"CMB"``, ``"Surface"`` and ``"OuterBoundary"``.
    gravitational_constant : float
        Finite positive ``G`` in a unit system consistent with radii and densities.
        No physical/nondimensional unit conversion is performed.
    solver_options : mapping or None, default=None
        PETSc options for the base Poisson system. A sheet correction KSP also uses
        ``ksp_type`` (default ``"gmres"``) and ``pc_type`` (default ``"gamg"``).
    tolerance : float, default=1e-10
        Finite positive base-system tolerance and relative sheet-correction tolerance.
        The correction also uses absolute tolerance 1e-12 and a 1000-iteration cap.

    Returns
    -------
    dict
        ``potential_field`` and ``poisson_system`` hold the P2 field and base system.
        ``outer_potential``/``inner_potential`` are primary-mode trace coefficients
        in potential units, not geoid heights. ``solver_reason`` is the sheet KSP
        reason when sheets exist, otherwise the base solve reason.
        ``load_diagnostics`` maps sheet labels (additional modes use
        ``"mode<index>:<label>"``) to :class:`InterfaceLoadDiagnostics`.
        ``volume_source_support`` is ``"regions"``, ``"radial_cutoff"`` or ``"none"``;
        ``volume_region_diagnostics`` reports labels and owned-cell counts, or is empty.

    Raises
    ------
    TypeError
        Harmonic or density-power types, or volume-region entries, are invalid.
    ValueError
        Mesh, source, radii, harmonic, exterior condition or region support is invalid.
    KeyError
        An additional sheet mode lacks ``sheet_densities``.
    RuntimeError
        Base or sheet correction solve fails or a P2 trace cannot be mapped.

    Notes
    -----
    Call collectively. The convention is ``-laplacian(Phi)=4*pi*G*rho``; regularity
    at the origin is represented by the Cartesian P2 space. Robin datum is
    ``(l+1)*Phi/radius_exterior``. Different orders of one degree may share it;
    mixed degrees require separate solves. Sheet assembly removes its discrete
    mean. Radial cutoffs can cut straight cells; explicit conforming regions avoid
    that integration ambiguity without rescaling the source. Exact Dirichlet
    comparison does not independently validate the analytical exterior kernel."""
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
