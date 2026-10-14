"""Physical modal spherical potential on caller-owned full-ball P2 meshes."""
from contextlib import ExitStack
from numbers import Integral
import numpy as np
import sympy as sp
from petsc4py import PETSc
import underworld3 as uw
from underworld3.maths.spherical_harmonics import real_harmonic_cartesian_sympy, solid_harmonic_sympy, validate_harmonic
from underworld3.analytic.spherical_gravity import spherical_sheet_potential_coefficient, spherical_volume_density_potential_coefficient
from .spherical_interface_load import assemble_spherical_harmonic_sheet_load, project_spherical_harmonic_trace_coefficient


def solve_spherical_potential(*, mesh, field_name, harmonic_degree,
        harmonic_order=0, harmonic_component='cos', harmonic_normalization='axisymmetric_unnormalized',
        sheet_densities=None, sheet_radii=None, volume_density=0., density_power=2,
        exterior_condition='robin',
        radius_inner, radius_outer, radius_exterior,
        inner_boundary='CMB', outer_boundary='Surface', exterior_boundary='OuterBoundary',
        gravitational_constant, solver_options=None, tolerance=1e-10):
    """Solve -laplacian(Phi)=4*pi*G*rho for one real mode on a full ball.

    sheet_densities maps caller-specified interface labels to density
    coefficients. Sheets use uniquely owned triangular P2 trace loads.
    Volume density is rho0*(r/r_outer)**k in radius_inner<=r<=radius_outer.
    The default exterior Robin term (ell+1)*Phi/r_exterior represents exact
    single-mode decaying continuation. exterior_condition=dirichlet_exact is
    an optional comparison using the analytical exterior value; that comparison
    checks the interior solve rather than independently testing the radial kernel. The origin is included, with regularity inherent
    in the continuous Cartesian P2 space. The caller selects solver policy.
    Returned potential coefficients use explicit interface trace integrals.
    """
    ell,m,component,normalization=validate_harmonic(harmonic_degree,harmonic_order,
                                                 harmonic_component,harmonic_normalization)
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
    density=sp.Piecewise((float(volume_density)*(radius/radius_outer)**int(density_power)*harmonic,
                         sp.And(radius>=radius_inner,radius<=radius_outer)),(0.,True))
    poisson.f=4*sp.pi*gravitational_constant*density if volume_density else 0.
    if exterior_condition=='robin':
        poisson.add_natural_bc(sp.Matrix([(ell+1)*phi.sym[0]/radius_exterior]),exterior_boundary)
    elif exterior_condition=='dirichlet_exact':
        radii=dict(sheet_radii or {})
        coefficient=0.
        for label,density in dict(sheet_densities or {}).items():
            if label not in radii:raise ValueError(f'Explicit source radius required for {label!r}.')
            coefficient+=spherical_sheet_potential_coefficient(source_radius=radii[label],
                target_radius=radius_exterior,harmonic_degree=ell,
                surface_density_coefficient=density,gravitational_constant=gravitational_constant)
        if volume_density:
            coefficient+=spherical_volume_density_potential_coefficient(radius_inner=radius_inner,
                radius_outer=radius_outer,harmonic_degree=ell,density_power=density_power,
                density_amplitude=volume_density,target_radius=radius_exterior,
                gravitational_constant=gravitational_constant)
        H=solid_harmonic_sympy(list(mesh.X),degree=ell,order=m,component=component,normalization=normalization)
        poisson.add_dirichlet_bc([coefficient*radius_exterior**(ell+1)*H/radius**(2*ell+1)],exterior_boundary)
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
        for label,coefficient in dict(sheet_densities or {}).items():
            vector,report=assemble_spherical_harmonic_sheet_load(mesh=mesh,load_variable=load,
                interface=label,harmonic_degree=ell,harmonic_order=m,harmonic_component=component,
                harmonic_normalization=normalization,density_coefficient=float(coefficient),
                gravitational_constant=gravitational_constant)
            cleanup.callback(vector.destroy)
            diagnostics[label]=report
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
                inner_potential=inner,solver_reason=reason,load_diagnostics=diagnostics)


__all__=['solve_spherical_potential']
