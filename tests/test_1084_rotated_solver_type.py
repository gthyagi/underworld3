"""Public solver policy preserves quadratic flow and invalidates warm workspaces."""
import os
import numpy as np
import pytest
import sympy as sp
import underworld3 as uw


def test_rotated_solver_policy_on_exact_quadratic_flow():
    root_pid=uw.mpi.comm.bcast(os.getpid() if uw.mpi.rank==0 else None,root=0)
    mesh=uw.meshing.UnstructuredSimplexBox(minCoords=(0.,0.),maxCoords=(1.,1.),
        cellSize=.25,regular=True,qdegree=2,filename=f'/tmp/uw-rotated-policy-{root_pid}.msh')
    u=uw.discretisation.MeshVariable('PolicyU',mesh,2,degree=2,vtype=uw.VarType.VECTOR)
    p=uw.discretisation.MeshVariable('PolicyP',mesh,1,degree=1)
    stokes=uw.systems.Stokes(mesh,velocityField=u,pressureField=p)
    stokes.constitutive_model=uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0=1.
    # u=(y-y^2,0), p=0: div(u)=0 and div(sigma)+(2,0)=0.
    stokes.bodyforce=sp.Matrix([[2.,0.]])
    x,y=mesh.X
    stokes.add_dirichlet_bc((0.,0.),'Bottom')
    stokes.add_rotated_freeslip_bc(0.,'Top',normal=sp.Matrix([[0.,1.]]))
    # Native natural data is minus the prescribed physical traction.
    stokes.add_natural_bc(sp.Matrix([[1.,0.]]),'Top')
    stokes.add_natural_bc(sp.Matrix([[0.,1.-2*y]]),'Left')
    stokes.add_natural_bc(sp.Matrix([[0.,2*y-1.]]),'Right')
    # Side traction fixes the pressure reference; its exact value is zero.
    stokes.petsc_use_pressure_nullspace=False
    stokes.petsc_options['snes_type']='ksponly'
    stokes.tolerance=1e-8
    assert stokes.rotated_solver_type=='iterative'
    stokes.solve(zero_init_guess=True)
    assert stokes.solve_report.converged
    np.testing.assert_allclose(u.data[:,0],u.coords[:,1]-u.coords[:,1]**2,rtol=0.,atol=1e-8)
    np.testing.assert_allclose(u.data[:,1],0.,rtol=0.,atol=1e-8)
    cache=stokes._rotated_linear_cache
    for invalid in ('lu','',None):
        with pytest.raises(ValueError,match='iterative or direct'):
            stokes.rotated_solver_type=invalid
        assert stokes._rotated_linear_cache is cache
    if uw.mpi.size==1:
        stokes.rotated_solver_type='direct'
        assert stokes._rotated_linear_cache is None
        stokes.solve(zero_init_guess=True)
        assert stokes.solve_report.converged
        np.testing.assert_allclose(u.data[:,0],u.coords[:,1]-u.coords[:,1]**2,rtol=0.,atol=1e-8)
        np.testing.assert_allclose(u.data[:,1],0.,rtol=0.,atol=1e-8)
    else:
        with pytest.raises(ValueError,match='restricted to serial'):
            stokes.rotated_solver_type='direct'
        assert stokes._rotated_linear_cache is cache
    if uw.mpi.size==1:
        # Conditions added after choosing direct must also be checked at solve.
        stokes.add_dirichlet_bc((y,0.),'Left')
        with pytest.raises(ValueError,match='homogeneous essential'):
            stokes.solve(zero_init_guess=True)
        with pytest.raises(ValueError,match='homogeneous essential'):
            stokes.rotated_solver_type='direct'
    stokes.rotated_solver_type='iterative'
    assert stokes._rotated_linear_cache is None
    assert stokes.tolerance==1e-8
