"""A Robin Jacobian must differentiate the residual, including in KSP-only solves."""
from contextlib import ExitStack
import numpy as np
import pytest
import sympy as sp
from mpi4py import MPI
import underworld3 as uw


@pytest.mark.level_1
@pytest.mark.tier_a
def test_robin_quadratic_solution_and_residual_derivative():
    # A P2-exact manufactured field on a flat boundary isolates assembly from
    # spherical faceting, harmonic projection and gravity-source quadrature.
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0., 0.), maxCoords=(1., 1.),
        cellSize=1/4, qdegree=4, regular=True)
    field = uw.discretisation.MeshVariable('RobinQuadratic', mesh, 1, degree=2)
    x, y = mesh.X
    exact = 1+x+x*x
    poisson = uw.systems.Poisson(mesh, u_Field=field)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.
    poisson.f = -2.
    poisson.add_dirichlet_bc([exact], 'Left')
    # du/dn=3 on Right, u=3; -du/dn=3*u-12.
    poisson.add_natural_bc(sp.Matrix([3*field.sym[0]-12]), 'Right')
    poisson.petsc_options['snes_type'] = 'ksponly'
    poisson.petsc_options['ksp_type'] = 'gmres'
    poisson.petsc_options['pc_type'] = 'jacobi'
    poisson.tolerance = 1e-11
    poisson.solve()
    assert poisson.solve_report.converged
    local_error = float(np.max(np.abs(field.data[:, 0]-(1+field.coords[:, 0]+field.coords[:, 0]**2)), initial=0.))
    assert mesh.dm.getComm().tompi4py().allreduce(local_error, op=MPI.MAX) < 1e-9
    with ExitStack() as cleanup:
        base = poisson.snes.getSolution().copy()
        cleanup.callback(base.destroy)
        direction = base.duplicate(); cleanup.callback(direction.destroy)
        direction.set(1.)
        perturbed = base.copy(); cleanup.callback(perturbed.destroy)
        perturbed.axpy(1., direction)
        residual = base.duplicate(); cleanup.callback(residual.destroy)
        shifted = base.duplicate(); cleanup.callback(shifted.destroy)
        action = base.duplicate(); cleanup.callback(action.destroy)
        poisson.snes.computeFunction(base, residual)
        poisson.snes.computeFunction(perturbed, shifted)
        J, P = poisson.snes.getJacobian()[:2]
        poisson.snes.computeJacobian(base, J, P)
        J.mult(direction, action)
        shifted.axpy(-1., residual)
        scale = shifted.norm()
        shifted.axpy(-1., action)
        assert shifted.norm()/scale < 1e-10
