"""Known lifting and linear sheet feedback must preserve an exact zero field."""

from mpi4py import MPI
import numpy as np
import pytest
import sympy as sp

import underworld3 as uw
from underworld3.postprocessing._density_potential import _solve_density_potential

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


@pytest.mark.parametrize('inner_feedback,outer_feedback', [(.01,0.), (0.,.02), (.01,.02)])
def test_nonzero_known_lift_with_feedback_preserves_zero_physical_solution(inner_feedback, outer_feedback):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(-2.5,-2.5), maxCoords=(2.5,2.5),
        cellSize=.4, qdegree=4, regular=True)
    x,y = mesh.N.x,mesh.N.y

    class HarmonicLift:
        expression = 1+x+x*y

        def prepare(self, mesh, vertices):
            pass

        def gradient(self, cell, points):
            return np.column_stack((1+points[:,1],points[:,0]))

    # Phi=0, u=-L is exactly P2. Sheet density alpha*Phi is zero, so any
    # nonzero physical solution exposes a missing/sign-reversed alpha*L load.
    field,poisson = _solve_density_potential(gravity_mesh=mesh, field_name='FeedbackRemainder',
        potential_degree=2, volume_density=sp.S.Zero, sheets=(),
        continuation=[(name,1.)for name in('Left','Right','Top','Bottom')],
        gravitational_constant=.01, tolerance=1e-11,
        solver_options={'snes_type':'ksponly','ksp_type':'preonly','pc_type':'lu','pc_factor_mat_solver_type':'mumps'},
        circular_sources=dict(radii=(1.1,1.5,2.1),
            sheets=((1.1,0.,inner_feedback),(1.5,0.,0.),(2.1,0.,outer_feedback)),
            quadrature_order=8, potential_lifting=HarmonicLift()))
    points=np.asarray(field.coords)
    lift_values=1+points[:,0]+points[:,0]*points[:,1]
    error=float(np.max(abs(field.array[:,0,0]+lift_values),initial=0.))
    scale=float(np.max(abs(lift_values),initial=0.))
    assert uw.mpi.comm.allreduce(error,op=MPI.MAX)<1e-9*uw.mpi.comm.allreduce(scale,op=MPI.MAX)
    assert poisson.solve_report.converged
