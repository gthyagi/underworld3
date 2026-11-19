"""Known lifting and linear sheet feedback must preserve an exact zero field."""

from types import SimpleNamespace

from mpi4py import MPI
import numpy as np
import pytest
import sympy as sp
from sympy.vector import CoordSys3D

import underworld3 as uw
from underworld3.postprocessing._density_potential import _solve_density_potential
from underworld3.postprocessing._circular_density_potential import _density

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


def test_distance_lifting_compiles_with_real_cartesian_coordinates():
    coordinates=CoordSys3D('DistanceLiftCoordinates')
    mesh=SimpleNamespace(N=coordinates,X=(coordinates.x,coordinates.y))
    x,y=mesh.X
    radius=sp.sqrt(x*x+y*y)
    density=_density((x*x+y*y)*sp.Abs(radius-sp.Rational(3,2)),mesh,
        np.array([[[1.,0.],[2.,0.],[1.,1.]]]))
    points=np.array([[1.2,.1],[1.8,.2]])
    radii=np.linalg.norm(points,axis=1)
    expected=radii**2*abs(radii-1.5)
    gradient=2*points*abs(radii-1.5)[:,None]+radii[:,None]*np.sign(radii-1.5)[:,None]*points
    assert density(0,points)==pytest.approx(expected,rel=1e-12)
    assert density.gradient(0,points)==pytest.approx(gradient,rel=1e-12)
