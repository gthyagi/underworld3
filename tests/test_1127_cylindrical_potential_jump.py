"""Normal-jump sign, analytic gradients, and manufactured weak lifting identity."""

from types import SimpleNamespace

from mpi4py import MPI
import numpy as np
import pytest
import sympy as sp
from sympy.vector import CoordSys3D
import underworld3 as uw
from underworld3.postprocessing._cylindrical_potential_jump import NormalDerivativeJump
from underworld3.postprocessing._density_potential import _solve_density_potential

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def test_prescribed_radial_derivative_jump_and_gradient():
    coordinates=CoordSys3D('JumpCoordinates');x,y=coordinates.x,coordinates.y
    mesh=SimpleNamespace(N=coordinates,X=(x,y))
    density=(x*x-y*y)/(x*x+y*y)
    lifting=NormalDerivativeJump(mesh,((1.5,density,0.),),.7)
    vertices=np.array([[[1.,0.],[2.,0.],[1.,1.]]])
    lifting.prepare(mesh,vertices)
    normal=np.array([np.cos(.2),np.sin(.2)])
    points=np.array([(1.5-1e-7)*normal,(1.5+1e-7)*normal])
    gradient=lifting.gradient(0,points)
    jump=(gradient[1]-gradient[0])@normal
    assert jump==pytest.approx(-4*np.pi*.7*np.cos(.4),abs=1e-8)
    evaluate=sp.lambdify((x,y),lifting.expression,modules='numpy',dummify=True)
    for point in np.array([[1.3,.1],[1.6,.2]]):
        fd=[]
        for axis in (0,1):
            delta=np.eye(2)[axis]*1e-6
            fd.append((evaluate(*(point+delta))-evaluate(*(point-delta)))/2e-6)
        assert lifting.gradient(0,point[None])[0]==pytest.approx(fd,rel=1e-7,abs=1e-8)


def test_quadratic_harmonic_lifting_preserves_zero_physical_solution():
    # Phi=0, L=1+x+xy: regular Phi=-L lies exactly in P2. This tests the
    # volume-gradient subtraction and nonzero Robin lift on the full box.
    mesh=uw.meshing.UnstructuredSimplexBox(minCoords=(-2.5,-2.5),maxCoords=(2.5,2.5),cellSize=.4,qdegree=4,regular=True)
    x,y=mesh.N.x,mesh.N.y

    class KnownHarmonicLift:
        expression=1+x+x*y

        def prepare(self, mesh, vertices):
            pass

        def gradient(self, cell, points):
            return np.column_stack((1+points[:,1],points[:,0]))

    field,poisson=_solve_density_potential(gravity_mesh=mesh,field_name='RegularHarmonic',potential_degree=2,
        volume_density=sp.S.Zero,sheets=(),continuation=[(name,1.)for name in ('Left','Right','Top','Bottom')],
        gravitational_constant=.01,tolerance=1e-11,
        solver_options={'snes_type':'ksponly','ksp_type':'preonly','pc_type':'lu','pc_factor_mat_solver_type':'mumps'},
        circular_sources=dict(radii=(1.1,1.5,2.1),sheets=((1.1,0.,0.),(1.5,0.,0.),(2.1,0.,0.)),quadrature_order=8,
                              potential_lifting=KnownHarmonicLift()))
    points=np.asarray(field.coords)
    error=float(np.max(abs(np.asarray(field.data)[:,0]+1+points[:,0]+points[:,0]*points[:,1]),initial=0.))
    assert uw.mpi.comm.allreduce(error,op=MPI.MAX)<1e-9
    assert poisson.solve_report.converged
