"""Normal-jump sign, analytic gradients, and manufactured weak lifting identity."""

from importlib import import_module
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


@pytest.mark.parametrize("side", [0, -1, 1])
def test_prescribed_radial_derivative_jump_and_gradient(side):
    coordinates=CoordSys3D('JumpCoordinates');x,y=coordinates.x,coordinates.y
    mesh=SimpleNamespace(N=coordinates,X=(x,y))
    density=(x*x-y*y)/(x*x+y*y)
    lifting=NormalDerivativeJump(mesh,((1.5,density,0.),),.7,{1.5:side})
    vertices=np.array([[[1.,0.],[2.,0.],[1.,1.]]])
    lifting.prepare(mesh,vertices)
    normal=np.array([np.cos(.2),np.sin(.2)])
    points=np.array([(1.5-1e-7)*normal,(1.5+1e-7)*normal])
    gradient=lifting.gradient(0,points)
    jump=(gradient[1]-gradient[0])@normal
    if side:
        mantle_point=(1.6 if side == -1 else 1.4)*normal
        assert lifting.gradient(0,mantle_point[None])[0]==pytest.approx([0.,0.],abs=1e-14)
        assert float(lifting.expression.subs({x:mantle_point[0],y:mantle_point[1]}))==pytest.approx(0.,abs=1e-14)
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
    previous=uw.meshing.UnstructuredSimplexBox(cellSize=.5,regular=True)
    a,b=previous.X
    previous_lift=1+a+a*b
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
    assert uw.function.expressions.extract_meshes(previous_lift)=={previous}


@pytest.mark.parametrize("mode", [2,32])
def test_compact_internal_jump_gradient_and_support(mode):
    coordinates=CoordSys3D('CompactJumpCoordinates');x,y=coordinates.x,coordinates.y
    mesh=SimpleNamespace(N=coordinates,X=(x,y))
    density=sp.cos(mode*sp.atan2(y,x))
    lift=NormalDerivativeJump(mesh,((1.5,density,0.),),.7,{1.5:0},{1.5:.2})
    lift.prepare(mesh,np.array([[[1.,0.],[2.,0.],[1.,1.]]]))
    normal=np.array([np.cos(.2),np.sin(.2)])
    points=np.array([(1.5-1e-7)*normal,(1.5+1e-7)*normal])
    assert (lift.gradient(0,points)[1]-lift.gradient(0,points)[0])@normal==pytest.approx(-4*np.pi*.7*np.cos(mode*.2),abs=1e-8)
    evaluate=sp.lambdify((x,y),lift.expression,modules='numpy',dummify=True)
    for radius in (1.2,1.3,1.7,1.8):
        point=radius*normal
        assert float(evaluate(*point))==pytest.approx(0.,abs=1e-12)
        assert lift.gradient(0,point[None])[0]==pytest.approx([0.,0.],abs=1e-10)
    for radius in (1.37,1.61):
        point=radius*normal;finite=[]
        for axis in (0,1):
            delta=np.eye(2)[axis]*1e-7
            finite.append((evaluate(*(point+delta))-evaluate(*(point-delta)))/2e-7)
        assert lift.gradient(0,point[None])[0]==pytest.approx(finite,rel=1e-6,abs=1e-7)
    assert lift.break_radii==pytest.approx((1.3,1.7))


@pytest.mark.parametrize("extract_internal", [True, False])
@pytest.mark.parametrize("feedback", [0., .01])
def test_selective_extraction_preserves_internal_mass(monkeypatch, extract_internal, feedback):
    # Selection changes the known representation, never the physical sheet RHS.
    module=import_module('underworld3.postprocessing.cylindrical_potential')
    coordinates=CoordSys3D('SelectedJumpCoordinates')
    mesh=SimpleNamespace(dim=2,N=coordinates)
    captured={}

    def capture_solve(**arguments):
        captured.update(arguments)
        return object(),object()

    monkeypatch.setattr(module,'_solve_density_potential',capture_solve)
    monkeypatch.setattr(module,'_interface_coefficient',lambda *args: (0.,0.))
    result=module.solve_cylindrical_spatial_potential(gravity_mesh=mesh,field_name='Selection',
        radius_inner=1.22,radius_internal=2.,radius_outer=2.22,radius_exterior=2.3,radius_interior=1.1,
        continuation_wavenumber=2,inner_sheet_density=1.,internal_sheet_density=-3.,outer_sheet_density=2.,
        volume_density=0.,inner_feedback_density=feedback,outer_feedback_density=feedback,gravitational_constant=1.,
        potential_degree=2,tolerance=1e-10,boundary_samples=32,inner_boundary='Lower',internal_boundary='Internal',
        outer_boundary='Upper',exterior_boundary='Exterior',interior_boundary='Interior',solver_options={},
        source_integration='circular',source_quadrature=8,normal_derivative_jump=True,
        jump_lifting_support='buffer',internal_lifting_half_width=0.,extract_internal_jump=extract_internal)
    sources=captured['circular_sources']
    assert sources['sheets'][1]==(2.,-3.,0.)
    assert captured['sheets'][1]==('Internal',-3.,0)
    lifted=dict(sources['potential_lifting'].sources)
    assert (2. in lifted)==extract_internal
    assert set(lifted)==({1.22,2.,2.22}if extract_internal else{1.22,2.22})
    assert result is not None
