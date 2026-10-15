"""Modal reaction mean removal, FE normalization, and physical gravity feedback."""
from types import SimpleNamespace
import numpy as np
import pytest
import sympy as sp
import underworld3 as uw
from underworld3.postprocessing.geoid import spherical_boundary_reaction_coefficient
from underworld3.maths.spherical_harmonics import real_harmonic_values
from underworld3.analytic.spherical.gravity import spherical_physical_shell_response

pytestmark=[pytest.mark.level_1,pytest.mark.tier_a]

@pytest.mark.parametrize('ell,m,component',[(2,0,'cos'),(3,1,'cos'),(3,2,'sin')])
def test_reaction_projection_uses_discrete_norm_and_removes_mean(monkeypatch,ell,m,component):
    cosine,weights=np.polynomial.legendre.leggauss(14)
    azimuth=np.arange(32)*2*np.pi/32
    C,P=np.meshgrid(cosine,azimuth,indexing='ij')
    points=np.column_stack((np.sqrt(1-C*C).ravel()*np.cos(P).ravel(),np.sqrt(1-C*C).ravel()*np.sin(P).ravel(),C.ravel()))
    w=np.repeat(weights,32)*2*np.pi/32*.731 # a different FE measure, not the analytical area
    coords=sp.symbols('x y z',real=True)
    Y=real_harmonic_values(points,degree=ell,order=m,component=component,normalization='real_orthonormal')
    values=-.73*Y+1.91
    def integral(expr):
        a=sp.lambdify(coords,expr,'numpy')(*points.T)
        return float(w@np.broadcast_to(a,len(points)))
    class BoundaryIntegral:
        def __init__(self,mesh,fn,boundary):self.fn=fn
        def evaluate(self):return integral(self.fn)
    class Stokes:
        mesh=SimpleNamespace(dim=3,X=sp.Matrix([coords]))
        def boundary_normal_traction(self,*args):raise AssertionError('No point gathering.')
        def boundary_normal_traction_integral(self,boundary,fn,remove_mean):
            assert remove_mean
            test=sp.lambdify(coords,fn,'numpy')(*points.T)
            return float(w@((values-w@values/w.sum())*test))
    monkeypatch.setattr(uw.maths,'BdIntegral',BoundaryIntegral)
    result=spherical_boundary_reaction_coefficient(Stokes(),'Surface',harmonic_degree=ell,
        harmonic_order=m,harmonic_component=component,harmonic_normalization='real_orthonormal')
    np.testing.assert_allclose(result,-.73,atol=2e-14)

@pytest.mark.parametrize('ell',[2,4])
def test_physical_load_operator_and_feedback(ell):
    ri,ro,G=.55,1.,.01
    h=np.array([.3,-.2]);contrast=np.array([1.7,2.3]);gravity=np.array([1.4,1.8])
    # Independently normalized sheet amplitudes at target/source radii.
    radii=(ro,ri)
    K=np.array([[4*np.pi*G*contrast[j]*source/(2*ell+1)*
        ((target/source)**ell if target<=source else (source/target)**(ell+1))
        for j,source in enumerate(radii)] for target in radii])
    expected=K@h
    sol=spherical_physical_shell_response(radius_inner=ri,radius_outer=ro,harmonic_degree=ell,
        outer_topography=h[0],inner_topography=h[1],outer_density_contrast=contrast[0],
        inner_density_contrast=contrast[1],outer_gravity=gravity[0],inner_gravity=gravity[1],
        gravitational_constant=G,include_self_gravity=True)
    np.testing.assert_allclose([sol['outer_potential'],sol['inner_potential']],expected,rtol=2e-14)
    corrected=np.linalg.solve(np.eye(2)-K/gravity[:,None],h)
    np.testing.assert_allclose([sol['outer_topography_self_gravity'],sol['inner_topography_self_gravity']],corrected,rtol=2e-14)
    assert sol['matrix_residual_norm']<1e-14


def test_trace_quadrature_preserves_constant_and_linear_moments():
    from underworld3.utilities._spherical_interface_load import _triangle_p2_load
    vertices=np.array([[1.,0,0],[0,1.,0],[0,0,1.]])
    b=_triangle_p2_load(vertices,harmonic_degree=0,harmonic_order=0,harmonic_component='cos',
        harmonic_normalization='axisymmetric_unnormalized',source_amplitude=1.,quadrature_points=8)
    area=np.linalg.norm(np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0]))/2
    np.testing.assert_allclose(b.sum(),area,atol=3e-14)
    nodes=np.vstack((vertices,(vertices[0]+vertices[1])/2,(vertices[1]+vertices[2])/2,(vertices[2]+vertices[0])/2))
    np.testing.assert_allclose(b@nodes,area*vertices.mean(axis=0),atol=3e-14)
