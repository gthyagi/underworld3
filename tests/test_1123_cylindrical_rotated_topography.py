"""Spatial rotated reactions preserve chord P2 traces and one pressure datum."""

import numpy as np
import pytest
import sympy as sp
import underworld3 as uw

from underworld3.postprocessing.cylindrical_rotated import PolygonTopographyTrace

pytestmark = pytest.mark.level_2


def trace_data():
    edges = np.array([.03, .21, .7, 1.9, 3.3, 4.8, .03+2*np.pi])
    starts, widths = edges[:-1], np.diff(edges)
    vertex = np.cos(2*starts)+.13*np.sin(3*starts)+.2
    midpoint = np.sin(starts+widths/2)+.17
    values = np.column_stack((vertex, np.roll(vertex, -1), midpoint))
    owned = np.arange(len(starts)) % uw.mpi.size == uw.mpi.rank
    return starts, widths, values, owned


@pytest.mark.tier_b
def test_radial_pullback_preserves_original_nonmodal_chord_polynomial():
    starts, widths, values, owned = trace_data()
    trace = PolygonTopographyTrace(2., starts[owned], widths[owned], values[owned], uw.mpi.comm)
    t = np.array([.13, .4, .78])
    a = np.column_stack((np.cos(starts), np.sin(starts)))
    b = np.column_stack((np.cos(starts+widths), np.sin(starts+widths)))
    points = (1-t[None,:,None])*a[:,None,:]+t[None,:,None]*b[:,None,:]
    angles = np.arctan2(points[...,1], points[...,0])
    basis = np.stack(((1-t)*(1-2*t), t*(2*t-1), 4*t*(1-t)), axis=-1)
    np.testing.assert_allclose(trace.evaluate(angles), values@basis.T, rtol=0, atol=2e-14)
    assert trace.evaluate(np.empty(0)).shape == (0,)


@pytest.mark.tier_b
def test_unequal_arc_constant_trace_retains_its_mean_and_has_no_higher_modes():
    starts, widths, values, owned = trace_data()
    trace = PolygonTopographyTrace(2., starts[owned], widths[owned],
                                  np.full_like(values[owned], 1.7), uw.mpi.comm)
    for mode in (1, 2, 8, 32):
        coefficient, mean = trace.cosine_coefficient(mode, 64)
        np.testing.assert_allclose([coefficient, mean], [0., 1.7], rtol=0, atol=2e-14)


@pytest.mark.tier_b
def test_rotated_spatial_reaction_conserves_load_and_pairs_pressure_shift():
    mesh = uw.meshing.Annulus(radiusInner=1.22, radiusOuter=2.22, cellSize=.125, qdegree=3)
    u = uw.discretisation.MeshVariable("RotTopoU", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("RotTopoP", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = 1.
    stokes.bodyforce = sp.Matrix([1, 0])
    stokes.tolerance = 1e-10
    stokes.petsc_use_pressure_nullspace = True
    radial = sp.Matrix(list(mesh.CoordinateSystem.unit_e_0)).T
    stokes.add_rotated_freeslip_bc(0., "Upper", normal=radial)
    stokes.add_rotated_freeslip_bc(0., "Lower", normal=-radial)
    stokes.solve()
    snapshots = [np.asarray(field.array).copy() for field in (u, p)]
    area = uw.maths.Integral(mesh, sp.S.One).evaluate()
    pressure_shift = uw.maths.Integral(mesh, p.sym[0]).evaluate()/area
    inputs = dict(inner_boundary="Lower", outer_boundary="Upper", radius_inner=1.22,
        radius_outer=2.22, inner_density_contrast=3., outer_density_contrast=2.,
        inner_reference_gravity=5., outer_reference_gravity=7., mass="lumped")
    result = stokes.cylindrical_rotated_topography(pressure_shift=pressure_shift, **inputs)
    shifted = stokes.cylindrical_rotated_topography(pressure_shift=pressure_shift+2.5, **inputs)
    angle = np.linspace(0, 2*np.pi, 37, endpoint=False)
    np.testing.assert_allclose(shifted.inner.evaluate(angle)-result.inner.evaluate(angle), 2.5/15, atol=2e-14)
    np.testing.assert_allclose(shifted.outer.evaluate(angle)-result.outer.evaluate(angle), -2.5/14, atol=2e-14)
    nodes, weights = np.polynomial.legendre.leggauss(48)
    for trace, label, scale in ((result.inner, "Lower", -1/15), (result.outer, "Upper", 1/14)):
        theta = trace.starts[:,None]+trace.widths[:,None]*(nodes+1)/2
        centre = trace.starts[:,None]+trace.widths[:,None]/2
        jacobian = trace.radius*np.cos(trace.widths[:,None]/2)/np.cos(theta-centre)**2
        measure = jacobian*trace.widths[:,None]*weights/2
        recovered = uw.mpi.comm.allreduce(float(np.sum(trace.evaluate(theta)*measure)))
        original = stokes.boundary_normal_traction_integral(label, sp.S.One, remove_mean=False)
        length = uw.maths.BdIntegral(mesh, fn=sp.S.One, boundary=label).evaluate()
        np.testing.assert_allclose(recovered, scale*(original-pressure_shift*length), atol=2e-12)
    for field, snapshot in zip((u,p), snapshots):
        np.testing.assert_array_equal(field.array, snapshot)
