"""Physical-circle reaction signs, angle-P2 transfer and MPI ownership."""

import numpy as np
import pytest
import sympy as sp
import underworld3 as uw
from mpi4py import MPI

from underworld3.postprocessing.cylindrical_gap import (
    CircularTopographyTrace, cylindrical_gap_topography,
)

pytestmark = pytest.mark.level_2


def test_angle_trace_preserves_nonmodal_data_on_unequal_arcs():
    """Direct transfer retains a nonharmonic piecewise-P2 trace, including empty ranks."""
    edges = np.array([.03, .1, .4, 1., 2., 4., 5., .03+2*np.pi])
    starts, widths = edges[:-1], np.diff(edges)
    nodes = starts[:, None]+widths[:, None]*np.array([0., 1., .5])
    values = np.cos(2*nodes)+.17*np.sin(3*nodes)+.11
    owned = np.arange(len(starts)) % uw.mpi.size == uw.mpi.rank
    trace = CircularTopographyTrace(2., starts[owned], widths[owned], values[owned], MPI.COMM_WORLD)
    for j in range(3):
        np.testing.assert_allclose(trace.evaluate(nodes[:, j]), values[:, j], atol=3e-15)
    assert trace.evaluate(np.empty(0)).shape == (0,)
    # Test the complete trace integral against independent dense sampling.
    theta = np.linspace(0, 2*np.pi, 100000, endpoint=False)
    sampled = trace.evaluate(theta)
    coefficient, mean = trace.cosine_coefficient(2, 32)
    np.testing.assert_allclose([coefficient, mean], [2*np.mean(sampled*np.cos(2*theta)), np.mean(sampled)], atol=2e-9)


@pytest.mark.tier_b
@pytest.mark.parametrize("zero_slip", [False, True])
def test_pressure_gradient_reactions_give_signed_topography_without_changing_stokes(zero_slip):
    """u=0,p=x gives lambda=x at both walls and opposite topography signs."""
    mesh = uw.meshing.Annulus(radiusInner=1.22, radiusOuter=2.22, cellSize=.125, qdegree=3)
    velocity = uw.discretisation.MeshVariable("TopoU", mesh, 2, degree=2)
    pressure = uw.discretisation.MeshVariable("TopoP", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = 1.
    stokes.bodyforce = sp.Matrix([1, 0])
    solve = stokes.solve_cylindrical_gap_zero_slip if zero_slip else stokes.solve_cylindrical_gap
    result = solve(inner_boundary="Lower", outer_boundary="Upper", radius_inner=1.22,
                   radius_outer=2.22, viscosity=1., pressure_jump=sp.S.Zero,
                   pressure_jump_integral=0., cap_bodyforce=sp.Matrix([1, 0]), quadrature=12,
                   linear_solver_options={"ksp_type":"preonly", "pc_type":"lu", "pc_factor_mat_solver_type":"mumps"})
    snapshots = [np.asarray(f.array).copy() for f in (velocity, pressure, result.normal_reaction)]
    topography = cylindrical_gap_topography(stokes=stokes, gap_result=result,
        inner_boundary="Lower", outer_boundary="Upper", radius_inner=1.22, radius_outer=2.22,
        inner_density_contrast=3., outer_density_contrast=2.,
        inner_reference_gravity=5., outer_reference_gravity=7.)
    inner, inner_mean = topography.inner.cosine_coefficient(1, 16)
    outer, outer_mean = topography.outer.cosine_coefficient(1, 16)
    np.testing.assert_allclose([inner, outer], [-1.22/15, 2.22/14], rtol=5e-7, atol=1e-10)
    np.testing.assert_allclose([inner_mean, outer_mean], 0, atol=2e-8)
    for field, old in zip((velocity, pressure, result.normal_reaction), snapshots):
        np.testing.assert_array_equal(np.asarray(field.array), old)
