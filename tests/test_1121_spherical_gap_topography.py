"""Physical-sphere multiplier pullbacks and unfiltered spatial source transfer."""

from itertools import product

import numpy as np
import pytest
import underworld3 as uw

from underworld3.postprocessing.spherical_gap import SphericalTopographyTrace

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _octahedron():
    return np.array([np.diag(signs) for signs in product((-1., 1.), repeat=3)])


@pytest.mark.parametrize("components", [1, 3])
@pytest.mark.parametrize("sign", [-1, 1])
def test_constant_physical_normal_reaction_and_opposite_buoyancy_signs(components, sign):
    radius = 1.7
    facets = radius*_octahedron()
    nodes = np.concatenate((facets, (facets[:, [0, 0, 1]]+facets[:, [1, 2, 2]])/2), axis=1)
    coefficients = np.ones((8, 6, 1)) if components == 1 else sign*nodes/radius
    if uw.mpi.rank != 0:
        facets, coefficients = facets[:0], coefficients[:0]
    trace = SphericalTopographyTrace(radius, facets, coefficients, sign,
                                     sign/(3.*5.), uw.mpi.comm)
    points = radius*np.random.default_rng(31).normal(size=(53, 3))
    np.testing.assert_allclose(trace.evaluate(points), sign/15., rtol=2e-14, atol=1e-15)
    coefficient, mean = trace.harmonic_coefficient(2, 1, "cos", "real_orthonormal", 24)
    assert abs(coefficient) < 1e-13
    np.testing.assert_allclose(mean, sign/15., rtol=2e-14)
    assert trace.evaluate(np.empty((0, 3))).shape == (0,)


def test_nonmodal_p2_trace_is_transferred_without_harmonic_reconstruction():
    facets = _octahedron()
    nodes = np.concatenate((facets, (facets[:, [0, 0, 1]]+facets[:, [1, 2, 2]])/2), axis=1)
    coefficients = (1+nodes[..., 0]+.2*nodes[..., 1]*nodes[..., 2])[..., None]
    if uw.mpi.rank != 0:
        facets, coefficients = facets[:0], coefficients[:0]
    trace = SphericalTopographyTrace(1., facets, coefficients, 1, 1., uw.mpi.comm)
    triangles = _octahedron()
    bary = np.array([[.1, .2, .7], [.6, .3, .1], [1., 0., 0.], [0., .5, .5]])
    chord = np.einsum("qi,fid->fqd", bary, triangles).reshape(-1, 3)
    expected = 1+chord[:, 0]+.2*chord[:, 1]*chord[:, 2]
    # Query rank counts differ, including ranks with no owned surface or queries.
    selected = np.arange(len(chord)) % uw.mpi.size == uw.mpi.rank
    np.testing.assert_allclose(trace.evaluate(chord[selected]), expected[selected], rtol=2e-14, atol=1e-14)


def test_invalid_query_is_rejected_collectively():
    facets = _octahedron() if uw.mpi.rank == 0 else np.empty((0, 3, 3))
    trace = SphericalTopographyTrace(1., facets, np.ones((len(facets), 6, 1)), 1, 1., uw.mpi.comm)
    points = np.zeros((1, 3)) if uw.mpi.rank == uw.mpi.size-1 else np.empty((0, 3))
    with pytest.raises(ValueError, match="nonzero"):
        trace.evaluate(points)
