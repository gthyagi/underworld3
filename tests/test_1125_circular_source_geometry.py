"""Physical circular measures and polynomial moments across arbitrary triangles."""

import numpy as np
import pytest
from scipy.spatial import Delaunay
from underworld3.utilities._circular_source_geometry import annulus_rule, circle_rule

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.mark.parametrize("order", [8, 12])
@pytest.mark.parametrize("rotation", [0., .173])
def test_physical_area_arcs_and_moments(order, rotation):
    # Nonconforming grid: neither physical circle is a polygon edge/tag.
    x, y = np.meshgrid(np.linspace(-2.5, 2.5, 10), np.linspace(-2.5, 2.5, 10))
    nodes = np.column_stack((x.ravel(), y.ravel()))
    c, s = np.cos(rotation), np.sin(rotation)
    nodes = nodes@np.array([[c, -s], [s, c]])
    triangles = nodes[Delaunay(nodes).simplices]
    area, radial_moment, length, harmonic = 0., 0., 0., 0.
    for triangle in triangles:
        points, weights = annulus_rule(triangle, (1.1, 1.5, 2.1), order)
        area += weights.sum()
        radial_moment += weights@(np.sum(points*points, axis=1)**4)
        points, weights = circle_rule(triangle, 1.5, order)
        length += weights.sum()
        theta = np.arctan2(points[:, 1], points[:, 0])
        harmonic += weights@np.cos(32*theta)
        if len(points):
            bary = np.linalg.solve((triangle[1:]-triangle[0]).T, (points-triangle[0]).T).T
            assert np.min(bary) >= -1e-11 and np.max(bary.sum(axis=1)) <= 1+1e-11
    assert area == pytest.approx(np.pi*(2.1**2-1.1**2), abs=2e-10)
    assert radial_moment == pytest.approx(2*np.pi*(2.1**10-1.1**10)/10, rel=2e-9)
    assert length == pytest.approx(3*np.pi, abs=2e-12)
    assert abs(harmonic) < 2e-9
