"""Positive quadrature for circles and circular volumes cut by affine triangles.

The circle-specific construction clips polar rays against each triangle's
half-planes. Angular panels end at edge/circle intersections and vertices;
radial integration is split at prescribed material radii. Every point belongs
to its actual containing triangle, including the thin circular crescents.
It follows the geometry-resolving principle of Saye (2015),
doi:10.1137/140966290, rather than applying that paper's Cartesian algorithm.
"""

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss


@lru_cache(maxsize=16)
def _gauss(order: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(order, bool) or not isinstance(order, int) or order < 2:
        raise ValueError("Circular source quadrature order must be an integer >= 2")
    return leggauss(order)


def radial_bounds(vertices: np.ndarray) -> tuple[float, float]:
    """Minimum/maximum distance from the origin to a closed triangle."""
    edges = np.roll(vertices, -1, axis=0) - vertices
    t = np.clip(-np.sum(vertices*edges, axis=1)/np.sum(edges*edges, axis=1), 0., 1.)
    minimum = np.linalg.norm(vertices+t[:, None]*edges, axis=1).min()
    cross = edges[:, 0]*(-vertices[:, 1])-edges[:, 1]*(-vertices[:, 0])
    if np.all(cross >= 0) or np.all(cross <= 0):
        minimum = 0.
    return float(minimum), float(np.linalg.norm(vertices, axis=1).max())


def angular_panels(vertices: np.ndarray, radii: tuple[float, ...]) -> np.ndarray:
    """Split where a ray changes edge or a circle crosses a triangle edge."""
    angles = list(np.linspace(0., 2*np.pi, 33))
    angles.extend(np.mod(np.arctan2(vertices[:, 1], vertices[:, 0]), 2*np.pi))
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        d = b-a
        aa, bb = d@d, 2*(a@d)
        for radius in radii:
            cc = a@a-radius**2
            discriminant = bb*bb-4*aa*cc
            if discriminant < 0:
                continue
            for t in ((-bb-np.sqrt(discriminant))/(2*aa), (-bb+np.sqrt(discriminant))/(2*aa)):
                if 0 <= t <= 1:
                    point = a+t*d
                    angles.append(float(np.arctan2(point[1], point[0]) % (2*np.pi)))
    return np.unique(angles)


def ray_intervals(vertices: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Intersect positive polar rays with all three oriented half-planes."""
    edges = np.roll(vertices, -1, axis=0)-vertices
    determinant = np.linalg.det(np.stack((vertices[1]-vertices[0], vertices[2]-vertices[0])))
    normals = np.sign(determinant)*np.column_stack((-edges[:, 1], edges[:, 0]))
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    a, b = directions@normals.T, np.sum(normals*vertices, axis=1)
    lower, upper = np.zeros(len(theta)), np.full(len(theta), np.inf)
    for i in range(3):
        positive, negative = a[:, i] > 1e-14, a[:, i] < -1e-14
        lower[positive] = np.maximum(lower[positive], b[i]/a[positive, i])
        upper[negative] = np.minimum(upper[negative], b[i]/a[negative, i])
        if b[i] > 1e-14:
            upper[~(positive | negative)] = -1.
    return lower, upper


def circle_rule(vertices: np.ndarray, radius: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Integrate only the arcs lying inside this triangle, with ds=R*dtheta."""
    lo, hi = radial_bounds(vertices)
    if radius < lo-1e-13 or radius > hi+1e-13:
        return np.empty((0, 2)), np.empty(0)
    cuts = angular_panels(vertices, (radius,))
    left, right = cuts[:-1], cuts[1:]
    lower, upper = ray_intervals(vertices, (left+right)/2)
    keep = (lower <= radius+1e-13) & (radius <= upper+1e-13)
    left, right = left[keep], right[keep]
    x, w = _gauss(order)
    theta = ((left+right)[:, None]+(right-left)[:, None]*x)/2
    points = radius*np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    weights = radius*(right-left)[:, None]*w/2
    return points.reshape(-1, 2), weights.ravel()


def annulus_rule(vertices: np.ndarray, radii: tuple[float, ...], order: int) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a triangle intersected with Ri<r<Ro, splitting at interior radii."""
    minimum, maximum = radial_bounds(vertices)
    if maximum <= radii[0] or minimum >= radii[-1]:
        return np.empty((0, 2)), np.empty(0)
    x, w = _gauss(order)
    # Uncut cells use a positive Duffy rule; no polar work is needed.
    if all(not minimum < radius < maximum for radius in radii):
        a, b = np.meshgrid((x+1)/2, (x+1)/2, indexing="ij")
        bary = np.stack((1-a, a*(1-b), a*b), axis=-1)
        determinant = abs(np.linalg.det(np.stack((vertices[1]-vertices[0], vertices[2]-vertices[0]))))
        weights = determinant*a*w[:, None]*w[None, :]/4
        return (bary@vertices).reshape(-1, 2), weights.ravel()
    cuts = angular_panels(vertices, radii)
    left, right = cuts[:-1], cuts[1:]
    theta = ((left+right)[:, None]+(right-left)[:, None]*x)/2
    angular_weights = (right-left)[:, None]*w/2
    theta, angular_weights = theta.ravel(), angular_weights.ravel()
    lower, upper = ray_intervals(vertices, theta)
    points, weights = [], []
    for r0, r1 in zip(radii[:-1], radii[1:]):
        lo, hi = np.maximum(lower, r0), np.minimum(upper, r1)
        keep = hi > lo
        radial = ((lo[keep]+hi[keep])[:, None]+(hi[keep]-lo[keep])[:, None]*x)/2
        measure = (hi[keep]-lo[keep])[:, None]*w/2*radial*angular_weights[keep, None]
        directions = np.column_stack((np.cos(theta[keep]), np.sin(theta[keep])))
        points.append((radial[:, :, None]*directions[:, None]).reshape(-1, 2))
        weights.append(measure.ravel())
    return np.concatenate(points), np.concatenate(weights)
