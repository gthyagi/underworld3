r"""Physical-sphere topography from the coupled Stokes multiplier.

The scalar free-slip trace and mapped Cartesian zero-slip trace are the spaces
defined by :mod:`underworld3.utilities._spherical_stokes_gap`. This adapter
preserves those spaces and their common physical-pressure gauge. It performs
no stress differentiation or spherical-harmonic reconstruction.

The weak normal constraint follows Urquiza, Garon and Farinas (2014),
doi:10.1016/j.jcp.2013.08.045; the gap-work principle follows Collins et al.
(2025), arXiv:2508.09613. Their results do not establish stability of this UW3
three-dimensional mapped multiplier space. The traction-to-displacement balance
and gravity conventions are documented in ``analytic.spherical.gravity``.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
from mpi4py import MPI

from underworld3.maths.spherical_harmonics import real_harmonic_values
from underworld3.utilities._spherical_gap import _spherical_facet_quadrature
from underworld3.utilities._spherical_stokes_gap import _canonical_rows, _wall_faces
from underworld3.utilities._spherical_wall_trace import _wall_cone_index

__all__ = ["SphericalTopographyTrace", "SphericalGapTopography", "spherical_gap_topography"]


def _triangle_p2(barycentric):
    """P2 basis in vertex, 01, 02, 12 edge ordering."""
    a, b, c = np.moveaxis(barycentric, -1, 0)
    return np.stack((a*(2*a-1), b*(2*b-1), c*(2*c-1), 4*a*b, 4*a*c, 4*b*c), axis=-1)


@dataclass(frozen=True)
class SphericalTopographyTrace:
    """Owned original facets and their scalar or mapped-vector reaction data.

    ``coefficients`` has shape (facets, 6, 1) for free slip or (facets, 6, 3)
    for zero slip. ``buoyancy_scale`` converts physical normal reaction to
    radial-outward displacement; its sign is negative on the inner sphere.
    Only rank bounding boxes and requested point values are communicated.
    All evaluation and integration calls are collective, including empty ranks.
    """

    radius: float
    facets: np.ndarray
    coefficients: np.ndarray
    normal_sign: int
    buoyancy_scale: float
    comm: object

    def _values(self, facet, barycentric):
        values = _triangle_p2(barycentric) @ self.coefficients[facet]
        if values.shape[-1] == 1:
            normal = values[..., 0]
        else:
            planar = barycentric @ self.facets[facet]
            distance = np.linalg.norm(planar, axis=-1)
            normal = np.sum(values*planar, axis=-1)*self.normal_sign*self.radius/distance**2
        return normal*self.buoyancy_scale

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the original trace along Cartesian radial directions.

        Radial projection also accepts chord midpoints on a gravity mesh.
        Exact cone containment chooses facets; proximity never chooses a trace.
        Sparse bounded point exchanges avoid replicating the spherical surface.
        """
        points = np.asarray(points, dtype=float)
        invalid = points.ndim != 2 or points.shape[1:] != (3,)
        if not invalid:
            invalid = not np.isfinite(points).all() or np.any(np.linalg.norm(points, axis=1) == 0)
        if self.comm.allreduce(invalid, op=MPI.LOR):
            raise ValueError("Spherical trace queries require finite nonzero (N, 3) coordinates on every rank")
        directions = points/np.linalg.norm(points, axis=1)[:, None]
        index = _wall_cone_index(self.facets) if len(self.facets) else None
        box = (np.stack((index[0].min(axis=0), index[1].max(axis=0))) if index is not None
               else np.array([[np.inf]*3, [-np.inf]*3]))
        boxes = self.comm.allgather(box)
        result = np.full(len(points), np.nan)
        tolerance = 1e-11
        batches = self.comm.allreduce((len(points)+2047)//2048, op=MPI.MAX)
        for batch in range(batches):
            first = 2048*batch
            local = directions[first:first+2048]
            queries = []
            for lower, upper in boxes:
                selected = np.flatnonzero(np.all((local >= lower-tolerance) & (local <= upper+tolerance), axis=1))
                queries.append((first+selected, local[selected]))
            responses = []
            for ids, requested in self.comm.alltoall(queries):
                records = []
                if index is not None:
                    _, _, pitch, bins = index
                    for point_id, direction in zip(ids, requested, strict=True):
                        lo = np.floor((direction-tolerance)/pitch).astype(int)
                        hi = np.floor((direction+tolerance)/pitch).astype(int)
                        candidates = set()
                        for key in product(*(range(lo[d], hi[d]+1) for d in range(3))):
                            candidates.update(bins.get(key, ()))
                        for facet in sorted(candidates):
                            radial = np.linalg.solve(self.facets[facet].T, direction)
                            total = radial.sum()
                            if total > 0 and np.min(radial/total) >= -tolerance:
                                value = float(self._values(facet, radial/total))
                                records.append((int(point_id), value))
                                break
                responses.append(records)
            for records in self.comm.alltoall(responses):
                for point_id, value in records:
                    if np.isnan(result[point_id]):
                        result[point_id] = value
                    elif not np.isclose(result[point_id], value, rtol=1e-10, atol=1e-12):
                        result[point_id] = np.inf
        failures = self.comm.allreduce(int(np.count_nonzero(~np.isfinite(result))), op=MPI.SUM)
        if failures:
            raise ArithmeticError(f"Spherical trace has {failures} missing or inconsistent facet queries")
        return result

    def quadrature_samples(self, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return owned physical-sphere points, area weights and topography."""
        rule = _spherical_facet_quadrature(self.facets, self.radius, order, 1)
        planar = rule["planar_points"]
        values = np.empty(planar.shape[:2])
        for facet in range(len(self.facets)):
            barycentric = np.linalg.solve(self.facets[facet].T, planar[facet].T).T
            values[facet] = self._values(facet, barycentric)
        return rule["surface_points"], rule["surface_weights"], values

    def harmonic_coefficient(self, degree: int, order: int, component: str,
                             normalization: str, quadrature: int) -> tuple[float, float]:
        """Integrate one real harmonic coefficient and the unfiltered mean."""
        points, weights, values = self.quadrature_samples(quadrature)
        harmonic = real_harmonic_values(points.reshape(-1, 3), degree=degree, order=order,
                                       component=component, normalization=normalization).reshape(weights.shape)
        totals = self.comm.allreduce(np.array([np.sum(weights*values*harmonic),
            np.sum(weights*harmonic**2), np.sum(weights*values), np.sum(weights)]), op=MPI.SUM)
        return float(totals[0]/totals[1]), float(totals[2]/totals[3])


@dataclass(frozen=True)
class SphericalGapTopography:
    """Both physical-sphere displacement traces and their original reaction."""

    inner: SphericalTopographyTrace
    outer: SphericalTopographyTrace
    normal_reaction: object


def spherical_gap_topography(*, stokes, gap_result, inner_boundary: str, outer_boundary: str,
    radius_inner: float, radius_outer: float, inner_density_contrast: float,
    outer_density_contrast: float, inner_reference_gravity: float,
    outer_reference_gravity: float) -> SphericalGapTopography:
    r"""Convert this solver's physical normal reaction to spatial topography.

    Density contrasts are smaller-radius density minus larger-radius density;
    gravity magnitudes are positive. With :math:`\lambda=-\sigma_{nn}`,
    radial-outward displacement is :math:`h_o=\lambda_o/(\Delta\rho_o g_o)` and
    :math:`h_i=-\lambda_i/(\Delta\rho_i g_i)`. The completed coupled solve has
    already applied the joint physical-pressure/reaction gauge. No additional
    pressure offset or independent wall-mean removal is performed.
    """
    mesh = stokes.mesh
    scales = np.array([inner_density_contrast, outer_density_contrast,
                       inner_reference_gravity, outer_reference_gravity], dtype=float)
    if mesh.dim != 3 or not 0 < radius_inner < radius_outer:
        raise ValueError("Require a three-dimensional mesh and ordered positive sphere radii")
    if not np.isfinite(scales).all() or np.any(scales[:2] == 0) or np.any(scales[2:] <= 0):
        raise ValueError("Require finite nonzero density contrasts and positive gravity magnitudes")
    normal = gap_result.normal_reaction
    if normal is not getattr(stokes, "_spherical_gap_normal_reaction", None):
        raise ValueError("Use the normal reaction returned by this Stokes spherical-gap solve")
    if gap_result.checks["relative_residual"] > 1e-9:
        raise ArithmeticError("The spherical-gap solution has not met its residual contract")
    components = gap_result.checks["constraint_components"]
    vertices = np.asarray(mesh._get_coords_for_basis(1, True))[mesh._cell_node_indices(1, True)]
    rows = _canonical_rows(mesh, normal, vertices)
    owned = np.flatnonzero(mesh._get_owned_cells_mask())
    traces = []
    for boundary, radius, sign, scale in (
        (inner_boundary, radius_inner, -1, -1/(scales[0]*scales[2])),
        (outer_boundary, radius_outer, 1, 1/(scales[1]*scales[3])),
    ):
        wall = _wall_faces(mesh, vertices, rows, owned, boundary, radius)
        indices = wall["rows"]
        values = np.asarray(normal.array)[indices, 0, 0, None]
        if components == 3:
            tangent_rows = _canonical_rows(mesh, gap_result.tangential_reaction, vertices)
            tangential = np.take_along_axis(tangent_rows[wall["cells"]], wall["positions"], axis=1)
            nodes = np.asarray(normal.coords)[indices]
            distance = np.linalg.norm(nodes, axis=-1)
            normals = sign*nodes/distance[..., None]
            values = (values*normals + np.asarray(gap_result.tangential_reaction.array)[tangential, 0, :])*distance[..., None]/radius
        traces.append(SphericalTopographyTrace(radius, wall["facets"].copy(), values.copy(),
                      sign, scale, mesh.dm.comm.tompi4py()))
    return SphericalGapTopography(traces[0], traces[1], normal)
