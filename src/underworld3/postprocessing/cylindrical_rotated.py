r"""Spatial topography from affine annulus rotated-constraint reactions.

Consistent boundary-flux recovery follows Zhong, Gurnis & Hulbert (1993),
doi:10.1016/0031-9201(93)90078-N. The assembled constraint load is
:math:`\lambda=-\sigma_{nn}` in this rotated backend. This adapter retains that
sign explicitly, rather than the legacy weak API's stress wording. It uses the
existing boundary-mass recovery and no projected volume stresses.

The chord-P2 field is evaluated by exact radial pullback to each physical circle:
:math:`t=\frac12[1+\tan(\theta-\theta_c)/\tan(\Delta\theta/2)]`.
This is a coordinate map, not a new curved-wall boundary condition or an angular
P2 approximation. All numerical modes remain available for spatial Poisson loads.
"""

from dataclasses import dataclass

import numpy as np

from .cylindrical_gap import CircularTopographyTrace, _p2_basis
from underworld3.utilities._cylindrical_dirichlet import _edge_key, _labelled_edge_keys
from underworld3.utilities.boundary_flux import _desmear, _key
from underworld3.utilities.rotated_bc import _wall_normal_loads


class PolygonTopographyTrace(CircularTopographyTrace):
    """Owned chord-P2 traces sampled on a circle by radial projection.

    The endpoints lie on the declared circle. Interpolation uses the chord's
    original P2 coordinate, including its midpoint, rather than fitting a
    polynomial in angle. Collective evaluation gathers only boundary arcs;
    quadrature reduces scalar moments, including ranks with no owned arcs.
    """

    def _quadrature_values(self, t):
        chord_t = (1+np.tan(self.widths[:,None]*(t-.5))/np.tan(self.widths[:,None]/2))/2
        return np.einsum("eqb,eb->eq", _p2_basis(chord_t), self.values)

    def _interpolation_coordinate(self, theta, starts, widths):
        return (1+np.tan(theta-starts-widths/2)/np.tan(widths/2))/2


@dataclass(frozen=True)
class CylindricalRotatedTopography:
    """Both spatial traces with one explicitly paired pressure-datum shift."""

    inner: PolygonTopographyTrace
    outer: PolygonTopographyTrace
    pressure_shift: float
    mass: str
    checks: dict


def _cylindrical_rotated_topography(*, stokes, inner_boundary, outer_boundary,
        radius_inner, radius_outer, inner_density_contrast, outer_density_contrast,
        inner_reference_gravity, outer_reference_gravity, pressure_shift, mass):
    """Recover original rotated loads without independent wall-mean removal."""
    mesh = stokes.mesh
    info = stokes._rotated_freeslip_info
    if mesh.dim != 2 or stokes.u.degree != 2:
        raise ValueError("Rotated annulus topography requires 2-D affine P2 velocity traces")
    if info is None or not info["converged"]:
        raise ValueError("Complete a converged rotated Stokes solve before reaction recovery")
    scales = np.asarray([inner_density_contrast, outer_density_contrast,
                         inner_reference_gravity, outer_reference_gravity, pressure_shift], dtype=float)
    if (not np.all(np.isfinite(scales)) or np.any(scales[:2] == 0)
        or np.any(scales[2:4] <= 0) or not 0 < radius_inner < radius_outer):
        raise ValueError("Require ordered radii, nonzero contrasts, positive gravity and a finite pressure shift")
    vertices = np.asarray(mesh._get_coords_for_basis(1, True))[mesh._cell_node_indices(1, True)]
    owned = np.flatnonzero(mesh._get_owned_cells_mask())
    radii = np.linalg.norm(vertices, axis=-1)
    traces = []
    for boundary, radius, scale in (
        (inner_boundary, radius_inner, -1/(inner_density_contrast*inner_reference_gravity)),
        (outer_boundary, radius_outer, 1/(outer_density_contrast*outer_reference_gravity)),
    ):
        xs, loads, _ = _wall_normal_loads(stokes, boundary, info)
        reaction = _desmear(stokes, boundary, xs, loads, mass,
                            remove_mean=False, partial_reaction=False)
        values_by_coordinate = {_key(x, 2):scale*(value-pressure_shift) for x,value in zip(xs, reaction)}
        keys = _labelled_edge_keys(mesh, boundary, radius, radius*1e-10)
        starts, widths, values = [], [], []
        for i,j in ((0,1), (1,2), (2,0)):
            selected = owned[np.all(abs(radii[owned][:,[i,j]]-radius)<1e-10, axis=1)]
            for cell in selected:
                edge = vertices[cell,[i,j]]
                if _edge_key(edge, radius*1e-10) not in keys:
                    continue
                a,b = np.arctan2(edge[:,1], edge[:,0])
                width = np.arctan2(np.sin(b-a), np.cos(b-a))
                if width < 0:
                    edge = edge[::-1]
                midpoint = edge.mean(axis=0)
                starts.append(float(np.mod(a if width > 0 else b, 2*np.pi)))
                widths.append(abs(width))
                values.append([values_by_coordinate[_key(x,2)] for x in (edge[0],edge[1],midpoint)])
        traces.append(PolygonTopographyTrace(radius, np.asarray(starts), np.asarray(widths),
            np.asarray(values).reshape(-1,3), mesh.dm.comm.tompi4py()))
    checks = dict(converged=bool(info["converged"]), ksp_reason=int(info["ksp_reason"]),
        fresh_projected_rnorm=float(info["fresh_projected_rnorm"]),
        fresh_reference_rnorm=float(info["fresh_reference_rnorm"]),
        fresh_residual_passed=bool(info["fresh_projected_residual_passed"]),
        ranks=mesh.dm.comm.getSize(), residual_definition="Original rotated projected momentum residual")
    return CylindricalRotatedTopography(traces[0], traces[1], float(pressure_shift), mass, checks)
