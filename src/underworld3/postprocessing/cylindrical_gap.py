r"""Topography from the coupled physical-circle Stokes multiplier.

The gap solve returns lambda=-sigma_nn in the physical pressure gauge.
For radial-outward displacement, h_outer=lambda/(Delta_rho*g) and
h_inner=-lambda/(Delta_rho*g). The inner domain normal is -e_r; this changes
buoyancy balance, not sigma_nn=sigma_rr. No stress derivative, rotated
residual or analytical Stokes field is used here.

The angle-P2 trace is the multiplier space of utilities._cylindrical_gap,
whose Collins et al. (2025), Urquiza et al. (2014) and Burman et al. (2020)
references document its variational origins. This reaction-to-topography
adapter is a UW3 extension. Cylindrical spectral gravity uses Simons (1996),
Appendix B, with the Poisson sign/normalization in analytic.cylindrical.gravity.
"""

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from underworld3.utilities._cylindrical_dirichlet import _edge_key, _labelled_edge_keys
from underworld3.utilities._cylindrical_gap import _ordered_rows


@dataclass
class CircularTopographyTrace:
    """Owned arc endpoints and three angle-P2 topography values per arc.

    All values use radial-outward displacement and the solver's common physical
    pressure gauge. Only boundary data are gathered by evaluate; coefficient
    integrates owned arcs and reduces scalars. Neither operation gathers volume
    fields. Call both operations collectively on comm, including empty ranks.
    """

    radius: float
    starts: np.ndarray
    widths: np.ndarray
    values: np.ndarray
    comm: object

    def cosine_coefficient(self, wavenumber: int, quadrature: int) -> tuple[float, float]:
        """Integrate the unnormalized cos(n theta) coefficient and constant mean."""
        if isinstance(wavenumber, bool) or int(wavenumber) != wavenumber or wavenumber < 1:
            raise ValueError("wavenumber must be a positive integer")
        if int(quadrature) != quadrature or quadrature < 3:
            raise ValueError("quadrature must be an integer at least three")
        nodes, weights = np.polynomial.legendre.leggauss(quadrature)
        t = (nodes + 1) / 2
        basis = np.stack(((1-t)*(1-2*t), t*(2*t-1), 4*t*(1-t)), axis=-1)
        angles = self.starts[:, None] + self.widths[:, None] * t
        samples = self.values @ basis.T
        measure = self.widths[:, None] * weights[None, :] / 2
        sums = np.array([np.sum(measure*samples*np.cos(wavenumber*angles)),
                         np.sum(measure*samples)])
        cosine, mean = self.comm.allreduce(sums, op=MPI.SUM)
        return float(cosine/np.pi), float(mean/(2*np.pi))

    def evaluate(self, angles: np.ndarray) -> np.ndarray:
        """Evaluate the spatial trace at arbitrary angles; no harmonic reconstruction.

        Gather only the owned boundary arcs, sort their periodic intervals and
        evaluate the original angle-P2 polynomial. Each rank may request its own
        angles, including an empty array. This preserves all numerical modes.
        """
        local = np.column_stack((self.starts, self.widths, self.values))
        table = np.concatenate(self.comm.allgather(local), axis=0)
        table = table[np.argsort(table[:, 0])]
        if not len(table):
            raise ValueError("The physical circle has no owned trace arcs")
        next_start = np.r_[table[1:, 0], table[0, 0]+2*np.pi]
        if not np.allclose(table[:, 0]+table[:, 1], next_start, rtol=0, atol=1e-10):
            raise ValueError("Owned trace arcs must cover one complete circle exactly once")
        theta = np.mod(np.asarray(angles, dtype=float)-table[0, 0], 2*np.pi)+table[0, 0]
        index = np.searchsorted(table[:, 0], theta, side="right")-1
        t = (theta-table[index, 0])/table[index, 1]
        basis = np.stack(((1-t)*(1-2*t), t*(2*t-1), 4*t*(1-t)), axis=-1)
        return np.sum(table[index, 2:]*basis, axis=-1)


@dataclass(frozen=True)
class CylindricalGapTopography:
    """Spatial topography at both circles, with unmodified physical reactions."""

    inner: CircularTopographyTrace
    outer: CircularTopographyTrace
    normal_reaction: object


def cylindrical_gap_topography(
    *, stokes, gap_result, inner_boundary: str, outer_boundary: str,
    radius_inner: float, radius_outer: float,
    inner_density_contrast: float, outer_density_contrast: float,
    inner_reference_gravity: float, outer_reference_gravity: float,
) -> CylindricalGapTopography:
    """Convert a completed coupled free/zero-slip normal reaction to topography.

    Density contrasts mean smaller-radius-side minus larger-radius-side density;
    gravity inputs are positive magnitudes. Displacements are radial-outward.
    The coupled solve has already paired the pressure and reaction gauge and
    included any prescribed pressure jump; do not subtract separate wall means.
    Zero-slip tangential reactions are not included in these normal loads.
    Call collectively. The Stokes velocity, pressure and reaction are unchanged.
    """
    mesh = stokes.mesh
    reaction = gap_result.normal_reaction
    scales = np.asarray([inner_density_contrast, outer_density_contrast,
                         inner_reference_gravity, outer_reference_gravity], dtype=float)
    if not np.all(np.isfinite(scales)) or np.any(scales[:2] == 0) or np.any(scales[2:] <= 0):
        raise ValueError("Require finite nonzero density contrasts and positive gravity magnitudes")
    if not 0 < radius_inner < radius_outer:
        raise ValueError("Require ordered positive inner/outer radii")
    if reaction is not getattr(stokes, "_cylindrical_gap_reaction", None):
        raise ValueError("Use the normal reaction returned by this Stokes coupled-gap solve")
    if gap_result.checks["relative_residual"] > 1e-9:
        raise ArithmeticError("The coupled-gap solve has not met its residual contract")
    vertices = np.asarray(mesh._get_coords_for_basis(1, True))[mesh._cell_node_indices(1, True)]
    rows = _ordered_rows(mesh, reaction, vertices)
    owned = np.flatnonzero(mesh._get_owned_cells_mask())
    radii = np.linalg.norm(vertices, axis=-1)
    field_values = np.asarray(reaction.array)[:, 0, 0]
    traces = []
    for boundary, radius, scale in (
        (inner_boundary, radius_inner, -1/(inner_density_contrast*inner_reference_gravity)),
        (outer_boundary, radius_outer, 1/(outer_density_contrast*outer_reference_gravity)),
    ):
        keys = _labelled_edge_keys(mesh, boundary, radius, radius*1e-10)
        starts, widths, values = [], [], []
        for i, j, mid in ((0, 1, 3), (1, 2, 4), (2, 0, 5)):
            selected = owned[np.all(abs(radii[owned][:, [i, j]]-radius)<1e-10, axis=1)]
            for cell in selected:
                edge = vertices[cell, [i, j]]
                if _edge_key(edge, radius*1e-10) not in keys:
                    continue
                a, b = np.arctan2(edge[:, 1], edge[:, 0])
                width = np.arctan2(np.sin(b-a), np.cos(b-a))
                indices = [i, j, mid] if width > 0 else [j, i, mid]
                starts.append(float(np.mod(a if width > 0 else b, 2*np.pi)))
                widths.append(abs(width))
                values.append(scale*field_values[rows[cell, indices]])
        traces.append(CircularTopographyTrace(radius, np.asarray(starts), np.asarray(widths),
                                              np.asarray(values).reshape(-1, 3), mesh.dm.comm.tompi4py()))
    return CylindricalGapTopography(traces[0], traces[1], reaction)
