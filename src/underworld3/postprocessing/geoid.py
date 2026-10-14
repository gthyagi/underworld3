r"""Spherical and cylindrical geoid and self-gravity response functions.

The pure coefficient functions in this module are independent of a particular
Stokes discretisation. They combine boundary and optional internal-load
coefficients through the appropriate radial Green's function. Separate
convenience adapters obtain topography coefficients from completed
rotated-free-slip Stokes solves.

The cylindrical sheet kernel follows Simons (1996), Appendix B, with its
normalisation fixed directly by potential continuity and the radial-derivative
jump condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

from mpi4py import MPI
import numpy as np


from underworld3.analytic.spherical.gravity import (
    GeoidResponse,
    SelfGravityResponse,
    spherical_shell_geoid_response,
    spherical_shell_self_gravity_response,
)

__all__ = [
    "GeoidResponse",
    "SelfGravityResponse",
    "SphericalShellResponse",
    "CylindricalGravityResponse",
    "CylindricalSelfGravityResponse",
    "CylindricalAnnulusResponse",
    "spherical_shell_geoid_response",
    "spherical_shell_self_gravity_response",
    "spherical_shell_response_from_rotated_stokes",
    "spherical_boundary_reaction_coefficient",
    "cylindrical_power_law_volume_potential_coefficient",
    "cylindrical_sheet_potential_coefficient",
    "cylindrical_sheet_radial_derivative_coefficient",
    "cylindrical_annulus_potential_operator",
    "cylindrical_annulus_geoid_response",
    "cylindrical_annulus_self_gravity_response",
    "cylindrical_cosine_boundary_coefficient",
    "cylindrical_annulus_response_from_rotated_stokes",
    "gather_unique_boundary_samples",
]


@dataclass(frozen=True)
class SphericalShellResponse:
    """Spherical-shell topography and corresponding geoid coefficients."""

    surface_topography: float
    cmb_topography: float
    surface_geoid: float
    cmb_geoid: float
    self_gravity: SelfGravityResponse | None = None


@dataclass(frozen=True)
class CylindricalAnnulusResponse:
    """Rotated-Stokes topography and cylindrical gravity coefficients."""

    outer_reaction: float
    inner_reaction: float
    outer_reaction_mean: float
    inner_reaction_mean: float
    outer_topography: float
    inner_topography: float
    outer_potential: float
    inner_potential: float
    outer_geoid: float
    inner_geoid: float
    self_gravity: CylindricalSelfGravityResponse | None = None


def _validate_geometry(
    radius_inner: float,
    radius_outer: float,
    harmonic_degree: int,
) -> tuple[float, float, int]:
    try:
        ri = float(radius_inner)
        ro = float(radius_outer)
    except (TypeError, ValueError) as error:
        raise TypeError("The shell radii must be real numbers.") from error
    if not np.all(np.isfinite((ri, ro))) or not 0.0 < ri < ro:
        raise ValueError("Expected finite radii ordered as 0 < radius_inner < radius_outer.")
    if isinstance(harmonic_degree, bool) or not isinstance(harmonic_degree, Integral):
        raise TypeError("harmonic_degree must be an integer.")
    degree = int(harmonic_degree)
    if degree < 0:
        raise ValueError("harmonic_degree must be non-negative.")
    return ri, ro, degree


from underworld3.analytic.cylindrical.gravity import (
    CylindricalGravityResponse,
    CylindricalSelfGravityResponse,
    cylindrical_sheet_potential_coefficient,
    cylindrical_sheet_radial_derivative_coefficient,
    cylindrical_annulus_potential_operator,
    cylindrical_annulus_geoid_response,
    cylindrical_annulus_self_gravity_response,
    cylindrical_power_law_volume_potential_coefficient,
)


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number.") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _validate_cylindrical_mode(wavenumber: int) -> int:
    if isinstance(wavenumber, bool) or not isinstance(wavenumber, Integral):
        raise TypeError("wavenumber must be an integer.")
    mode = int(wavenumber)
    if mode < 0:
        raise ValueError("wavenumber must be non-negative.")
    if mode == 0:
        raise ValueError(
            "The n=0 cylindrical mode has a logarithmic radial solution and "
            "requires a potential gauge."
        )
    return mode


def _validate_cylindrical_annulus(
    radius_inner: float,
    radius_outer: float,
    wavenumber: int,
) -> tuple[float, float, int]:
    radius_inner = _positive_float(radius_inner, "radius_inner")
    radius_outer = _positive_float(radius_outer, "radius_outer")
    if radius_inner >= radius_outer:
        raise ValueError("Expected radius_inner < radius_outer.")
    return (
        radius_inner,
        radius_outer,
        _validate_cylindrical_mode(wavenumber),
    )


def _spherical_triangle_area(a, b, c, radius: float) -> float:
    determinant = abs(float(np.dot(a, np.cross(b, c))))
    denominator = float(1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a))
    return 2.0 * np.arctan2(determinant, denominator) * radius**2


def _project_spherical_harmonic_samples(
    coords: np.ndarray,
    values: np.ndarray,
    radius: float,
    harmonic_degree: int,
    response_sign: float,
    harmonic_order=0,
    harmonic_component="cos",
    harmonic_normalization="axisymmetric_unnormalized",
) -> float:
    from scipy.spatial import ConvexHull
    from underworld3.maths.spherical_harmonics import real_harmonic_values, harmonic_norm

    unit_coords = coords / np.linalg.norm(coords, axis=1)[:, None]
    hull = ConvexHull(unit_coords, qhull_options="QJ")
    projected_integral = 0.0

    for simplex in hull.simplices:
        area = _spherical_triangle_area(
            unit_coords[simplex[0]],
            unit_coords[simplex[1]],
            unit_coords[simplex[2]],
            radius,
        )
        centroid = unit_coords[simplex].mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        projected_integral += (
            area * float(values[simplex].mean()) * real_harmonic_values(
                centroid, degree=harmonic_degree, order=harmonic_order,
                component=harmonic_component, normalization=harmonic_normalization)
        )

    angular_norm = harmonic_norm(harmonic_degree, harmonic_order,
                                harmonic_component, harmonic_normalization)
    return float(response_sign * projected_integral / (radius**2 * angular_norm))


def spherical_boundary_reaction_coefficient(
    stokes, boundary, *, harmonic_degree, harmonic_order=0,
    harmonic_component="cos", harmonic_normalization="axisymmetric_unnormalized",
):
    """Project wall reaction with a real harmonic and the matching FE norm.

    The assembled reaction mean is removed before contraction. Return the
    modal coefficient, whose sign is reaction_nn=-fluid_sigma_rr. Angular
    basis values use a Cartesian solid harmonic and remain defined at poles.
    """
    from underworld3.maths import BdIntegral
    from underworld3.maths.spherical_harmonics import real_harmonic_cartesian_sympy
    if stokes.mesh.dim != 3:
        raise ValueError("Spherical reaction projection requires a 3D mesh.")
    harmonic = real_harmonic_cartesian_sympy(list(stokes.mesh.X), degree=harmonic_degree,
        order=harmonic_order, component=harmonic_component, normalization=harmonic_normalization)
    numerator = float(stokes.boundary_normal_traction_integral(boundary, harmonic, remove_mean=True))
    norm = float(BdIntegral(stokes.mesh, fn=harmonic**2, boundary=boundary).evaluate())
    if not np.isfinite(numerator) or not np.isfinite(norm) or norm <= 0:
        raise ValueError("Spherical reaction projection requires finite data and a positive FE norm.")
    return numerator/norm


def _rotated_topography_coefficient(
    stokes,
    boundary: str,
    radius: float,
    harmonic_degree: int,
    buoyancy_scale: float,
    response_sign: float,
    projection: str,
    harmonic_order=0,
    harmonic_component="cos",
    harmonic_normalization="axisymmetric_unnormalized",
) -> float:
    buoyancy_scale = float(buoyancy_scale)
    if not np.isfinite(buoyancy_scale) or buoyancy_scale == 0.0:
        raise ValueError("Boundary buoyancy scales must be finite and nonzero.")

    if projection == "reaction":
        coefficient = spherical_boundary_reaction_coefficient(stokes, boundary,
            harmonic_degree=harmonic_degree, harmonic_order=harmonic_order,
            harmonic_component=harmonic_component, harmonic_normalization=harmonic_normalization)
        return float(-response_sign*coefficient/buoyancy_scale)
    if projection != "centroid":
        raise ValueError("projection must be 'centroid' or 'reaction'.")

    coords, sigma_nn = stokes.boundary_normal_traction(boundary, mass="auto")
    local_rows = np.column_stack(
        (
            np.asarray(coords, dtype=float),
            -np.asarray(sigma_nn, dtype=float) / buoyancy_scale,
        )
    )
    gathered_rows = MPI.COMM_WORLD.gather(local_rows, root=0)

    coefficient = None
    root_error = None
    if MPI.COMM_WORLD.rank == 0:
        try:
            if gathered_rows is None:
                raise RuntimeError("MPI gather returned no root payload.")
            nonempty_rows = [rows for rows in gathered_rows if rows.size]
            if not nonempty_rows:
                raise RuntimeError(f"No samples found on boundary {boundary!r}.")
            merged = {}
            for row in np.vstack(nonempty_rows):
                key = tuple(np.round(row[:3], 12))
                if key not in merged:
                    merged[key] = [row[:3].copy(), 0.0, 0]
                merged[key][1] += float(row[3])
                merged[key][2] += 1
            global_coords = np.array([item[0] for item in merged.values()])
            global_values = np.array([item[1] / item[2] for item in merged.values()])
            coefficient = _project_spherical_harmonic_samples(
                global_coords,
                global_values,
                radius,
                harmonic_degree,
                response_sign,
                harmonic_order, harmonic_component, harmonic_normalization,
            )
        except Exception as error:
            root_error = f"{type(error).__name__}: {error}"

    root_error = MPI.COMM_WORLD.bcast(root_error, root=0)
    if root_error is not None:
        raise RuntimeError(f"Spherical-harmonic projection failed: {root_error}")
    return float(MPI.COMM_WORLD.bcast(coefficient, root=0))


def spherical_shell_response_from_rotated_stokes(
    *,
    stokes,
    radius_inner: float,
    radius_outer: float,
    harmonic_degree: int,
    internal_load_radius: float | None = None,
    internal_load_coefficient: float = 0.0,
    surface_boundary: str = "Upper",
    cmb_boundary: str = "Lower",
    surface_buoyancy_scale: float = 1.0,
    cmb_buoyancy_scale: float = 1.0,
    include_self_gravity: bool = False,
    surface_density_contrast: float | None = None,
    cmb_density_contrast: float | None = None,
    planet_radius: float | None = None,
    gravity: float | None = None,
    gravitational_constant: float = 6.67430e-11,
    projection: str = "centroid",
    harmonic_order: int = 0,
    harmonic_component: str = "cos",
    harmonic_normalization: str = "axisymmetric_unnormalized",
) -> SphericalShellResponse:
    r"""Compute spherical-shell response from a rotated-free-slip Stokes solve.

    ``projection="centroid"`` (default) recovers pointwise traction and fits it
    over a triangulation of the boundary samples. ``projection="reaction"``
    contracts the assembled nodal reaction directly with the harmonic test
    function. The latter is a distributed weak/integral quantity that avoids
    consuming slowly converging P2 vertex values on curved boundaries (issue
    #414). Explicit order, component and normalization support non-axisymmetric
    real harmonics while preserving the zonal defaults. Density contrasts, planet radius,
    and gravity are required when ``include_self_gravity`` is true.
    """

    ri, ro, degree = _validate_geometry(
        radius_inner,
        radius_outer,
        harmonic_degree,
    )
    if not isinstance(include_self_gravity, bool):
        raise TypeError("include_self_gravity must be True or False.")
    if projection not in ("centroid", "reaction"):
        raise ValueError("projection must be 'centroid' or 'reaction'.")
    if degree == 0:
        raise ValueError(
            "The rotated-Stokes adapter requires harmonic_degree >= 1 because "
            "boundary_normal_traction() removes the degree-zero mean."
        )
    from underworld3.maths.spherical_harmonics import validate_harmonic
    validate_harmonic(degree, harmonic_order, harmonic_component, harmonic_normalization)
    if include_self_gravity:
        missing = [
            name
            for name, value in (
                ("surface_density_contrast", surface_density_contrast),
                ("cmb_density_contrast", cmb_density_contrast),
                ("planet_radius", planet_radius),
                ("gravity", gravity),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "Self-gravity requires explicit values for " + ", ".join(missing) + "."
            )

    surface_topography = _rotated_topography_coefficient(
        stokes,
        surface_boundary,
        ro,
        degree,
        surface_buoyancy_scale,
        1.0,
        projection,
        harmonic_order, harmonic_component, harmonic_normalization,
    )
    cmb_topography = _rotated_topography_coefficient(
        stokes,
        cmb_boundary,
        ri,
        degree,
        cmb_buoyancy_scale,
        -1.0,
        projection,
        harmonic_order, harmonic_component, harmonic_normalization,
    )
    geoid = spherical_shell_geoid_response(
        radius_inner=ri,
        radius_outer=ro,
        harmonic_degree=degree,
        surface_topography_coefficient=surface_topography,
        cmb_topography_coefficient=cmb_topography,
        internal_load_radius=internal_load_radius,
        internal_load_coefficient=internal_load_coefficient,
    )
    self_gravity = None
    if include_self_gravity:
        assert surface_density_contrast is not None
        assert cmb_density_contrast is not None
        assert planet_radius is not None
        assert gravity is not None
        self_gravity = spherical_shell_self_gravity_response(
            radius_inner=ri,
            radius_outer=ro,
            harmonic_degree=degree,
            surface_topography_coefficient=surface_topography,
            cmb_topography_coefficient=cmb_topography,
            internal_load_radius=internal_load_radius,
            internal_load_coefficient=internal_load_coefficient,
            surface_density_contrast=float(surface_density_contrast),
            cmb_density_contrast=float(cmb_density_contrast),
            planet_radius=float(planet_radius),
            gravity=float(gravity),
            gravitational_constant=gravitational_constant,
        )

    return SphericalShellResponse(
        surface_topography=surface_topography,
        cmb_topography=cmb_topography,
        surface_geoid=geoid.surface_geoid,
        cmb_geoid=geoid.cmb_geoid,
        self_gravity=self_gravity,
    )


def gather_unique_boundary_samples(
    coords,
    values,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather diagnostic boundary samples; production reactions use FE integrals.

    This replicated sample utility is intended for plotting and pointwise stress
    diagnostics, not for fitting production reaction/topography coefficients.
    """

    local_rows = np.column_stack((np.asarray(coords, dtype=float), np.asarray(values, dtype=float)))
    gathered_rows = MPI.COMM_WORLD.allgather(local_rows)
    nonempty_rows = [rows for rows in gathered_rows if rows.size]
    if not nonempty_rows:
        raise RuntimeError("No boundary samples were recovered.")

    merged = {}
    for row in np.vstack(nonempty_rows):
        key = tuple(np.round(row[:dimension], 12))
        if key not in merged:
            merged[key] = [row[:dimension].copy(), 0.0, 0]
        merged[key][1] += float(row[dimension])
        merged[key][2] += 1
    global_coords = np.array([entry[0] for entry in merged.values()])
    global_values = np.array([entry[1] / entry[2] for entry in merged.values()], dtype=float)
    return global_coords, global_values


def _trapezoidal_integral(values: np.ndarray, coordinates: np.ndarray) -> float:
    """Integrate samples without requiring NumPy's version-specific helpers."""

    widths = np.diff(coordinates)
    averages = 0.5 * (values[:-1] + values[1:])
    return float(np.sum(widths * averages))


def cylindrical_cosine_boundary_coefficient(
    coords,
    values,
    wavenumber: int,
) -> tuple[float, float]:
    r"""Project sampled circular data onto :math:`\cos(n\theta)` and the mean.

    ``coords`` must contain at least three two-dimensional Cartesian boundary
    points. The samples may begin at any angle and need not include a duplicate
    endpoint; this function sorts and closes the periodic interval. This helper
    is intended for external sampled data. Finite-element reaction loads use
    :meth:`Stokes.boundary_normal_traction_integral` instead, so their projection
    follows the actual boundary facets without a global gather or reconstructed
    angular ordering.

    Parameters
    ----------
    coords : array-like, shape (n, 2)
        Cartesian circular-boundary coordinates.
    values : array-like, shape (n,)
        Scalar values at ``coords``.
    wavenumber : int
        Positive azimuthal Fourier wavenumber.

    Returns
    -------
    coefficient, mean : tuple of float
        Cosine-mode coefficient and degree-zero mean.
    """

    mode = _validate_cylindrical_mode(wavenumber)
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n, 2).")
    if coords.shape[0] != values.size:
        raise ValueError("coords and values must contain the same number of samples.")
    if coords.shape[0] < 3:
        raise ValueError("At least three circular-boundary samples are required.")
    if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(values)):
        raise ValueError("Boundary coordinates and values must be finite.")

    theta = np.mod(np.arctan2(coords[:, 1], coords[:, 0]), 2.0 * np.pi)
    order = np.argsort(theta)
    theta = theta[order]
    values = values[order]
    theta = np.append(theta, theta[0] + 2.0 * np.pi)
    values = np.append(values, values[0])

    coefficient = (
        _trapezoidal_integral(
            values * np.cos(mode * theta),
            theta,
        )
        / np.pi
    )
    mean = _trapezoidal_integral(values, theta) / (2.0 * np.pi)
    return float(coefficient), float(mean)


def _rotated_cylindrical_boundary_response(
    *,
    stokes,
    boundary: str,
    wavenumber: int,
    buoyancy_scale: float,
) -> tuple[float, float, float]:
    """Project one boundary's assembled reaction and topography coefficients."""

    mode = _validate_cylindrical_mode(wavenumber)
    buoyancy_scale = _finite_float(buoyancy_scale, "buoyancy_scale")
    if buoyancy_scale == 0.0:
        raise ValueError("Boundary buoyancy scales must be nonzero.")
    if getattr(stokes.mesh, "dim", None) != 2:
        raise ValueError("The cylindrical adapter requires a two-dimensional mesh.")

    import sympy
    from underworld3.maths import BdIntegral

    theta = stokes.mesh.CoordinateSystem.xR[1]
    harmonic = sympy.cos(mode * theta)
    reaction_integral = float(
        stokes.boundary_normal_traction_integral(
            boundary,
            harmonic,
            remove_mean=True,
        )
    )
    reaction_total = float(
        stokes.boundary_normal_traction_integral(
            boundary,
            1.0,
            remove_mean=False,
        )
    )
    boundary_measure = float(BdIntegral(stokes.mesh, fn=1.0, boundary=boundary).evaluate())
    harmonic_norm = float(BdIntegral(stokes.mesh, fn=harmonic**2, boundary=boundary).evaluate())
    if not np.all(
        np.isfinite((reaction_integral, reaction_total, boundary_measure, harmonic_norm))
    ):
        raise RuntimeError("Cylindrical reaction projection produced non-finite data.")
    if boundary_measure <= 0.0 or harmonic_norm <= 0.0:
        raise RuntimeError("Cylindrical reaction projection requires positive boundary integrals.")

    reaction_coefficient = reaction_integral / harmonic_norm
    reaction_mean = reaction_total / boundary_measure
    return (
        float(reaction_coefficient),
        float(reaction_mean),
        float(-reaction_coefficient / buoyancy_scale),
    )


def cylindrical_annulus_response_from_rotated_stokes(
    *,
    stokes,
    radius_inner: float,
    radius_outer: float,
    wavenumber: int,
    outer_density_contrast: float,
    inner_density_contrast: float,
    outer_reference_gravity: float,
    inner_reference_gravity: float,
    internal_load_radius: float | None = None,
    internal_surface_density_coefficient: float = 0.0,
    volume_density_radial_exponent: float | None = None,
    volume_density_coefficient: float = 0.0,
    outer_boundary: str = "Upper",
    inner_boundary: str = "Lower",
    outer_buoyancy_scale: float = 1.0,
    inner_buoyancy_scale: float = -1.0,
    gravitational_constant: float = 1.0,
    include_self_gravity: bool = False,
    outer_feedback_factor: float | None = None,
    inner_feedback_factor: float | None = None,
) -> CylindricalAnnulusResponse:
    r"""Compute a cylindrical response from rotated-free-slip wall reactions.

    :meth:`Stokes.boundary_normal_traction_integral` contracts the assembled
    wall reaction directly with a boundary test function. This adapter uses

    .. math::

       h=-reaction_{nn}/signed\_buoyancy\_scale

    The scales are caller-supplied sign conventions. If the recovered reaction
    is ``-sigma_rr`` and h is positive radially outward, hydrostatic balance
    requires outer scale ``-Delta_rho*g`` and inner scale ``+Delta_rho*g``.
    Historical defaults are retained for compatibility; density-coupled
    benchmarks pass their physical signed scales explicitly.

    and projects each boundary onto the unnormalised real basis
    :math:`\cos(n\theta)`. The numerator and harmonic norm use the same finite-
    element boundary measure. Owned reaction degrees of freedom are reduced on
    the mesh communicator; no pointwise traction recovery, angular sorting, or
    rank-zero boundary gather is performed.

    Parameters
    ----------
    stokes : underworld3.systems.Stokes
        Completed two-dimensional rotated-free-slip Stokes solve.
    radius_inner, radius_outer : float
        Positive annulus radii ordered from inner to outer.
    wavenumber : int
        Positive azimuthal Fourier wavenumber.
    outer_density_contrast, inner_density_contrast : float
        Signed density contrasts at the two boundaries.
    outer_reference_gravity, inner_reference_gravity : float
        Positive gravity magnitudes used to convert potential to geoid.
    internal_load_radius : float, optional
        Radius of an internal sheet, strictly inside the annulus.
    internal_surface_density_coefficient : float, default=0
        Density coefficient of the optional internal sheet.
    outer_boundary, inner_boundary : str
        Mesh boundary labels used for reaction recovery.
    outer_buoyancy_scale, inner_buoyancy_scale : float
        Signed scales converting wall reaction to dynamic topography.
    gravitational_constant : float, default=1
        Positive gravitational constant in the selected unit system.
    include_self_gravity : bool, default=False
        Return the self-gravity-corrected response when true.
    outer_feedback_factor, inner_feedback_factor : float, optional
        Explicit self-gravity feedback factors.

    Returns
    -------
    CylindricalAnnulusResponse
        Recovered reactions, topography, gravity response, and optional
        self-gravity correction, identical on every MPI rank.
    """

    radius_inner, radius_outer, mode = _validate_cylindrical_annulus(
        radius_inner,
        radius_outer,
        wavenumber,
    )
    if not isinstance(include_self_gravity, bool):
        raise TypeError("include_self_gravity must be True or False.")

    outer_reaction, outer_mean, outer_topography = _rotated_cylindrical_boundary_response(
        stokes=stokes,
        boundary=outer_boundary,
        wavenumber=mode,
        buoyancy_scale=outer_buoyancy_scale,
    )
    inner_reaction, inner_mean, inner_topography = _rotated_cylindrical_boundary_response(
        stokes=stokes,
        boundary=inner_boundary,
        wavenumber=mode,
        buoyancy_scale=inner_buoyancy_scale,
    )
    gravity = cylindrical_annulus_geoid_response(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        wavenumber=mode,
        outer_topography_coefficient=outer_topography,
        inner_topography_coefficient=inner_topography,
        outer_density_contrast=outer_density_contrast,
        inner_density_contrast=inner_density_contrast,
        outer_reference_gravity=outer_reference_gravity,
        inner_reference_gravity=inner_reference_gravity,
        internal_load_radius=internal_load_radius,
        internal_surface_density_coefficient=internal_surface_density_coefficient,
        volume_density_radial_exponent=volume_density_radial_exponent,
        volume_density_coefficient=volume_density_coefficient,
        gravitational_constant=gravitational_constant,
    )
    self_gravity = None
    if include_self_gravity:
        self_gravity = cylindrical_annulus_self_gravity_response(
            radius_inner=radius_inner,
            radius_outer=radius_outer,
            wavenumber=mode,
            outer_topography_coefficient=outer_topography,
            inner_topography_coefficient=inner_topography,
            outer_density_contrast=outer_density_contrast,
            inner_density_contrast=inner_density_contrast,
            outer_reference_gravity=outer_reference_gravity,
            inner_reference_gravity=inner_reference_gravity,
            internal_load_radius=internal_load_radius,
            internal_surface_density_coefficient=internal_surface_density_coefficient,
            volume_density_radial_exponent=volume_density_radial_exponent,
            volume_density_coefficient=volume_density_coefficient,
            gravitational_constant=gravitational_constant,
            outer_feedback_factor=outer_feedback_factor,
            inner_feedback_factor=inner_feedback_factor,
        )

    return CylindricalAnnulusResponse(
        outer_reaction=outer_reaction,
        inner_reaction=inner_reaction,
        outer_reaction_mean=outer_mean,
        inner_reaction_mean=inner_mean,
        outer_topography=outer_topography,
        inner_topography=inner_topography,
        outer_potential=gravity.outer_potential,
        inner_potential=gravity.inner_potential,
        outer_geoid=gravity.outer_geoid,
        inner_geoid=gravity.inner_geoid,
        self_gravity=self_gravity,
    )
