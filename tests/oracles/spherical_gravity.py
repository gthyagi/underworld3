r"""Independent spherical-harmonic gravity reference for the Zhong benchmark.

This module deliberately has no Underworld dependency.  It derives the radial
potential of one spherical density sheet from

    laplacian(Phi) = -4*pi*G*sigma_lm*delta(r-r_s)*Y_lm,

using the regular interior branch ``r**ell`` and decaying exterior branch
``r**(-(ell+1))``.  The derivative jump fixes the source-radius amplitude to

    Phi_lm(r_s) = 4*pi*G*sigma_lm*r_s/(2*ell+1).

The normalized two-boundary operator factors out ``4*pi*G/g`` and therefore
matches the coefficient convention used by the Zhong et al. (2008) response
benchmark.  A positive outward internal-load coefficient is represented by a
negative density-sheet coefficient and consequently enters the load vector
with a negative sign.

Reference context and provenance
--------------------------------
Zhong, S., McNamara, A., Tan, E., Moresi, L. and Gurnis, M. (2008),
A benchmark study on mantle convection in a 3-D spherical shell using CitcomS,
Geochemistry, Geophysics, Geosystems 9, Q10017.
https://doi.org/10.1029/2008GC002048

Appendix A supplies related spherical response conventions. This file is
a project mesh, Poisson, harmonic, regression or MPI implementation check.
Manufactured volume sources and operator tests derive their references from
the stated Poisson problem; they are not claimed as published Zhong tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


__all__ = [
    "CONVENTIONS",
    "SphericalGeoidResponse",
    "SphericalSelfGravityResponse",
    "axisymmetric_harmonic_norm",
    "recover_axisymmetric_coefficient",
    "spherical_sheet_source_amplitude",
    "spherical_sheet_radial_factor",
    "spherical_sheet_potential_coefficient",
    "spherical_sheet_finite_dirichlet_potential_coefficient",
    "spherical_sheet_radial_derivative_coefficient",
    "spherical_volume_density_potential_coefficient",
    "spherical_volume_density_finite_dirichlet_potential_coefficient",
    "spherical_volume_density_potential_coefficient_quadrature",
    "spherical_volume_density_radial_derivative_coefficient",
    "spherical_shell_potential_operator",
    "spherical_shell_geoid_response",
    "spherical_shell_self_gravity_response",
]


CONVENTIONS = {
    "harmonic_basis": (
        "Y_ell0=P_ell(cos(theta)); integral(Y_ell0^2 dOmega)="
        "4*pi/(2*ell+1)"
    ),
    "potential_convention": "laplacian(Phi)=-4*pi*G*delta_rho",
    "poisson_source_sign": "UW3 -div(grad(Phi))=+4*pi*G*delta_rho",
    "surface_normal": "+e_r",
    "cmb_normal": "-e_r for the mantle domain",
    "topography_convention": "h_surface=-sigma_rr_surface; h_cmb=+sigma_rr_cmb",
    "internal_load_convention": (
        "positive outward load enters the density-sheet load vector with a "
        "negative sign"
    ),
    "geoid_convention": "N=Phi/g; normalized operator factors out 4*pi*G/g",
    "coefficient_dimensions": (
        "stored response coefficients are nondimensional; planet_radius, "
        "gravity, density contrasts, and G retain the SI units named by "
        "their metadata keys"
    ),
}


@dataclass(frozen=True)
class SphericalGeoidResponse:
    """Normalized surface and CMB geoid coefficients without self-gravity."""

    surface_geoid: float
    cmb_geoid: float


@dataclass(frozen=True)
class SphericalSelfGravityResponse:
    """Self-gravity-corrected topography and geoid coefficients."""

    surface_topography: float
    cmb_topography: float
    surface_geoid: float
    cmb_geoid: float
    q_surface: float
    q_cmb: float
    matrix_condition_number: float
    matrix_residual_norm: float


def _finite_float(value: Real, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number.") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_float(value: Real, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _validate_degree(harmonic_degree: int) -> int:
    if isinstance(harmonic_degree, bool) or not isinstance(harmonic_degree, Integral):
        raise TypeError("harmonic_degree must be an integer.")
    degree = int(harmonic_degree)
    if degree < 1:
        raise ValueError("harmonic_degree must be at least 1.")
    return degree


def _validate_geometry(
    radius_inner: Real,
    radius_outer: Real,
    harmonic_degree: int,
) -> tuple[float, float, int]:
    radius_inner = _positive_float(radius_inner, "radius_inner")
    radius_outer = _positive_float(radius_outer, "radius_outer")
    if radius_inner >= radius_outer:
        raise ValueError("Expected radius_inner < radius_outer.")
    return radius_inner, radius_outer, _validate_degree(harmonic_degree)


def axisymmetric_harmonic_norm(harmonic_degree: int) -> float:
    r"""Return ``integral(P_ell(cos(theta))**2 dOmega)`` on the unit sphere."""

    degree = _validate_degree(harmonic_degree)
    return float(4.0 * np.pi / (2 * degree + 1))


def recover_axisymmetric_coefficient(
    *,
    weighted_surface_integral: Real,
    sphere_radius: Real,
    harmonic_degree: int,
) -> float:
    r"""Recover ``f_ell0`` from ``integral(f P_ell dS)`` on a sphere."""

    integral = _finite_float(weighted_surface_integral, "weighted_surface_integral")
    radius = _positive_float(sphere_radius, "sphere_radius")
    return float(
        integral / (radius**2 * axisymmetric_harmonic_norm(harmonic_degree))
    )


def spherical_sheet_source_amplitude(
    *,
    source_radius: Real,
    harmonic_degree: int,
    surface_density_coefficient: Real,
    gravitational_constant: Real = 1.0,
) -> float:
    r"""Return the potential coefficient at a spherical density sheet."""

    source_radius = _positive_float(source_radius, "source_radius")
    degree = _validate_degree(harmonic_degree)
    density = _finite_float(
        surface_density_coefficient, "surface_density_coefficient"
    )
    gravity_constant = _positive_float(
        gravitational_constant, "gravitational_constant"
    )
    return float(
        4.0
        * np.pi
        * gravity_constant
        * density
        * source_radius
        / (2 * degree + 1)
    )


def spherical_sheet_radial_factor(
    *,
    source_radius: Real,
    target_radius: Real,
    harmonic_degree: int,
) -> float:
    r"""Return the regular/decaying radial continuation factor."""

    source_radius = _positive_float(source_radius, "source_radius")
    target_radius = _positive_float(target_radius, "target_radius")
    degree = _validate_degree(harmonic_degree)
    if target_radius <= source_radius:
        return float((target_radius / source_radius) ** degree)
    return float((source_radius / target_radius) ** (degree + 1))


def spherical_sheet_potential_coefficient(
    *,
    source_radius: Real,
    target_radius: Real,
    harmonic_degree: int,
    surface_density_coefficient: Real,
    gravitational_constant: Real = 1.0,
) -> float:
    r"""Return ``Phi_lm`` at one target radius for a spherical density sheet."""

    amplitude = spherical_sheet_source_amplitude(
        source_radius=source_radius,
        harmonic_degree=harmonic_degree,
        surface_density_coefficient=surface_density_coefficient,
        gravitational_constant=gravitational_constant,
    )
    factor = spherical_sheet_radial_factor(
        source_radius=source_radius,
        target_radius=target_radius,
        harmonic_degree=harmonic_degree,
    )
    return float(amplitude * factor)


def spherical_sheet_radial_derivative_coefficient(
    *,
    source_radius: Real,
    target_radius: Real,
    harmonic_degree: int,
    surface_density_coefficient: Real,
    gravitational_constant: Real = 1.0,
    source_side: str | None = None,
) -> float:
    r"""Return ``dPhi_lm/dr`` on an interior or exterior radial branch.

    ``source_side`` is required only when ``target_radius == source_radius``;
    use ``"inside"`` for the regular branch and ``"outside"`` for the
    decaying branch.
    """

    source_radius = _positive_float(source_radius, "source_radius")
    target_radius = _positive_float(target_radius, "target_radius")
    degree = _validate_degree(harmonic_degree)
    potential = spherical_sheet_potential_coefficient(
        source_radius=source_radius,
        target_radius=target_radius,
        harmonic_degree=degree,
        surface_density_coefficient=surface_density_coefficient,
        gravitational_constant=gravitational_constant,
    )

    if np.isclose(target_radius, source_radius, rtol=0.0, atol=0.0):
        if source_side not in {"inside", "outside"}:
            raise ValueError(
                "source_side must be 'inside' or 'outside' at the source radius."
            )
        inside = source_side == "inside"
    else:
        if source_side is not None:
            raise ValueError("source_side is valid only at the source radius.")
        inside = target_radius < source_radius

    radial_power = degree if inside else -(degree + 1)
    return float(radial_power * potential / target_radius)


def _power_integral(lower: float, upper: float, exponent: int) -> float:
    """Return the exact integral of ``s**exponent`` on positive bounds."""

    if upper <= lower:
        return 0.0
    if exponent == -1:
        return float(np.log(upper / lower))
    power = exponent + 1
    return float((upper**power - lower**power) / power)


def _validate_volume_density_parameters(
    *,
    radius_inner: Real,
    radius_outer: Real,
    target_radius: Real,
    harmonic_degree: int,
    density_power: int,
    density_amplitude: Real,
    gravitational_constant: Real,
) -> tuple[float, float, float, int, int, float, float]:
    radius_inner, radius_outer, degree = _validate_geometry(
        radius_inner, radius_outer, harmonic_degree
    )
    target = _positive_float(target_radius, "target_radius")
    if isinstance(density_power, bool) or not isinstance(density_power, Integral):
        raise TypeError("density_power must be an integer.")
    power = int(density_power)
    amplitude = _finite_float(density_amplitude, "density_amplitude")
    gravity_constant = _positive_float(
        gravitational_constant, "gravitational_constant"
    )
    return (
        radius_inner,
        radius_outer,
        target,
        degree,
        power,
        amplitude,
        gravity_constant,
    )


def spherical_volume_density_potential_coefficient(
    *,
    radius_inner: Real,
    radius_outer: Real,
    target_radius: Real,
    harmonic_degree: int,
    density_power: int = 2,
    density_amplitude: Real = 1.0,
    gravitational_constant: Real = 1.0,
) -> float:
    r"""Return the potential coefficient for a compact radial volume source.

    The source coefficient is

    ``rho_ell(r) = density_amplitude * (r/radius_outer)**density_power``

    between ``radius_inner`` and ``radius_outer`` and is zero elsewhere.  The
    result follows directly from the regular/decaying spherical Green kernel
    and uses the convention ``laplacian(Phi) = -4*pi*G*rho``.
    """

    ri, ro, radius, degree, power, amplitude, gravity_constant = (
        _validate_volume_density_parameters(
            radius_inner=radius_inner,
            radius_outer=radius_outer,
            target_radius=target_radius,
            harmonic_degree=harmonic_degree,
            density_power=density_power,
            density_amplitude=density_amplitude,
            gravitational_constant=gravitational_constant,
        )
    )
    scale = amplitude / ro**power
    inside_integral = scale * _power_integral(
        ri,
        min(radius, ro),
        degree + 2 + power,
    )
    outside_integral = scale * _power_integral(
        max(radius, ri),
        ro,
        1 - degree + power,
    )
    prefactor = 4.0 * np.pi * gravity_constant / (2 * degree + 1)
    return float(
        prefactor
        * (
            radius ** (-(degree + 1)) * inside_integral
            + radius**degree * outside_integral
        )
    )


def spherical_sheet_finite_dirichlet_potential_coefficient(
    *,
    source_radius: Real,
    exterior_radius: Real,
    target_radius: Real,
    harmonic_degree: int,
    surface_density_coefficient: Real,
    gravitational_constant: Real = 1.0,
) -> float:
    r"""Return one sheet's regular full-ball solution with ``Phi(R_ext)=0``."""

    source = _positive_float(source_radius, "source_radius")
    exterior = _positive_float(exterior_radius, "exterior_radius")
    target = _positive_float(target_radius, "target_radius")
    degree = _validate_degree(harmonic_degree)
    if exterior <= source:
        raise ValueError("exterior_radius must be greater than source_radius.")
    if target > exterior:
        raise ValueError("target_radius must not exceed exterior_radius.")
    infinite_target = spherical_sheet_potential_coefficient(
        source_radius=source,
        target_radius=target,
        harmonic_degree=degree,
        surface_density_coefficient=surface_density_coefficient,
        gravitational_constant=gravitational_constant,
    )
    infinite_exterior = spherical_sheet_potential_coefficient(
        source_radius=source,
        target_radius=exterior,
        harmonic_degree=degree,
        surface_density_coefficient=surface_density_coefficient,
        gravitational_constant=gravitational_constant,
    )
    return float(infinite_target - infinite_exterior * (target / exterior) ** degree)


def spherical_volume_density_radial_derivative_coefficient(
    *,
    radius_inner: Real,
    radius_outer: Real,
    target_radius: Real,
    harmonic_degree: int,
    density_power: int = 2,
    density_amplitude: Real = 1.0,
    gravitational_constant: Real = 1.0,
) -> float:
    r"""Return ``dPhi_ell/dr`` for the compact radial volume source."""

    ri, ro, radius, degree, power, amplitude, gravity_constant = (
        _validate_volume_density_parameters(
            radius_inner=radius_inner,
            radius_outer=radius_outer,
            target_radius=target_radius,
            harmonic_degree=harmonic_degree,
            density_power=density_power,
            density_amplitude=density_amplitude,
            gravitational_constant=gravitational_constant,
        )
    )
    scale = amplitude / ro**power
    inside_integral = scale * _power_integral(
        ri,
        min(radius, ro),
        degree + 2 + power,
    )
    outside_integral = scale * _power_integral(
        max(radius, ri),
        ro,
        1 - degree + power,
    )
    prefactor = 4.0 * np.pi * gravity_constant / (2 * degree + 1)
    return float(
        prefactor
        * (
            -(degree + 1) * radius ** (-(degree + 2)) * inside_integral
            + degree * radius ** (degree - 1) * outside_integral
        )
    )


def spherical_volume_density_finite_dirichlet_potential_coefficient(
    *,
    radius_inner: Real,
    radius_outer: Real,
    exterior_radius: Real,
    target_radius: Real,
    harmonic_degree: int,
    density_power: int = 2,
    density_amplitude: Real = 1.0,
    gravitational_constant: Real = 1.0,
) -> float:
    r"""Return the exact full-ball solution with ``Phi(R_ext)=0``.

    The infinite-domain solution is corrected by the regular homogeneous
    branch ``C*r**ell``. This preserves regularity at the centre and enforces
    the finite exterior Dirichlet condition exactly.
    """

    exterior = _positive_float(exterior_radius, "exterior_radius")
    target = _positive_float(target_radius, "target_radius")
    degree = _validate_degree(harmonic_degree)
    radius_outer_value = _positive_float(radius_outer, "radius_outer")
    if exterior <= radius_outer_value:
        raise ValueError("exterior_radius must be greater than radius_outer.")
    if target > exterior:
        raise ValueError("target_radius must not exceed exterior_radius.")
    infinite_target = spherical_volume_density_potential_coefficient(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        target_radius=target,
        harmonic_degree=degree,
        density_power=density_power,
        density_amplitude=density_amplitude,
        gravitational_constant=gravitational_constant,
    )
    infinite_exterior = spherical_volume_density_potential_coefficient(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        target_radius=exterior,
        harmonic_degree=degree,
        density_power=density_power,
        density_amplitude=density_amplitude,
        gravitational_constant=gravitational_constant,
    )
    return float(infinite_target - infinite_exterior * (target / exterior) ** degree)


def spherical_volume_density_potential_coefficient_quadrature(
    *,
    radius_inner: Real,
    radius_outer: Real,
    target_radius: Real,
    harmonic_degree: int,
    density_power: int = 2,
    density_amplitude: Real = 1.0,
    gravitational_constant: Real = 1.0,
    quadrature_points: int = 128,
) -> float:
    r"""Evaluate the same volume-source Green integral by Gauss--Legendre."""

    ri, ro, radius, degree, power, amplitude, gravity_constant = (
        _validate_volume_density_parameters(
            radius_inner=radius_inner,
            radius_outer=radius_outer,
            target_radius=target_radius,
            harmonic_degree=harmonic_degree,
            density_power=density_power,
            density_amplitude=density_amplitude,
            gravitational_constant=gravitational_constant,
        )
    )
    if (
        isinstance(quadrature_points, bool)
        or not isinstance(quadrature_points, Integral)
        or quadrature_points < 2
    ):
        raise ValueError("quadrature_points must be an integer of at least 2.")
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_points))

    def integrate(lower: float, upper: float, exponent: int) -> float:
        if upper <= lower:
            return 0.0
        points = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
        return float(0.5 * (upper - lower) * np.dot(weights, points**exponent))

    scale = amplitude / ro**power
    inside_integral = scale * integrate(
        ri,
        min(radius, ro),
        degree + 2 + power,
    )
    outside_integral = scale * integrate(
        max(radius, ri),
        ro,
        1 - degree + power,
    )
    prefactor = 4.0 * np.pi * gravity_constant / (2 * degree + 1)
    return float(
        prefactor
        * (
            radius ** (-(degree + 1)) * inside_integral
            + radius**degree * outside_integral
        )
    )


def spherical_shell_potential_operator(
    *,
    radius_inner: Real,
    radius_outer: Real,
    harmonic_degree: int,
    internal_load_radius: Real | None = None,
    internal_load_coefficient: Real = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the normalized surface/CMB sheet operator and load vector.

    Rows target the surface then CMB. Columns source unit surface then CMB
    density/topography coefficients. The normalization has ``4*pi*G/g = 1``.
    """

    radius_inner, radius_outer, degree = _validate_geometry(
        radius_inner, radius_outer, harmonic_degree
    )
    load_coefficient = _finite_float(
        internal_load_coefficient, "internal_load_coefficient"
    )
    denominator = float(2 * degree + 1)

    def normalized_sheet(source_radius: float, target_radius: float) -> float:
        return float(
            source_radius
            * spherical_sheet_radial_factor(
                source_radius=source_radius,
                target_radius=target_radius,
                harmonic_degree=degree,
            )
            / denominator
        )

    operator = np.array(
        [
            [
                normalized_sheet(radius_outer, radius_outer),
                normalized_sheet(radius_inner, radius_outer),
            ],
            [
                normalized_sheet(radius_outer, radius_inner),
                normalized_sheet(radius_inner, radius_inner),
            ],
        ],
        dtype=float,
    )

    load = np.zeros(2, dtype=float)
    if internal_load_radius is None:
        if load_coefficient != 0.0:
            raise ValueError(
                "internal_load_radius is required for a nonzero internal load."
            )
    else:
        load_radius = _positive_float(internal_load_radius, "internal_load_radius")
        if not radius_inner < load_radius < radius_outer:
            raise ValueError(
                "internal_load_radius must lie strictly between the shell radii."
            )
        load = -load_coefficient * np.array(
            [
                normalized_sheet(load_radius, radius_outer),
                normalized_sheet(load_radius, radius_inner),
            ],
            dtype=float,
        )
    return operator, load


def spherical_shell_geoid_response(
    *,
    radius_inner: Real,
    radius_outer: Real,
    harmonic_degree: int,
    surface_topography_coefficient: Real,
    cmb_topography_coefficient: Real,
    internal_load_radius: Real | None = None,
    internal_load_coefficient: Real = 0.0,
) -> SphericalGeoidResponse:
    """Return normalized no-self-gravity surface and CMB geoid coefficients."""

    operator, load = spherical_shell_potential_operator(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        harmonic_degree=harmonic_degree,
        internal_load_radius=internal_load_radius,
        internal_load_coefficient=internal_load_coefficient,
    )
    topography = np.array(
        [
            _finite_float(
                surface_topography_coefficient, "surface_topography_coefficient"
            ),
            _finite_float(cmb_topography_coefficient, "cmb_topography_coefficient"),
        ],
        dtype=float,
    )
    geoid = operator @ topography + load
    return SphericalGeoidResponse(float(geoid[0]), float(geoid[1]))


def spherical_shell_self_gravity_response(
    *,
    radius_inner: Real,
    radius_outer: Real,
    harmonic_degree: int,
    surface_topography_coefficient: Real,
    cmb_topography_coefficient: Real,
    surface_density_contrast: Real,
    cmb_density_contrast: Real,
    planet_radius: Real,
    gravity: Real,
    internal_load_radius: Real | None = None,
    internal_load_coefficient: Real = 0.0,
    gravitational_constant: Real = 6.67e-11,
) -> SphericalSelfGravityResponse:
    """Return the independent two-interface self-gravity response."""

    operator, load = spherical_shell_potential_operator(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        harmonic_degree=harmonic_degree,
        internal_load_radius=internal_load_radius,
        internal_load_coefficient=internal_load_coefficient,
    )
    topography = np.array(
        [
            _finite_float(
                surface_topography_coefficient, "surface_topography_coefficient"
            ),
            _finite_float(cmb_topography_coefficient, "cmb_topography_coefficient"),
        ],
        dtype=float,
    )
    density_contrasts = np.array(
        [
            _finite_float(surface_density_contrast, "surface_density_contrast"),
            _finite_float(cmb_density_contrast, "cmb_density_contrast"),
        ],
        dtype=float,
    )
    planet_radius = _positive_float(planet_radius, "planet_radius")
    gravity = _positive_float(gravity, "gravity")
    gravity_constant = _positive_float(
        gravitational_constant, "gravitational_constant"
    )

    q = 4.0 * np.pi * gravity_constant * planet_radius * density_contrasts / gravity
    q_matrix = np.diag(q)
    system = np.eye(2) - q_matrix @ operator
    right_hand_side = topography + q_matrix @ load
    corrected_topography = np.linalg.solve(system, right_hand_side)
    corrected_geoid = operator @ corrected_topography + load
    residual = system @ corrected_topography - right_hand_side

    return SphericalSelfGravityResponse(
        surface_topography=float(corrected_topography[0]),
        cmb_topography=float(corrected_topography[1]),
        surface_geoid=float(corrected_geoid[0]),
        cmb_geoid=float(corrected_geoid[1]),
        q_surface=float(q[0]),
        q_cmb=float(q[1]),
        matrix_condition_number=float(np.linalg.cond(system)),
        matrix_residual_norm=float(np.linalg.norm(residual)),
    )
