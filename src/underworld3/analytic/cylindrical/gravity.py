r"""Analytical cylindrical sheet/volume gravity and coefficient responses.

The source convention is laplacian(Phi)=-4*pi*G*delta_rho and N=Phi/g.
These coefficient kernels need no mesh, Stokes solver or MPI imports.
The n=0 logarithmic monopole is unsupported; n>=1 is accepted.

Source provenance
-----------------
The cylindrical sheet/topography construction follows Simons, M. (1996),
Localization of Gravity and Topography: Constraints on the Tectonics and Mantle
Dynamics of Earth and Venus, PhD thesis, Massachusetts Institute of Technology,
Appendix B. Volume-density and feedback support are UW3 extensions derived from
the stated Poisson equation; they are not claimed as published Simons results.
Related Stokes benchmark: Kramer, Davies and Wilson (2021), Analytical solutions
for mantle flow in cylindrical and spherical shells.
https://doi.org/10.5194/gmd-14-1899-2021
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CylindricalGravityResponse:
    r"""Cylindrical potential and geoid coefficients without interface feedback.

    Attributes
    ----------
    outer_potential, inner_potential : float
        Signed coefficients of ``cos(n*theta)`` at the annulus boundaries, in
        potential units (length^2/time^2 for consistent physical inputs).
    outer_geoid, inner_geoid : float
        Potential divided by the corresponding reference gravity. Units are length
        for physical inputs, or the caller's consistently nondimensional units."""

    outer_potential: float
    inner_potential: float
    outer_geoid: float
    inner_geoid: float


@dataclass(frozen=True)
class CylindricalSelfGravityResponse:
    r"""Cylindrical interface response after potential-displacement feedback.

    Attributes
    ----------
    outer_topography, inner_topography : float
        Corrected signed outward-displacement cosine coefficients, in length units.
    outer_potential, inner_potential : float
        Corrected potential cosine coefficients, in potential units.
    outer_geoid, inner_geoid : float
        Corrected potential divided by each boundary's reference gravity.
    q_outer, q_inner : float
        Feedback factors converting potential to displacement, normally ``1/g``.
    matrix_residual_norm : float
        Euclidean norm of the two-interface displacement-equation residual.
        This is not a spatial field error or an accuracy acceptance test.

    Notes
    -----
    The same attributes may use consistent nondimensional units when all inputs
    are nondimensional. Gravity represents an axially invariant cylindrical source."""

    outer_topography: float
    inner_topography: float
    outer_potential: float
    inner_potential: float
    outer_geoid: float
    inner_geoid: float
    q_outer: float
    q_inner: float
    matrix_residual_norm: float


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


def cylindrical_sheet_potential_coefficient(
    *,
    source_radius: float,
    target_radius: float,
    wavenumber: int,
    surface_density_coefficient: float,
    gravitational_constant: float = 1.0,
) -> float:
    r"""Return one cylindrical mass sheet's potential coefficient.

    The result multiplies the unnormalised real Fourier basis
    :math:`\cos(n\theta)`. For :math:`n\geq 1`, a sheet at radius :math:`r_s`
    has

    .. math::

       \Phi_n(r_s) = \frac{2\pi G r_s\sigma_n}{n},

    with radial factors :math:`(r/r_s)^n` inside and :math:`(r_s/r)^n`
    outside. Positive density gives positive potential under the convention
    :math:`\nabla^2\Phi=-4\pi G\rho` and :math:`\mathbf{g}=\nabla\Phi`.

    Radii, density, and ``gravitational_constant`` may be dimensional or
    nondimensional, but they must use one mutually consistent unit system.
    The axisymmetric ``n=0`` mode is intentionally excluded because its
    logarithmic exterior branch requires a separate potential gauge.

    Parameters
    ----------
    source_radius, target_radius : real
        Positive source-sheet and evaluation radii.
    wavenumber : int
        Positive azimuthal Fourier wavenumber.
    surface_density_coefficient : real
        Sheet-density coefficient multiplying :math:`\cos(n\theta)`.
    gravitational_constant : real, default=1
        Positive gravitational constant in the selected unit system.

    Returns
    -------
    float
        Potential coefficient at ``target_radius``.

    Raises
    ------
    TypeError
        Wavenumber is not an integer or numeric inputs cannot be converted.
    ValueError
        Geometry, mode, finite coefficients, source or positive-constant restrictions fail.

    Notes
    -----
    Use one consistent physical or nondimensional unit system; no conversion is
    performed. Radii and topography have length units, surface-density coefficients
    have mass/area units, volume densities/contrasts have mass/volume units,
    G has length^3/(mass*time^2), and potential has length^2/time^2. Geoid
    coefficients, where returned, are potential divided by positive reference gravity.
    The unnormalized cosine basis is shared by every source and response."""

    source_radius = _positive_float(source_radius, "source_radius")
    target_radius = _positive_float(target_radius, "target_radius")
    mode = _validate_cylindrical_mode(wavenumber)
    density = _finite_float(
        surface_density_coefficient,
        "surface_density_coefficient",
    )
    gravity_constant = _positive_float(
        gravitational_constant,
        "gravitational_constant",
    )

    radial_factor = (min(source_radius, target_radius) / max(source_radius, target_radius)) ** mode
    source_amplitude = 2.0 * np.pi * gravity_constant * source_radius * density / mode
    return float(source_amplitude * radial_factor)


def cylindrical_sheet_radial_derivative_coefficient(
    *,
    source_radius: float,
    target_radius: float,
    wavenumber: int,
    surface_density_coefficient: float,
    gravitational_constant: float = 1.0,
    source_side: str | None = None,
) -> float:
    r"""Return :math:`d\Phi_n/dr` on either side of a cylindrical sheet.

    At ``target_radius == source_radius``, ``source_side`` must be
    ``"inside"`` or ``"outside"`` because the derivative is discontinuous.
    The returned branches satisfy
    :math:`[d\Phi_n/dr]_{outside-inside}=-4\pi G\sigma_n`.

    Parameters
    ----------
    source_radius, target_radius : real
        Positive source-sheet and evaluation radii.
    wavenumber : int
        Positive azimuthal Fourier wavenumber.
    surface_density_coefficient : real
        Sheet-density coefficient multiplying :math:`\cos(n\theta)`.
    gravitational_constant : real, default=1
        Positive gravitational constant in the selected unit system.
    source_side : {"inside", "outside"}, optional
        Radial branch when evaluating exactly on the sheet.

    Returns
    -------
    float
        Radial derivative coefficient at ``target_radius``.

    Raises
    ------
    TypeError
        Wavenumber is not an integer or numeric inputs cannot be converted.
    ValueError
        Geometry, mode, finite coefficients, source or positive-constant restrictions fail.

    Notes
    -----
    Use one consistent physical or nondimensional unit system; no conversion is
    performed. Radii and topography have length units, surface-density coefficients
    have mass/area units, volume densities/contrasts have mass/volume units,
    G has length^3/(mass*time^2), and potential has length^2/time^2. Geoid
    coefficients, where returned, are potential divided by positive reference gravity.
    The unnormalized cosine basis is shared by every source and response.
    Radial derivative coefficients have potential/length units."""

    source_radius = _positive_float(source_radius, "source_radius")
    target_radius = _positive_float(target_radius, "target_radius")
    mode = _validate_cylindrical_mode(wavenumber)
    potential = cylindrical_sheet_potential_coefficient(
        source_radius=source_radius,
        target_radius=target_radius,
        wavenumber=mode,
        surface_density_coefficient=surface_density_coefficient,
        gravitational_constant=gravitational_constant,
    )

    if target_radius == source_radius:
        if source_side not in ("inside", "outside"):
            raise ValueError("source_side must be 'inside' or 'outside' at the sheet radius.")
        branch_sign = 1.0 if source_side == "inside" else -1.0
    else:
        if source_side is not None:
            raise ValueError("source_side is only valid when target_radius equals source_radius.")
        branch_sign = 1.0 if target_radius < source_radius else -1.0
    return float(branch_sign * mode * potential / target_radius)


def cylindrical_annulus_potential_operator(
    *,
    radius_inner: float,
    radius_outer: float,
    wavenumber: int,
    outer_density_contrast: float,
    inner_density_contrast: float,
    gravitational_constant: float = 1.0,
) -> np.ndarray:
    r"""Return the two-boundary operator :math:`\Phi=G_n h`.

    Rows are target boundaries ``[outer, inner]`` and columns are topographic
    sheet sources ``[outer, inner]``. Density contrasts are signed as density
    on the smaller-radius side minus density on the larger-radius side. Thus
    positive outward topography creates sheet density
    :math:`\Delta\rho\,h`.

    Parameters
    ----------
    radius_inner, radius_outer : real
        Positive annulus radii ordered from inner to outer.
    wavenumber : int
        Positive azimuthal Fourier wavenumber.
    outer_density_contrast, inner_density_contrast : real
        Signed density contrasts at the two boundaries.
    gravitational_constant : real, default=1
        Positive gravitational constant in the selected unit system.

    Returns
    -------
    numpy.ndarray
        Two-by-two potential operator with targets in rows and sources in
        columns, both ordered ``[outer, inner]``.

    Raises
    ------
    TypeError
        Wavenumber is not an integer or numeric inputs cannot be converted.
    ValueError
        Geometry, mode, finite coefficients, source or positive-constant restrictions fail.

    Notes
    -----
    Use one consistent physical or nondimensional unit system; no conversion is
    performed. Radii and topography have length units, surface-density coefficients
    have mass/area units, volume densities/contrasts have mass/volume units,
    G has length^3/(mass*time^2), and potential has length^2/time^2. Geoid
    coefficients, where returned, are potential divided by positive reference gravity.
    The unnormalized cosine basis is shared by every source and response."""

    radius_inner, radius_outer, mode = _validate_cylindrical_annulus(
        radius_inner,
        radius_outer,
        wavenumber,
    )
    density_contrasts = np.array(
        [
            _finite_float(
                outer_density_contrast,
                "outer_density_contrast",
            ),
            _finite_float(
                inner_density_contrast,
                "inner_density_contrast",
            ),
        ]
    )
    gravity_constant = _positive_float(
        gravitational_constant,
        "gravitational_constant",
    )
    source_radii = (radius_outer, radius_inner)
    target_radii = (radius_outer, radius_inner)

    operator = np.empty((2, 2), dtype=float)
    for row, target_radius in enumerate(target_radii):
        for column, (source_radius, density) in enumerate(zip(source_radii, density_contrasts)):
            operator[row, column] = cylindrical_sheet_potential_coefficient(
                source_radius=source_radius,
                target_radius=target_radius,
                wavenumber=mode,
                surface_density_coefficient=density,
                gravitational_constant=gravity_constant,
            )
    return operator


def _cylindrical_internal_load_vector(
    *,
    radius_inner: float,
    radius_outer: float,
    wavenumber: int,
    internal_load_radius: float | None,
    internal_surface_density_coefficient: float,
    gravitational_constant: float,
) -> np.ndarray:
    """Return boundary potential coefficients from an internal mass sheet."""

    load_density = _finite_float(
        internal_surface_density_coefficient,
        "internal_surface_density_coefficient",
    )
    load = np.zeros(2, dtype=float)
    if internal_load_radius is None:
        if load_density != 0.0:
            raise ValueError("internal_load_radius is required for a nonzero internal load.")
        return load

    internal_load_radius = _positive_float(
        internal_load_radius,
        "internal_load_radius",
    )
    if not radius_inner < internal_load_radius < radius_outer:
        raise ValueError("internal_load_radius must lie strictly inside the annulus.")
    for index, target_radius in enumerate((radius_outer, radius_inner)):
        load[index] = cylindrical_sheet_potential_coefficient(
            source_radius=internal_load_radius,
            target_radius=target_radius,
            wavenumber=wavenumber,
            surface_density_coefficient=load_density,
            gravitational_constant=gravitational_constant,
        )
    return load


def cylindrical_annulus_geoid_response(
    *,
    radius_inner: float,
    radius_outer: float,
    wavenumber: int,
    outer_topography_coefficient: float,
    inner_topography_coefficient: float,
    outer_density_contrast: float,
    inner_density_contrast: float,
    outer_reference_gravity: float,
    inner_reference_gravity: float,
    internal_load_radius: float | None = None,
    internal_surface_density_coefficient: float = 0.0,
    volume_density_radial_exponent: float | None = None,
    volume_density_coefficient: float = 0.0,
    gravitational_constant: float = 1.0,
) -> CylindricalGravityResponse:
    r"""Assemble annulus potential and geoid coefficients for one mode.

    Potential and topography retain their physical signs. Geoid is defined as
    :math:`N_n=\Phi_n/g_{reference}` independently at the outer and inner
    boundaries. The optional internal source is a cylindrical sheet-density
    coefficient in the same Fourier normalisation and unit system.

    Parameters
    ----------
    radius_inner, radius_outer : real
        Positive annulus radii ordered from inner to outer.
    wavenumber : int
        Positive azimuthal Fourier wavenumber.
    outer_topography_coefficient, inner_topography_coefficient : real
        Signed boundary topography coefficients.
    outer_density_contrast, inner_density_contrast : real
        Signed density contrasts at the two boundaries.
    outer_reference_gravity, inner_reference_gravity : real
        Positive gravity magnitudes used to convert potential to geoid.
    internal_load_radius : real, optional
        Radius of an internal sheet, strictly inside the annulus.
    internal_surface_density_coefficient : real, default=0
        Density coefficient of the optional internal sheet.
    volume_density_radial_exponent : real or None, default=None
        Finite power in ``rho(r)=volume_density_coefficient*(r/radius_outer)**k``.
        Required for nonzero volume density; source support is the annulus.
    volume_density_coefficient : real, default=0
        Finite signed volume-density amplitude at the outer source radius.
    gravitational_constant : real, default=1
        Positive gravitational constant in the selected unit system.

    Returns
    -------
    CylindricalGravityResponse
        Outer and inner potential and geoid coefficients.

    Raises
    ------
    TypeError
        Wavenumber is not an integer or numeric inputs cannot be converted.
    ValueError
        Geometry, mode, finite coefficients, source or positive-constant restrictions fail.

    Notes
    -----
    Use one consistent physical or nondimensional unit system; no conversion is
    performed. Radii and topography have length units, surface-density coefficients
    have mass/area units, volume densities/contrasts have mass/volume units,
    G has length^3/(mass*time^2), and potential has length^2/time^2. Geoid
    coefficients, where returned, are potential divided by positive reference gravity.
    The unnormalized cosine basis is shared by every source and response."""

    radius_inner, radius_outer, mode = _validate_cylindrical_annulus(
        radius_inner,
        radius_outer,
        wavenumber,
    )
    topography = np.array(
        [
            _finite_float(
                outer_topography_coefficient,
                "outer_topography_coefficient",
            ),
            _finite_float(
                inner_topography_coefficient,
                "inner_topography_coefficient",
            ),
        ],
        dtype=float,
    )
    reference_gravity = np.array(
        [
            _positive_float(
                outer_reference_gravity,
                "outer_reference_gravity",
            ),
            _positive_float(
                inner_reference_gravity,
                "inner_reference_gravity",
            ),
        ],
        dtype=float,
    )
    gravity_constant = _positive_float(
        gravitational_constant,
        "gravitational_constant",
    )
    operator = cylindrical_annulus_potential_operator(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        wavenumber=mode,
        outer_density_contrast=outer_density_contrast,
        inner_density_contrast=inner_density_contrast,
        gravitational_constant=gravity_constant,
    )
    load = _cylindrical_internal_load_vector(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        wavenumber=mode,
        internal_load_radius=internal_load_radius,
        internal_surface_density_coefficient=(internal_surface_density_coefficient),
        gravitational_constant=gravity_constant,
    )

    load += _volume_load_vector(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        wavenumber=mode,
        volume_density_radial_exponent=volume_density_radial_exponent,
        volume_density_coefficient=volume_density_coefficient,
        gravitational_constant=gravity_constant,
    )
    potential = operator @ topography + load
    geoid = potential / reference_gravity
    return CylindricalGravityResponse(
        outer_potential=float(potential[0]),
        inner_potential=float(potential[1]),
        outer_geoid=float(geoid[0]),
        inner_geoid=float(geoid[1]),
    )


def cylindrical_annulus_self_gravity_response(
    *,
    radius_inner: float,
    radius_outer: float,
    wavenumber: int,
    outer_topography_coefficient: float,
    inner_topography_coefficient: float,
    outer_density_contrast: float,
    inner_density_contrast: float,
    outer_reference_gravity: float,
    inner_reference_gravity: float,
    internal_load_radius: float | None = None,
    internal_surface_density_coefficient: float = 0.0,
    volume_density_radial_exponent: float | None = None,
    volume_density_coefficient: float = 0.0,
    gravitational_constant: float = 1.0,
    outer_feedback_factor: float | None = None,
    inner_feedback_factor: float | None = None,
) -> CylindricalSelfGravityResponse:
    r"""Return the two-boundary cylindrical self-gravity correction.

    Holding the hydrodynamic traction fixed gives
    :math:`h_{sg}=h+Q\Phi_{sg}`. For positive reference-gravity magnitudes the
    default factors are :math:`Q=diag(1/g_o,1/g_i)`. Explicit factors may be
    supplied to test a signed convention or disable either feedback row. The
    solved equation is

    .. math::

       (I-QG_n)h_{sg}=h+Q\phi_{load}.

    Parameters
    ----------
    radius_inner, radius_outer : real
        Positive annulus radii ordered from inner to outer.
    wavenumber : int
        Positive azimuthal Fourier wavenumber.
    outer_topography_coefficient, inner_topography_coefficient : real
        Hydrodynamic topography coefficients before self-gravity feedback.
    outer_density_contrast, inner_density_contrast : real
        Signed density contrasts at the two boundaries.
    outer_reference_gravity, inner_reference_gravity : real
        Positive reference-gravity magnitudes.
    internal_load_radius : real, optional
        Radius of an internal sheet, strictly inside the annulus.
    internal_surface_density_coefficient : real, default=0
        Density coefficient of the optional internal sheet.
    volume_density_radial_exponent : real or None, default=None
        Finite power in ``rho(r)=volume_density_coefficient*(r/radius_outer)**k``.
        Required for nonzero volume density; source support is the annulus.
    volume_density_coefficient : real, default=0
        Finite signed volume-density amplitude at the outer source radius.
    gravitational_constant : real, default=1
        Positive gravitational constant in the selected unit system.
    outer_feedback_factor, inner_feedback_factor : real, optional
        Explicit diagonal entries of :math:`Q`; defaults are reciprocal
        reference-gravity magnitudes.

    Returns
    -------
    CylindricalSelfGravityResponse
        Corrected topography, potential, geoid, feedback, and residual values.

    Raises
    ------
    TypeError
        Wavenumber is not an integer or numeric inputs cannot be converted.
    ValueError
        Geometry, mode, finite coefficients, source or positive-constant restrictions fail.

    Notes
    -----
    Use one consistent physical or nondimensional unit system; no conversion is
    performed. Radii and topography have length units, surface-density coefficients
    have mass/area units, volume densities/contrasts have mass/volume units,
    G has length^3/(mass*time^2), and potential has length^2/time^2. Geoid
    coefficients, where returned, are potential divided by positive reference gravity.
    The unnormalized cosine basis is shared by every source and response.
    A singular feedback matrix raises ValueError; default feedback factors are 1/g."""

    radius_inner, radius_outer, mode = _validate_cylindrical_annulus(
        radius_inner,
        radius_outer,
        wavenumber,
    )
    reference_gravity = np.array(
        [
            _positive_float(
                outer_reference_gravity,
                "outer_reference_gravity",
            ),
            _positive_float(
                inner_reference_gravity,
                "inner_reference_gravity",
            ),
        ],
        dtype=float,
    )
    topography = np.array(
        [
            _finite_float(
                outer_topography_coefficient,
                "outer_topography_coefficient",
            ),
            _finite_float(
                inner_topography_coefficient,
                "inner_topography_coefficient",
            ),
        ],
        dtype=float,
    )
    gravity_constant = _positive_float(
        gravitational_constant,
        "gravitational_constant",
    )
    operator = cylindrical_annulus_potential_operator(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        wavenumber=mode,
        outer_density_contrast=outer_density_contrast,
        inner_density_contrast=inner_density_contrast,
        gravitational_constant=gravity_constant,
    )
    load = _cylindrical_internal_load_vector(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        wavenumber=mode,
        internal_load_radius=internal_load_radius,
        internal_surface_density_coefficient=(internal_surface_density_coefficient),
        gravitational_constant=gravity_constant,
    )
    load += _volume_load_vector(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        wavenumber=mode,
        volume_density_radial_exponent=volume_density_radial_exponent,
        volume_density_coefficient=volume_density_coefficient,
        gravitational_constant=gravity_constant,
    )
    feedback = np.array(
        [
            (
                1.0 / reference_gravity[0]
                if outer_feedback_factor is None
                else _finite_float(
                    outer_feedback_factor,
                    "outer_feedback_factor",
                )
            ),
            (
                1.0 / reference_gravity[1]
                if inner_feedback_factor is None
                else _finite_float(
                    inner_feedback_factor,
                    "inner_feedback_factor",
                )
            ),
        ],
        dtype=float,
    )
    q_matrix = np.diag(feedback)
    system_matrix = np.eye(2) - q_matrix @ operator
    right_hand_side = topography + q_matrix @ load
    try:
        corrected_topography = np.linalg.solve(
            system_matrix,
            right_hand_side,
        )
    except np.linalg.LinAlgError as error:
        raise ValueError("The self-gravity feedback matrix is singular.") from error
    corrected_potential = operator @ corrected_topography + load
    corrected_geoid = corrected_potential / reference_gravity
    residual = system_matrix @ corrected_topography - right_hand_side

    return CylindricalSelfGravityResponse(
        outer_topography=float(corrected_topography[0]),
        inner_topography=float(corrected_topography[1]),
        outer_potential=float(corrected_potential[0]),
        inner_potential=float(corrected_potential[1]),
        outer_geoid=float(corrected_geoid[0]),
        inner_geoid=float(corrected_geoid[1]),
        q_outer=float(feedback[0]),
        q_inner=float(feedback[1]),
        matrix_residual_norm=float(np.linalg.norm(residual)),
    )


def _power_integral(lower: float, upper: float, exponent: float) -> float:
    r"""Return ``integral(lower**upper, r**exponent dr)`` stably."""

    shifted_exponent = exponent + 1.0
    if np.isclose(shifted_exponent, 0.0, rtol=0.0, atol=1.0e-14):
        return float(np.log(upper / lower))
    return float(
        lower**shifted_exponent
        * np.expm1(shifted_exponent * np.log(upper / lower))
        / shifted_exponent
    )


def cylindrical_power_law_volume_potential_coefficient(
    *,
    radius_inner: float,
    radius_outer: float,
    target_radius: float,
    wavenumber: int,
    radial_exponent: float,
    density_coefficient: float = 1.0,
    gravitational_constant: float = 1.0,
) -> float:
    r"""Return cylindrical potential for a compact power-law volume-density source.

    Parameters
    ----------
    radius_inner, radius_outer : float
        Positive source-support radii in consistent length units, ordered inner < outer.
    target_radius : float
        Positive evaluation radius; it may be inside, within or outside the source annulus.
    wavenumber : int
        Positive Fourier mode ``n >= 1`` for the unnormalized ``cos(n*theta)`` basis.
    radial_exponent : float
        Finite real power ``k`` in ``rho_n(s)=density_coefficient*(s/radius_outer)**k``.
    density_coefficient : float, default=1
        Finite signed volume-density amplitude at ``radius_outer``; physical units
        are mass per volume for the axially invariant source.
    gravitational_constant : float, default=1
        Finite positive G in units consistent with radii and density.

    Returns
    -------
    float
        Potential cosine coefficient at ``target_radius``. For physical inputs its
        units are length^2/time^2; positive density produces positive potential
        under ``laplacian(Phi)=-4*pi*G*rho``.

    Raises
    ------
    TypeError
        Mode has an invalid type or other parameters cannot be converted to numbers.
    ValueError
        Radius ordering, positive mode/G or finite coefficient/exponent checks fail.

    Notes
    -----
    Superposition of sheets gives
    ``Phi_n(r)=(2*pi*G/n)*integral(s*rho_n(s)*K_n(r,s), ds)`` with
    ``K_n=(s/r)**n`` for ``s<r`` and ``(r/s)**n`` for ``s>r``.
    The closed form includes logarithmic exponent limits. No mesh, MPI operation
    or physical/nondimensional unit conversion is involved."""

    radius_inner, radius_outer, mode = _validate_cylindrical_annulus(
        radius_inner, radius_outer, wavenumber
    )
    target_radius = _positive_float(target_radius, "target_radius")
    exponent = _finite_float(radial_exponent, "radial_exponent")
    density = _finite_float(density_coefficient, "density_coefficient")
    gravity_constant = _positive_float(gravitational_constant, "gravitational_constant")

    density_scale = density / radius_outer**exponent
    potential_integral = 0.0
    source_below_target = min(target_radius, radius_outer)
    if source_below_target > radius_inner:
        potential_integral += (
            _power_integral(
                radius_inner,
                source_below_target,
                exponent + mode + 1.0,
            )
            / target_radius**mode
        )

    source_above_target = max(target_radius, radius_inner)
    if source_above_target < radius_outer:
        potential_integral += target_radius**mode * _power_integral(
            source_above_target,
            radius_outer,
            exponent - mode + 1.0,
        )

    return float(2.0 * np.pi * gravity_constant * density_scale * potential_integral / mode)


def cylindrical_power_law_volume_radial_derivative_coefficient(
    *,
    radius_inner: float,
    radius_outer: float,
    wavenumber: int,
    radial_exponent: float,
    density_coefficient: float,
    target_radius: float,
    gravitational_constant: float = 1.0,
) -> float:
    r"""Return the radial derivative of the continuous power-law volume potential.

    Differentiate the split radial integrals analytically. The moving-endpoint
    terms cancel, including at both density-domain boundaries. Resonant power
    integrals retain their logarithmic limits.

    Parameters
    ----------
    radius_inner : real
        Finite positive inner source/shell radius in consistent length units.
    radius_outer : real
        Finite positive outer source/shell radius, strictly larger than radius_inner.
    wavenumber : int
        Positive cylindrical Fourier mode n >= 1 in the unnormalized cosine basis.
    radial_exponent : real
        Finite real power k of compact cylindrical density coefficient*(r/radius_outer)**k.
    density_coefficient : real
        Finite signed cylindrical volume-density amplitude at radius_outer; physical units
        are mass/volume.
    target_radius : real
        Finite positive evaluation radius, inside or outside the source support.
    gravitational_constant : real, default=1.0
        Finite positive G in units consistent with radii and densities. No unit conversion
        is performed.

    Returns
    -------
    float
        Radial potential derivative coefficient in potential/length units. Compact
        volume-source derivatives are continuous at the source-support boundaries.

    Raises
    ------
    TypeError
        The harmonic degree/mode or density power has an invalid type, or a real input
        cannot be converted.
    ValueError
        A numeric input is nonfinite, a required positive value is nonpositive, or the
        harmonic degree/mode, radius ordering or source support is invalid.

    Notes
    -----
    The convention is laplacian(Phi)=-4*pi*G*rho. All density coefficients and returned coefficients
    share the selected angular basis/normalization. No mesh, MPI operation or PDE solve is
    performed.
    """
    ri, ro, mode = _validate_cylindrical_annulus(radius_inner, radius_outer, wavenumber)
    target = _positive_float(target_radius, "target_radius")
    exponent = _finite_float(radial_exponent, "radial_exponent")
    density = _finite_float(density_coefficient, "density_coefficient")
    G = _positive_float(gravitational_constant, "gravitational_constant")
    below = _power_integral(ri, min(target, ro), exponent + mode + 1) if target > ri else 0.0
    above = _power_integral(max(target, ri), ro, exponent - mode + 1) if target < ro else 0.0
    return float(
        2
        * np.pi
        * G
        * density
        / ro**exponent
        * (-below / target ** (mode + 1) + target ** (mode - 1) * above)
    )


def _volume_load_vector(
    *,
    radius_inner: float,
    radius_outer: float,
    wavenumber: int,
    volume_density_radial_exponent: float | None,
    volume_density_coefficient: float,
    gravitational_constant: float,
) -> np.ndarray:
    """Return boundary potentials from an optional power-law volume load."""

    density = _finite_float(volume_density_coefficient, "volume_density_coefficient")
    if volume_density_radial_exponent is None:
        if density != 0.0:
            raise ValueError(
                "volume_density_radial_exponent is required for a nonzero " "volume density."
            )
        return np.zeros(2, dtype=float)

    return np.array(
        [
            cylindrical_power_law_volume_potential_coefficient(
                radius_inner=radius_inner,
                radius_outer=radius_outer,
                target_radius=target_radius,
                wavenumber=wavenumber,
                radial_exponent=volume_density_radial_exponent,
                density_coefficient=density,
                gravitational_constant=gravitational_constant,
            )
            for target_radius in (radius_outer, radius_inner)
        ],
        dtype=float,
    )


__all__ = [
    "CylindricalGravityResponse",
    "CylindricalSelfGravityResponse",
    "cylindrical_sheet_potential_coefficient",
    "cylindrical_sheet_radial_derivative_coefficient",
    "cylindrical_annulus_potential_operator",
    "cylindrical_annulus_geoid_response",
    "cylindrical_annulus_self_gravity_response",
    "cylindrical_power_law_volume_potential_coefficient",
    "cylindrical_power_law_volume_radial_derivative_coefficient",
]
