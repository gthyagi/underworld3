"""Independent sheet-jump, continuation and radial-quadrature gravity tests."""

import numpy as np
import pytest

from underworld3.analytic.cylindrical.gravity import (
    cylindrical_annulus_geoid_response,
    cylindrical_annulus_potential_operator,
    cylindrical_annulus_self_gravity_response,
    cylindrical_power_law_volume_potential_coefficient,
    cylindrical_power_law_volume_radial_derivative_coefficient,
    cylindrical_sheet_potential_coefficient,
    cylindrical_sheet_radial_derivative_coefficient,
)


RADIUS_INNER = 1.22
RADIUS_INTERNAL = 2.0
RADIUS_OUTER = 2.22
WAVENUMBER = 2
GRAVITATIONAL_CONSTANT = 3.7


@pytest.mark.parametrize("target", [0.8, 1.6, 2.7])
@pytest.mark.parametrize("mode,power", [(2, 2), (4, 2)])
def test_volume_radial_derivative_matches_centered_difference(target, mode, power):
    kwargs = dict(
        radius_inner=RADIUS_INNER,
        radius_outer=RADIUS_OUTER,
        wavenumber=mode,
        radial_exponent=power,
        density_coefficient=-0.43,
        gravitational_constant=0.61,
    )
    step = 1e-5 * target
    plus = cylindrical_power_law_volume_potential_coefficient(target_radius=target + step, **kwargs)
    minus = cylindrical_power_law_volume_potential_coefficient(
        target_radius=target - step, **kwargs
    )
    actual = cylindrical_power_law_volume_radial_derivative_coefficient(
        target_radius=target, **kwargs
    )
    np.testing.assert_allclose(actual, (plus - minus) / (2 * step), rtol=2e-8, atol=2e-10)


def test_sheet_potential_is_continuous_at_source_radius():
    kwargs = {
        "source_radius": RADIUS_INTERNAL,
        "wavenumber": WAVENUMBER,
        "surface_density_coefficient": -0.73,
        "gravitational_constant": GRAVITATIONAL_CONSTANT,
    }
    source_value = cylindrical_sheet_potential_coefficient(target_radius=RADIUS_INTERNAL, **kwargs)
    expected = (
        2.0
        * np.pi
        * GRAVITATIONAL_CONSTANT
        * RADIUS_INTERNAL
        * kwargs["surface_density_coefficient"]
        / WAVENUMBER
    )

    np.testing.assert_allclose(source_value, expected, rtol=1.0e-15)
    for fractional_offset in (-1.0e-8, 1.0e-8):
        adjacent_value = cylindrical_sheet_potential_coefficient(
            target_radius=RADIUS_INTERNAL * (1.0 + fractional_offset),
            **kwargs,
        )
        np.testing.assert_allclose(adjacent_value, source_value, rtol=3.0e-8)


def test_sheet_derivative_has_poisson_jump():
    surface_density = 0.41
    kwargs = {
        "source_radius": RADIUS_INTERNAL,
        "target_radius": RADIUS_INTERNAL,
        "wavenumber": WAVENUMBER,
        "surface_density_coefficient": surface_density,
        "gravitational_constant": GRAVITATIONAL_CONSTANT,
    }
    derivative_inside = cylindrical_sheet_radial_derivative_coefficient(
        source_side="inside", **kwargs
    )
    derivative_outside = cylindrical_sheet_radial_derivative_coefficient(
        source_side="outside", **kwargs
    )

    expected_jump = -4.0 * np.pi * GRAVITATIONAL_CONSTANT * surface_density
    np.testing.assert_allclose(
        derivative_outside - derivative_inside,
        expected_jump,
        rtol=1.0e-15,
    )


@pytest.mark.parametrize("wavenumber", [2, 3, 4, 8])
def test_supported_modes_satisfy_jump_and_radial_continuation(wavenumber):
    surface_density = -0.37
    kwargs = {
        "source_radius": RADIUS_INTERNAL,
        "wavenumber": wavenumber,
        "surface_density_coefficient": surface_density,
        "gravitational_constant": GRAVITATIONAL_CONSTANT,
    }
    at_source = cylindrical_sheet_potential_coefficient(target_radius=RADIUS_INTERNAL, **kwargs)
    inner_radius = 0.7 * RADIUS_INTERNAL
    outer_radius = 1.4 * RADIUS_INTERNAL
    np.testing.assert_allclose(
        cylindrical_sheet_potential_coefficient(target_radius=inner_radius, **kwargs),
        at_source * (inner_radius / RADIUS_INTERNAL) ** wavenumber,
    )
    np.testing.assert_allclose(
        cylindrical_sheet_potential_coefficient(target_radius=outer_radius, **kwargs),
        at_source * (RADIUS_INTERNAL / outer_radius) ** wavenumber,
    )
    derivative_inside = cylindrical_sheet_radial_derivative_coefficient(
        target_radius=RADIUS_INTERNAL, source_side="inside", **kwargs
    )
    derivative_outside = cylindrical_sheet_radial_derivative_coefficient(
        target_radius=RADIUS_INTERNAL, source_side="outside", **kwargs
    )
    np.testing.assert_allclose(
        derivative_outside - derivative_inside,
        -4.0 * np.pi * GRAVITATIONAL_CONSTANT * surface_density,
        rtol=2.0e-15,
    )


def test_sheet_has_regular_inner_and_decaying_outer_radial_branches():
    kwargs = {
        "source_radius": RADIUS_INTERNAL,
        "wavenumber": WAVENUMBER,
        "surface_density_coefficient": 1.0,
        "gravitational_constant": GRAVITATIONAL_CONSTANT,
    }
    at_source = cylindrical_sheet_potential_coefficient(target_radius=RADIUS_INTERNAL, **kwargs)
    inner_radius = 0.2 * RADIUS_INTERNAL
    outer_radius = 5.0 * RADIUS_INTERNAL

    inner = cylindrical_sheet_potential_coefficient(target_radius=inner_radius, **kwargs)
    outer = cylindrical_sheet_potential_coefficient(target_radius=outer_radius, **kwargs)
    np.testing.assert_allclose(inner, at_source * (inner_radius / RADIUS_INTERNAL) ** WAVENUMBER)
    np.testing.assert_allclose(outer, at_source * (RADIUS_INTERNAL / outer_radius) ** WAVENUMBER)


@pytest.mark.parametrize(
    ("wavenumber", "radial_exponent", "target_radius"),
    [
        (2, 2.0, 0.8 * RADIUS_INNER),
        (3, 2.0, 0.5 * (RADIUS_INNER + RADIUS_OUTER)),
        (4, 2.0, RADIUS_INNER),
        (8, 2.0, 1.4 * RADIUS_OUTER),
    ],
)
def test_power_law_volume_potential_matches_sheet_quadrature(
    wavenumber, radial_exponent, target_radius
):
    density_coefficient = -0.63
    nodes, weights = np.polynomial.legendre.leggauss(128)
    intervals = [RADIUS_INNER, RADIUS_OUTER]
    if RADIUS_INNER < target_radius < RADIUS_OUTER:
        intervals.insert(1, target_radius)
    source_radii = np.concatenate(
        [
            0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
            for lower, upper in zip(intervals[:-1], intervals[1:])
        ]
    )
    source_weights = np.concatenate(
        [0.5 * (upper - lower) * weights for lower, upper in zip(intervals[:-1], intervals[1:])]
    )
    density = density_coefficient * (source_radii / RADIUS_OUTER) ** radial_exponent
    kernel = (
        np.minimum(source_radii, target_radius) / np.maximum(source_radii, target_radius)
    ) ** wavenumber
    expected = (
        2.0
        * np.pi
        * GRAVITATIONAL_CONSTANT
        / wavenumber
        * np.sum(source_weights * source_radii * density * kernel)
    )

    actual = cylindrical_power_law_volume_potential_coefficient(
        radius_inner=RADIUS_INNER,
        radius_outer=RADIUS_OUTER,
        target_radius=target_radius,
        wavenumber=wavenumber,
        radial_exponent=radial_exponent,
        density_coefficient=density_coefficient,
        gravitational_constant=GRAVITATIONAL_CONSTANT,
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-13)


def test_power_law_outer_boundary_has_closed_form():
    mode = 3
    exponent = 2.0
    density = 0.71
    expected = (
        2.0
        * np.pi
        * GRAVITATIONAL_CONSTANT
        * density
        * RADIUS_OUTER**2
        / (mode * (mode + exponent + 2.0))
        * (1.0 - (RADIUS_INNER / RADIUS_OUTER) ** (mode + exponent + 2.0))
    )
    actual = cylindrical_power_law_volume_potential_coefficient(
        radius_inner=RADIUS_INNER,
        radius_outer=RADIUS_OUTER,
        target_radius=RADIUS_OUTER,
        wavenumber=mode,
        radial_exponent=exponent,
        density_coefficient=density,
        gravitational_constant=GRAVITATIONAL_CONSTANT,
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-15)


def test_power_law_inner_boundary_handles_logarithmic_resonance():
    mode = 4
    exponent = 2.0
    density = 0.71
    expected = (
        2.0
        * np.pi
        * GRAVITATIONAL_CONSTANT
        * density
        / mode
        * RADIUS_INNER**mode
        / RADIUS_OUTER**exponent
        * np.log(RADIUS_OUTER / RADIUS_INNER)
    )
    actual = cylindrical_power_law_volume_potential_coefficient(
        radius_inner=RADIUS_INNER,
        radius_outer=RADIUS_OUTER,
        target_radius=RADIUS_INNER,
        wavenumber=mode,
        radial_exponent=exponent,
        density_coefficient=density,
        gravitational_constant=GRAVITATIONAL_CONSTANT,
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-15)


def test_two_boundary_operator_matches_isolated_sheet_responses():
    outer_density = 0.8
    inner_density = -1.3
    operator = cylindrical_annulus_potential_operator(
        radius_inner=RADIUS_INNER,
        radius_outer=RADIUS_OUTER,
        wavenumber=WAVENUMBER,
        outer_density_contrast=outer_density,
        inner_density_contrast=inner_density,
        gravitational_constant=GRAVITATIONAL_CONSTANT,
    )

    for row, target_radius in enumerate((RADIUS_OUTER, RADIUS_INNER)):
        for column, (source_radius, density) in enumerate(
            ((RADIUS_OUTER, outer_density), (RADIUS_INNER, inner_density))
        ):
            expected = cylindrical_sheet_potential_coefficient(
                source_radius=source_radius,
                target_radius=target_radius,
                wavenumber=WAVENUMBER,
                surface_density_coefficient=density,
                gravitational_constant=GRAVITATIONAL_CONSTANT,
            )
            np.testing.assert_allclose(operator[row, column], expected)


def test_geoid_response_superposes_boundaries_and_internal_load():
    outer_topography = -0.77
    inner_topography = -0.32
    outer_density = 0.6
    inner_density = -0.9
    internal_density = 0.27
    outer_gravity = 1.7
    inner_gravity = 2.4
    operator = cylindrical_annulus_potential_operator(
        radius_inner=RADIUS_INNER,
        radius_outer=RADIUS_OUTER,
        wavenumber=WAVENUMBER,
        outer_density_contrast=outer_density,
        inner_density_contrast=inner_density,
        gravitational_constant=GRAVITATIONAL_CONSTANT,
    )
    expected_potential = operator @ np.array([outer_topography, inner_topography])
    for index, target_radius in enumerate((RADIUS_OUTER, RADIUS_INNER)):
        expected_potential[index] += cylindrical_sheet_potential_coefficient(
            source_radius=RADIUS_INTERNAL,
            target_radius=target_radius,
            wavenumber=WAVENUMBER,
            surface_density_coefficient=internal_density,
            gravitational_constant=GRAVITATIONAL_CONSTANT,
        )

    response = cylindrical_annulus_geoid_response(
        radius_inner=RADIUS_INNER,
        radius_outer=RADIUS_OUTER,
        wavenumber=WAVENUMBER,
        outer_topography_coefficient=outer_topography,
        inner_topography_coefficient=inner_topography,
        outer_density_contrast=outer_density,
        inner_density_contrast=inner_density,
        outer_reference_gravity=outer_gravity,
        inner_reference_gravity=inner_gravity,
        internal_load_radius=RADIUS_INTERNAL,
        internal_surface_density_coefficient=internal_density,
        gravitational_constant=GRAVITATIONAL_CONSTANT,
    )

    np.testing.assert_allclose(
        [response.outer_potential, response.inner_potential], expected_potential
    )
    np.testing.assert_allclose(
        [response.outer_geoid, response.inner_geoid],
        expected_potential / np.array([outer_gravity, inner_gravity]),
    )


def test_geoid_response_superposes_power_law_volume_load():
    exponent = 2.0
    density = -0.37
    outer_gravity = 1.7
    inner_gravity = 2.4
    expected = np.array(
        [
            cylindrical_power_law_volume_potential_coefficient(
                radius_inner=RADIUS_INNER,
                radius_outer=RADIUS_OUTER,
                target_radius=target_radius,
                wavenumber=WAVENUMBER,
                radial_exponent=exponent,
                density_coefficient=density,
                gravitational_constant=GRAVITATIONAL_CONSTANT,
            )
            for target_radius in (RADIUS_OUTER, RADIUS_INNER)
        ]
    )
    response = cylindrical_annulus_geoid_response(
        radius_inner=RADIUS_INNER,
        radius_outer=RADIUS_OUTER,
        wavenumber=WAVENUMBER,
        outer_topography_coefficient=0.0,
        inner_topography_coefficient=0.0,
        outer_density_contrast=1.0,
        inner_density_contrast=1.0,
        outer_reference_gravity=outer_gravity,
        inner_reference_gravity=inner_gravity,
        volume_density_radial_exponent=exponent,
        volume_density_coefficient=density,
        gravitational_constant=GRAVITATIONAL_CONSTANT,
    )
    np.testing.assert_allclose([response.outer_potential, response.inner_potential], expected)
    np.testing.assert_allclose(
        [response.outer_geoid, response.inner_geoid],
        expected / np.array([outer_gravity, inner_gravity]),
    )


@pytest.mark.parametrize("wavenumber", [-1, 0])
def test_negative_and_monopole_modes_are_rejected(wavenumber):
    with pytest.raises(ValueError, match="non-negative|logarithmic"):
        cylindrical_sheet_potential_coefficient(
            source_radius=2.0,
            target_radius=2.2,
            wavenumber=wavenumber,
            surface_density_coefficient=1.0,
        )


def test_nonzero_internal_load_requires_a_radius():
    with pytest.raises(ValueError, match="internal_load_radius"):
        cylindrical_annulus_geoid_response(
            radius_inner=RADIUS_INNER,
            radius_outer=RADIUS_OUTER,
            wavenumber=WAVENUMBER,
            outer_topography_coefficient=0.0,
            inner_topography_coefficient=0.0,
            outer_density_contrast=1.0,
            inner_density_contrast=-1.0,
            outer_reference_gravity=1.0,
            inner_reference_gravity=1.0,
            internal_surface_density_coefficient=1.0,
        )


def _self_gravity_kwargs():
    return {
        "radius_inner": RADIUS_INNER,
        "radius_outer": RADIUS_OUTER,
        "wavenumber": WAVENUMBER,
        "outer_topography_coefficient": -0.77,
        "inner_topography_coefficient": -0.32,
        "outer_density_contrast": 0.06,
        "inner_density_contrast": 0.09,
        "outer_reference_gravity": 1.7,
        "inner_reference_gravity": 2.4,
        "internal_load_radius": RADIUS_INTERNAL,
        "internal_surface_density_coefficient": 0.027,
        "gravitational_constant": 0.1,
    }


def test_self_gravity_satisfies_two_boundary_matrix_equation():
    kwargs = _self_gravity_kwargs()
    response = cylindrical_annulus_self_gravity_response(**kwargs)
    operator = cylindrical_annulus_potential_operator(
        radius_inner=kwargs["radius_inner"],
        radius_outer=kwargs["radius_outer"],
        wavenumber=kwargs["wavenumber"],
        outer_density_contrast=kwargs["outer_density_contrast"],
        inner_density_contrast=kwargs["inner_density_contrast"],
        gravitational_constant=kwargs["gravitational_constant"],
    )
    load = np.array(
        [
            cylindrical_sheet_potential_coefficient(
                source_radius=kwargs["internal_load_radius"],
                target_radius=target_radius,
                wavenumber=kwargs["wavenumber"],
                surface_density_coefficient=kwargs["internal_surface_density_coefficient"],
                gravitational_constant=kwargs["gravitational_constant"],
            )
            for target_radius in (RADIUS_OUTER, RADIUS_INNER)
        ]
    )
    q_matrix = np.diag([response.q_outer, response.q_inner])
    corrected_topography = np.array([response.outer_topography, response.inner_topography])
    original_topography = np.array(
        [
            kwargs["outer_topography_coefficient"],
            kwargs["inner_topography_coefficient"],
        ]
    )

    residual = (
        (np.eye(2) - q_matrix @ operator) @ corrected_topography
        - original_topography
        - q_matrix @ load
    )
    np.testing.assert_allclose(residual, 0.0, atol=2.0e-16)
    assert response.matrix_residual_norm < 2.0e-16
    assert response.q_outer == pytest.approx(1.0 / kwargs["outer_reference_gravity"])
    assert response.q_inner == pytest.approx(1.0 / kwargs["inner_reference_gravity"])


def test_zero_feedback_recovers_no_self_gravity_response():
    kwargs = _self_gravity_kwargs()
    no_self_gravity = cylindrical_annulus_geoid_response(**kwargs)
    response = cylindrical_annulus_self_gravity_response(
        **kwargs,
        outer_feedback_factor=0.0,
        inner_feedback_factor=0.0,
    )

    np.testing.assert_allclose(
        [response.outer_topography, response.inner_topography],
        [
            kwargs["outer_topography_coefficient"],
            kwargs["inner_topography_coefficient"],
        ],
    )
    np.testing.assert_allclose(
        [response.outer_geoid, response.inner_geoid],
        [no_self_gravity.outer_geoid, no_self_gravity.inner_geoid],
    )


@pytest.mark.parametrize(
    "disabled_factor,unchanged_topography",
    [
        ("outer_feedback_factor", "outer_topography"),
        ("inner_feedback_factor", "inner_topography"),
    ],
)
def test_disabling_one_interface_removes_only_its_feedback_row(
    disabled_factor, unchanged_topography
):
    kwargs = _self_gravity_kwargs()
    response = cylindrical_annulus_self_gravity_response(**kwargs, **{disabled_factor: 0.0})
    original_name = unchanged_topography.replace("topography", "topography_coefficient")

    assert getattr(response, unchanged_topography) == pytest.approx(kwargs[original_name])
    other_boundary = "inner" if unchanged_topography.startswith("outer") else "outer"
    assert getattr(response, f"{other_boundary}_topography") != pytest.approx(
        kwargs[f"{other_boundary}_topography_coefficient"]
    )


def test_zero_interface_density_and_no_load_has_no_self_gravity_correction():
    kwargs = _self_gravity_kwargs()
    kwargs.update(
        outer_density_contrast=0.0,
        inner_density_contrast=0.0,
        internal_load_radius=None,
        internal_surface_density_coefficient=0.0,
    )
    response = cylindrical_annulus_self_gravity_response(**kwargs)

    np.testing.assert_allclose(
        [response.outer_topography, response.inner_topography],
        [
            kwargs["outer_topography_coefficient"],
            kwargs["inner_topography_coefficient"],
        ],
    )
    np.testing.assert_allclose([response.outer_potential, response.inner_potential], 0.0)
