"""Independent radial integration and harmonic convention checks."""

import numpy as np
import pytest
import sympy as sp

from underworld3.analytic import spherical_gravity as gravity
from underworld3.maths import spherical_harmonics as harmonics


@pytest.mark.parametrize("degree", [0, 1, 2, 5, 8])
def test_sheet_continuity_jump_and_continuation(degree):
    source, density, G = 0.8, -0.37, 0.61
    common = dict(
        source_radius=source,
        harmonic_degree=degree,
        surface_density_coefficient=density,
        gravitational_constant=G,
    )
    expected = 4 * np.pi * G * density * source / (2 * degree + 1)
    for target in (0.4, source, 1.6):
        factor = (
            (target / source) ** degree if target <= source else (source / target) ** (degree + 1)
        )
        np.testing.assert_allclose(
            gravity.spherical_sheet_potential_coefficient(target_radius=target, **common),
            expected * factor,
            rtol=2e-15,
        )
    inside = gravity.spherical_sheet_radial_derivative_coefficient(
        target_radius=source, source_side="inside", **common
    )
    outside = gravity.spherical_sheet_radial_derivative_coefficient(
        target_radius=source, source_side="outside", **common
    )
    np.testing.assert_allclose(outside - inside, -4 * np.pi * G * density, rtol=2e-15)


@pytest.mark.parametrize("degree,power", [(1, 2), (2, 2), (5, 2), (8, 5)])
@pytest.mark.parametrize("target", [0.3, 0.7, 1.4])
def test_volume_matches_independent_piecewise_quadrature(degree, power, target):
    ri, ro, density, G = 0.5, 1.0, -0.43, 0.61
    nodes, weights = np.polynomial.legendre.leggauss(96)
    intervals = sorted({ri, ro} | ({target} if ri < target < ro else set()))
    integral = 0.0
    for lo, hi in zip(intervals[:-1], intervals[1:]):
        sources = (hi - lo) * nodes / 2 + (hi + lo) / 2
        continuation = np.where(
            sources <= target, (sources / target) ** (degree + 1), (target / sources) ** degree
        )
        values = sources * density * (sources / ro) ** power * continuation
        integral += (hi - lo) * float(weights @ values) / 2
    expected = 4 * np.pi * G * integral / (2 * degree + 1)
    actual = gravity.spherical_volume_density_potential_coefficient(
        radius_inner=ri,
        radius_outer=ro,
        harmonic_degree=degree,
        density_power=power,
        density_amplitude=density,
        target_radius=target,
        gravitational_constant=G,
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-15)


@pytest.mark.parametrize(
    "degree,order,component,normalization",
    [
        (2, 0, "cos", "axisymmetric_unnormalized"),
        (2, 0, "cos", "real_orthonormal"),
        (3, 2, "cos", "real_orthonormal"),
        (5, 3, "sin", "real_orthonormal"),
    ],
)
def test_harmonic_symbolic_numeric_and_angular_norm(degree, order, component, normalization):
    theta, phi = sp.symbols("theta phi", real=True)
    expression = harmonics.real_harmonic_sympy(
        theta, phi, degree=degree, order=order, component=component, normalization=normalization
    )
    evaluate = sp.lambdify((theta, phi), expression, "numpy", cse=True)
    cosine, weights = np.polynomial.legendre.leggauss(40)
    azimuth = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    T, P = np.meshgrid(np.arccos(cosine), azimuth, indexing="ij")
    points = np.column_stack(
        (
            np.sin(T).ravel() * np.cos(P).ravel(),
            np.sin(T).ravel() * np.sin(P).ravel(),
            np.cos(T).ravel(),
        )
    )
    values = harmonics.real_harmonic_values(
        points, degree=degree, order=order, component=component, normalization=normalization
    )
    np.testing.assert_allclose(values, evaluate(T, P).ravel(), rtol=2e-12, atol=2e-14)
    norm = (weights @ (values.reshape(T.shape) ** 2).sum(axis=1)) * 2 * np.pi / len(azimuth)
    np.testing.assert_allclose(
        norm, harmonics.harmonic_norm(degree, order, component, normalization), rtol=2e-13
    )


@pytest.mark.parametrize("degree", [True, 2.1])
def test_harmonic_rejects_silent_integer_truncation(degree):
    with pytest.raises(TypeError, match="integer"):
        harmonics.validate_harmonic(degree, 0, "cos", "real_orthonormal")
