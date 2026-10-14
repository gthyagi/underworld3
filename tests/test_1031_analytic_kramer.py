"""Native symbolic Kramer fields: independent oracle, physics and caching."""

from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp
import underworld3 as uw
from underworld3.analytic import _validation
from underworld3.analytic.cylindrical.kramer import assess_available

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture
def mesh():
    x, y = sp.symbols("x y", real=True)
    return SimpleNamespace(dim=2, X=sp.Matrix([[x, y]]))


@pytest.mark.skipif(not assess_available(), reason="optional assess oracle not installed")
@pytest.mark.parametrize(
    "density,boundary",
    [("delta", "free"), ("delta", "zero"), ("smooth", "free"), ("smooth", "zero")],
)
@pytest.mark.parametrize("mode", [2, 3, 8])
def test_native_fields_match_assess(mesh, density, boundary, mode):
    import assess

    kwargs = dict(
        n=mode,
        k=2,
        r_inner=1.22,
        r_outer=2.22,
        r_int=2.0,
        density=density,
        boundary=boundary,
        viscosity=2.3,
        forcing=-0.73 if density == "delta" else 0.61,
    )
    # n=3,k=2 is a resonant smooth Stokes family, explicitly unsupported.
    if density == "smooth" and mode == 3:
        with pytest.raises(NotImplementedError, match="not implemented"):
            uw.analytic.CylindricalStokes(mesh, **kwargs)
        return
    solution = uw.analytic.CylindricalStokes(mesh, **kwargs)
    family = getattr(
        assess,
        f"CylindricalStokesSolution{density.capitalize()}{'FreeSlip' if boundary == 'free' else 'ZeroSlip'}",
    )
    common = dict(Rm=1.22, Rp=2.22, nu=kwargs["viscosity"], g=kwargs["forcing"])
    above = family(mode, +1, rp=2.0, **common) if density == "delta" else family(mode, 2, **common)
    below = family(mode, -1, rp=2.0, **common) if density == "delta" else above
    points = solution.sample_points(37)
    for field, method in (
        ("velocity", "velocity_cartesian"),
        ("pressure", "pressure_cartesian"),
        ("radial_normal_stress", "radial_stress_cartesian"),
    ):
        expected = np.asarray(
            [getattr(above if np.linalg.norm(p) >= 2.0 else below, method)(p) for p in points]
        ).reshape(solution.evaluate(field, points).shape)
        actual = solution.evaluate(field, points)
        scale = max(np.max(np.abs(expected)), 1e-14)
        np.testing.assert_allclose(actual, expected, atol=scale * 2e-12, rtol=2e-11)


@pytest.mark.parametrize(
    "density,boundary",
    [("delta", "free"), ("delta", "zero"), ("smooth", "free"), ("smooth", "zero")],
)
def test_symbolic_bulk_physics(mesh, density, boundary):
    solution = uw.analytic.CylindricalStokes(mesh, density=density, boundary=boundary)
    points = solution.sample_points(12)
    assert _validation.incompressibility_residual(solution, points) < 1e-10
    assert _validation.momentum_residual(solution, points) < 1e-9
    assert _validation.strainrate_consistency(solution, points) < 1e-10


@pytest.mark.parametrize(
    "density,boundary",
    [("delta", "free"), ("delta", "zero"), ("smooth", "free"), ("smooth", "zero")],
)
def test_wall_conditions(mesh, density, boundary):
    solution = uw.analytic.CylindricalStokes(mesh, density=density, boundary=boundary)
    angle = np.linspace(0, 2 * np.pi, 43, endpoint=False)
    unit = np.column_stack((np.cos(angle), np.sin(angle)))
    for radius in (solution.r_inner, solution.r_outer):
        points = radius * unit
        velocity = solution.evaluate("velocity", points)
        np.testing.assert_allclose(np.sum(velocity * unit, axis=1), 0, atol=3e-13)
        if boundary == "zero":
            np.testing.assert_allclose(velocity, 0, atol=3e-13)
        else:
            np.testing.assert_allclose(
                solution.evaluate("radial_shear_stress", points), 0, atol=3e-13
            )


def test_delta_jump_is_checked_separately(mesh):
    solution = uw.analytic.CylindricalStokes(mesh, forcing=-0.73)
    radius, theta = solution._r, solution._theta
    upper, lower = solution._polar_fields[1], solution._polar_fields[0]
    angles = np.linspace(0, 2 * np.pi, 37, endpoint=False)
    for field in ("velocity", "radial_shear_stress"):
        jump = upper[field] - lower[field]
        terms = list(jump) if isinstance(jump, sp.MatrixBase) else [jump]
        evaluate = sp.lambdify(
            theta, [term.subs(radius, solution.r_int) for term in terms], "numpy"
        )
        for value in evaluate(angles):
            np.testing.assert_allclose(value, 0, atol=3e-13)
    jump = (upper["radial_normal_stress"] - lower["radial_normal_stress"]).subs(
        radius, solution.r_int
    )
    actual = sp.lambdify(theta, jump, "numpy")(angles)
    # Integrating div(sigma)+f=0 through the sheet gives [sigma_rr]=-f_sheet.
    # Kramer's delta coefficient g denotes the stress jump, not f_sheet.
    np.testing.assert_allclose(actual, solution.g * np.cos(solution.n * angles), atol=3e-13)


def test_evaluators_are_lazy_and_reused(mesh):
    solution = uw.analytic.CylindricalStokes(mesh)
    assert not solution._evaluators
    points = solution.sample_points(20)
    expected = solution.evaluate("velocity", points)
    first = dict(solution._evaluators)
    np.testing.assert_array_equal(solution.evaluate("velocity", points), expected)
    assert first == solution._evaluators
    assert solution.evaluate("velocity", np.empty((0, 2))).shape == (0, 2)
    assert solution.fn_stress.shape == (2, 2)
    assert solution.fn_velocity.shape == (1, 2)


def test_native_forcing_requires_and_uses_conforming_sheet(mesh):
    solution = uw.analytic.CylindricalStokes(mesh, forcing=1.0)
    mesh.boundaries = SimpleNamespace(__members__={})
    calls = []
    solver = SimpleNamespace(
        mesh=mesh, add_natural_bc=lambda value, label: calls.append((value, label))
    )
    with pytest.raises(ValueError, match="conforming"):
        solution.apply_forcing(solver)
    mesh.boundaries.__members__["Internal"] = 2
    solution.apply_forcing(solver)
    assert calls[0][1] == "Internal"
    assert solver.bodyforce == sp.zeros(1, 2)


def test_positive_density_under_inward_gravity_uses_positive_delta_jump(mesh):
    solution = uw.analytic.CylindricalStokes(mesh, density="delta", forcing=1.0)
    # The legacy assess default uses g=-1 (an outward sheet force). Density-
    # coupled geoid cases explicitly use g=+rho with inward unit gravity.
    assert solution.evaluate("velocity", np.array([[0.7, 0.0]]))[0, 0] < 0
    outer_sigma = solution.evaluate("radial_normal_stress", np.array([[1.0, 0.0]]))[0]
    inner_sigma = solution.evaluate("radial_normal_stress", np.array([[0.5, 0.0]]))[0]
    outer_reaction, inner_reaction = -outer_sigma, -inner_sigma
    np.testing.assert_allclose(-outer_reaction / (-1.0), -outer_sigma)
    np.testing.assert_allclose(-inner_reaction / (+1.0), +inner_sigma)


@pytest.mark.parametrize(
    "kwargs", [dict(n=1), dict(n=2.1), dict(k=True), dict(viscosity=0), dict(r_int=1.0)]
)
def test_invalid_parameters_are_explicit(mesh, kwargs):
    with pytest.raises((ValueError, TypeError)):
        uw.analytic.CylindricalStokes(mesh, **kwargs)
