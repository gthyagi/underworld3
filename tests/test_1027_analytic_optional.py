r"""The native cylindrical reference works without the optional assess oracle.

Legacy numerical-oracle checks use analytic.kramer.CylindricalStokes explicitly;
the public analytic.CylindricalStokes is the native symbolic implementation.
Absence is simulated so dependency independence is checked on every install.
"""

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import importlib.util
import sys

import numpy as np
import underworld3 as uw

from underworld3.analytic import kramer
from underworld3.analytic.kramer import assess_available

_real_find_spec = importlib.util.find_spec

needs_assess = pytest.mark.skipif(
    not assess_available(), reason="the optional 'assess' package is not installed"
)


def test_importing_underworld_does_not_need_the_optional_package():
    """The whole point of a lazy import: absence must cost nothing at import."""

    import underworld3.analytic.kramer  # noqa: F401

    assert kramer.CylindricalStokes is not None


def test_native_solution_is_listed_without_the_optional_oracle():
    assert "CylindricalStokes" in uw.analytic.available()
    assert uw.analytic.is_available("CylindricalStokes")
    assert "CylindricalStokes" in uw.analytic.available(installed_only=True)


def test_every_other_solution_is_unconditionally_available():
    for name in uw.analytic.available():
        if name == "CylindricalStokes":
            continue
        assert uw.analytic.is_available(name), name
        assert getattr(uw.analytic, name).requires is None


@pytest.fixture
def without_assess(monkeypatch):
    """Make `assess` unimportable, whether or not it is installed.

    The missing-dependency path is the one a normal install takes and the one
    the previous arrangement got wrong, so it must be tested *everywhere* — not
    only on machines that happen to lack the package. `assess` is now a dev
    dependency in pixi.toml, so skipping when it is present would mean these
    tests never run in CI and the wrapper's whole purpose goes unchecked.
    """

    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "assess" or name.startswith("assess."):
            raise ImportError("No module named 'assess'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "assess", raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: None
        if name == "assess"
        else _real_find_spec(name, *a, **k)
    )


def test_the_absence_simulation_actually_blocks(without_assess):
    """Negative control for the fixture itself.

    If the monkeypatching silently stopped working — a Python version changing
    how imports resolve, say — every test below would pass by importing the real
    package, and would look like coverage of a path they never touched.
    """

    from underworld3.analytic import kramer

    assert uw.analytic.is_available("CylindricalStokes")
    with pytest.raises(ImportError):
        kramer.require_assess()


def test_describe_identifies_native_reference_without_missing_dependency(without_assess):
    summary = uw.analytic.describe("CylindricalStokes")

    assert "unavailable" not in summary
    assert "assess" not in summary


def test_native_reference_remains_in_installed_only_listing(without_assess):
    assert "CylindricalStokes" in uw.analytic.available()
    assert "CylindricalStokes" in uw.analytic.available(installed_only=True)


def test_constructing_it_explains_rather_than_tracebacks(without_assess):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)

    with pytest.raises(ImportError) as raised:
        kramer.CylindricalStokes(mesh)

    message = str(raised.value)
    assert "pip install" in message
    assert "assess" in message
    # and it cites the paper, so the user knows what they are installing
    assert "Kramer" in message


def test_native_reference_constructs_with_assess_blocked(without_assess):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)
    solution = uw.analytic.CylindricalStokes(mesh)
    assert solution.symbolic and solution.requires is None
    assert np.isfinite(solution.evaluate("pressure", np.array([[.7, 0.]]))).all()


def test_legacy_oracle_declares_its_optional_dependency():
    """A numeric oracle is not a validated solution, and must not pass as one.

    Every other solution is SymPy and clears the physics residual. This one
    cannot: there is no expression to differentiate. The declaration is what
    keeps the conformance sweep from either skipping it silently or reporting a
    pass it never earned.
    """

    assert kramer.CylindricalStokes.symbolic is False
    assert kramer.CylindricalStokes.requires == "assess"

    for name in uw.analytic.available():
        if name != "CylindricalStokes":
            assert getattr(uw.analytic, name).symbolic, name


def test_it_rejects_unknown_cases_before_reaching_the_dependency(monkeypatch):
    """Argument validation must not be hidden behind the optional import.

    Otherwise a typo in `density=` reports a missing package on machines without
    `assess` and a wrong answer on machines with it.
    """

    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)

    with pytest.raises(ValueError, match="density must be"):
        kramer.CylindricalStokes(mesh, density="gaussian")

    with pytest.raises(ValueError, match="boundary must be"):
        kramer.CylindricalStokes(mesh, boundary="sticky")


@pytest.fixture(scope="module")
def annulus():
    return uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.15)


def _interior_points(count=60, seed=7, r_inner=0.5, r_outer=1.0):
    """Points well inside the shell, away from both walls."""

    rng = np.random.default_rng(seed)
    margin = 0.1 * (r_outer - r_inner)
    theta = rng.uniform(0.0, 2 * np.pi, count)
    radius = rng.uniform(r_inner + margin, r_outer - margin, count)

    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def _wall_points(radius, count=40):
    theta = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def _divergence(sol, points, step=1e-6):
    r"""Numerical :math:`\nabla\cdot\mathbf u` by central differences.

    The suite's other solutions are differentiated symbolically. This one is
    numeric, so the only way to ask whether it solves anything is to
    finite-difference it — weaker than the SymPy gates, but the same idea, and
    much stronger than checking that the values are finite.
    """

    total = np.zeros(len(points))

    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = step
        total += (
            sol.evaluate("velocity", points + offset)[:, axis]
            - sol.evaluate("velocity", points - offset)[:, axis]
        ) / (2 * step)

    return total


@needs_assess
@pytest.mark.parametrize("density", ["delta", "smooth"])
@pytest.mark.parametrize("boundary", ["free", "zero"])
def test_all_four_cases_construct_and_evaluate(annulus, density, boundary):
    sol = kramer.CylindricalStokes(
        annulus, n=2, k=2, density=density, boundary=boundary
    )

    coords = annulus.X.coords
    velocity = sol.evaluate("velocity", coords)
    pressure = sol.evaluate("pressure", coords)

    assert velocity.shape == (len(coords), 2)
    assert pressure.shape == (len(coords),)
    assert np.isfinite(velocity).all()
    assert np.isfinite(pressure).all()


@needs_assess
@pytest.mark.parametrize("density", ["delta", "smooth"])
@pytest.mark.parametrize("boundary", ["free", "zero"])
def test_the_velocity_is_incompressible(annulus, density, boundary):
    sol = kramer.CylindricalStokes(
        annulus, n=2, k=2, density=density, boundary=boundary
    )

    points = _interior_points()
    scale = np.abs(sol.evaluate("velocity", points)).max()

    assert np.abs(_divergence(sol, points)).max() / scale < 1e-7


@needs_assess
def test_the_divergence_probe_actually_fires():
    """Negative control: the check above must fail on a compressible field.

    Without this, `div = 0` says nothing — a probe that returns zero for
    everything would pass all four cases above and look like a validation.
    """

    class Expanding:
        """u = (x, y), divergence 2 everywhere."""

        def evaluate(self, field, coords):
            return np.asarray(coords, dtype=float)

    points = _interior_points()
    assert np.abs(_divergence(Expanding(), points) - 2.0).max() < 1e-5


@needs_assess
@pytest.mark.parametrize("density", ["delta", "smooth"])
def test_free_slip_stops_wall_normal_flow_but_not_tangential(annulus, density):
    r""":math:`\mathbf u\cdot\hat n = 0` on both arcs, with :math:`|u| \neq 0`.

    The second half is the control. A wall where the whole velocity vanishes
    would pass a `u.n = 0` test trivially, and that is the zero-slip case, not
    this one.
    """

    sol = kramer.CylindricalStokes(
        annulus, n=2, k=2, density=density, boundary="free"
    )

    for radius in (sol.r_inner, sol.r_outer):
        wall = _wall_points(radius)
        velocity = sol.evaluate("velocity", wall)
        normal = wall / radius

        assert np.abs((velocity * normal).sum(axis=1)).max() < 1e-14
        assert np.abs(velocity).max() > 1e-4, "wall is not slipping at all"


@needs_assess
@pytest.mark.parametrize("density", ["delta", "smooth"])
def test_zero_slip_stops_the_wall_entirely(annulus, density):
    sol = kramer.CylindricalStokes(
        annulus, n=2, k=2, density=density, boundary="zero"
    )

    for radius in (sol.r_inner, sol.r_outer):
        velocity = sol.evaluate("velocity", _wall_points(radius))
        assert np.abs(velocity).max() < 1e-14

    # ... and it is not zero everywhere, which would make the above vacuous
    assert np.abs(sol.evaluate("velocity", _interior_points())).max() > 1e-5


@needs_assess
def test_the_delta_case_is_genuinely_two_sided(annulus):
    """The branch selection is the benchmark, so it has to actually branch."""

    sol = kramer.CylindricalStokes(annulus, density="delta", r_int=0.8)
    assert sol._above is not sol._below

    radii = np.array([[0.6, 0.0], [0.9, 0.0]])
    above, below = sol._split(radii)
    assert below[0] and above[1]


@needs_assess
def test_it_rejects_fields_it_does_not_have(annulus):
    sol = kramer.CylindricalStokes(annulus)

    with pytest.raises(ValueError, match="velocity"):
        sol.evaluate("temperature", annulus.X.coords)

    velocity = uw.discretisation.MeshVariable("Vk", annulus, annulus.dim, degree=2)
    with pytest.raises(ValueError, match="no symbolic form"):
        sol.error("velocity", velocity, norm="integral")
