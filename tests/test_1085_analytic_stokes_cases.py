"""Common case contract, Thieulot physics/oracles and Kramer response wiring.

No FE model/mesh is run here; fake meshes supply only dimension/coordinate
symbols. The existing independent gravity and assess suites own kernel accuracy.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp
import underworld3 as uw
from underworld3.analytic import _validation

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def mesh(dim):
    return SimpleNamespace(dim=dim, X=sp.Matrix([sp.symbols(f"x0:{dim}", real=True)]))


@pytest.mark.parametrize(
    "geometry,case,name",
    [
        ("cylindrical", "kramer", "KramerCylindricalStokes"),
        ("cylindrical", "thieulot", "ThieulotCylindricalStokes"),
        ("spherical", "kramer", "KramerSphericalStokes"),
        ("spherical", "thieulot", "ThieulotSphericalStokes"),
        ("spherical", "zhong", "Zhong"),
    ],
)
def test_geometry_package_owns_each_case(geometry, case, name):
    import importlib
    import inspect

    module = importlib.import_module(f"underworld3.analytic.{geometry}.{case}")
    cls = getattr(module, name)
    assert cls is getattr(uw.analytic, name)
    assert cls is getattr(getattr(uw.analytic, geometry), name)
    assert cls.__name__ == name
    assert cls.__module__ == module.__name__
    assert inspect.getsourcefile(cls).endswith(f"/analytic/{geometry}/{case}.py")
    assert "https://" in module.__doc__
    if case == "zhong":
        assert module.__all__ == ["Zhong", "ZhongResponse"]
        assert not hasattr(module, "Zhong2008")


@pytest.mark.parametrize(
    "old,current",
    [
        ("kramer_cylindrical", "cylindrical.kramer"),
        ("kramer_spherical", "spherical.kramer"),
        ("zhong", "spherical.zhong"),
    ],
)
def test_flat_modules_only_reexport_canonical_objects(old, current):
    import importlib

    previous = importlib.import_module(f"underworld3.analytic.{old}")
    canonical = importlib.import_module(f"underworld3.analytic.{current}")
    for name in previous.__all__:
        assert getattr(previous, name) is getattr(canonical, name)


def test_shared_shell_field_helper_and_flat_imports():
    from underworld3.analytic import thieulot
    from underworld3.analytic._shell_fields import _PrescribedShellStokes

    assert (
        thieulot.ThieulotCylindricalStokes
        is uw.analytic.cylindrical.ThieulotCylindricalStokes
    )
    assert (
        thieulot.ThieulotSphericalStokes
        is uw.analytic.spherical.ThieulotSphericalStokes
    )
    assert issubclass(thieulot.ThieulotCylindricalStokes, _PrescribedShellStokes)
    assert issubclass(thieulot.ThieulotSphericalStokes, _PrescribedShellStokes)


@pytest.mark.parametrize(
    "old,canonical",
    [
        ("CylindricalStokes", "KramerCylindricalStokes"),
        ("SphericalStokes", "KramerSphericalStokes"),
        ("AssessCylindricalStokes", "AssessKramerCylindricalStokes"),
    ],
)
def test_legacy_names_resolve_without_duplicate_discovery(old, canonical):
    assert getattr(uw.analytic, old) is getattr(uw.analytic, canonical)
    assert getattr(uw.analytic, canonical).__name__ == canonical
    assert canonical in uw.analytic.available()
    assert old not in uw.analytic.available()
    assert uw.analytic.describe(old) == uw.analytic.describe(canonical)
    assert uw.analytic.is_available(old) == uw.analytic.is_available(canonical)


def test_selector_returns_named_case_classes():
    for geometry, dimension in (("cylindrical", 2), ("spherical", 3)):
        for case in ("kramer", "thieulot"):
            name = case.capitalize() + geometry.capitalize() + "Stokes"
            solution = uw.analytic.StokesSolution(
                mesh(dimension), geometry=geometry, case=case
            )
            assert type(solution) is getattr(uw.analytic, name)
            assert solution.case == case
            assert solution.geometry == geometry


@pytest.mark.parametrize(
    "geometry,dim,cls",
    [
        ("cylindrical", 2, uw.analytic.CylindricalStokes),
        ("spherical", 3, uw.analytic.SphericalStokes),
    ],
)
def test_kramer_compatibility(geometry, dim, cls):
    sol = uw.analytic.StokesSolution(mesh(dim), geometry=geometry)
    assert type(sol) is cls
    assert sol.geometry == geometry and sol.case == "kramer"
    expected_cases = (
        ("kramer", "thieulot") if dim == 2 else ("kramer", "thieulot", "zhong")
    )
    assert uw.analytic.stokes_cases(geometry) == expected_cases
    from underworld3.analytic.kramer import CylindricalStokes

    assert uw.analytic.CylindricalStokes is CylindricalStokes


@pytest.mark.parametrize(
    "geometry,kwargs",
    [
        ("cylindrical", dict(k=0)),
        ("cylindrical", dict(k=1, viscosity=2.3, rho_0=0.17)),
        ("cylindrical", dict(k=4, amplitude=0.61, r_inner=0.7, r_outer=1.9)),
        ("spherical", dict(m=-1)),
        ("spherical", dict(m=3, mu_0=2.3, gamma=0.61)),
        ("spherical", dict(m=-3)),
        ("spherical", dict(m=0, r_inner=0.7, r_outer=1.9)),
    ],
)
def test_thieulot_bulk_walls_and_constitutive_stress(geometry, kwargs):
    dim = 2 if geometry == "cylindrical" else 3
    sol = uw.analytic.StokesSolution(
        mesh(dim), geometry=geometry, case="thieulot", **kwargs
    )
    points = sol.sample_points(7)
    assert _validation.incompressibility_residual(sol, points) < 2e-10
    assert _validation.momentum_residual(sol, points) < 2e-10
    # An independent numerical gradient checks the Cartesian tensor/evaluation
    # implementation, including radial/toroidal components in three dimensions.
    eps = 2e-6
    jac = np.stack(
        [
            (
                sol.evaluate("velocity", points + eps * np.eye(dim)[i])
                - sol.evaluate("velocity", points - eps * np.eye(dim)[i])
            )
            / (2 * eps)
            for i in range(dim)
        ],
        axis=2,
    )
    eta = sol.evaluate("viscosity", points)
    p = sol.evaluate("pressure", points)
    expected = eta[:, None, None] * (jac + np.swapaxes(jac, 1, 2)) - p[
        :, None, None
    ] * np.eye(dim)
    np.testing.assert_allclose(
        sol.evaluate("stress", points), expected, rtol=2e-7, atol=3e-8
    )
    directions = points / np.linalg.norm(points, axis=1)[:, None]
    for radius in (sol.r_inner, sol.r_outer):
        np.testing.assert_allclose(
            np.sum(sol.evaluate("velocity", radius * directions) * directions, axis=1),
            0,
            atol=2e-11,
        )
    assert not sol.publishes_both_stress_and_strainrate


@pytest.mark.parametrize("geometry", ["cylindrical", "spherical"])
def test_force_sign_negative_control(geometry):
    dim = 2 if geometry == "cylindrical" else 3
    sol = uw.analytic.StokesSolution(mesh(dim), geometry=geometry, case="thieulot")
    sol.fn_bodyforce = -sol.fn_bodyforce
    assert _validation.momentum_residual(sol, sol.sample_points(6)) > 0.01


@pytest.mark.parametrize("m", [-1, 3])
def test_thieulot_spherical_published_fields_and_poles(m):
    sol = uw.analytic.StokesSolution(
        mesh(3), geometry="spherical", case="thieulot", m=m
    )
    pts = np.vstack((sol.sample_points(12), [[0, 0, 0.75], [0, 0, -0.75]]))
    r = np.linalg.norm(pts, axis=1)
    ri, ro = 0.5, 1.0
    if m == -1:
        alpha = -(ro**3 - ri**3) / (ro**3 * np.log(ri) - ri**3 * np.log(ro))
        beta = (
            -3 * (np.log(ro) - np.log(ri)) / (ri**3 * np.log(ro) - ro**3 * np.log(ri))
        )
        g = -2 * (alpha * np.log(r) + beta * r**3 / 3 + 1) / r**2
    else:
        alpha = (m + 1) * (ri**-3 - ro**-3) / (ri ** (-(m + 4)) - ro ** (-(m + 4)))
        beta = -3 * (ri ** (m + 1) - ro ** (m + 1)) / (ri ** (m + 4) - ro ** (m + 4))
        g = -2 * (-alpha * r ** (-(m + 1)) / (m + 1) + beta * r**3 / 3 + 1) / r**2
    f = alpha * r ** (-(m + 3)) + beta * r
    theta, phi = np.arccos(pts[:, 2] / r), np.arctan2(pts[:, 1], pts[:, 0])
    er = pts / r[:, None]
    et = np.column_stack(
        (np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta))
    )
    ep = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros(len(r))))
    expected = g[:, None] * np.cos(theta)[:, None] * er + f[:, None] * np.sin(theta)[
        :, None
    ] * (et + ep)
    np.testing.assert_allclose(sol.evaluate("velocity", pts), expected, atol=3e-13)
    np.testing.assert_allclose(
        sol.evaluate("pressure", pts), (m + 3) * r**m * g * np.cos(theta), atol=3e-13
    )
    for field in ("stress", "bodyforce", "radial_normal_stress"):
        assert np.all(np.isfinite(sol.evaluate(field, pts)))


@pytest.mark.parametrize("k", [0, 1, 4])
def test_thieulot_annulus_published_fields(k):
    sol = uw.analytic.StokesSolution(
        mesh(2), geometry="cylindrical", case="thieulot", k=k
    )
    pts = sol.sample_points(12)
    r, theta = np.linalg.norm(pts, axis=1), np.arctan2(pts[:, 1], pts[:, 0])
    # R1=1, R2=2, C=-1: independent reduced ASPECT coefficients.
    A, B = 2.0, -3 / np.log(2)
    f, g = A * r + B / r, A * r / 2 + B * np.log(r) / r - 1 / r
    vr, vt = k * g * np.sin(k * theta), f * np.cos(k * theta)
    expected = np.column_stack(
        (
            vr * np.cos(theta) - vt * np.sin(theta),
            vr * np.sin(theta) + vt * np.cos(theta),
        )
    )
    np.testing.assert_allclose(sol.evaluate("velocity", pts), expected, atol=2e-13)
    np.testing.assert_allclose(
        sol.evaluate("pressure", pts),
        k * (2 * g - f) / r * np.sin(k * theta),
        atol=2e-13,
    )


@pytest.mark.parametrize(
    "geometry,density,expected",
    [
        ("cylindrical", "delta", (-0.6636011701874045, -0.5024941444939373)),
        ("cylindrical", "smooth", (-0.3593167704006861, -0.3254846780244329)),
        ("spherical", "delta", (-0.0032409460113843915, -0.004028851872138672)),
        ("spherical", "smooth", (-0.0006311619859870657, -0.0007384106749212236)),
    ],
)
def test_geoid_matches_previous_native_reference(geometry, density, expected):
    dim = 2 if geometry == "cylindrical" else 3
    parameters = (
        dict(r_inner=1.22, r_outer=2.22, r_int=2.0)
        if dim == 2
        else dict(
            r_inner=0.55, r_outer=1.0, r_int=0.775, normalization="real_orthonormal"
        )
    )
    sol = uw.analytic.StokesSolution(
        mesh(dim), geometry=geometry, density=density, forcing=1.0, **parameters
    )
    response = sol.geoid_response(
        outer_density_contrast=1.0,
        inner_density_contrast=1.0,
        outer_reference_gravity=1.0,
        inner_reference_gravity=1.0,
        gravitational_constant=1.0 if dim == 2 else 0.01,
        source_density_coefficient=1.0,
        include_self_gravity=True,
    )
    np.testing.assert_allclose(
        [response["outer_geoid"], response["inner_geoid"]], expected, atol=2e-13
    )
    assert response["matrix_residual_norm"] < 1e-13
    assert "outer_geoid_self_gravity" in response


def test_error_reports_preserve_sign_and_zero_reference():
    report = uw.analytic.compare_coefficients(
        {"negative": -2.2, "zero": 0.1}, {"negative": -2.0, "zero": 0.0}
    )
    assert report["negative"]["relative_error"] == pytest.approx(0.1)
    assert report["zero"]["absolute_error"] == 0.1
    assert report["zero"]["relative_error"] is None
    with pytest.raises(ValueError, match="finite"):
        uw.analytic.compare_coefficients({"bad": np.nan}, {"bad": 1.0})
    sol = uw.analytic.StokesSolution(
        mesh(2), geometry="cylindrical", case="thieulot", k=0
    )
    pts = sol.sample_points(6)
    variable = SimpleNamespace(coords=pts, array=1.1 * sol.evaluate("velocity", pts))
    assert sol.error_report("velocity", variable)["relative_l2"] == pytest.approx(0.1)
    variable.array = np.ones((len(pts), 1))
    assert sol.error_report("pressure", variable)["relative_l2"] is None


def test_case_validation_and_wall_contract():
    with pytest.raises(ValueError, match="geometry"):
        uw.analytic.StokesSolution(mesh(2), geometry="box")
    with pytest.raises(ValueError, match="unsupported"):
        uw.analytic.StokesSolution(mesh(2), geometry="cylindrical", case="zhong")
    with pytest.raises(ValueError, match="3D"):
        uw.analytic.StokesSolution(mesh(2), geometry="spherical", case="thieulot")
    with pytest.raises(NotImplementedError, match="m=-4"):
        uw.analytic.StokesSolution(mesh(3), geometry="spherical", case="thieulot", m=-4)
    sol = uw.analytic.StokesSolution(mesh(3), geometry="spherical", case="thieulot")
    calls = []
    solver = SimpleNamespace(
        add_dirichlet_bc=lambda value, wall: calls.append((value, wall))
    )
    sol.apply_boundary_conditions(solver)
    sol.apply_forcing(solver)
    assert [wall for _, wall in calls] == ["Upper", "Lower"]
    assert calls[0][0] == sol.fn_velocity
    assert solver.bodyforce == sol.fn_bodyforce
    assert solver.petsc_use_pressure_nullspace
    assert sol.evaluate("velocity", np.empty((0, 3))).shape == (0, 3)
    with pytest.raises(ValueError, match="origin"):
        sol.evaluate("velocity", [[0, 0, 0]])


def test_global_error_report_with_empty_ranks():
    sol = uw.analytic.StokesSolution(
        mesh(2), geometry="cylindrical", case="thieulot", k=0
    )
    points = np.array([[1.5, 0.0]]) if uw.mpi.rank == 0 else np.empty((0, 2))
    exact = sol.evaluate("velocity", points)
    variable = SimpleNamespace(coords=points, array=1.2 * exact)
    report = sol.error_report("velocity", variable)
    expected = np.linalg.norm(sol.evaluate("velocity", [[1.5, 0.0]]))
    assert report["reference_l2"] == pytest.approx(expected)
    assert report["absolute_l2"] == pytest.approx(0.2 * expected)
    assert report["relative_l2"] == pytest.approx(0.2)


@pytest.mark.parametrize("geometry,dim", [("cylindrical", 2), ("spherical", 3)])
def test_geoid_rejects_invalid_physical_inputs(geometry, dim):
    sol = uw.analytic.StokesSolution(mesh(dim), geometry=geometry)
    parameters = dict(
        outer_density_contrast=1.0,
        inner_density_contrast=1.0,
        outer_reference_gravity=1.0,
        inner_reference_gravity=1.0,
        gravitational_constant=0.01,
        source_density_coefficient=1.0,
    )
    with pytest.raises(ValueError, match="contrasts"):
        sol.geoid_response(**dict(parameters, inner_density_contrast=0))
    with pytest.raises(ValueError, match="gravity magnitudes"):
        sol.geoid_response(**dict(parameters, outer_reference_gravity=-1))
    sol.boundary = "zero"
    with pytest.raises(ValueError, match="free-slip"):
        sol.geoid_response(**parameters)
