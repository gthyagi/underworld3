"""Discovery, optional-oracle independence and pure gravity API boundaries.

These tests exercise public imports and response calculations without FE solves.
Published coefficient and field accuracy remain covered by their oracle suites.
"""

import builtins
from dataclasses import asdict
import importlib
import importlib.util
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def test_discovery_distinguishes_fields_from_responses():
    name = "ZhongSphericalStokes"
    fields = uw.analytic.available()
    responses = uw.analytic.available(kind="responses")
    assert "KramerCylindricalStokes" in fields
    assert "AssessKramerCylindricalStokes" in fields
    assert name not in fields
    assert responses == [name]
    assert uw.analytic.available(kind="all") == sorted(set(fields) | set(responses))
    assert "ZhongSphericalStokesResponse" not in uw.analytic.available(kind="all")
    assert uw.analytic.is_available(name)
    assert "Propagator-matrix" in uw.analytic.describe(name)
    assert name in uw.analytic.available(installed_only=True, kind="responses")
    selected = uw.analytic.StokesSolution(geometry="spherical", case="zhong")
    assert isinstance(selected, getattr(uw.analytic, responses[0]))
    assert not hasattr(selected, "fn_velocity")


@pytest.mark.parametrize("lookup", ["describe", "is_available"])
def test_unknown_reference_has_a_consistent_error(lookup):
    with pytest.raises(ValueError, match="no analytic solution named"):
        getattr(uw.analytic, lookup)("not_a_reference")


def test_invalid_discovery_kind_is_rejected():
    with pytest.raises(ValueError, match="kind must be"):
        uw.analytic.available(kind="unknown")


@pytest.mark.parametrize(
    "geometry,case,name",
    [("cylindrical", "kramer", "KramerCylindricalStokes"),
     ("spherical", "kramer", "KramerSphericalStokes"),
     ("cylindrical", "thieulot", "ThieulotCylindricalStokes"),
     ("spherical", "thieulot", "ThieulotSphericalStokes")],
)
def test_case_wildcard_imports_exclude_implementation_details(geometry, case, name):
    namespace = {}
    exec(f"from underworld3.analytic.{geometry}.{case} import *", namespace)
    assert namespace[name] is getattr(uw.analytic, name)
    for internal in ("np", "sp", "AnalyticSolution", "KramerGeoidMixin",
                     "_KramerGeoidMixin", "_KramerCylindricalBase", "_ThieulotStokesBase"):
        assert internal not in namespace
    if case == "kramer":
        assert callable(namespace["build_delta_solution"])
        assert callable(namespace["analytical_pressure_sympy"])


def test_native_annulus_is_independent_of_optional_assess(monkeypatch):
    original = builtins.__import__

    def without_assess(name, *args, **kwargs):
        if name == "assess" or name.startswith("assess."):
            raise ImportError("assess intentionally unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_assess)
    mesh = SimpleNamespace(dim=2, X=sp.Matrix([sp.symbols("x y", real=True)]))
    native = uw.analytic.KramerCylindricalStokes(mesh)
    assert not isinstance(native, uw.analytic.AssessKramerCylindricalStokes)
    assert np.isfinite(native.evaluate("velocity", np.array([[0.7, 0.1]]))).all()
    with pytest.raises(ImportError, match="assess"):
        uw.analytic.AssessKramerCylindricalStokes(mesh)


def test_normalized_gravity_works_without_framework_imports(monkeypatch):
    from underworld3.analytic.spherical import gravity
    from underworld3.postprocessing import geoid

    options = dict(radius_inner=0.55, radius_outer=1.0, harmonic_degree=5,
                   surface_topography_coefficient=0.4, cmb_topography_coefficient=0.46,
                   surface_density_contrast=3300., cmb_density_contrast=5400.,
                   planet_radius=6370000., gravity=9.8, gravitational_constant=6.67e-11,
                   internal_load_radius=0.775, internal_load_coefficient=1.0)
    expected = gravity.spherical_shell_self_gravity_response(**options)
    assert geoid.SelfGravityResponse is gravity.SelfGravityResponse
    assert geoid.GeoidResponse is gravity.GeoidResponse
    assert geoid.spherical_shell_geoid_response is gravity.spherical_shell_geoid_response
    assert geoid.spherical_shell_self_gravity_response is gravity.spherical_shell_self_gravity_response
    assert pickle.loads(pickle.dumps(expected)) == expected
    # Older serialized responses resolve their original postprocessing class path.
    legacy = pickle.dumps(expected, protocol=0).replace(
        b"cunderworld3.analytic.spherical.gravity\nSelfGravityResponse\n",
        b"cunderworld3.postprocessing.geoid\nSelfGravityResponse\n",
    )
    assert b"cunderworld3.postprocessing.geoid\nSelfGravityResponse\n" in legacy
    assert pickle.loads(legacy) == expected

    spec = importlib.util.spec_from_file_location("_standalone_spherical_gravity", gravity.__file__)
    standalone = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, standalone)
    original = builtins.__import__

    def without_framework(name, *args, **kwargs):
        if name.split(".")[0] in ("underworld3", "mpi4py", "petsc4py"):
            raise ImportError("pure gravity must not import the framework")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_framework)
    spec.loader.exec_module(standalone)
    assert asdict(standalone.spherical_shell_self_gravity_response(**options)) == asdict(expected)
