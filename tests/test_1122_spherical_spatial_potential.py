"""Spherical spatial-source and three-dimensional continuation contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp
import underworld3 as uw

from underworld3.postprocessing import spherical_potential

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def test_spatial_sources_feedback_and_spherical_robin_coefficients(monkeypatch):
    phi, x, y, z = sp.symbols("phi x y z")
    variable = SimpleNamespace(sym=(phi,))

    class Poisson:
        def __init__(self, mesh, u_Field):
            self.petsc_options, self.boundaries = {}, []

        def add_natural_bc(self, value, boundary):
            self.boundaries.append((boundary, value[0]))

        def solve(self, verbose):
            assert self.petsc_options == {"ksp_type": "preonly", "pc_type": "lu"}

    monkeypatch.setattr(uw.discretisation, "MeshVariable", lambda *a, **k: variable)
    monkeypatch.setattr(uw.systems, "Poisson", Poisson)
    monkeypatch.setattr(uw.constitutive_models, "DiffusionModel", SimpleNamespace(Parameters=SimpleNamespace()))
    monkeypatch.setattr(spherical_potential, "spherical_potential_coefficient", lambda *a: (1.2, .3))
    result = spherical_potential.solve_spherical_spatial_potential(
        gravity_mesh=SimpleNamespace(dim=3), field_name="Phi", radius_inner=.55,
        radius_internal=.775, radius_outer=1., radius_exterior=1.25, radius_interior=.25,
        continuation_degree=2, harmonic_order=1, component="cos", normalization="real_orthonormal",
        inner_sheet_density=x+y*z, internal_sheet_density=z, outer_sheet_density=x*x-y*y,
        volume_density=x*y*z, inner_feedback_density=.02, outer_feedback_density=.03,
        gravitational_constant=.01, potential_degree=2, tolerance=1e-10, angular_order=12,
        inner_boundary="Core", internal_boundary="Load", outer_boundary="Surface",
        exterior_boundary="Exterior", interior_boundary="Interior",
        solver_options={"ksp_type": "preonly", "pc_type": "lu"})
    bc = dict(result.poisson_system.boundaries)
    assert sp.simplify(bc["Core"]+sp.pi/25*(x+y*z+phi/50)) == 0
    assert sp.simplify(bc["Load"]+sp.pi*z/25) == 0
    assert sp.simplify(bc["Surface"]+sp.pi/25*(x*x-y*y+3*phi/100)) == 0
    assert sp.simplify(result.poisson_system.f-sp.pi*x*y*z/25) == 0
    assert sp.simplify(bc["Exterior"]-sp.Rational(12, 5)*phi) == 0
    assert sp.simplify(bc["Interior"]-8*phi) == 0


def test_physical_sphere_harmonic_projection_has_declared_normalization(monkeypatch):
    """Y21=x*z times its known normalization; add a constant and other mode."""
    radius = .7

    def values(expression, points):
        unit = points/radius
        y21 = -np.sqrt(15/(4*np.pi))*unit[:, 0]*unit[:, 2]
        return 2.3*y21+.17+.2*unit[:, 1]

    monkeypatch.setattr(uw.function, "global_evaluate", values)
    coefficient, mean = spherical_potential.spherical_potential_coefficient(
        SimpleNamespace(sym=(sp.Symbol("Phi"),)), radius, 2, 1, "cos", "real_orthonormal", 16)
    np.testing.assert_allclose([coefficient, mean], [2.3, .17], rtol=1e-13, atol=1e-14)
