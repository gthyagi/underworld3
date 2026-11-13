"""Validate public cylindrical potential inputs and caller-owned solver policy."""

from types import SimpleNamespace

import pytest
import sympy as sp
import underworld3 as uw
from underworld3.postprocessing import cylindrical_potential as potential

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def parameters():
    return dict(
        gravity_mesh=SimpleNamespace(dim=2),
        field_name="Phi",
        radius_inner=1.22,
        radius_internal=2.0,
        radius_outer=2.22,
        radius_exterior=2.47,
        radius_interior=0.97,
        wavenumber=2,
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(wavenumber=1),
        dict(wavenumber=2.1),
        dict(radius_exterior=2.0),
        dict(radius_interior=1.5),
        dict(gravity_mesh=SimpleNamespace(dim=3)),
    ],
)
def test_rejects_invalid_inputs_before_allocating_solver(change):
    with pytest.raises((ValueError, TypeError)):
        potential.solve_cylindrical_sheet_potential(**(parameters() | change))


def test_labels_and_options_are_supplied_by_caller(monkeypatch):
    phi = sp.Symbol("phi")
    radius, theta = sp.symbols("r theta", real=True)
    mesh = SimpleNamespace(dim=2, CoordinateSystem=SimpleNamespace(xR=(radius, theta)))
    variable = SimpleNamespace(sym=(phi,))

    class FakePoisson:
        def __init__(self, mesh, u_Field):
            self.petsc_options, self.boundaries = {}, []

        def add_natural_bc(self, value, boundary):
            self.boundaries.append((boundary, value))

        def solve(self, verbose):
            assert self.petsc_options == {"ksp_type": "preonly", "pc_type": "lu"}

    model = SimpleNamespace(Parameters=SimpleNamespace(diffusivity=None))
    monkeypatch.setattr(uw.discretisation, "MeshVariable", lambda *a, **k: variable)
    monkeypatch.setattr(uw.systems, "Poisson", FakePoisson)
    monkeypatch.setattr(uw.constitutive_models, "DiffusionModel", model)
    monkeypatch.setattr(potential, "_interface_coefficient", lambda *a: (0.73, 0.0))
    solved = potential.solve_cylindrical_sheet_potential(
        **(parameters() | dict(gravity_mesh=mesh)),
        inner_sheet_density=0.1,
        internal_sheet_density=0.2,
        outer_sheet_density=0.3,
        inner_boundary="Core",
        internal_boundary="Sheet",
        outer_boundary="Mantle",
        exterior_boundary="Vacuum",
        interior_boundary="Truncation",
        solver_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    assert [label for label, _ in solved.poisson_system.boundaries] == [
        "Core",
        "Sheet",
        "Mantle",
        "Vacuum",
        "Truncation",
    ]
    assert solved.outer_potential == 0.73


def test_spatial_loads_and_feedback_enter_the_same_weak_source(monkeypatch):
    """Nonmodal spatial loads stay intact; only explicit feedback depends on Phi."""
    phi, x, y = sp.symbols("phi x y")
    mesh = SimpleNamespace(dim=2)
    variable = SimpleNamespace(sym=(phi,))

    class FakePoisson:
        def __init__(self, mesh, u_Field):
            self.petsc_options, self.boundaries = {}, []

        def add_natural_bc(self, value, boundary):
            self.boundaries.append((boundary, value[0]))

        def solve(self, verbose):
            pass

    model = SimpleNamespace(Parameters=SimpleNamespace(diffusivity=None))
    monkeypatch.setattr(uw.discretisation, "MeshVariable", lambda *a, **k: variable)
    monkeypatch.setattr(uw.systems, "Poisson", FakePoisson)
    monkeypatch.setattr(uw.constitutive_models, "DiffusionModel", model)
    monkeypatch.setattr(potential, "_interface_coefficient", lambda *a: (0.73, 0.0))
    solved = potential.solve_cylindrical_spatial_potential(
        gravity_mesh=mesh, field_name="Phi", radius_inner=1.22, radius_internal=2.,
        radius_outer=2.22, radius_exterior=2.5, radius_interior=1.,
        continuation_wavenumber=2, inner_sheet_density=x+2*y,
        internal_sheet_density=sp.S.Zero, outer_sheet_density=y*y, volume_density=x-y,
        inner_feedback_density=.03, outer_feedback_density=0., gravitational_constant=.01,
        potential_degree=2, tolerance=1e-10, boundary_samples=32,
        inner_boundary="Lower", internal_boundary="Internal", outer_boundary="Upper",
        exterior_boundary="Exterior", interior_boundary="Interior", solver_options={},
    )
    boundaries = dict(solved.poisson_system.boundaries)
    assert sp.simplify(boundaries["Lower"]+.04*sp.pi*(x+2*y+.03*phi)) == 0
    assert sp.simplify(boundaries["Upper"]+.04*sp.pi*y*y) == 0
    assert sp.simplify(solved.poisson_system.f-.04*sp.pi*(x-y)) == 0
    assert sp.simplify(sp.diff(boundaries["Lower"], phi)+.0012*sp.pi) == 0
