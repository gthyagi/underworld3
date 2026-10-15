"""Restricted Poisson sheet API rejects invalid operators before any solve."""
from types import SimpleNamespace
import pytest
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


@pytest.mark.parametrize("case, error, match", [
    ("solution_degree", ValueError, "P2 solution"),
    ("load_degree", ValueError, "distinct scalar continuous P2"),
    ("same_field", ValueError, "distinct scalar continuous P2"),
    ("diffusivity", ValueError, "unit diffusivity"),
    ("constitutive_model", TypeError, "DiffusionModel"),
    ("tolerance", ValueError, "positive correction tolerance"),
    ("mixed_degree", ValueError, "share harmonic_degree"),
])
def test_poisson_sheet_method_rejects_unsupported_inputs_before_solving(case, error, match, monkeypatch):
    # A valid communicator/chart is enough: rejection must precede field assembly.
    mesh = SimpleNamespace(dim=3, dm=SimpleNamespace(
        getComm=lambda: SimpleNamespace(tompi4py=lambda: uw.mpi.comm),
        getChart=lambda: (0, 1)))
    solution = SimpleNamespace(mesh=mesh, degree=2, continuous=True, field_id=0)
    load = SimpleNamespace(mesh=mesh, degree=2, continuous=True,
                           num_components=1, field_id=1)
    class DiffusionModel:
        Parameters = SimpleNamespace(diffusivity=1)
    monkeypatch.setattr(uw.constitutive_models, "DiffusionModel", DiffusionModel)
    constitutive_model = DiffusionModel()
    constitutive_model.Parameters.diffusivity = 1
    def unexpected_solve():
        pytest.fail("An unsupported sheet configuration reached the PDE solve")
    solver = SimpleNamespace(mesh=mesh, u=solution, constitutive_model=constitutive_model,
                             tolerance=1e-8, solve=unexpected_solve)
    arguments = dict(load_variable=load, harmonic_degree=2)
    if case == "solution_degree": solution.degree = 1
    elif case == "load_degree": load.degree = 1
    elif case == "same_field": load.field_id = solution.field_id
    elif case == "diffusivity": constitutive_model.Parameters.diffusivity = 2
    elif case == "constitutive_model": solver.constitutive_model = object()
    elif case == "tolerance": arguments["tolerance"] = 0
    elif case == "mixed_degree": arguments["sheet_modes"] = [dict(harmonic_degree=4, sheet_densities={})]
    with pytest.raises(error, match=match):
        uw.systems.Poisson.solve_linear_spherical_sheet_sources(solver, **arguments)


def test_poisson_sheet_method_preserves_exact_base_solution_without_sheets():
    import os
    import numpy as np
    import sympy as sp
    root_pid = uw.mpi.comm.bcast(os.getpid() if uw.mpi.rank == 0 else None, root=0)
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0., 0., 0.), maxCoords=(1., 1., 1.), cellSize=.5,
        regular=True, qdegree=4, filename=f"/tmp/uw-sheet-base-{root_pid}.msh")
    potential = uw.discretisation.MeshVariable("ExactSheetBase", mesh, 1, degree=2)
    load = uw.discretisation.MeshVariable("EmptySheetBaseLoad", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=potential)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    x, y, z = mesh.X
    exact = x*x + y*y + z*z
    poisson.f = -6
    poisson.tolerance = 1e-10
    for wall in ("Top", "Bottom", "Left", "Right", "Front", "Back"):
        poisson.add_dirichlet_bc([exact], wall)
    result = poisson.solve_linear_spherical_sheet_sources(
        load_variable=load, harmonic_degree=2)
    assert result["solver_reason"] > 0 and result["load_diagnostics"] == {}
    expected = np.sum(np.asarray(potential.coords)**2, axis=1)
    np.testing.assert_allclose(potential.array[:, 0, 0], expected, rtol=1e-9, atol=1e-11)
