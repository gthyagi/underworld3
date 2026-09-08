"""Predictor-corrector composition with the scalar transport solver."""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


@pytest.fixture
def fields():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25,
        qdegree=3,
    )
    temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    return mesh, temperature, sympy.zeros(1, mesh.dim)


def test_manager_rejects_a_different_solver_unknown(fields):
    mesh, temperature, velocity = fields
    other = uw.discretisation.MeshVariable("Other", mesh, 1, degree=1)
    transport = uw.systems.ddt.EulerianSUPGPC(mesh, temperature, velocity)
    with pytest.raises(ValueError, match="unknown|u_Field|field"):
        uw.systems.AdvDiffusion(mesh, other, velocity, DuDt=transport)


@pytest.mark.parametrize("method", ["citcoms", "pc_converged"])
def test_manager_uses_live_boundary_conditions_and_timestep(fields, method):
    mesh, temperature, velocity = fields
    temperature.array[:, 0, 0] = 1.0
    transport = uw.systems.ddt.EulerianSUPGPC(
        mesh, temperature, velocity, method=method,
    )
    thermal = uw.systems.AdvDiffusion(mesh, temperature, velocity, DuDt=transport)
    thermal.constitutive_model.Parameters.diffusivity = 0.1
    # Conditions are added after the manager is bound to the solver.
    for boundary in ("Left", "Right", "Top", "Bottom"):
        thermal.add_dirichlet_bc(1.0, boundary)
    thermal.solve(timestep=0.001)
    thermal.solve()
    assert thermal.DuDt is transport
    assert float(transport.delta_t.sym) == 0.001
    np.testing.assert_allclose(temperature.array, 1.0, rtol=0, atol=1e-12)
