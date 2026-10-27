"""Analytical serial/MPI gates for physical-circle mixed Stokes integration.

The independent pressure-gradient problem has u=0, p=x and f=(1,0).
No Kramer reference or spectral coefficient is used by the solver.
"""

import numpy as np
import pytest
import sympy as sp
import underworld3 as uw

pytestmark = pytest.mark.level_2


@pytest.mark.tier_b
def test_cylindrical_gap_preserves_pressure_gradient_and_null_modes():
    mesh = uw.meshing.Annulus(
        radiusInner=1.22, radiusOuter=2.22, cellSize=0.125, qdegree=3
    )
    velocity = uw.discretisation.MeshVariable("GapU", mesh, 2, degree=2)
    pressure = uw.discretisation.MeshVariable("GapP", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.bodyforce = sp.Matrix([1, 0])
    result = stokes.solve_cylindrical_gap(
        inner_boundary="Lower",
        outer_boundary="Upper",
        radius_inner=1.22,
        radius_outer=2.22,
        viscosity=1.0,
        pressure_jump=sp.S.Zero,
        pressure_jump_integral=0.0,
        cap_bodyforce=sp.Matrix([1, 0]),
        quadrature=8,
        linear_solver_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    area = uw.maths.Integral(mesh, sp.S.One).evaluate()
    velocity_rms = np.sqrt(
        uw.maths.Integral(mesh, velocity.sym.dot(velocity.sym)).evaluate() / area
    )
    pressure_rms = np.sqrt(
        uw.maths.Integral(mesh, (pressure.sym[0] - mesh.N.x) ** 2).evaluate() / area
    )
    assert (
        velocity_rms < 2e-8
    ), "A pure pressure gradient must not drive appreciable flow"
    assert (
        pressure_rms < 3e-8
    ), "Physical pressure must reproduce p=x with zero circular-domain mean"
    assert result.checks["relative_residual"] < 1e-10
    assert result.checks["pressure_null_relative_residual"] < 1e-12
    assert result.checks["rotation_null_relative_residual"] < 1e-12
    assert abs(result.checks["physical_area"] - np.pi * (2.22**2 - 1.22**2)) < 1e-11
