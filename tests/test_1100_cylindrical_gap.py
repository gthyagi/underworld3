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


@pytest.mark.tier_b
def test_circular_cap_area_and_empty_partition():
    from underworld3.utilities._cylindrical_gap import _cap_quadrature

    radius, half_angle = 2.22, 0.06
    edges = radius * np.array(
        [
            [
                [np.cos(half_angle), -np.sin(half_angle)],
                [np.cos(half_angle), np.sin(half_angle)],
            ]
        ]
    )
    rule = _cap_quadrature(edges, radius, 12, 4)
    exact = radius**2 * (2 * half_angle - np.sin(2 * half_angle)) / 2
    np.testing.assert_allclose(rule["cap_weight"].sum(), exact, atol=1e-14, rtol=0)
    empty = _cap_quadrature(np.empty((0, 2, 2)), radius, 8, 4)
    assert empty["cap_points"].shape == (0, 32, 2)


@pytest.mark.tier_b
def test_mixed_element_transpose_is_inserted_in_row_order():
    """PETSc Mat.setValues must receive C-contiguous rectangular block data."""
    from petsc4py import PETSc
    from underworld3.utilities._cylindrical_gap import _insert_blocks

    matrix = PETSc.Mat().createAIJ(size=(15, 15), nnz=15, comm=PETSc.COMM_SELF)
    b = np.arange(36, dtype=float).reshape(1, 3, 12)
    _insert_blocks(
        matrix,
        np.arange(12, dtype=PETSc.IntType)[None],
        np.arange(12, 15, dtype=PETSc.IntType)[None],
        {"A": np.zeros((1, 12, 12)), "B": b},
        1.0,
        1,
    )
    matrix.assemble()
    actual = matrix.getValues(
        np.arange(12, dtype=PETSc.IntType), np.arange(12, 15, dtype=PETSc.IntType)
    )
    np.testing.assert_array_equal(actual, b[0].T)
    assert matrix.isSymmetric(tol=1e-14)
    matrix.destroy()
