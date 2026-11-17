"""FEM operator and preconditioner assembly agree on ranks without cells.

The exact solution is u=0, p=x-1/2 under body force (1,0), for either free
or zero slip. Four cells force empty partitions at MPI16. PETSc 3.25.4's
two-matrix FEM callback skipped an operator assembly on those ranks, pairing
later collectives incorrectly (MPI_ERR_BUFFER at the rotated PtAP).
"""

import pytest
import sympy as sp

import underworld3 as uw


@pytest.mark.level_2
@pytest.mark.tier_b
@pytest.mark.parametrize("boundary", ("rotated", "zero"))
def test_linear_pressure_with_empty_cell_partitions(boundary):
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(2, 2), minCoords=(0, 0), maxCoords=(1, 1), qdegree=3)
    velocity = uw.discretisation.MeshVariable("v_empty", mesh, 2, degree=2)
    pressure = uw.discretisation.MeshVariable("p_empty", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = 1.
    stokes.bodyforce = sp.Matrix([[1, 0]])
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-10
    for wall in ("Top", "Bottom", "Left", "Right"):
        if boundary == "rotated":
            stokes.add_rotated_freeslip_bc(0, wall)
        else:
            stokes.add_dirichlet_bc((0, 0), wall)
    first, last = mesh.dm.getHeightStratum(0)
    empty = uw.mpi.comm.allgather(first == last)
    if uw.mpi.size > 4:
        assert any(empty), "The four-cell fixture must exercise an empty rank"
    stokes.solve()
    x, _ = mesh.X
    area = uw.maths.Integral(mesh, sp.S.One).evaluate()
    mean = uw.maths.Integral(mesh, pressure.sym[0]).evaluate()/area
    pressure_error = uw.maths.Integral(
        mesh, (pressure.sym[0]-mean-x+sp.Rational(1, 2))**2).evaluate()
    velocity_error = uw.maths.Integral(mesh, velocity.sym.dot(velocity.sym)).evaluate()
    assert pressure_error < 1e-20, pressure_error
    assert velocity_error < 1e-20, velocity_error
