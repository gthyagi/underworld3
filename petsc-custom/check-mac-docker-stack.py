"""Check native PETSc's serial/MPI direct solvers against manufactured solutions.

Run with the environment's Python and MPI launcher. True residuals are required:
a positive KSP convergence reason alone did not detect the Darwin COMMON-symbol
linker failure in MUMPS's distributed-RHS path.
"""

# HDF5 imports MPI before PETSc in UW3; retain that order in this regression.
import h5py
import numpy as np
from petsc4py import PETSc

assert h5py.get_config().mpi
assert PETSc.Sys.getVersion() == (3, 25, 4)
assert np.dtype(PETSc.IntType).itemsize == 4
assert np.dtype(PETSc.ScalarType) == np.dtype("float64")

comm = PETSc.COMM_WORLD
matrix = PETSc.Mat().createAIJ(size=(32, 32), nnz=3, comm=comm)
start, end = matrix.getOwnershipRange()
for row in range(start, end):
    columns = [col for col in (row - 1, row, row + 1) if 0 <= col < 32]
    matrix.setValues(
        row, columns, [4 + row / 32 if col == row else -1 for col in columns]
    )
matrix.assemble()
exact, rhs = matrix.createVecs()
solution, residual = exact.duplicate(), rhs.duplicate()

for solver in ("mumps", "superlu_dist"):
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(matrix)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType(solver)
    for shift in (0.0, 0.7):
        exact.getArray()[:] = np.sin(np.arange(start, end) + shift) + 2
        matrix.mult(exact, rhs)
        ksp.solve(rhs, solution)
        matrix.mult(solution, residual)
        residual.axpy(-1, rhs)
        relative_residual = residual.norm() / rhs.norm()
        solution.axpy(-1, exact)
        error = solution.norm(PETSc.NormType.NORM_INFINITY)
        assert ksp.getConvergedReason() > 0
        assert error < 1e-12 and relative_residual < 1e-12, (
            solver,
            error,
            relative_residual,
        )
    ksp.destroy()
    if comm.getRank() == 0:
        print(
            f"DIRECT_SOLVER_OK solver={solver} ranks={comm.getSize()} error={error:.3e} residual={relative_residual:.3e}",
            flush=True,
        )

for obj in (matrix, exact, rhs, solution, residual):
    obj.destroy()
