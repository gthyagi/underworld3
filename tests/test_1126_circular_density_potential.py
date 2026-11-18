"""Native serial/MPI circular loads, spatial FE sources and fixed Jacobian."""

from mpi4py import MPI
import numpy as np
import pytest
import underworld3 as uw
from underworld3.postprocessing._density_potential import _solve_density_potential
from underworld3.postprocessing._cylindrical_potential_jump import NormalDerivativeJump

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.mark.parametrize("feedback,use_lifting", [(0.,False), (.01,False), (0.,True), (0.,"buffer")])
def test_spatial_field_and_coordinate_loads_agree(feedback, use_lifting):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(-2.5, -2.5), maxCoords=(2.5, 2.5),
        cellSize=.4, qdegree=4, regular=True)
    field = uw.discretisation.MeshVariable("SourcePolynomial", mesh, 1, degree=2)
    with mesh.access(field):
        field.data[:, 0] = np.sum(field.coords**2, axis=1)
    x, y = mesh.X
    solutions = []
    for name, density in (("CoordinateLoad", x*x+y*y), ("FieldLoad", field.sym[0])):
        sheets = ((1.1,density,feedback),(1.5,-density,0.),(2.1,density,feedback))
        lifting = NormalDerivativeJump(mesh,sheets,.01,{1.1:-1,1.5:0,2.1:1} if use_lifting == "buffer" else None) if use_lifting else None
        phi, poisson = _solve_density_potential(gravity_mesh=mesh, field_name=name,
            potential_degree=2, volume_density=density, sheets=(),
            continuation=[(label, 1.) for label in ("Left", "Right", "Top", "Bottom")],
            gravitational_constant=.01, tolerance=1e-11,
            solver_options={"snes_type":"ksponly", "ksp_type":"preonly", "pc_type":"lu", "pc_factor_mat_solver_type":"mumps"},
            circular_sources=dict(radii=(1.1, 1.5, 2.1),
                sheets=sheets, quadrature_order=8, potential_lifting=lifting))
        assert poisson.solve_report.converged
        assert poisson.circular_source_checks["coverage_relative_error"] < 1e-10
        vector = poisson.snes.getSolution()
        residual = vector.duplicate()
        poisson.snes.computeFunction(vector, residual)
        assert residual.norm() < 1e-9
        J, P = poisson.snes.getJacobian()[:2]
        direction = vector.duplicate(); direction.set(1.)
        shifted = vector.copy(); shifted.axpy(1., direction)
        difference = residual.duplicate()
        poisson.snes.computeFunction(shifted, difference); difference.axpy(-1., residual)
        poisson.snes.computeJacobian(vector, J, P)
        J.mult(direction, residual); difference.axpy(-1., residual)
        assert difference.norm() < 1e-9
        solutions.append(np.asarray(phi.array).copy())
        for temporary in (residual, direction, shifted, difference): temporary.destroy()
    error = float(np.max(abs(solutions[1]-solutions[0]), initial=0.))
    assert uw.mpi.comm.allreduce(error, op=MPI.MAX) < 1e-9
