"""Exact affine stress recovery in the common discontinuous tensor space."""

import sympy as sp
import underworld3 as uw
from underworld3.postprocessing.stress import project_deviatoric_stress


def test_symmetric_affine_tensor_is_exact_without_smoothing():
    mesh = uw.meshing.Annulus(radiusInner=1.22, radiusOuter=2.22, cellSize=.125, qdegree=3)
    x, y = mesh.X
    exact = sp.Matrix([[2*x, x-y], [x-y, -2*x]])
    field = project_deviatoric_stress(mesh, exact, "AffineTauDG", 1e-11)
    error = sum((field.sym[i,j]-exact[i,j])**2 for i in range(2) for j in range(2))
    assert uw.maths.Integral(mesh, error).evaluate() < 1e-18
    assert field.degree == 1 and not field.continuous
