"""Independent mesh coordinates must remain independent under SymPy caching."""

import numpy as np
import pytest
import sympy as sp

import underworld3 as uw
from underworld3.function.expressions import extract_meshes, _unwrap_for_compilation
from underworld3.postprocessing._circular_density_potential import _density

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def test_two_live_meshes_have_independent_symbolic_coordinates():
    first = uw.meshing.UnstructuredSimplexBox(cellSize=0.5, regular=True)
    x, y = first.X
    warm_radius = sp.sqrt(x*x + y*y)
    second = uw.meshing.UnstructuredSimplexBox(cellSize=0.5, regular=True)
    a, b = second.X
    radius = sp.sqrt(a*a + b*b)
    raw_radius = sp.sqrt(second.N.x**2 + second.N.y**2)

    assert first.N.x != second.N.x
    assert sp.diff(first.N.x, second.N.x) == 0
    assert sp.diff(second.N.x**2, a) == 2*second.N.x
    assert extract_meshes(warm_radius) == {first}
    assert extract_meshes(radius) == {second}
    assert _unwrap_for_compilation(radius, keep_constants=False, return_self=False) == raw_radius
    point = {second.N.x: 3, second.N.y: 4}
    assert raw_radius.subs(point) == 5


def test_circular_density_compiles_after_another_mesh_warms_expression_cache():
    first = uw.meshing.UnstructuredSimplexBox(cellSize=0.5, regular=True)
    x, y = first.X
    warm_expression = (x*x + y*y)*sp.cos(2*sp.atan2(y, x))
    second = uw.meshing.UnstructuredSimplexBox(cellSize=0.5, regular=True)
    a, b = second.X
    radius = sp.sqrt(a*a + b*b)
    expression = sp.Piecewise(((a*a + b*b)*sp.cos(2*sp.atan2(b, a)), radius < 2), (0, True))
    vertices = np.array([[[0., 0.], [1., 0.], [0., 1.]]])
    density = _density(expression, second, vertices)
    points = np.array([[.3, .4], [.6, .2]])
    np.testing.assert_allclose(density(0, points), points[:, 0]**2-points[:, 1]**2, rtol=1e-13)
    assert extract_meshes(warm_expression) == {first}


def test_solver_still_rejects_a_foreign_mesh_coordinate():
    first = uw.meshing.UnstructuredSimplexBox(cellSize=0.5, regular=True)
    second = uw.meshing.UnstructuredSimplexBox(cellSize=0.5, regular=True)
    unknown = uw.discretisation.MeshVariable("IdentityPotential", second, 1, degree=2)
    poisson = uw.systems.Poisson(second, u_Field=unknown)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.add_natural_bc(sp.Matrix([first.X[0]]), "Top")
    with pytest.raises(ValueError, match="foreign mesh"):
        poisson._check_expression_meshes()
