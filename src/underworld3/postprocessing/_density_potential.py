"""Shared conforming sheet/volume Poisson assembly; geometry supplies continuation."""

import sympy as sp
import underworld3 as uw


def _solve_density_potential(*, gravity_mesh, field_name, potential_degree,
    volume_density, sheets, continuation, gravitational_constant, tolerance,
    solver_options, circular_sources=None):
    """Assemble spatial loads with -laplacian(Phi)=4*pi*G*rho.

    Each sheet is (boundary, prescribed density, density per unit potential).
    Each continuation is (boundary, positive Robin coefficient). The caller
    supplies conforming interface labels and all geometry-specific scales.
    """
    phi = uw.discretisation.MeshVariable(field_name, gravity_mesh, 1, degree=potential_degree)
    poisson = uw.systems.Poisson(gravity_mesh, u_Field=phi)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = sp.Integer(1)
    scale = 4*sp.pi*sp.nsimplify(gravitational_constant, rational=True)
    poisson.f = scale*volume_density if circular_sources is None else sp.S.Zero
    for boundary, density, feedback in (() if circular_sources is not None else sheets):
        source = density+sp.nsimplify(feedback, rational=True)*phi.sym[0]
        if source != 0:
            poisson.add_natural_bc(sp.Matrix([-scale*source]), boundary)
    for boundary, coefficient in continuation:
        poisson.add_natural_bc(sp.Matrix([sp.nsimplify(coefficient, rational=True)*phi.sym[0]]), boundary)
    poisson.tolerance = tolerance
    for key, value in solver_options.items():
        poisson.petsc_options[key] = value
    if circular_sources is not None:
        from ._circular_density_potential import install_circular_sources
        install_circular_sources(poisson=poisson, volume_density=volume_density,
                                 gravitational_constant=gravitational_constant, **circular_sources)
    poisson.solve(verbose=False)
    return phi, poisson
