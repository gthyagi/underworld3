"""Physical-space cylindrical sheet-potential solves on a gravity parent mesh."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import sympy as sp
import underworld3 as uw

from .geoid import cylindrical_cosine_boundary_coefficient


__all__ = ["PhysicalPotentialSolve", "solve_cylindrical_sheet_potential"]


@dataclass
class PhysicalPotentialSolve:
    """One physical-space Poisson solution and its interface coefficients."""

    potential_field: object
    poisson_system: object
    outer_potential: float
    inner_potential: float
    outer_mean: float
    inner_mean: float


def _interface_coefficient(field, radius, wavenumber, samples):
    theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    coords = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    values = np.asarray(uw.function.global_evaluate(field.sym[0], coords), dtype=float).reshape(-1)
    return cylindrical_cosine_boundary_coefficient(coords, values, wavenumber)


def solve_cylindrical_sheet_potential(
    *,
    gravity_mesh,
    field_name,
    radius_inner,
    radius_internal,
    radius_outer,
    radius_exterior,
    radius_interior=None,
    wavenumber,
    inner_sheet_density=0.0,
    internal_sheet_density=0.0,
    outer_sheet_density=0.0,
    gravitational_constant=1.0,
    potential_degree=2,
    tolerance=1.0e-10,
    boundary_samples=2048,
    inner_boundary="Lower",
    internal_boundary="Internal",
    outer_boundary="Upper",
    exterior_boundary="Exterior",
    interior_boundary="Interior",
    solver_options=None,
):
    r"""Solve for three conforming cylindrical sheets without regularisation.

    UW solves ``-laplacian(Phi)=f`` and its natural datum is
    ``-grad(Phi).n=g``.  For the benchmark convention
    ``laplacian(Phi)=-4*pi*G*rho``, each interface sheet is therefore supplied
    as ``g=-4*pi*G*sigma*cos(n*theta)``.  The exterior condition is the exact
    single-mode Robin continuation ``dPhi/dr+n*Phi/r=0``.  When
    ``radius_interior`` is supplied, regular continuation toward the omitted
    origin is imposed as ``dPhi/dr-n*Phi/r=0`` on ``Interior``.
    Boundary labels are explicit keyword inputs. Solver options are supplied by
    the caller; this helper does not select a benchmark/HPC preconditioner.
    """

    if isinstance(wavenumber, bool) or not isinstance(wavenumber, Integral):
        raise TypeError("wavenumber must be an integer.")
    mode = int(wavenumber)
    if mode < 2:
        raise ValueError("wavenumber must be at least 2.")
    if int(boundary_samples) < 16:
        raise ValueError("boundary_samples must be at least 16.")

    if gravity_mesh.dim != 2:
        raise ValueError("The cylindrical potential solve requires a two-dimensional mesh.")
    if not 0 < radius_inner < radius_internal < radius_outer < radius_exterior:
        raise ValueError("Expected ordered positive sheet radii and an exterior truncation.")
    if radius_interior is not None and not 0 < radius_interior < radius_inner:
        raise ValueError("radius_interior must lie strictly below radius_inner.")

    phi = uw.discretisation.MeshVariable(
        str(field_name), gravity_mesh, 1, degree=int(potential_degree)
    )
    _radius, theta = gravity_mesh.CoordinateSystem.xR
    cosine_mode = sp.cos(mode * theta)
    sheet_scale = -4.0 * sp.pi * float(gravitational_constant) * cosine_mode

    poisson = uw.systems.Poisson(gravity_mesh, u_Field=phi)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    for boundary, density in (
        (inner_boundary, inner_sheet_density),
        (internal_boundary, internal_sheet_density),
        (outer_boundary, outer_sheet_density),
    ):
        density = float(density)
        if density != 0.0:
            poisson.add_natural_bc(sp.Matrix([sheet_scale * density]), boundary)
    poisson.add_natural_bc(
        sp.Matrix([mode * phi.sym[0] / float(radius_exterior)]), exterior_boundary
    )
    if radius_interior is not None:
        # On the inner truncation n_domain=-e_r, so UW's natural flux
        # -grad(Phi).n equals +dPhi/dr.  Regular Phi_n proportional to r**n
        # therefore gives the same positive n*Phi/r datum as the exterior.
        poisson.add_natural_bc(
            sp.Matrix([mode * phi.sym[0] / float(radius_interior)]),
            interior_boundary,
        )
    poisson.tolerance = float(tolerance)
    if solver_options is not None:
        for option, value in solver_options.items():
            poisson.petsc_options[str(option)] = value
    poisson.solve(verbose=False)

    outer_potential, outer_mean = _interface_coefficient(
        phi, float(radius_outer), mode, int(boundary_samples)
    )
    inner_potential, inner_mean = _interface_coefficient(
        phi, float(radius_inner), mode, int(boundary_samples)
    )
    return PhysicalPotentialSolve(
        potential_field=phi,
        poisson_system=poisson,
        outer_potential=outer_potential,
        inner_potential=inner_potential,
        outer_mean=outer_mean,
        inner_mean=inner_mean,
    )
