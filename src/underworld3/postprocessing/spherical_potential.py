r"""Spatial spherical Poisson gravity on conforming source meshes.

The sign is :math:`-\nabla^2\Phi=4\pi G\rho`, with geoid :math:`N=\Phi/g`.
Regular interior and decaying exterior harmonic branches give Robin coefficients
ell/r_inner and (ell+1)/r_outer, respectively. These are three-dimensional
conditions, distinct from cylindrical continuation. Analytical references and
normalization are in :mod:`underworld3.analytic.spherical.gravity`.
The numerical spatial loading is a UW3 implementation. Stokes gap corrections
do not automatically correct the faceted Poisson source geometry.
"""

from numbers import Integral

import numpy as np
import underworld3 as uw

from underworld3.maths.spherical_harmonics import real_harmonic_values, validate_harmonic
from ._density_potential import _solve_density_potential
from .cylindrical_potential import PhysicalPotentialSolve

__all__ = ["solve_spherical_spatial_potential", "spherical_potential_coefficient"]


def spherical_potential_coefficient(field, radius: float, degree: int, order: int,
    component: str, normalization: str, angular_order: int) -> tuple[float, float]:
    """Project the solved field on a physical sphere; coefficients are diagnostics.

    Gauss-Legendre polar and uniform azimuthal quadrature integrates the declared
    real harmonic. Evaluate through global owning-cell lookup, including when
    the target sphere cuts the surrounding gravity elements.
    """
    if isinstance(angular_order, bool) or not isinstance(angular_order, Integral) or not np.isfinite(radius) or radius <= 0 or angular_order < 2*degree+2:
        raise ValueError("Require positive radius and angular_order at least 2*degree+2")
    z, weights = np.polynomial.legendre.leggauss(angular_order)
    azimuth = np.arange(2*angular_order)*np.pi/angular_order
    zz, phi = np.meshgrid(z, azimuth, indexing="ij")
    radial = np.sqrt(1-zz**2)
    points = radius*np.stack((radial*np.cos(phi), radial*np.sin(phi), zz), axis=-1).reshape(-1, 3)
    harmonic = real_harmonic_values(points, degree=degree, order=order,
                                    component=component, normalization=normalization)
    values = np.asarray(uw.function.global_evaluate(field.sym[0], points)).reshape(-1)
    measure = np.broadcast_to(weights[:, None]*np.pi/angular_order, zz.shape).ravel()
    return float(np.sum(measure*values*harmonic)/np.sum(measure*harmonic**2)), float(np.sum(measure*values)/(4*np.pi))


def solve_spherical_spatial_potential(*, gravity_mesh, field_name: str,
    radius_inner: float, radius_internal: float, radius_outer: float,
    radius_exterior: float, radius_interior: float | None,
    continuation_degree: int, harmonic_order: int, component: str, normalization: str,
    inner_sheet_density, internal_sheet_density, outer_sheet_density, volume_density,
    inner_feedback_density: float, outer_feedback_density: float,
    gravitational_constant: float, potential_degree: int, tolerance: float,
    angular_order: int, inner_boundary: str, internal_boundary: str,
    outer_boundary: str, exterior_boundary: str, interior_boundary: str,
    solver_options: dict) -> PhysicalPotentialSolve:
    r"""Solve spherical potential from spatial FE/coordinate density sources.

    Numerical topography enters as a spatial density expression; no harmonic
    reconstruction is performed. Robin continuation is exact for the declared
    degree and approximate for other degrees present in a numerical field.
    Set radius_interior=None for a full-ball mesh with regularity at the centre.
    Volume density must already vanish outside its intended source region.

    Feedback adds sheet density ``alpha*Phi`` at each wall while keeping Stokes
    traction fixed. Use zero alpha for independent no-feedback verification.
    Density, geometry, G and gravity scales must use one consistent unit system;
    no unit conversion or density/topography sign inference occurs here.
    """
    ell = continuation_degree
    if isinstance(ell, bool) or not isinstance(ell, Integral) or ell < 1:
        raise ValueError("continuation_degree must be an integer at least one")
    if isinstance(harmonic_order, bool) or not isinstance(harmonic_order, Integral) or not 0 <= harmonic_order <= ell:
        raise ValueError("Require integer harmonic order between zero and degree")
    if component not in ("cos", "sin") or (harmonic_order == 0 and component != "cos"):
        raise ValueError("Require cos/sin component, with cos for order zero")
    validate_harmonic(ell, harmonic_order, component, normalization)
    if gravity_mesh.dim != 3 or not 0 < radius_inner < radius_internal < radius_outer < radius_exterior:
        raise ValueError("Require a 3-D gravity mesh and ordered source/exterior radii")
    if radius_interior is not None and not 0 < radius_interior < radius_inner:
        raise ValueError("Inner truncation must lie strictly below the inner source sphere")
    if not np.isfinite([gravitational_constant, tolerance, inner_feedback_density, outer_feedback_density]).all() or gravitational_constant <= 0 or not 0 < tolerance < 1:
        raise ValueError("Require positive finite G/tolerance and finite feedback densities")
    if potential_degree < 1 or angular_order < 2*ell+2:
        raise ValueError("Require positive potential degree and sufficient angular quadrature")
    continuation = [(exterior_boundary, (ell+1)/radius_exterior)]
    if radius_interior is not None:
        continuation.append((interior_boundary, ell/radius_interior))
    phi, poisson = _solve_density_potential(gravity_mesh=gravity_mesh,
        field_name=field_name, potential_degree=potential_degree,
        volume_density=volume_density, sheets=(
            (inner_boundary, inner_sheet_density, inner_feedback_density),
            (internal_boundary, internal_sheet_density, 0),
            (outer_boundary, outer_sheet_density, outer_feedback_density)),
        continuation=continuation, gravitational_constant=gravitational_constant,
        tolerance=tolerance, solver_options=solver_options)
    outer, outer_mean = spherical_potential_coefficient(phi, radius_outer, ell,
        harmonic_order, component, normalization, angular_order)
    inner, inner_mean = spherical_potential_coefficient(phi, radius_inner, ell,
        harmonic_order, component, normalization, angular_order)
    return PhysicalPotentialSolve(phi, poisson, outer, inner, outer_mean, inner_mean)
