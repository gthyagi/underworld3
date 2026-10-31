"""Physical-space cylindrical sheet-potential solves on a gravity parent mesh."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import sympy as sp
import underworld3 as uw

from .geoid import cylindrical_cosine_boundary_coefficient


__all__ = ["PhysicalPotentialSolve", "solve_cylindrical_sheet_potential", "solve_cylindrical_spatial_potential"]


@dataclass
class PhysicalPotentialSolve:
    r"""Physical cylindrical Poisson solution and sampled interface coefficients.

    Attributes
    ----------
    potential_field : underworld3.discretisation.MeshVariable
        Scalar potential field on the caller's gravity mesh.
    poisson_system : underworld3.systems.Poisson
        System used to compute ``potential_field``.
    outer_potential, inner_potential : float
        Coefficients of ``cos(n*theta)`` at the outer and inner mantle radii.
        Units match the potential field; they are not geoid heights.
    outer_mean, inner_mean : float
        Degree-zero means of the sampled interface potential in the same units."""

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
    r"""Solve physical potential for three conforming cylindrical density sheets.

    Parameters
    ----------
    gravity_mesh : underworld3.discretisation.Mesh
        Two-dimensional gravity parent mesh containing the three conforming sheet
        interfaces and an exterior truncation. Call collectively on solver ranks.
    field_name : str
        Name of the scalar potential MeshVariable created on ``gravity_mesh``.
    radius_inner, radius_internal, radius_outer : float
        Positive mantle/sheet radii in mesh units, ordered strictly inner <
        internal < outer. They must match the caller's interface geometry.
    radius_exterior : float
        Exterior truncation radius, strictly larger than ``radius_outer``.
    radius_interior : float or None, default=None
        Optional inner truncation satisfying ``0 < radius_interior < radius_inner``.
        ``None`` omits the inner Robin datum; the caller supplies a suitable mesh
        containing the origin or its intended interior treatment.
    wavenumber : int
        Azimuthal mode ``n >= 2`` multiplying the unnormalized ``cos(n*theta)`` basis.
    inner_sheet_density, internal_sheet_density, outer_sheet_density : float, default=0
        Signed surface-mass density coefficients at the three radii. In physical
        units these are mass per area for the axially invariant cylindrical model.
    gravitational_constant : float, default=1
        Gravitational constant in a unit system consistent with radii and densities.
        The convention is ``laplacian(Phi)=-4*pi*G*rho``; no unit conversion is done.
    potential_degree : int, default=2
        Polynomial degree of the continuous scalar potential field.
    tolerance : float, default=1e-10
        Tolerance assigned to the Poisson system.
    boundary_samples : int, default=2048
        At least 16 equally spaced angular samples per circle for returned cosine
        coefficients and means. This controls diagnostic sampling, not mesh resolution.
    inner_boundary, internal_boundary, outer_boundary : str
        Sheet labels, defaulting to ``"Lower"``, ``"Internal"`` and ``"Upper"``.
    exterior_boundary : str, default="Exterior"
        Label for the exterior Robin condition.
    interior_boundary : str, default="Interior"
        Label for the inner Robin condition, used only with ``radius_interior``.
    solver_options : mapping or None, default=None
        PETSc option names/values passed to the Poisson system. No benchmark-specific
        preconditioner is selected by this helper.

    Returns
    -------
    PhysicalPotentialSolve
        Potential field, Poisson system, interface cosine coefficients and means.
        For consistent physical units, potential has units length squared/time squared;
        geoid height requires a separate division by reference gravity.

    Raises
    ------
    TypeError
        ``wavenumber`` is not an integer.
    ValueError
        Mode, sample count, mesh dimension or radius ordering is invalid.

    Notes
    -----
    UW solves ``-laplacian(Phi)=f`` with natural datum ``-grad(Phi).n=g``.
    Each sheet uses ``g=-4*pi*G*sigma*cos(n*theta)``. Exterior continuation is
    ``dPhi/dr+n*Phi/r=0``; optional interior continuation is
    ``dPhi/dr-n*Phi/r=0``. Samples use global field evaluation. The returned
    system remains available for callers to inspect its solve report."""

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

    _radius, theta = gravity_mesh.CoordinateSystem.xR
    cosine = sp.cos(mode*theta)
    return solve_cylindrical_spatial_potential(
        gravity_mesh=gravity_mesh, field_name=str(field_name),
        radius_inner=radius_inner, radius_internal=radius_internal,
        radius_outer=radius_outer, radius_exterior=radius_exterior,
        radius_interior=radius_interior, continuation_wavenumber=mode,
        inner_sheet_density=float(inner_sheet_density)*cosine,
        internal_sheet_density=float(internal_sheet_density)*cosine,
        outer_sheet_density=float(outer_sheet_density)*cosine, volume_density=sp.S.Zero,
        inner_feedback_density=0., outer_feedback_density=0.,
        gravitational_constant=gravitational_constant, potential_degree=int(potential_degree),
        tolerance=float(tolerance), boundary_samples=int(boundary_samples),
        inner_boundary=inner_boundary, internal_boundary=internal_boundary,
        outer_boundary=outer_boundary, exterior_boundary=exterior_boundary,
        interior_boundary=interior_boundary,
        solver_options={} if solver_options is None else solver_options,
    )


def solve_cylindrical_spatial_potential(
    *, gravity_mesh, field_name: str, radius_inner: float, radius_internal: float,
    radius_outer: float, radius_exterior: float, radius_interior: float | None,
    continuation_wavenumber: int, inner_sheet_density, internal_sheet_density,
    outer_sheet_density, volume_density, inner_feedback_density: float,
    outer_feedback_density: float, gravitational_constant: float,
    potential_degree: int, tolerance: float, boundary_samples: int,
    inner_boundary: str, internal_boundary: str, outer_boundary: str,
    exterior_boundary: str, interior_boundary: str, solver_options: dict,
) -> PhysicalPotentialSolve:
    r"""Solve -laplacian(Phi)=4*pi*G*rho using spatial density expressions.

    The three sheets are conforming gravity-mesh interfaces. Supply numerical
    spatial loads directly as expressions on that mesh; no harmonic coefficient
    enters their construction. Positive density uses the natural interface datum
    -4*pi*G*sigma in UW3's weak convention. volume_density must already be zero
    outside its intended source region. Parameters are explicit, nondimensional
    or consistently scaled; this API performs no unit conversion.

    Robin continuation uses continuation_wavenumber at the truncations. It is
    exact for that angular mode and an approximation for other modes in a
    numerical spatial field. inner_feedback_density and outer_feedback_density
    specify additional sheet density per unit potential, normally Delta_rho/g;
    use zero for no feedback. The solve includes sigma=sigma_0+alpha*Phi on each
    wall, retaining the original Stokes traction without another Stokes solve.
    This is the spatial form of the analytical two-interface feedback equation.
    The returned coefficients are post-solve diagnostics
    in that mode, not sources. Use exterior-domain controls when assessing error.

    The PDE sign follows analytic.cylindrical.gravity and Simons (1996), Appendix
    B. Conforming finite-element spatial loading is a UW3 implementation; the
    circular Stokes gap correction is not automatically a Poisson correction.
    """
    mode = continuation_wavenumber
    if isinstance(mode, bool) or not isinstance(mode, Integral) or mode < 2:
        raise ValueError("continuation_wavenumber must be an integer at least two")
    if gravity_mesh.dim != 2 or not 0 < radius_inner < radius_internal < radius_outer < radius_exterior:
        raise ValueError("Require a two-dimensional mesh and ordered source/exterior radii")
    if radius_interior is not None and not 0 < radius_interior < radius_inner:
        raise ValueError("The inner truncation must lie below the inner source circle")
    if not np.isfinite(gravitational_constant) or gravitational_constant <= 0:
        raise ValueError("gravitational_constant must be finite and positive")
    if potential_degree < 1 or boundary_samples < 16:
        raise ValueError("Require positive potential degree and at least 16 boundary samples")
    if not np.all(np.isfinite([inner_feedback_density, outer_feedback_density])):
        raise ValueError("Feedback density coefficients must be finite")
    phi = uw.discretisation.MeshVariable(field_name, gravity_mesh, 1, degree=potential_degree)
    poisson = uw.systems.Poisson(gravity_mesh, u_Field=phi)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.
    scale = 4*sp.pi*float(gravitational_constant)
    poisson.f = scale*volume_density
    for boundary, density in (
        (inner_boundary, inner_sheet_density+inner_feedback_density*phi.sym[0]),
        (internal_boundary, internal_sheet_density),
        (outer_boundary, outer_sheet_density+outer_feedback_density*phi.sym[0]),
    ):
        if density != 0:
            poisson.add_natural_bc(sp.Matrix([-scale*density]), boundary)
    poisson.add_natural_bc(sp.Matrix([mode*phi.sym[0]/radius_exterior]), exterior_boundary)
    if radius_interior is not None:
        poisson.add_natural_bc(sp.Matrix([mode*phi.sym[0]/radius_interior]), interior_boundary)
    poisson.tolerance = tolerance
    for key, value in solver_options.items():
        poisson.petsc_options[key] = value
    poisson.solve(verbose=False)
    outer, outer_mean = _interface_coefficient(phi, radius_outer, mode, boundary_samples)
    inner, inner_mean = _interface_coefficient(phi, radius_inner, mode, boundary_samples)
    return PhysicalPotentialSolve(phi, poisson, outer, inner, outer_mean, inner_mean)
