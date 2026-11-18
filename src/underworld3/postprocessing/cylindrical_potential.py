"""Physical-space cylindrical sheet-potential solves on a gravity parent mesh."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import sympy as sp
import underworld3 as uw

from .geoid import cylindrical_cosine_boundary_coefficient
from ._density_potential import _solve_density_potential
from ._cylindrical_potential_jump import NormalDerivativeJump


__all__ = ["PhysicalPotentialSolve", "solve_cylindrical_sheet_potential", "solve_cylindrical_spatial_potential"]


@dataclass
class PhysicalPotentialSolve:
    r"""Physical cylindrical Poisson solution and sampled interface coefficients.

    Attributes
    ----------
    potential_field : underworld3.discretisation.MeshVariable
        Scalar FE field on the caller's gravity mesh; the regular remainder
        when potential_lifting is present. Use potential_expression for the
        total physical potential.
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
    potential_lifting: object = None

    @property
    def potential_expression(self):
        """Total physical potential; potential_field is regular when lifted."""
        return self.potential_field.sym[0]+(sp.S.Zero if self.potential_lifting is None else self.potential_lifting.expression)


def _interface_coefficient(field, radius, wavenumber, samples, lifting=None):
    theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    coords = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    expression = field.sym[0]+(sp.S.Zero if lifting is None else lifting.expression)
    values = np.asarray(uw.function.global_evaluate(expression, coords), dtype=float).reshape(-1)
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
    source_integration: str = "mesh", source_quadrature: int = 8,
    normal_derivative_jump: bool = False, jump_lifting_support: str = "global",
    internal_lifting_half_width: float = 0.,
) -> PhysicalPotentialSolve:
    r"""Solve -laplacian(Phi)=4*pi*G*rho using spatial density expressions.

    The three sheets are conforming gravity-mesh interfaces. Supply numerical
    spatial loads directly as expressions on that mesh; no harmonic coefficient
    enters their construction. Positive density uses the natural interface datum
    -4*pi*G*sigma in UW3's weak convention. volume_density must already be zero
    outside its intended source region. Parameters are explicit, nondimensional
    or consistently scaled; this API performs no unit conversion.

    ``source_integration="mesh"`` retains the conforming polygon load. Select
    ``"circular"`` to integrate sheets on their physical circles and volume
    density inside radius_inner<r<radius_outer, clipped to each containing
    affine triangle with positive Gauss rules of order source_quadrature.
    This supports coordinate expressions and gravity-mesh FE density values;
    the native diffusion/Robin operator is unchanged. The resulting linear
    system is frozen: recreate it after changing sources, mesh or coefficients.
    Coverage, signed loads and assembly timing are available through
    poisson_system.circular_source_checks. See Saye (2015),
    doi:10.1137/140966290, for geometry-resolving quadrature principles.

    ``normal_derivative_jump=True`` solves for the regular remainder after
    extracting L=-2*pi*G*sum(sigma_j(x)*abs(r-R_j)). It enforces [Phi]=0 and
    [d_r Phi]=-4*pi*G*sigma on the physical circles (outside minus inside).
    Both -int grad(L).grad(test) and the native -int beta*L*test Robin load
    enter the right-hand side. This currently supports prescribed sheets on
    annular gravity meshes, without feedback. potential_field stores the
    regular remainder; potential_expression includes L and must be used for
    physical evaluation. Never reconstruct the total as a plain nodal FE
    projection. See Guzman, Sanchez & Sarkis (2016), doi:10.1051/m2an/2015093;
    these distance lifting/Robin adaptations need their own tests.
    ``jump_lifting_support="buffer"`` confines wall lifts to the core/vacuum
    side: L_j=-2*pi*G*sigma_j*(abs(r-R_j)+s_j*(r-R_j)), s_inner=-1,
    s_outer=+1. L=0 in the mantle, preserving its original trial space.
    Internal sheets still use the two-sided distance lift. The default
    ``"global"`` preserves the original representation.
    A positive ``internal_lifting_half_width=w`` localizes the internal-sheet
    lift to |r-radius_internal|<w using the C2 quintic cutoff documented in
    _cylindrical_potential_jump. The cutoff gradient and both band edges enter
    the weak-load integration. Zero retains the original global internal lift;
    choose w strictly below both distances to the mantle walls. This parameter
    changes only the known lifting representation, not the physical sheet.

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
    if source_integration not in ("mesh", "circular"):
        raise ValueError("source_integration must be 'mesh' or 'circular'")
    if isinstance(source_quadrature, bool) or not isinstance(source_quadrature, int) or source_quadrature < 2:
        raise ValueError("source_quadrature must be an integer >= 2")
    if normal_derivative_jump and (source_integration != "circular" or radius_interior is None
                                  or inner_feedback_density != 0 or outer_feedback_density != 0):
        raise ValueError("Normal-derivative correction requires circular integration, an annular gravity domain and prescribed sheets without feedback")
    if jump_lifting_support not in ("global", "buffer"):
        raise ValueError("jump_lifting_support must be 'global' or 'buffer'")
    if not normal_derivative_jump and jump_lifting_support != "global":
        raise ValueError("Buffer-side lifting requires normal_derivative_jump=True")
    if not np.isfinite(internal_lifting_half_width) or internal_lifting_half_width < 0:
        raise ValueError("internal_lifting_half_width must be finite and nonnegative")
    if internal_lifting_half_width and (not normal_derivative_jump or
            internal_lifting_half_width >= min(radius_internal-radius_inner,radius_outer-radius_internal)):
        raise ValueError("Compact internal lifting requires jump extraction and a band strictly inside both mantle walls")
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
    continuation = [(exterior_boundary, mode/radius_exterior)]
    if radius_interior is not None:
        continuation.append((interior_boundary, mode/radius_interior))
    circular_sources = None
    lifting = None
    if source_integration == "circular":
        circular_sources = dict(radii=(radius_inner, radius_internal, radius_outer),
            sheets=((radius_inner, inner_sheet_density, inner_feedback_density),
                    (radius_internal, internal_sheet_density, 0.),
                    (radius_outer, outer_sheet_density, outer_feedback_density)),
            quadrature_order=source_quadrature)
        if normal_derivative_jump:
            sides = {radius_inner: -1, radius_internal: 0, radius_outer: 1} if jump_lifting_support == "buffer" else None
            widths = {radius_internal:internal_lifting_half_width} if internal_lifting_half_width else None
            lifting = NormalDerivativeJump(gravity_mesh, circular_sources["sheets"], gravitational_constant, sides, widths)
            circular_sources["potential_lifting"] = lifting
    phi, poisson = _solve_density_potential(gravity_mesh=gravity_mesh,
        field_name=field_name, potential_degree=potential_degree,
        volume_density=volume_density, sheets=(
            (inner_boundary, inner_sheet_density, inner_feedback_density),
            (internal_boundary, internal_sheet_density, 0),
            (outer_boundary, outer_sheet_density, outer_feedback_density)),
        continuation=continuation, gravitational_constant=gravitational_constant,
        tolerance=tolerance, solver_options=solver_options, circular_sources=circular_sources)
    outer, outer_mean = _interface_coefficient(phi, radius_outer, mode, boundary_samples, lifting)
    inner, inner_mean = _interface_coefficient(phi, radius_inner, mode, boundary_samples, lifting)
    return PhysicalPotentialSolve(phi, poisson, outer, inner, outer_mean, inner_mean, lifting)
