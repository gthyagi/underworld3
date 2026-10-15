r"""Spherical trace projection and compatibility sheet-load entry points.

Finite-element interface assembly is owned by utilities. Solver configuration
is exposed through Stokes and Poisson methods; trace projection remains a
postprocessing operation. Existing free-function names delegate without
maintaining a second assembly implementation.
"""
from __future__ import annotations
import numpy as np
from petsc4py import PETSc
from underworld3.maths.spherical_harmonics import AXISYMMETRIC_UNNORMALIZED
from underworld3.utilities._spherical_interface_load import (
    InterfaceLoadDiagnostics,
    _assemble_spherical_harmonic_sheet_load,
    _project_spherical_harmonic_trace_coefficient,
)

__all__ = [
    "InterfaceLoadDiagnostics", "assemble_axisymmetric_sheet_load",
    "assemble_spherical_harmonic_sheet_load", "project_axisymmetric_trace_coefficient",
    "project_spherical_harmonic_trace_coefficient",
    "install_linear_spherical_stokes_sheet_load",
]

def assemble_spherical_harmonic_sheet_load(
    *,
    mesh,
    load_variable,
    interface: str,
    harmonic_degree: int,
    harmonic_order: int = 0,
    harmonic_component: str = "cos",
    harmonic_normalization: str = AXISYMMETRIC_UNNORMALIZED,
    density_coefficient: float,
    gravitational_constant: float = 1.0 / (4.0 * np.pi),
    quadrature_points: int = 8,
) -> tuple[PETSc.Vec, InterfaceLoadDiagnostics]:
    r"""Assemble a scalar spherical sheet source on uniquely owned P2 facets.

    Parameters
    ----------
    mesh : underworld3.discretisation.Mesh
        Three-dimensional simplicial mesh with straight triangular interface
        facets. Assembly is collective on its communicator.
    load_variable : underworld3.discretisation.MeshVariable
        Continuous scalar P2 variable on ``mesh``. Its local/global vectors are
        overwritten with weak-load entries, not pointwise density samples.
    interface : str
        Member name in ``mesh.boundaries`` identifying the sheet facets.
    harmonic_degree : int
        Spherical-harmonic degree, at least 1.
    harmonic_order : int, default=0
        Order satisfying ``0 <= harmonic_order <= harmonic_degree``.
    harmonic_component : {"cos", "sin"}, default="cos"
        Real angular component. Order zero requires ``"cos"``.
    harmonic_normalization : {"axisymmetric_unnormalized", "real_orthonormal"}, default="axisymmetric_unnormalized"
        Angular normalization used by both sources and projections. The default
        is ``P_l(cos(theta))`` and is restricted to order zero. Use
        ``"real_orthonormal"`` for non-axisymmetric modes.
    density_coefficient : float
        Finite signed surface-density coefficient multiplying the chosen harmonic.
        In physical units this is mass per area; use consistent mesh/G units.
    gravitational_constant : float, default=1/(4*pi)
        Finite source scaling constant. The weak source is ``4*pi*G*sigma*Y``;
        the default makes ``4*pi*G=1``. No unit conversion is performed.
    quadrature_points : int, default=8
        At least two Gauss points per transformed triangle coordinate; each facet
        uses ``quadrature_points**2`` evaluations.

    Returns
    -------
    vector : petsc4py.PETSc.Vec
        Independent copy of the global scalar weak-load vector in the variable's
        sub-DM layout. The caller owns it and must call ``destroy()`` when finished.
    diagnostics : InterfaceLoadDiagnostics
        Global source, ownership, norm and mean-correction diagnostics.

    Raises
    ------
    TypeError
        Harmonic degree or order has an invalid type.
    ValueError
        Geometry, harmonic, field, interface, source or quadrature inputs are invalid.
    RuntimeError
        P2 facet nodes cannot be mapped or the global interface area is nonpositive.
    AssertionError
        Copies of a shared nodal load disagree beyond the diagnostic tolerance.

    Notes
    -----
    The sign is for ``-laplacian(Phi)=4*pi*G*rho``. The discrete uniform mean
    of the assembled source is removed, including quadrature/geometry leakage
    of a nonzero-degree harmonic. This is not a degree-zero mass-sheet API.
    Owned facets contribute once; empty local interface strata are supported."""

    return _assemble_spherical_harmonic_sheet_load(
        mesh=mesh,
        load_variable=load_variable,
        interface=interface,
        harmonic_degree=harmonic_degree,
        harmonic_order=harmonic_order,
        harmonic_component=harmonic_component,
        harmonic_normalization=harmonic_normalization,
        density_coefficient=density_coefficient,
        gravitational_constant=gravitational_constant,
        quadrature_points=quadrature_points,
    )


def project_spherical_harmonic_trace_coefficient(
    *,
    mesh,
    variable,
    interface: str,
    interface_radius: float,
    harmonic_degree: int,
    harmonic_order: int = 0,
    harmonic_component: str = "cos",
    harmonic_normalization: str = AXISYMMETRIC_UNNORMALIZED,
    harmonic_norm_value: float | None = None,
    quadrature_points: int = 8,
) -> float:
    r"""Project a continuous scalar P2 trace using uniquely owned interface facets.

    Parameters
    ----------
    mesh : underworld3.discretisation.Mesh
        Three-dimensional simplicial mesh with straight triangular interface facets.
    variable : underworld3.discretisation.MeshVariable
        Continuous scalar P2 field on ``mesh`` whose trace is projected.
    interface : str
        Member name in ``mesh.boundaries`` identifying the integration facets.
    interface_radius : float
        Positive reference radius, in mesh coordinate units. Its square converts
        the supplied unit-sphere angular norm to a spherical surface norm.
    harmonic_degree : int
        Spherical-harmonic degree, at least 1.
    harmonic_order : int, default=0
        Order satisfying ``0 <= harmonic_order <= harmonic_degree``.
    harmonic_component : {"cos", "sin"}, default="cos"
        Real angular component. Order zero requires ``"cos"``.
    harmonic_normalization : {"axisymmetric_unnormalized", "real_orthonormal"}, default="axisymmetric_unnormalized"
        Angular normalization used by both sources and projections. The default
        is ``P_l(cos(theta))`` and is restricted to order zero. Use
        ``"real_orthonormal"`` for non-axisymmetric modes.
    harmonic_norm_value : float or None, default=None
        Positive unit-sphere integral of ``Y**2``. ``None`` obtains it from the
        selected harmonic normalization; an explicit value overrides that norm.
    quadrature_points : int, default=8
        Number of Gauss points in each transformed triangle coordinate.

    Returns
    -------
    float
        ``integral(variable*Y dS)/(interface_radius**2*harmonic_norm_value)``,
        reduced on the mesh communicator and returned on every rank. Its units
        match the input scalar field.

    Raises
    ------
    TypeError
        Harmonic degree or order has an invalid type.
    ValueError
        Field, geometry, harmonic, radius, norm or interface inputs are invalid.
    RuntimeError
        A facet's P2 trace nodes cannot be mapped to the field.

    Notes
    -----
    Call collectively on the mesh communicator. Empty local interface strata
    are allowed. Integration uses discrete facets, while normalization uses the
    specified spherical radius and angular norm; the field mean is not removed."""

    return _project_spherical_harmonic_trace_coefficient(
        mesh=mesh,
        variable=variable,
        interface=interface,
        interface_radius=interface_radius,
        harmonic_degree=harmonic_degree,
        harmonic_order=harmonic_order,
        harmonic_component=harmonic_component,
        harmonic_normalization=harmonic_normalization,
        harmonic_norm_value=harmonic_norm_value,
        quadrature_points=quadrature_points,
    )


def assemble_axisymmetric_sheet_load(**kwargs):
    r"""Assemble a sheet using the unnormalized zonal basis ``P_l(cos(theta))``.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments for :func:`assemble_spherical_harmonic_sheet_load`:
        ``mesh``, ``load_variable``, ``interface``, ``harmonic_degree`` and
        ``density_coefficient`` are required. ``gravitational_constant`` defaults
        to ``1/(4*pi)`` and ``quadrature_points`` to 8. Their types, units and
        restrictions are described by that function. Do not supply
        ``harmonic_order``, ``harmonic_component`` or ``harmonic_normalization``;
        this wrapper fixes them to 0, ``"cos"`` and ``"axisymmetric_unnormalized"``.

    Returns
    -------
    vector : petsc4py.PETSc.Vec
        Caller-owned weak-load vector; destroy it when finished.
    diagnostics : InterfaceLoadDiagnostics
        Source and mean-correction diagnostics.

    Raises
    ------
    TypeError
        Required keywords are missing, fixed harmonic keywords are duplicated,
        or degree has an invalid type.
    ValueError, RuntimeError, AssertionError
        See :func:`assemble_spherical_harmonic_sheet_load` for input/assembly errors.

    Notes
    -----
    This collective compatibility wrapper performs the same discrete mean removal
    and owned-facet assembly as the general real-harmonic routine."""

    return assemble_spherical_harmonic_sheet_load(
        **kwargs,
        harmonic_order=0,
        harmonic_component="cos",
        harmonic_normalization=AXISYMMETRIC_UNNORMALIZED,
    )


def project_axisymmetric_trace_coefficient(*, harmonic_norm: float, **kwargs):
    r"""Project a P2 trace onto the unnormalized zonal spherical harmonic.

    Parameters
    ----------
    harmonic_norm : float
        Positive unit-sphere angular norm, normally ``4*pi/(2*l+1)``.
    **kwargs : dict
        Keyword arguments for :func:`project_spherical_harmonic_trace_coefficient`:
        ``mesh``, ``variable``, ``interface``, ``interface_radius`` and
        ``harmonic_degree`` are required; ``quadrature_points`` defaults to 8.
        See that function for types and units. Do not supply its
        ``harmonic_order``, ``harmonic_component``, ``harmonic_normalization`` or
        ``harmonic_norm_value`` arguments, which this wrapper fixes explicitly.

    Returns
    -------
    float
        Collective modal coefficient with the same units as the scalar field.

    Raises
    ------
    TypeError
        Required keywords are missing, fixed keywords are duplicated, or degree
        has an invalid type.
    ValueError, RuntimeError
        See :func:`project_spherical_harmonic_trace_coefficient` for input/trace errors."""

    return project_spherical_harmonic_trace_coefficient(
        **kwargs,
        harmonic_order=0,
        harmonic_component="cos",
        harmonic_normalization=AXISYMMETRIC_UNNORMALIZED,
        harmonic_norm_value=harmonic_norm,
    )


def install_linear_spherical_stokes_sheet_load(
    *, stokes, interface, harmonic_degree, density_coefficient,
    harmonic_order=0, harmonic_component='cos',
    harmonic_normalization=AXISYMMETRIC_UNNORMALIZED, quadrature_points=8,
    modes=None,
):
    r"""Install a radial-sheet residual in a linear continuous-P2 Stokes system.

    Parameters
    ----------
    stokes : underworld3.systems.Stokes
        Configured three-dimensional Stokes system with continuous P2 velocity,
        ``ViscousFlowModel``, zero bulk body force and solution-independent
        viscosity. Fixed spatial/layered viscosity is allowed. Configure walls,
        viscosity and solver policy before calling; no natural BCs and only
        homogeneous essential data are accepted. The mesh must use a communicator
        congruent to UW3's world communicator and have a nonempty chart on every rank.
    interface : str
        Member name in ``stokes.mesh.boundaries`` for the straight triangular facets.
    harmonic_degree : int
        Spherical-harmonic degree, at least 1.
    harmonic_order : int, default=0
        Order satisfying ``0 <= harmonic_order <= harmonic_degree``.
    harmonic_component : {"cos", "sin"}, default="cos"
        Real angular component. Order zero requires ``"cos"``.
    harmonic_normalization : {"axisymmetric_unnormalized", "real_orthonormal"}, default="axisymmetric_unnormalized"
        Angular normalization used by both sources and projections. The default
        is ``P_l(cos(theta))`` and is restricted to order zero. Use
        ``"real_orthonormal"`` for non-axisymmetric modes.
    density_coefficient : float
        Finite signed amplitude of the radial sheet term in Stokes residual units.
        No ``4*pi*G`` factor is applied. It is added to the residual as
        ``density_coefficient*Y*e_r``; see the residual sign convention below.
    quadrature_points : int, default=8
        At least two Gauss points per transformed triangle coordinate.
    modes : sequence of dict or None, default=None
        Optional nonempty multi-mode specification replacing the single mode.
        Each mapping requires ``harmonic_degree`` and ``density_coefficient``;
        ``harmonic_order``, ``harmonic_component`` and ``harmonic_normalization``
        default to 0, ``"cos"`` and the function's normalization respectively.
        Different degrees may share one Stokes operator. The single-mode arguments
        remain required by the signature but are ignored when ``modes`` is supplied.

    Returns
    -------
    petsc4py.PETSc.Vec
        Coupled global vector ``b_sheet`` with sheet entries in the velocity block
        and zeros in the pressure block. The Stokes callback owns this vector:
        callers may inspect it but must not mutate or destroy it. Reinstallation,
        reset or a DM rebuild releases the previous vector.

    Raises
    ------
    TypeError
        Harmonic degree or order has an invalid type, or a mode has malformed types.
    KeyError
        A mode specification lacks a required entry.
    ValueError
        Mesh, field, modes, interface, quadrature, viscosity, body force or boundary
        data do not satisfy the restricted linear-adapter contract.

    Notes
    -----
    Compatibility entry point; prefer
    ``stokes.install_linear_spherical_sheet_load(...)``.
    Call collectively, instead of adding the same internal sheet as a natural BC.
    The native Jacobian is retained and the callback evaluates
    ``F(x)=J*x+b_sheet``; the algebraic equation is therefore ``J*x=-b_sheet``.
    The vector is a residual load, not a separately sign-reversed force RHS.
    Consistent facet coordinates and radial directions give zero work against
    rigid rotations. Empty local interface strata are allowed, empty mesh charts
    and proper subcommunicators are not. Reinstall after model/mesh setup changes."""
    return stokes.install_linear_spherical_sheet_load(
        interface=interface,
        harmonic_degree=harmonic_degree,
        density_coefficient=density_coefficient,
        harmonic_order=harmonic_order,
        harmonic_component=harmonic_component,
        harmonic_normalization=harmonic_normalization,
        quadrature_points=quadrature_points,
        modes=modes,
    )
