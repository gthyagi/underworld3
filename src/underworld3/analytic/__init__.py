r"""Exact solutions for validating Underworld3 solves.

Symbolic field solutions share :class:`AnalyticSolution`, which supplies
boundary conditions, evaluation and error measurement. A validation run reads::

    sol = uw.analytic.SolCx(mesh, eta_A=1.0, eta_B=1.0e6)

    stokes.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    stokes.bodyforce = sol.fn_bodyforce
    sol.apply_boundary_conditions(stokes)

    stokes.solve()
    rel = sol.error("velocity", stokes.u)

Numeric field oracles, such as the optional assess wrapper, provide their
documented field subset. Coefficient references such as Zhong instead return
boundary responses without mesh fields. :func:`available` lists field references
by default; use ``kind="responses"`` or ``kind="all"`` to discover coefficient
references too. :func:`describe` summarises either kind without constructing it.

These are worth reaching for whenever a solver change needs evidence. The SolCx
port alone caught a direct ``ksponly + lu`` solve silently mangling the singular
saddle on a free-slip problem — a wrong answer that no residual norm reported.

See Also
--------
underworld3.analytic._base : the field-validation base.
"""

from . import cylindrical, spherical
from ._base import (
    AnalyticSolution,
    free_slip,
    prescribed_scalar,
    prescribed_velocity,
)

from .barr_houseman_96 import FaultedMedium
from .inclusion import EllipticalInclusion
from .cylindrical import (
    AssessKramerCylindricalStokes,
    KramerCylindricalStokes,
    AssessCylindricalStokes,
    CylindricalStokes,
    ThieulotCylindricalStokes,
)
from .spherical import (
    KramerSphericalStokes,
    SphericalStokes,
    ThieulotSphericalStokes,
    ZhongSphericalStokes,
    ZhongSphericalStokesResponse,
)
from .richards import GardnerSteady, GardnerTransient
from .transport import AdvectedFront, ErfcDiffusion, Poisson1D, RotatingGaussian, TwoLayerDarcy
from .velic import (
    SolA,
    SolB,
    SolC,
    SolCx,
    SolDA,
    SolDB2d,
    SolDB3d,
    SolH,
    SolKx,
    SolKz,
    SolM,
    SolNL,
)
from .stokes import StokesSolution, stokes_cases, compare_coefficients

__all__ = [
    "cylindrical",
    "spherical",
    "AnalyticSolution",
    "free_slip",
    "prescribed_scalar",
    "prescribed_velocity",
    "AdvectedFront",
    "EllipticalInclusion",
    "ErfcDiffusion",
    "FaultedMedium",
    "KramerCylindricalStokes",
    "KramerSphericalStokes",
    "AssessKramerCylindricalStokes",
    "CylindricalStokes",
    "SphericalStokes",
    "StokesSolution",
    "stokes_cases",
    "compare_coefficients",
    "ThieulotCylindricalStokes",
    "ThieulotSphericalStokes",
    "AssessCylindricalStokes",
    "GardnerSteady",
    "GardnerTransient",
    "Poisson1D",
    "RotatingGaussian",
    "SolA",
    "SolB",
    "SolC",
    "SolCx",
    "SolDA",
    "SolDB2d",
    "SolDB3d",
    "SolH",
    "SolKx",
    "SolKz",
    "SolM",
    "SolNL",
    "TwoLayerDarcy",
    "ZhongSphericalStokes",
    "ZhongSphericalStokesResponse",
    "available",
    "describe",
    "is_available",
]

# The solutions this namespace offers. Explicit rather than introspected, so the
# listing stays truthful while solutions are still being migrated onto the
# contract, and so a solution needing an optional dependency can be listed
# without being importable.
_SOLUTIONS = {
    "AssessKramerCylindricalStokes": AssessKramerCylindricalStokes,
    "AdvectedFront": AdvectedFront,
    "KramerCylindricalStokes": KramerCylindricalStokes,
    "KramerSphericalStokes": KramerSphericalStokes,
    "ThieulotCylindricalStokes": ThieulotCylindricalStokes,
    "ThieulotSphericalStokes": ThieulotSphericalStokes,
    "EllipticalInclusion": EllipticalInclusion,
    "ErfcDiffusion": ErfcDiffusion,
    "FaultedMedium": FaultedMedium,
    "GardnerSteady": GardnerSteady,
    "GardnerTransient": GardnerTransient,
    "Poisson1D": Poisson1D,
    "RotatingGaussian": RotatingGaussian,
    "SolA": SolA,
    "SolB": SolB,
    "SolC": SolC,
    "SolCx": SolCx,
    "SolDA": SolDA,
    "SolDB2d": SolDB2d,
    "SolDB3d": SolDB3d,
    "SolH": SolH,
    "SolKx": SolKx,
    "SolKz": SolKz,
    "SolM": SolM,
    "SolNL": SolNL,
    "TwoLayerDarcy": TwoLayerDarcy,
}

# Coefficient references have no symbolic mesh-field contract. Keep them out
# of the default field conformance sweep, but include them in explicit discovery.
_RESPONSE_REFERENCES = {"ZhongSphericalStokes": ZhongSphericalStokes}
_REFERENCE_CLASSES = {**_SOLUTIONS, **_RESPONSE_REFERENCES}


# Resolve old lookup names without listing any concrete solution twice.
_SOLUTION_ALIASES = {
    "AssessCylindricalStokes": "AssessKramerCylindricalStokes",
    "CylindricalStokes": "KramerCylindricalStokes",
    "SphericalStokes": "KramerSphericalStokes",
}


def _reference_class(name):
    try:
        return _REFERENCE_CLASSES[_SOLUTION_ALIASES.get(name, name)]
    except KeyError:
        raise ValueError(
            f"no analytic solution named {name!r}; available: "
            f"{', '.join(available(kind='all'))}"
        ) from None


def is_available(name):
    """Whether *name* can actually be constructed here and now.

    False only when a solution needs an optional package that is not installed.
    Kept separate from :func:`available` because a missing optional dependency
    should not make a solution disappear — silently shortening the listing hides
    the very thing the user needs to be told.

    Parameters
    ----------
    name : str
        A field or coefficient-reference name, or a supported legacy alias.

    Returns
    -------
    bool
    """

    solution = _reference_class(name)
    requires = getattr(solution, "requires", None)

    if requires is None:
        return True

    import importlib.util

    return importlib.util.find_spec(requires) is not None


def available(installed_only=False, *, kind="fields"):
    """Canonical reference names, filtered by their contract and dependency.

    Parameters
    ----------
    installed_only : bool
        Omit references whose optional dependency is missing. Otherwise list
        every reference of the selected kind; use :func:`is_available`, or
        :func:`describe`, which says so in words.
    kind : {"fields", "responses", "all"}
        ``fields`` preserves the existing field-solution list, including the
        numeric assess oracle. ``responses`` lists mesh-free coefficient
        references. ``all`` includes both, without result dataclasses or aliases.

    Returns
    -------
    list of str
    """

    if kind == "fields":
        references = _SOLUTIONS
    elif kind == "responses":
        references = _RESPONSE_REFERENCES
    elif kind == "all":
        references = _REFERENCE_CLASSES
    else:
        raise ValueError("kind must be 'fields', 'responses', or 'all'.")
    names = sorted(references)

    if installed_only:
        return [name for name in names if is_available(name)]

    return names


def describe(name):
    """One-line summary of a solution, without constructing it.

    Parameters
    ----------
    name : str
        A field or coefficient-reference name, or a supported legacy alias.

    Returns
    -------
    str
        The first line of the solution's docstring.
    """

    solution = _reference_class(name)

    docstring = solution.__doc__ or ""
    summary = docstring.strip().split("\n")[0]

    if not is_available(name):
        return f"{summary}  [unavailable: needs the '{solution.requires}' package]"

    return summary
