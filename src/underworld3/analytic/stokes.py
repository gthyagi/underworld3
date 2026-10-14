"""Geometry/case selection and scalar comparisons for shell Stokes benchmarks."""

import math

from .kramer_cylindrical import CylindricalStokes
from .kramer_spherical import SphericalStokes
from .thieulot import ThieulotCylindricalStokes, ThieulotSphericalStokes


_CASES = {
    "cylindrical": {"kramer": CylindricalStokes, "thieulot": ThieulotCylindricalStokes},
    "spherical": {"kramer": SphericalStokes, "thieulot": ThieulotSphericalStokes},
}


def stokes_cases(geometry):
    """Supported case names for ``cylindrical`` or ``spherical`` geometry."""
    if geometry not in _CASES:
        raise ValueError(f"geometry must be one of {tuple(_CASES)}; got {geometry!r}")
    return tuple(sorted(_CASES[geometry]))


def StokesSolution(mesh, *, geometry, case="kramer", **parameters):
    """Construct a shell solution through one geometry/case interface.

    The returned object is the original concrete solution, with SymPy ``fn_*``
    fields, cached ``evaluate`` and global ``error``/``error_report`` methods.
    Parameters remain case-specific: Kramer cylindrical uses ``n`` (mode) and
    ``k`` (density power); Thieulot cylindrical uses ``k`` (mode). Spherical
    Kramer uses ``ell, m`` (harmonic); spherical Thieulot uses ``m`` (viscosity
    exponent parameter, eta=mu_0*r**(m+1)). No parameters are silently translated.

    Existing ``CylindricalStokes`` and ``SphericalStokes`` remain Kramer classes.
    Zhong2008 is a mesh-free coefficient propagator with a different contract;
    use ``Zhong2008(...).response()`` for that reference.
    """
    names = stokes_cases(geometry)
    if case not in names:
        raise ValueError(f"unsupported {geometry} case {case!r}; choose from {names}")
    return _CASES[geometry][case](mesh, **parameters)


def compare_coefficients(numerical, analytical):
    """JSON-ready signed coefficient comparisons, keyed like ``analytical``.

    Relative errors divide by abs(reference). A zero reference has an absolute
    error and ``None`` relative/percentage errors. Missing/non-finite values are
    rejected. Compare matching harmonic coefficients, units and normalisations;
    an RMS magnitude is not a signed coefficient.
    """
    report = {}
    for name, reference in analytical.items():
        value, reference = float(numerical[name]), float(reference)
        if not math.isfinite(value) or not math.isfinite(reference):
            raise ValueError(f"coefficient {name!r} must have finite values")
        absolute = abs(value - reference)
        relative = absolute / abs(reference) if reference != 0 else None
        report[name] = dict(
            numerical=value,
            analytical=reference,
            absolute_error=absolute,
            relative_error=relative,
            percentage_error=None if relative is None else 100 * relative,
        )
    return report
