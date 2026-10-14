"""Compatibility imports for the Kramer analytical solutions.

The cylindrical coefficients, symbolic fields and cached numerical evaluators
are maintained together in :mod:`underworld3.analytic.kramer_cylindrical`.
This module preserves the existing class and optional-dependency import paths.
"""

from .kramer_cylindrical import (
    AssessCylindricalStokes,
    CylindricalStokes,
    assess_available,
    require_assess,
)

__all__ = ["AssessCylindricalStokes", "CylindricalStokes", "assess_available", "require_assess"]
