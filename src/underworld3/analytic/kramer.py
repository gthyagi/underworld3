"""Compatibility imports for the Kramer analytical solutions.

The cylindrical coefficients, symbolic fields and cached numerical evaluators
are maintained together in :mod:`underworld3.analytic.kramer_cylindrical`.
This module preserves the existing class and optional-dependency import paths.

Reference: Kramer, S. C., Davies, D. R. and Wilson, C. R. (2021), Analytical
solutions for mantle flow in cylindrical and spherical shells, Geoscientific
Model Development 14, 1899-1919. https://doi.org/10.5194/gmd-14-1899-2021
"""

from .kramer_cylindrical import (
    AssessCylindricalStokes,
    CylindricalStokes,
    assess_available,
    require_assess,
)

__all__ = ["AssessCylindricalStokes", "CylindricalStokes", "assess_available", "require_assess"]
