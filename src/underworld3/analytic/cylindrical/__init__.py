"""Cylindrical analytical cases, reached through StokesSolution or case modules.

Kramer, Davies and Wilson (2021), Analytical solutions for mantle flow in
cylindrical and spherical shells: https://doi.org/10.5194/gmd-14-1899-2021
Thieulot static annulus formulas, ASPECT documentation:
https://aspect-documentation.readthedocs.io/en/latest/user/benchmarks/benchmarks/annulus/doc/annulus.html
Each case module owns its symbolic construction and evaluation class.
"""

from . import kramer, thieulot
from .kramer import AssessCylindricalStokes, CylindricalStokes
from .thieulot import ThieulotCylindricalStokes

__all__ = [
    "kramer",
    "thieulot",
    "CylindricalStokes",
    "AssessCylindricalStokes",
    "ThieulotCylindricalStokes",
]
