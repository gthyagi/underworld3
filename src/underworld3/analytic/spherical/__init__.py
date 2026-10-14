"""Spherical analytical cases: native fields and the Zhong response reference.

Kramer, Davies and Wilson (2021), Analytical solutions for mantle flow in
cylindrical and spherical shells: https://doi.org/10.5194/gmd-14-1899-2021
Thieulot (2017), Analytical solution for viscous incompressible Stokes flow in
a spherical shell: https://doi.org/10.5194/se-8-1181-2017
Zhong et al. (2008), A benchmark study on mantle convection in a 3-D spherical
shell using CitcomS: https://doi.org/10.1029/2008GC002048
Zhong exposes response coefficients; Kramer/Thieulot expose symbolic fields.
"""

from . import kramer, thieulot, zhong
from .kramer import KramerSphericalStokes
from .thieulot import ThieulotSphericalStokes
from .zhong import Zhong, ZhongResponse

# Compatibility alias for the former unqualified Kramer name.
SphericalStokes = KramerSphericalStokes

__all__ = [
    "kramer",
    "thieulot",
    "zhong",
    "KramerSphericalStokes",
    "SphericalStokes",
    "ThieulotSphericalStokes",
    "Zhong",
    "ZhongResponse",
]
