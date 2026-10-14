"""Compatibility imports for the geometry-specific Thieulot cases.

Annulus formulas: ASPECT documentation (Thieulot, Puckett, Gassmoeller):
https://aspect-documentation.readthedocs.io/en/latest/user/benchmarks/benchmarks/annulus/doc/annulus.html
Spherical fields: Thieulot (2017), Analytical solution for viscous incompressible
Stokes flow in a spherical shell. https://doi.org/10.5194/se-8-1181-2017
"""

from .cylindrical.thieulot import ThieulotCylindricalStokes
from .spherical.thieulot import ThieulotSphericalStokes

__all__ = ["ThieulotCylindricalStokes", "ThieulotSphericalStokes"]
