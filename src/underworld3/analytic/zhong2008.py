"""Compatibility imports; the maintained Zhong implementation is in ``zhong``.

Reference: Zhong, S., McNamara, A., Tan, E., Moresi, L. and Gurnis, M. (2008),
A benchmark study on mantle convection in a 3-D spherical shell using CitcomS,
Geochemistry, Geophysics, Geosystems 9, Q10017.
https://doi.org/10.1029/2008GC002048
"""

from .zhong import Zhong, ZhongResponse, Zhong2008, Zhong2008Response

__all__ = ["Zhong", "ZhongResponse", "Zhong2008", "Zhong2008Response"]
