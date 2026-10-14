"""Compatibility imports; the maintained propagator lives in spherical.zhong.

Reference: Zhong, McNamara, Tan, Moresi and Gurnis (2008), A benchmark study on
mantle convection in a 3-D spherical shell using CitcomS, G3 9, Q10017.
https://doi.org/10.1029/2008GC002048
Only the yearless Zhong/ZhongResponse names are exported.
"""

from .spherical.zhong import Zhong, ZhongResponse

__all__ = ["Zhong", "ZhongResponse"]
