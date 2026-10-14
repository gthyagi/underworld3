"""Thieulot spherical shell: symbolic fields and cached numerical evaluation.

Reference: Thieulot, C. (2017), Analytical solution for viscous incompressible
Stokes flow in a spherical shell, Solid Earth 8, 1181-1191.
https://doi.org/10.5194/se-8-1181-2017
Exact wall velocity is prescribed. The Cartesian representation is finite at
the poles; shared stress/array evaluation machinery lives in analytic._thieulot_fields.
"""


__all__ = [
    "ThieulotSphericalStokes",
]

from numbers import Integral

import numpy as np
import sympy as sp

from .._thieulot_fields import _ThieulotStokesBase


class ThieulotSphericalStokes(_ThieulotStokesBase):
    """Thieulot (2017) shell with eta=mu_0*r**(m+1) and prescribed wall velocity.

    ``m=-1`` is constant viscosity; ``m=3`` is the common variable-viscosity
    case. ``m`` is NOT spherical-harmonic order. The poloidal and toroidal
    velocity components follow the published solution. Cartesian expressions
    avoid pole singularities. The degenerate m=-4 family is not implemented.
    """

    dim = 3
    geometry = "spherical"
    case = "thieulot"
    reference = "Thieulot (2017), Solid Earth 8, 1181–1191, doi:10.5194/se-8-1181-2017."

    def __init__(self, mesh, m=-1, r_inner=0.5, r_outer=1.0, gamma=1.0, mu_0=1.0):
        ri, ro = self._geometry(mesh, r_inner, r_outer)
        if isinstance(m, bool) or not isinstance(m, Integral):
            raise ValueError("m must be an integer.")
        self.m = int(m)
        if self.m == -4:
            raise NotImplementedError(
                "The degenerate m=-4 logarithmic family is not implemented."
            )
        if not np.all(np.isfinite((gamma, mu_0))) or mu_0 <= 0:
            raise ValueError("Expected finite gamma and positive mu_0.")
        self.gamma, self.mu_0 = float(gamma), float(mu_0)
        gamma, mu_0 = sp.Rational(str(gamma)), sp.Rational(str(mu_0))
        r = sp.Symbol("_thieulot_r", positive=True)
        m = self.m
        eta = mu_0 * r ** (m + 1)
        if m == -1:
            denominator = ro**3 * sp.log(ri) - ri**3 * sp.log(ro)
            alpha = -gamma * (ro**3 - ri**3) / denominator
            beta = 3 * gamma * (sp.log(ro) - sp.log(ri)) / denominator
            integral = alpha * sp.log(r) + beta * r**3 / 3 + gamma
        else:
            alpha = (
                gamma
                * (m + 1)
                * (ri**-3 - ro**-3)
                / (ri ** (-(m + 4)) - ro ** (-(m + 4)))
            )
            beta = (
                -3
                * gamma
                * (ri ** (m + 1) - ro ** (m + 1))
                / (ri ** (m + 4) - ro ** (m + 4))
            )
            integral = -alpha * r ** (-(m + 1)) / (m + 1) + beta * r**3 / 3 + gamma
        f = alpha * r ** (-(m + 3)) + beta * r
        g = -2 * integral / r**2
        h = (m + 3) * eta * g / r
        # Radial momentum balance in the common inward-gravity convention.
        # The Cartesian stress divergence is checked independently in tests.
        rho = (
            eta * (sp.diff(g, r, 2) + 2 * sp.diff(g, r) / r - 4 * (g + f) / r**2)
            + 2 * sp.diff(eta, r) * sp.diff(g, r)
            - sp.diff(h, r)
        )
        self._xyz = sp.symbols("_thieulot_x _thieulot_y _thieulot_z", real=True)
        x, y, z = self._xyz
        velocity = sp.Matrix(
            [
                (g + f) * z * x / r**2 - f * y / r,
                (g + f) * z * y / r**2 + f * x / r,
                (g + f) * z * z / r**2 - f,
            ]
        )
        radius = sp.sqrt(x * x + y * y + z * z)
        self._finish(
            velocity.subs(r, radius),
            (h * z / r).subs(r, radius),
            eta.subs(r, radius),
            (rho * z / r).subs(r, radius),
        )
