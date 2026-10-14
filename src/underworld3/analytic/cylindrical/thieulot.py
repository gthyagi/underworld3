"""Thieulot's static annulus: symbolic fields and cached numerical evaluation.

Formula source: ASPECT documentation, The 2D annulus benchmark, contributed by
C. Thieulot, E. G. Puckett and R. Gassmoeller:
https://aspect-documentation.readthedocs.io/en/latest/user/benchmarks/benchmarks/annulus/doc/annulus.html
This is the documented static manufactured solution; no separate paper is
claimed. Exact wall velocity is prescribed. Shared Cartesian stress/array
evaluation machinery lives in the private analytic._thieulot_fields module.
"""

from numbers import Integral

import numpy as np
import sympy as sp

from .._thieulot_fields import _ThieulotStokesBase


class ThieulotCylindricalStokes(_ThieulotStokesBase):
    """Thieulot's isoviscous annulus, with prescribed tangential wall velocity.

    ``k`` is the nonnegative Fourier mode; ``amplitude`` is the integration
    constant C (default -1). ``rho_0`` adds hydrostatic background density.
    Pressure/density scale with ``viscosity`` for fixed manufactured velocity.
    """

    dim = 2
    geometry = "cylindrical"
    case = "thieulot"
    reference = (
        "ASPECT annulus benchmark, Thieulot, Puckett and Gassmoeller. "
        "https://aspect-documentation.readthedocs.io/en/latest/user/benchmarks/benchmarks/annulus/doc/annulus.html"
    )

    def __init__(
        self,
        mesh,
        k=2,
        r_inner=1.0,
        r_outer=2.0,
        amplitude=-1.0,
        viscosity=1.0,
        rho_0=0.0,
    ):
        ri, ro = self._geometry(mesh, r_inner, r_outer)
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 0:
            raise ValueError("k must be an integer >=0.")
        self.k = int(k)
        if not np.all(np.isfinite((amplitude, viscosity, rho_0))) or viscosity <= 0:
            raise ValueError("Expected finite amplitude/rho_0 and positive viscosity.")
        self.amplitude, self.viscosity, self.rho_0 = map(
            float, (amplitude, viscosity, rho_0)
        )
        C, eta, background = map(
            lambda v: sp.Rational(str(v)), (amplitude, viscosity, rho_0)
        )
        r, theta = sp.symbols("_thieulot_r _thieulot_theta", positive=True)
        denominator = ro**2 * sp.log(ri) - ri**2 * sp.log(ro)
        A = -2 * C * (sp.log(ri) - sp.log(ro)) / denominator
        B = -C * (ro**2 - ri**2) / denominator
        f = A * r + B / r
        g = A * r / 2 + B * sp.log(r) / r + C / r
        h = (2 * g - f) / r
        radial_density = (
            sp.diff(g, r, 2)
            - sp.diff(g, r) / r
            - g * (self.k**2 - 1) / r**2
            + f / r**2
            + sp.diff(f, r) / r
        )
        vr, vt = self.k * g * sp.sin(self.k * theta), f * sp.cos(self.k * theta)
        velocity = sp.Matrix(
            [
                vr * sp.cos(theta) - vt * sp.sin(theta),
                vr * sp.sin(theta) + vt * sp.cos(theta),
            ]
        )
        pressure = eta * self.k * h * sp.sin(self.k * theta) + background * (ro - r)
        density = eta * self.k * radial_density * sp.sin(self.k * theta) + background
        self._xyz = sp.symbols("_thieulot_x _thieulot_y", real=True)
        x, y = self._xyz
        substitution = {r: sp.sqrt(x * x + y * y), theta: sp.atan2(y, x)}
        self._finish(
            velocity.subs(substitution),
            pressure.subs(substitution),
            eta,
            density.subs(substitution),
        )
