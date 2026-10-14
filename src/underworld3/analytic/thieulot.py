"""Thieulot annulus and spherical fields, with cached array evaluation.

The annulus follows the ASPECT annulus benchmark documentation (Thieulot,
Puckett and Gassmoeller). The shell follows Thieulot (2017),
doi:10.5194/se-8-1181-2017. Both prescribe the exact wall velocity: tangential
velocity alone does not imply zero tangential traction. Density is defined by
div(sigma)+f=0 with f=-rho*e_r and unit inward gravity throughout.
"""

from numbers import Integral

import numpy as np
import sympy as sp

from ._base import AnalyticSolution, prescribed_velocity


class _ThieulotStokes(AnalyticSolution):
    boundaries = ("Upper", "Lower")
    case = "thieulot"
    evaluation_cse = True
    _fields = dict(
        AnalyticSolution._fields,
        density="fn_density",
        radial_normal_stress="fn_radial_normal_stress",
    )

    def _geometry(self, mesh, r_inner, r_outer):
        super().__init__(mesh)
        self.r_inner, self.r_outer = float(r_inner), float(r_outer)
        if (
            not np.all(np.isfinite((self.r_inner, self.r_outer)))
            or not 0 < self.r_inner < self.r_outer
        ):
            raise ValueError("Expected finite 0 < r_inner < r_outer.")
        return sp.Rational(str(self.r_inner)), sp.Rational(str(self.r_outer))

    def _finish(self, velocity, pressure, viscosity, density):
        X = sp.Matrix(self._xyz)
        radius = sp.sqrt(X.dot(X))
        strain = (velocity.jacobian(X) + velocity.jacobian(X).T) / 2
        stress = 2 * viscosity * strain - pressure * sp.eye(self.dim)
        force = -density * X / radius
        radial = (X.T * stress * X)[0] / radius**2
        self._expressions = dict(
            velocity=velocity,
            pressure=pressure,
            viscosity=viscosity,
            density=density,
            bodyforce=force,
            strainrate=strain,
            stress=stress,
            radial_normal_stress=radial,
        )
        substitution = dict(zip(self._xyz, self.mesh.X))
        spatial = lambda value: value.subs(substitution)
        self.set_fields(
            velocity=spatial(velocity),
            pressure=spatial(pressure),
            viscosity=spatial(viscosity),
            bodyforce=spatial(force),
            strainrate=spatial(strain),
        )
        self.fn_density = spatial(density)
        self.fn_radial_normal_stress = spatial(radial)
        self._evaluators = {}

    def apply_forcing(self, solver):
        """Set the smooth radial body force; no internal sheet is present."""
        solver.bodyforce = self.fn_bodyforce

    def apply_boundary_conditions(self, solver):
        prescribed_velocity(solver, self.boundaries, self.fn_velocity)
        solver.petsc_use_pressure_nullspace = True

    def compile_evaluators(self, *fields):
        for field in fields:
            if field not in self._expressions:
                raise ValueError(f"Unknown Thieulot field {field!r}.")
            key = (field, bool(self.evaluation_cse))
            if key not in self._evaluators:
                expression = self._expressions[field]
                terms = (
                    list(expression)
                    if isinstance(expression, sp.MatrixBase)
                    else [expression]
                )
                self._evaluators[key] = sp.lambdify(
                    self._xyz,
                    terms,
                    "numpy",
                    cse=bool(self.evaluation_cse),
                    docstring_limit=0,
                )

    def evaluate(self, field, coords):
        if not isinstance(field, str):
            return super().evaluate(field, coords)
        points = np.asarray(coords, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != self.dim
            or not np.all(np.isfinite(points))
        ):
            raise ValueError(f"coords must be a finite array of shape (N,{self.dim}).")
        if np.any(np.linalg.norm(points, axis=1) == 0):
            raise ValueError("The shell solution is undefined at the origin.")
        self.compile_evaluators(field)
        expression = self._expressions[field]
        shape = expression.shape if isinstance(expression, sp.MatrixBase) else ()
        if field in ("velocity", "bodyforce"):
            shape = (self.dim,)
        if len(points) == 0:
            return np.empty((0, *shape))
        values = self._evaluators[(field, bool(self.evaluation_cse))](*points.T)
        return np.column_stack(
            [np.broadcast_to(v, (len(points),)) for v in values]
        ).reshape((len(points), *shape))

    def sample_points(self, count=12):
        radii = (
            self.r_inner
            + (self.r_outer - self.r_inner) * (np.arange(count) + 0.37) / count
        )
        angle = np.arange(count) * 2.399963229728653
        if self.dim == 2:
            directions = np.column_stack((np.cos(angle), np.sin(angle)))
        else:
            z = 1 - 2 * (np.arange(count) + 0.5) / count
            directions = np.column_stack(
                (
                    np.sqrt(1 - z * z) * np.cos(angle),
                    np.sqrt(1 - z * z) * np.sin(angle),
                    z,
                )
            )
        return radii[:, None] * directions


class ThieulotCylindricalStokes(_ThieulotStokes):
    """Thieulot's isoviscous annulus, with prescribed tangential wall velocity.

    ``k`` is the nonnegative Fourier mode; ``amplitude`` is the integration
    constant C (default -1). ``rho_0`` adds hydrostatic background density.
    Pressure/density scale with ``viscosity`` for fixed manufactured velocity.
    """

    dim = 2
    geometry = "cylindrical"
    reference = "ASPECT annulus benchmark, Thieulot, Puckett and Gassmoeller."

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


class ThieulotSphericalStokes(_ThieulotStokes):
    """Thieulot (2017) shell with eta=mu_0*r**(m+1) and prescribed wall velocity.

    ``m=-1`` is constant viscosity; ``m=3`` is the common variable-viscosity
    case. ``m`` is NOT spherical-harmonic order. The poloidal and toroidal
    velocity components follow the published solution. Cartesian expressions
    avoid pole singularities. The degenerate m=-4 family is not implemented.
    """

    dim = 3
    geometry = "spherical"
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
