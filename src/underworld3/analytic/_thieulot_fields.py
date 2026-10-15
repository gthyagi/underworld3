"""Shared Thieulot stress assembly, prescribed forcing and cached field evaluation.

Geometry-specific radial formulas live in cylindrical/thieulot.py and
spherical/thieulot.py. Thieulot uses this helper for its two geometries. Kramer owns its branched
field/evaluation machinery in each geometry module.
Sources: ASPECT static annulus documentation (Thieulot, Puckett, Gassmoeller),
https://aspect-documentation.readthedocs.io/en/latest/user/benchmarks/benchmarks/annulus/doc/annulus.html
and Thieulot (2017), Analytical solution for viscous incompressible Stokes flow
in a spherical shell, Solid Earth 8, 1181-1191.
https://doi.org/10.5194/se-8-1181-2017
"""

import numpy as np
import sympy as sp

from ._base import AnalyticSolution, prescribed_velocity


class _ThieulotStokesBase(AnalyticSolution):
    boundaries = ("Upper", "Lower")
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
        r"""Set the smooth radial body force; no internal sheet is present.

        Parameters
        ----------
        solver : underworld3.systems.Stokes
            Solver on the reference mesh whose bodyforce is configured before solving.

        Returns
        -------
        None
            Mutates solver forcing; it does not run the solver.

        Notes
        -----
        Call separately from applying the prescribed wall velocities. Thieulot has no internal
        sheet; the smooth body force follows the manufactured density and unit inward gravity.
        """
        solver.bodyforce = self.fn_bodyforce

    def apply_boundary_conditions(self, solver):
        r"""Apply boundary conditions.

        Parameters
        ----------
        solver : underworld3.systems.Stokes
            Solver receiving the reference wall data on Upper and Lower before solving.

        Returns
        -------
        None
            Mutates wall conditions and enables the pressure nullspace.

        Notes
        -----
        Prescribe the exact manufactured velocity on both walls. This performs no solve.
        """
        prescribed_velocity(solver, self.boundaries, self.fn_velocity)
        solver.petsc_use_pressure_nullspace = True

    def compile_evaluators(self, *fields):
        r"""Compile evaluators.

        Parameters
        ----------
        *fields : str
            Names of fields to prepare; use the same names accepted by evaluate.
            An empty argument list compiles nothing.

        Returns
        -------
        None
            Cached evaluators are prepared without sampling points or solving a PDE.

        Raises
        ------
        ValueError
            A requested field is unknown.

        Notes
        -----
        The cache includes evaluation_cse; toggling it selects separately compiled
        common-subexpression-elimination evaluators.
        """
        for field in fields:
            if field not in self._expressions:
                raise ValueError(f"Unknown analytical field {field!r}.")
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
        r"""Evaluate the analytical reference field.

        Parameters
        ----------
        field : str or sympy expression
            Named field or symbolic expression. Native names include velocity,
            pressure, viscosity, bodyforce, strainrate, stress and radial_normal_stress;
            Thieulot also provides density.
        coords : array-like, shape (N, dim)
            Finite Cartesian coordinates in the reference scaling; the origin is
            excluded. Shell membership is the caller's responsibility.

        Returns
        -------
        numpy.ndarray
            One value per point: scalar (N,), vector (N, dim), or tensor
            (N, dim, dim). Arbitrary expressions use the UW3
            evaluator's expression shape. Empty arrays retain the corresponding shape.

        Raises
        ------
        ValueError
            Coordinates or the requested field are invalid.

        Notes
        -----
        Named fields use cached vectorized NumPy evaluators; symbolic expressions delegate to the
        shared JIT evaluator. This performs no Stokes solve. Cartesian spherical evaluators are
        finite at both poles.
        """
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
        r"""Sample points.

        Parameters
        ----------
        count : int, default=12
            Requested number of interior diagnostic points; use a positive count.

        Returns
        -------
        numpy.ndarray, shape (count, dim)
            Deterministic Cartesian points strictly between the two reference radii.

        Notes
        -----
        This samples the continuous reference independently of mesh nodes.
        """
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
