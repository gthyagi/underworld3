r"""Cylindrical Kramer (2021) Stokes solutions: symbolic fields and array evaluation.

The separated streamfunction is transcribed from the benchmark formulas and
validated independently against assess and the Stokes equations. Coefficients,
symbolic expressions and cached numerical evaluators have one implementation
here. Only the optional AssessKramerCylindricalStokes oracle requires assess.

Reference
---------
Kramer, S. C., Davies, D. R. and Wilson, C. R. (2021), Analytical solutions
for mantle flow in cylindrical and spherical shells, Geoscientific Model
Development 14, 1899-1919. https://doi.org/10.5194/gmd-14-1899-2021
"""


__all__ = [
    "coefficients_cylinder_delta_fs",
    "coefficients_cylinder_delta_ns",
    "coefficients_cylinder_smooth_fs",
    "coefficients_cylinder_smooth_ns",
    "build_delta_solution",
    "build_smooth_solution",
    "analytical_velocity_cartesian_sympy",
    "analytical_pressure_sympy",
    "analytical_radial_deviatoric_stress_sympy",
    "analytical_radial_normal_stress_sympy",
    "analytical_radial_shear_stress_sympy",
    "require_assess",
    "assess_available",
    "AssessKramerCylindricalStokes",
    "KramerCylindricalStokes",
]

from types import SimpleNamespace
from numbers import Integral

import numpy as np
import sympy as sp
import underworld3 as uw

from .._base import AnalyticSolution, free_slip, prescribed_velocity
from .._kramer_geoid import _KramerGeoidMixin


def coefficients_cylinder_delta_fs(Rp, Rm, rp, n, g, nu, sign):
    alpha_pm, alpha_mp = [Rp / rp, Rm / rp][:: int(sign)]
    pm = sign

    A = (
        -0.125
        * (alpha_mp ** (2 * n - 2) - 1)
        * g
        * pm
        * rp ** (-n + 2)
        / ((alpha_mp ** (2 * n - 2) - alpha_pm ** (2 * n - 2)) * (n - 1) * nu)
    )
    B = (
        -0.125
        * (alpha_mp ** (2 * n + 2) - 1)
        * alpha_pm ** (2 * n + 2)
        * g
        * pm
        * rp ** (n + 2)
        / ((alpha_mp ** (2 * n + 2) - alpha_pm ** (2 * n + 2)) * (n + 1) * nu)
    )
    C = (
        0.125
        * (alpha_mp ** (2 * n + 2) - 1)
        * g
        * pm
        / ((alpha_mp ** (2 * n + 2) - alpha_pm ** (2 * n + 2)) * (n + 1) * nu * rp**n)
    )
    D = (
        0.125
        * (alpha_mp ** (2 * n - 2) - 1)
        * alpha_pm ** (2 * n - 2)
        * g
        * pm
        * rp**n
        / ((alpha_mp ** (2 * n - 2) - alpha_pm ** (2 * n - 2)) * (n - 1) * nu)
    )
    return A, B, C, D


def coefficients_cylinder_delta_ns(Rp, Rm, rp, n, g, nu, sign):
    alpha_p, alpha_m = [Rp / rp, Rm / rp]
    pm, mp = sign, -sign

    A = (
        -0.125
        * (
            (
                (alpha_m**2 - alpha_p**2) * n
                - (n + 1) * pm
                + 1 / alpha_m ** (2 * n)
                - 1 / alpha_p ** (2 * n)
            )
            * (n - 1)
            + (alpha_m**2 / alpha_p ** (2 * n) - alpha_p**2 / alpha_m ** (2 * n)) * n
            + (n**2 * (alpha_m / alpha_p) ** (2 * mp) - (alpha_m / alpha_p) ** (2 * n * pm)) * pm
        )
        * g
        * rp ** (-n + 2)
        / (
            (
                n**2 * (alpha_m / alpha_p - alpha_p / alpha_m) ** 2
                - ((alpha_m / alpha_p) ** n - 1 / (alpha_m / alpha_p) ** n) ** 2
            )
            * (n - 1)
            * nu
        )
    )
    B = (
        -0.125
        * (
            ((alpha_m**2 - alpha_p**2) * n - (n - 1) * pm - alpha_m ** (2 * n) + alpha_p ** (2 * n))
            * (n + 1)
            - (alpha_m**2 * alpha_p ** (2 * n) - alpha_m ** (2 * n) * alpha_p**2) * n
            + (n**2 * (alpha_m / alpha_p) ** (2 * mp) - (alpha_m / alpha_p) ** (2 * mp * n)) * pm
        )
        * g
        * rp ** (n + 2)
        / (
            (
                n**2 * (alpha_m / alpha_p - alpha_p / alpha_m) ** 2
                - ((alpha_m / alpha_p) ** n - 1 / (alpha_m / alpha_p) ** n) ** 2
            )
            * (n + 1)
            * nu
        )
    )
    C = (
        -0.125
        * (
            (n**2 * (alpha_m / alpha_p) ** (2 * pm) - (alpha_m / alpha_p) ** (2 * n * pm)) * mp
            - (
                mp * (n - 1)
                + n * (1 / alpha_m**2 - 1 / alpha_p**2)
                - 1 / alpha_m ** (2 * n)
                + 1 / alpha_p ** (2 * n)
            )
            * (n + 1)
            - n * (1 / (alpha_m ** (2 * n) * alpha_p**2) - 1 / (alpha_m**2 * alpha_p ** (2 * n)))
        )
        * g
        / (
            (
                n**2 * (alpha_m / alpha_p - alpha_p / alpha_m) ** 2
                - ((alpha_m / alpha_p) ** n - 1 / (alpha_m / alpha_p) ** n) ** 2
            )
            * (n + 1)
            * nu
            * rp**n
        )
    )
    D = (
        -0.125
        * (
            (n**2 * (alpha_m / alpha_p) ** (2 * pm) - (alpha_m / alpha_p) ** (2 * mp * n)) * mp
            - (
                mp * (n + 1)
                + n * (1 / alpha_m**2 - 1 / alpha_p**2)
                + alpha_m ** (2 * n)
                - alpha_p ** (2 * n)
            )
            * (n - 1)
            + n * (alpha_m ** (2 * n) / alpha_p**2 - alpha_p ** (2 * n) / alpha_m**2)
        )
        * g
        * rp**n
        / (
            (
                n**2 * (alpha_m / alpha_p - alpha_p / alpha_m) ** 2
                - ((alpha_m / alpha_p) ** n - 1 / (alpha_m / alpha_p) ** n) ** 2
            )
            * (n - 1)
            * nu
        )
    )
    return A, B, C, D


def coefficients_cylinder_smooth_fs(Rp, Rm, k, n, g, nu):
    alpha = Rm / Rp
    A = (
        -0.25
        * (alpha**2 - alpha ** (k + n + 3))
        * Rp ** (-n + 3)
        * g
        / ((alpha + alpha**n) * (alpha**n - alpha) * (k + n + 1) * (k - n + 3) * nu)
    )
    B = (
        0.25
        * Rp ** (n + 3)
        * (alpha ** (k + n + 3) - alpha ** (2 * n + 2))
        * g
        / ((alpha ** (n + 1) + 1) * (alpha ** (n + 1) - 1) * (k + n + 3) * (k - n + 1) * nu)
    )
    C = (
        -0.25
        * Rp ** (-n + 1)
        * (alpha ** (k + n + 3) - 1)
        * g
        / ((alpha ** (n + 1) + 1) * (alpha ** (n + 1) - 1) * (k + n + 3) * (k - n + 1) * nu)
    )
    D = (
        -0.25
        * Rp ** (n + 1)
        * (alpha ** (k + n + 3) - alpha ** (2 * n))
        * g
        / ((alpha + alpha**n) * (alpha**n - alpha) * (k + n + 1) * (k - n + 3) * nu)
    )
    E = g * n / (((k + 3) ** 2 - n**2) * ((k + 1) ** 2 - n**2) * Rp**k * nu)
    return A, B, C, D, E


def coefficients_cylinder_smooth_ns(Rp, Rm, k, n, g, nu):
    alpha = Rm / Rp
    denom = (
        ((alpha ** (n + 1) - alpha ** (n - 1)) ** 2 * n**2 - (alpha ** (2 * n) - 1) ** 2)
        * ((k + 3) ** 2 - n**2)
        * ((k + 1) ** 2 - n**2)
        * nu
    )
    A = (
        0.5
        * (
            (alpha ** (k + n + 3) + alpha ** (2 * n)) * (k + n + 1) * (n + 1)
            - (alpha ** (k + n + 1) + alpha ** (2 * n + 2)) * (k + n + 3) * n
            - (alpha ** (k + 3 * n + 3) + 1) * (k - n + 1)
        )
        * Rp ** (-n + 3)
        * g
        * n
        / denom
    )
    B = (
        -0.5
        * (
            (alpha ** (k + 3 * n + 3) + alpha ** (2 * n)) * (k - n + 1) * (n - 1)
            - (alpha ** (k + 3 * n + 1) + alpha ** (2 * n + 2)) * (k - n + 3) * n
            + (alpha ** (k + n + 3) + alpha ** (4 * n)) * (k + n + 1)
        )
        * Rp ** (n + 3)
        * g
        * n
        / denom
    )
    C = (
        0.5
        * (
            (alpha ** (k + n + 1) + alpha ** (2 * n)) * (k + n + 3) * (n - 1)
            - (alpha ** (k + n + 3) + alpha ** (2 * n - 2)) * (k + n + 1) * n
            + (alpha ** (k + 3 * n + 1) + 1) * (k - n + 3)
        )
        * Rp ** (-n + 1)
        * g
        * n
        / denom
    )
    D = (
        -0.5
        * (
            (alpha ** (k + 3 * n + 1) + alpha ** (2 * n)) * (k - n + 3) * (n + 1)
            - (alpha ** (k + 3 * n + 3) + alpha ** (2 * n - 2)) * (k - n + 1) * n
            - (alpha ** (k + n + 1) + alpha ** (4 * n)) * (k + n + 3)
        )
        * Rp ** (n + 1)
        * g
        * n
        / denom
    )
    E = g * n / (((k + 3) ** 2 - n**2) * ((k + 1) ** 2 - n**2) * Rp**k * nu)
    return A, B, C, D, E


def build_delta_solution(Rp, Rm, rp, n, g, nu, sign, no_slip):
    if no_slip:
        ABCD = coefficients_cylinder_delta_ns(Rp, Rm, rp, n, g, nu, sign)
    else:
        ABCD = coefficients_cylinder_delta_fs(Rp, Rm, rp, n, g, nu, sign)
    _, _, C, D = ABCD
    return SimpleNamespace(
        n=n,
        g=g,
        nu=nu,
        ABCD=ABCD,
        G=-4 * nu * C * (n + 1),
        H=-4 * nu * D * (n - 1),
    )


def build_smooth_solution(Rp, Rm, k, n, g, nu, no_slip):
    if abs(k + 3) == n or abs(k + 1) == n:
        raise NotImplementedError(f"Smooth solution not implemented for k={k}, n={n}")
    if no_slip:
        ABCDE = coefficients_cylinder_smooth_ns(Rp, Rm, k, n, g, nu)
    else:
        ABCDE = coefficients_cylinder_smooth_fs(Rp, Rm, k, n, g, nu)
    _, _, C, D, _ = ABCDE
    F = -g * (k + 1) * Rp ** (-k) / ((k + 1) ** 2 - n**2)
    return SimpleNamespace(
        n=n,
        k=k,
        g=g,
        nu=nu,
        ABCDE=ABCDE,
        G=-4 * nu * C * (n + 1),
        H=-4 * nu * D * (n - 1),
        F=F,
    )


def analytical_velocity_cartesian_sympy(soln, r_sym, th_sym, rrotN):
    """Build a symbolic Cartesian velocity from analytical cylindrical coefficients."""
    n_sol = int(soln.n)

    if hasattr(soln, "ABCD"):
        A, B, C, D = soln.ABCD
        psi_r = (
            A * r_sym**n_sol
            + B * r_sym ** (-n_sol)
            + C * r_sym ** (n_sol + 2)
            + D * r_sym ** (-n_sol + 2)
        )
        dpsi_rdr = (
            A * n_sol * r_sym ** (n_sol - 1)
            + B * (-n_sol) * r_sym ** (-n_sol - 1)
            + C * (n_sol + 2) * r_sym ** (n_sol + 1)
            + D * (-n_sol + 2) * r_sym ** (-n_sol + 1)
        )
    elif hasattr(soln, "ABCDE"):
        A, B, C, D, E = soln.ABCDE
        k_sol = int(soln.k)
        psi_r = (
            A * r_sym**n_sol
            + B * r_sym ** (-n_sol)
            + C * r_sym ** (n_sol + 2)
            + D * r_sym ** (-n_sol + 2)
            + E * r_sym ** (k_sol + 3)
        )
        dpsi_rdr = (
            A * n_sol * r_sym ** (n_sol - 1)
            + B * (-n_sol) * r_sym ** (-n_sol - 1)
            + C * (n_sol + 2) * r_sym ** (n_sol + 1)
            + D * (-n_sol + 2) * r_sym ** (-n_sol + 1)
            + E * (k_sol + 3) * r_sym ** (k_sol + 2)
        )
    else:
        raise TypeError(f"Unsupported analytical solution type: {type(soln)}")

    u_r = -(n_sol * sp.cos(n_sol * th_sym) * psi_r) / r_sym
    u_theta = sp.sin(n_sol * th_sym) * dpsi_rdr
    return rrotN.T * sp.Matrix([u_r, u_theta])


def analytical_pressure_sympy(soln, r_sym, th_sym):
    """Build a symbolic pressure expression from analytical cylindrical coefficients."""
    n_sol = int(soln.n)
    p_expr = (soln.G * r_sym**n_sol + soln.H * r_sym ** (-n_sol)) * sp.cos(n_sol * th_sym)
    if hasattr(soln, "F"):
        p_expr += soln.F * r_sym ** (int(soln.k) + 1) * sp.cos(n_sol * th_sym)
    return p_expr


def analytical_radial_deviatoric_stress_sympy(soln, r_sym, th_sym):
    r"""Return the analytical cylindrical deviatoric stress :math:`\tau_{rr}`.

    The Kramer streamfunction has the separated form
    :math:`\psi(r,\theta)=\psi_r(r)\sin(n\theta)`. With the paper's velocity
    convention, the radial deviatoric stress is

    .. math::

       \tau_{rr} = 2\eta n\cos(n\theta)
       \left(\frac{\psi_r}{r^2}-\frac{1}{r}\frac{d\psi_r}{dr}\right).
    """

    n_sol = int(soln.n)

    if hasattr(soln, "ABCD"):
        A, B, C, D = soln.ABCD
        psi_r = (
            A * r_sym**n_sol
            + B * r_sym ** (-n_sol)
            + C * r_sym ** (n_sol + 2)
            + D * r_sym ** (-n_sol + 2)
        )
        dpsi_rdr = (
            A * n_sol * r_sym ** (n_sol - 1)
            + B * (-n_sol) * r_sym ** (-n_sol - 1)
            + C * (n_sol + 2) * r_sym ** (n_sol + 1)
            + D * (-n_sol + 2) * r_sym ** (-n_sol + 1)
        )
    elif hasattr(soln, "ABCDE"):
        A, B, C, D, E = soln.ABCDE
        k_sol = int(soln.k)
        psi_r = (
            A * r_sym**n_sol
            + B * r_sym ** (-n_sol)
            + C * r_sym ** (n_sol + 2)
            + D * r_sym ** (-n_sol + 2)
            + E * r_sym ** (k_sol + 3)
        )
        dpsi_rdr = (
            A * n_sol * r_sym ** (n_sol - 1)
            + B * (-n_sol) * r_sym ** (-n_sol - 1)
            + C * (n_sol + 2) * r_sym ** (n_sol + 1)
            + D * (-n_sol + 2) * r_sym ** (-n_sol + 1)
            + E * (k_sol + 3) * r_sym ** (k_sol + 2)
        )
    else:
        raise TypeError(f"Unsupported analytical solution type: {type(soln)}")

    angular_factor = n_sol * sp.cos(n_sol * th_sym)
    return 2 * soln.nu * angular_factor * (psi_r / r_sym**2 - dpsi_rdr / r_sym)


def analytical_radial_normal_stress_sympy(soln, r_sym, th_sym):
    r"""Return total radial normal stress :math:`\sigma_{rr}=\tau_{rr}-p`."""

    tau_rr = analytical_radial_deviatoric_stress_sympy(soln, r_sym, th_sym)
    pressure = analytical_pressure_sympy(soln, r_sym, th_sym)
    return tau_rr - pressure


def analytical_radial_shear_stress_sympy(soln, r_sym, th_sym):
    r"""Return the cylindrical shear stress :math:`\tau_{r\theta}`.

    This is primarily a boundary-condition diagnostic. For the Kramer
    streamfunction convention it is

    .. math::

       \tau_{r\theta} = \eta\sin(n\theta)\left(
       \psi_r'' - \frac{\psi_r'}{r} + \frac{n^2\psi_r}{r^2}\right).
    """

    n_sol = int(soln.n)
    if hasattr(soln, "ABCD"):
        coefficients = soln.ABCD
        powers = (n_sol, -n_sol, n_sol + 2, -n_sol + 2)
    elif hasattr(soln, "ABCDE"):
        coefficients = soln.ABCDE
        powers = (n_sol, -n_sol, n_sol + 2, -n_sol + 2, int(soln.k) + 3)
    else:
        raise TypeError(f"Unsupported analytical solution type: {type(soln)}")

    psi_r = sum(coefficient * r_sym**power for coefficient, power in zip(coefficients, powers))
    dpsi_rdr = sum(
        coefficient * power * r_sym ** (power - 1)
        for coefficient, power in zip(coefficients, powers)
    )
    d2psi_rdr2 = sum(
        coefficient * power * (power - 1) * r_sym ** (power - 2)
        for coefficient, power in zip(coefficients, powers)
    )
    return (
        soln.nu
        * sp.sin(n_sol * th_sym)
        * (d2psi_rdr2 - dpsi_rdr / r_sym + n_sol**2 * psi_r / r_sym**2)
    )


_INSTALL_MESSAGE = (
    "The Kramer benchmark solutions need the 'assess' package, which "
    "Underworld3 does not depend on.\n"
    '    pip install "underworld3[benchmarks]"\n'
    "or  pip install assess\n"
    "See Kramer, Davies & Wilson (2021), doi:10.5194/gmd-14-1899-2021."
)


def require_assess():
    """Import `assess`, or explain how to get it.

    Deferred to construction rather than to module import so that
    ``import underworld3`` works without it and
    :func:`underworld3.analytic.available` can still list the solution.

    Public because the curved-geometry benchmark scripts under
    ``docs/examples/`` call it directly: they use `assess` for cases this
    wrapper does not cover, and should still fail with an install message
    rather than a bare ``ModuleNotFoundError``.

    Returns
    -------
    module
        The imported `assess` module.

    Raises
    ------
    ImportError
        With installation instructions, if `assess` is not present.
    """

    try:
        import assess
    except ImportError as missing:
        raise ImportError(_INSTALL_MESSAGE) from missing

    return assess


def assess_available():
    """Whether the optional `assess` dependency can be imported."""

    try:
        require_assess()
    except ImportError:
        return False

    return True


class _KramerCylindricalBase(AnalyticSolution):
    """Shared annulus branch selection and wall policy for native/assess siblings."""

    dim = 2
    boundaries = ("Upper", "Lower")

    def _split(self, coords):
        """Indices of the points above and below the anomaly radius."""

        radius = np.linalg.norm(np.asarray(coords, dtype=float), axis=1)
        above = radius >= self.r_int
        return above, ~above

    def apply_boundary_conditions(self, solver):
        """Free-slip or zero-slip on both arcs, as the case declares.

        The two boundaries are curved, and the constraint uses the solver's own
        geometric normal rather than the analytic radial one. That is deliberate:
        the geometric normal is the direction the straight-facet boundary
        integral actually sees, so the constant pressure stays a null vector to
        machine precision, where ``X/|X|`` leaves a consistency error that grows
        with facet non-uniformity. Reach for ``normal=`` only when the constraint
        must follow the true circle rather than the mesh — see the "Which normal
        to use" section of ``docs/developer/subsystems/rotated-freeslip.md``.
        """

        # The pressure nullspace belongs to the DOMAIN, not to the wall type.
        # Both cases here are an enclosed annulus — velocity prescribed
        # everywhere on both arcs, or wall-normal flow zero on both — so in
        # either the pressure is determined only up to a constant and the
        # nullspace has to be removed. It used to be set after an early
        # `return` that the zero-slip branch took, so that case ran a singular
        # saddle and could return a quiet, wrong answer (#577).
        if self.boundary == "zero":
            prescribed_velocity(solver, self.boundaries, (0.0, 0.0))
        else:
            free_slip(solver, self.boundaries)

        solver.petsc_use_pressure_nullspace = True


class AssessKramerCylindricalStokes(_KramerCylindricalBase):
    r"""Stokes flow in a cylindrical annulus, after Kramer et al. (2021).

    Four cases, chosen by *density* and *boundary*:

    ==========  ===========  ====================================================
    ``density`` ``boundary`` anomaly
    ==========  ===========  ====================================================
    ``"delta"`` ``"free"``   a delta-function density shell at radius ``r_int``
    ``"delta"`` ``"zero"``   the same, with zero-slip walls
    ``"smooth"`` ``"free"``  :math:`r^k` radial dependence, wavenumber ``n``
    ``"smooth"`` ``"zero"``  the same, with zero-slip walls
    ==========  ===========  ====================================================

    The delta cases are solved as two solutions, above and below the anomaly, and
    :meth:`evaluate` selects between them by radius — that discontinuity is the
    point of the benchmark, and flattening it would remove what is being tested.

    Parameters
    ----------
    mesh : Mesh
        A 2D annulus mesh.
    n : int
        Azimuthal wavenumber.
    k : int
        Radial wavenumber. Used by the smooth cases only.
    r_inner, r_outer : float
        Shell radii.
    r_int : float
        Radius of the delta-function anomaly. Delta cases only.
    density : {"delta", "smooth"}
    boundary : {"free", "zero"}

    Notes
    -----
    Built on the `assess` calls the curved-geometry benchmark scripts in
    ``docs/examples/`` already make — ``CylindricalStokesSolution*(n, sign,
    Rp=, Rm=, rp=, nu=, g=)`` with ``.velocity_cartesian`` and
    ``.pressure_cartesian``. The wrapper's own tests cover the missing-dependency
    path; the working path is only as good as those signatures.
    """

    nonlinear = False
    symbolic = False
    solves = "stokes"
    requires = "assess"
    reference = (
        "Kramer, Davies & Wilson (2021), Geosci. Model Dev. 14, 1899-1919, "
        "doi:10.5194/gmd-14-1899-2021. Numeric, via the external 'assess' package."
    )

    def __init__(
        self,
        mesh,
        n=2,
        k=2,
        r_inner=0.5,
        r_outer=1.0,
        r_int=0.8,
        density="delta",
        boundary="free",
    ):
        super().__init__(mesh)

        if density not in ("delta", "smooth"):
            raise ValueError(f"density must be 'delta' or 'smooth'; got {density!r}")
        if boundary not in ("free", "zero"):
            raise ValueError(f"boundary must be 'free' or 'zero'; got {boundary!r}")

        assess = require_assess()

        self.n = int(n)
        self.k = int(k)
        self.r_inner = float(r_inner)
        self.r_outer = float(r_outer)
        self.r_int = float(r_int)
        self.density = density
        self.boundary = boundary

        slip = "FreeSlip" if boundary == "free" else "ZeroSlip"
        family = getattr(assess, f"CylindricalStokesSolution{density.capitalize()}{slip}")

        if density == "delta":
            shared = dict(Rp=self.r_outer, Rm=self.r_inner, rp=self.r_int, nu=1.0, g=-1.0)
            self._above = family(self.n, +1, **shared)
            self._below = family(self.n, -1, **shared)
        else:
            shared = dict(Rp=self.r_outer, Rm=self.r_inner, nu=1.0, g=1.0)
            self._above = family(self.n, self.k, **shared)
            self._below = self._above


    def evaluate(self, field, coords):
        """Exact *field* at *coords*, choosing the branch by radius.

        Parameters
        ----------
        field : {"velocity", "pressure"}
        coords : array, shape (n, 2)
            Cartesian coordinates.

        Returns
        -------
        ndarray
            Shape ``(n, 2)`` for velocity, ``(n,)`` for pressure.
        """

        if field not in ("velocity", "pressure"):
            raise ValueError(
                f"{type(self).__name__} provides 'velocity' and 'pressure'; " f"got {field!r}"
            )

        coords = np.asarray(coords, dtype=float)
        above, below = self._split(coords)

        width = 2 if field == "velocity" else 1
        values = np.zeros((len(coords), width))

        for mask, solution in ((above, self._above), (below, self._below)):
            if not mask.any():
                continue
            call = getattr(solution, f"{field}_cartesian")
            values[mask] = np.array([call(point) for point in coords[mask]]).reshape(-1, width)

        return values if width == 2 else values[:, 0]

    def error(self, field, meshvar, norm="l2"):
        """Relative error of *meshvar* against the exact *field*.

        The mesh-variable coordinates are used directly, so this is the discrete
        :math:`\\ell_2` norm over nodes rather than an integral — `assess` gives
        no symbolic form to integrate.
        """

        if norm != "l2":
            raise ValueError(
                f"{type(self).__name__} has no symbolic form, so only the "
                f"discrete 'l2' norm is available; got {norm!r}"
            )

        from underworld3 import mpi

        coords = meshvar.coords
        exact = np.atleast_2d(self.evaluate(field, coords).T).T
        computed = meshvar.data.reshape(exact.shape)

        local = np.array([((computed - exact) ** 2).sum(), (exact**2).sum()], dtype=float)
        difference, magnitude = mpi.comm.allreduce(local)

        return float(np.sqrt(difference / magnitude))


class KramerCylindricalStokes(_KramerGeoidMixin, _KramerCylindricalBase):
    r"""Native symbolic Kramer annulus solution with cached array evaluation.

    The four delta/smooth, free/zero-slip cases use the same parameter defaults
    as the former assess wrapper. Velocity, pressure and total stress are SymPy
    expressions on ``mesh.X`` and can be passed to the UW3 JIT. Named fields are
    evaluated by lazy, cached NumPy functions over whole coordinate arrays.
    ``assess`` is needed only by the separate :class:`AssessKramerCylindricalStokes`
    oracle, not by this solution.

    Parameters
    ----------
    mesh : Mesh
        A two-dimensional mesh. The solution domain is the specified annulus.
    n, k : int
        Fourier mode (n>=2) and radial power for smooth density.
    r_inner, r_outer, r_int : float
        Annulus and internal-sheet radii. The sheet must be strictly inside.
    density : {"delta", "smooth"}
        Delta sheet or volume density proportional to ``(r/r_outer)**k``.
    boundary : {"free", "zero"}
        Free-slip or zero-slip on both bounding arcs.
    viscosity : float
        Positive constant viscosity.
    forcing : float, optional
        Kramer coefficient g: default -1 for delta and +1 for smooth. The
        smooth bulk force is ``-g*(r/r_outer)**k*cos(n*theta)*e_r``.

    Notes
    -----
    Delta bulk forcing is zero. Its sheet is encoded by the jump of the two
    branches, not by differentiating a smoothed density. The outward normal
    stress jump is ``g*cos(n*theta)``; in ``div(sigma)+f=0`` the sheet force is
    its negative. At the sheet radius
    point evaluation selects the outer branch. Bulk residual checks must sample
    away from the interface; interface jump checks are separate.
    ``evaluation_cse`` controls common-subexpression elimination before a cached
    evaluator is first generated. It does not change the symbolic fields.
    """

    symbolic = True
    requires = None
    reference = "Kramer, Davies & Wilson (2021), GMD 14, 1899–1919, doi:10.5194/gmd-14-1899-2021."
    evaluation_cse = True
    _fields = dict(
        AnalyticSolution._fields,
        radial_normal_stress="fn_radial_normal_stress",
        radial_shear_stress="fn_radial_shear_stress",
    )

    geometry = "cylindrical"

    def __init__(
        self,
        mesh,
        n=2,
        k=2,
        r_inner=0.5,
        r_outer=1.0,
        r_int=0.8,
        density="delta",
        boundary="free",
        viscosity=1.0,
        forcing=None,
    ):
        super().__init__(mesh)
        if density not in ("delta", "smooth"):
            raise ValueError(f"density must be 'delta' or 'smooth'; got {density!r}")
        if boundary not in ("free", "zero"):
            raise ValueError(f"boundary must be 'free' or 'zero'; got {boundary!r}")
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 2:
            raise ValueError(
                "n must be an integer >=2; the exceptional n=0/1 Stokes families are not implemented."
            )
        if isinstance(k, bool) or not isinstance(k, Integral):
            raise TypeError("k must be an integer.")
        self.n, self.k = int(n), int(k)
        self.r_inner, self.r_outer, self.r_int = map(float, (r_inner, r_outer, r_int))
        self.viscosity = float(viscosity)
        self.g = float((-1.0 if density == "delta" else 1.0) if forcing is None else forcing)
        if not np.all(
            np.isfinite((self.r_inner, self.r_outer, self.r_int, self.viscosity, self.g))
        ):
            raise ValueError("Radii, viscosity and forcing must be finite.")
        if not 0 < self.r_inner < self.r_outer or self.viscosity <= 0:
            raise ValueError("Expected 0 < r_inner < r_outer and viscosity > 0.")
        if density == "delta" and not self.r_inner < self.r_int < self.r_outer:
            raise ValueError("r_int must lie strictly inside the annulus for a delta sheet.")
        self.density, self.boundary = density, boundary
        args = (self.r_outer, self.r_inner)
        if density == "delta":
            self._above = build_delta_solution(
                *args, self.r_int, self.n, self.g, self.viscosity, +1, boundary == "zero"
            )
            self._below = build_delta_solution(
                *args, self.r_int, self.n, self.g, self.viscosity, -1, boundary == "zero"
            )
        else:
            self._above = build_smooth_solution(
                *args, self.k, self.n, self.g, self.viscosity, boundary == "zero"
            )
            self._below = self._above
        self._r, self._theta = sp.symbols("_kramer_r _kramer_theta", positive=True, real=True)
        r, theta = self._r, self._theta
        rotation = sp.Matrix([[sp.cos(theta), -sp.sin(theta)], [sp.sin(theta), sp.cos(theta)]])
        self._polar_fields = []
        for branch in (self._below, self._above):
            velocity = analytical_velocity_cartesian_sympy(branch, r, theta, rotation.T)
            pressure = analytical_pressure_sympy(branch, r, theta)
            tau = analytical_radial_deviatoric_stress_sympy(branch, r, theta)
            shear = analytical_radial_shear_stress_sympy(branch, r, theta)
            stress = (
                rotation
                * sp.Matrix([[tau - pressure, shear], [shear, -tau - pressure]])
                * rotation.T
            )
            force = (
                sp.zeros(2, 1)
                if density == "delta"
                else -self.g
                * (r / self.r_outer) ** self.k
                * sp.cos(self.n * theta)
                * rotation[:, 0]
            )
            self._polar_fields.append(
                dict(
                    velocity=velocity,
                    pressure=pressure,
                    stress=stress,
                    strainrate=(stress + pressure * sp.eye(2)) / (2 * self.viscosity),
                    viscosity=sp.Float(self.viscosity),
                    bodyforce=force,
                    radial_normal_stress=tau - pressure,
                    radial_shear_stress=shear,
                )
            )
        x, y = mesh.X[0], mesh.X[1]
        radius = sp.sqrt(x * x + y * y)
        substitutions = {r: radius, theta: sp.atan2(y, x)}

        def spatial(field):
            expressions = [item[field] for item in self._polar_fields]
            if density == "smooth":
                return expressions[0].subs(substitutions)
            if isinstance(expressions[0], sp.MatrixBase):
                return sp.Matrix(
                    expressions[0].rows,
                    expressions[0].cols,
                    lambda i, j: sp.Piecewise(
                        (expressions[1][i, j], r >= self.r_int), (expressions[0][i, j], True)
                    ),
                ).subs(substitutions)
            return sp.Piecewise((expressions[1], r >= self.r_int), (expressions[0], True)).subs(
                substitutions
            )

        self.set_fields(
            velocity=list(spatial("velocity")),
            pressure=spatial("pressure"),
            viscosity=self.viscosity,
            bodyforce=list(spatial("bodyforce")),
            stress=spatial("stress"),
        )
        self.fn_radial_normal_stress = spatial("radial_normal_stress")
        self.fn_radial_shear_stress = spatial("radial_shear_stress")
        self.fn_interface_stress_jump = (
            self.g * sp.cos(self.n * sp.atan2(y, x)) if density == "delta" else sp.S.Zero
        )
        self._evaluators = {}

    def apply_forcing(self, solver, internal_boundary="Internal"):
        """Apply bulk forcing and, for delta density, its conforming weak load.

        Call once, separately from apply_boundary_conditions. A delta case
        requires the labelled conforming internal interface. Its natural datum
        is the stress jump g*cos(n*theta)*e_r; in div(sigma)+f=0 the physical
        sheet force is its negative. No Gaussian regularisation is introduced.
        """
        if solver.mesh is not self.mesh:
            raise ValueError("The solver and analytic solution must share the same mesh.")
        if self.density == "delta" and internal_boundary not in self.mesh.boundaries.__members__:
            raise ValueError(f"Delta forcing needs a conforming {internal_boundary!r} interface.")
        solver.bodyforce = self.fn_bodyforce
        if self.density == "delta":
            x, y = self.mesh.X[0], self.mesh.X[1]
            unit_r = sp.Matrix([x, y]) / sp.sqrt(x * x + y * y)
            solver.add_natural_bc(self.fn_interface_stress_jump * unit_r, internal_boundary)

    def evaluate(self, field, coords):
        """Evaluate a named field in batches; arbitrary expressions use UW3 JIT."""
        if not isinstance(field, str):
            return AnalyticSolution.evaluate(self, field, coords)
        if field not in self._polar_fields[0]:
            raise ValueError(f"Unknown field {field!r}; available: {sorted(self._polar_fields[0])}")
        points = np.asarray(coords, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("coords must be a finite array of shape (N, 2).")
        radius = np.linalg.norm(points, axis=1)
        if np.any(radius == 0):
            raise ValueError("The Kramer shell solution is undefined at the origin.")
        theta = np.arctan2(points[:, 1], points[:, 0])
        expression = self._polar_fields[0][field]
        shape = expression.shape if isinstance(expression, sp.MatrixBase) else ()
        if field in ("velocity", "bodyforce"):
            shape = (2,)
        result = np.empty((len(points), *shape))
        above = (
            radius >= self.r_int if self.density == "delta" else np.zeros(len(points), dtype=bool)
        )
        for branch, mask in ((0, ~above), (1, above)):
            if not np.any(mask):
                continue
            key = (field, branch, bool(self.evaluation_cse))
            if key not in self._evaluators:
                self._compile_evaluator(field, branch)
            values = self._evaluators[key](radius[mask], theta[mask])
            flat = np.column_stack(
                [np.broadcast_to(value, (np.count_nonzero(mask),)) for value in values]
            )
            result[mask] = flat.reshape((np.count_nonzero(mask), *shape))
        return result

    def _compile_evaluator(self, field, branch):
        key = (field, branch, bool(self.evaluation_cse))
        if key not in self._evaluators:
            expression = self._polar_fields[branch][field]
            terms = list(expression) if isinstance(expression, sp.MatrixBase) else [expression]
            self._evaluators[key] = sp.lambdify(
                (self._r, self._theta),
                terms,
                "numpy",
                cse=bool(self.evaluation_cse),
                docstring_limit=0,
            )

    def compile_evaluators(self, *fields):
        """Prepare named-field NumPy evaluators once, without evaluating points."""
        for field in fields:
            if field not in self._polar_fields[0]:
                raise ValueError(f"Unknown Kramer field {field!r}.")
            for branch in range(2 if self.density == "delta" else 1):
                self._compile_evaluator(field, branch)

    def error(self, field, meshvar, norm="l2"):
        """Use cached named-field nodal evaluation or the symbolic integral norm."""
        if norm != "l2" or not isinstance(field, str):
            return AnalyticSolution.error(self, field, meshvar, norm)
        computed = np.asarray(meshvar.array).reshape(len(meshvar.coords), -1)
        exact = self.evaluate(field, meshvar.coords).reshape(computed.shape)
        sums = np.array([np.sum((computed - exact) ** 2), np.sum(exact**2)])
        difference, magnitude = uw.mpi.comm.allreduce(sums)
        return float(np.sqrt(difference / magnitude))

    def sample_points(self, count=12):
        """Deterministic bulk points on both sides, excluding the sheet itself."""
        radii = self.r_inner + (self.r_outer - self.r_inner) * (np.arange(count) + 0.37) / count
        radii[np.isclose(radii, self.r_int)] += 1e-4 * (self.r_outer - self.r_inner)
        theta = np.arange(count) * 2.399963229728653
        return radii[:, None] * np.column_stack((np.cos(theta), np.sin(theta)))


# Historical names refer to the canonical classes; no second implementation.
AssessCylindricalStokes = AssessKramerCylindricalStokes
CylindricalStokes = KramerCylindricalStokes
