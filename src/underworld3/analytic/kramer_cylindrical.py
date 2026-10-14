r"""Symbolic cylindrical Kramer (2021) fields and radial stress expressions.

The separated streamfunction is transcribed from the benchmark formulas and
validated independently against assess and the Stokes equations. No mesh,
solver or optional assess dependency is required by these building blocks.
"""

from types import SimpleNamespace

import sympy as sp


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
