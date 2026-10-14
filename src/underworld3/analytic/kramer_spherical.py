"""Compatibility imports; maintained formulas/evaluators live in spherical.kramer.

Reference: Kramer, Davies and Wilson (2021), Analytical solutions for mantle
flow in cylindrical and spherical shells, GMD 14, 1899-1919.
https://doi.org/10.5194/gmd-14-1899-2021
"""

from .spherical.kramer import (
    coefficients_sphere_delta_fs,
    coefficients_sphere_delta_ns,
    coefficients_sphere_smooth_fs,
    coefficients_sphere_smooth_ns,
    build_delta_solution,
    build_smooth_solution,
    analytical_velocity_cartesian_sympy,
    analytical_pressure_sympy,
    analytical_Pl_sympy,
    analytical_radial_normal_stress_sympy,
    SphericalStokes,
    analytical_radial_stress_assess_sympy,
)

__all__ = [
    "coefficients_sphere_delta_fs",
    "coefficients_sphere_delta_ns",
    "coefficients_sphere_smooth_fs",
    "coefficients_sphere_smooth_ns",
    "build_delta_solution",
    "build_smooth_solution",
    "analytical_velocity_cartesian_sympy",
    "analytical_pressure_sympy",
    "analytical_Pl_sympy",
    "analytical_radial_normal_stress_sympy",
    "SphericalStokes",
    "analytical_radial_stress_assess_sympy",
]
