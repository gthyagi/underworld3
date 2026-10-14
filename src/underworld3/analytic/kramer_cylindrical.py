"""Compatibility imports; maintained formulas/evaluators live in cylindrical.kramer.

Reference: Kramer, Davies and Wilson (2021), Analytical solutions for mantle
flow in cylindrical and spherical shells, GMD 14, 1899-1919.
https://doi.org/10.5194/gmd-14-1899-2021
"""

from .cylindrical.kramer import (
    AssessKramerCylindricalStokes,
    KramerCylindricalStokes,
    coefficients_cylinder_delta_fs,
    coefficients_cylinder_delta_ns,
    coefficients_cylinder_smooth_fs,
    coefficients_cylinder_smooth_ns,
    build_delta_solution,
    build_smooth_solution,
    analytical_velocity_cartesian_sympy,
    analytical_pressure_sympy,
    analytical_radial_deviatoric_stress_sympy,
    analytical_radial_normal_stress_sympy,
    analytical_radial_shear_stress_sympy,
    require_assess,
    assess_available,
    AssessCylindricalStokes,
    CylindricalStokes,
)

__all__ = [
    "AssessKramerCylindricalStokes",
    "KramerCylindricalStokes",
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
    "AssessCylindricalStokes",
    "CylindricalStokes",
]
