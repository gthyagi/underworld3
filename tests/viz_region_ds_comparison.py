# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Region DS Verification: Rock-Only vs Air Layer Comparison

Two Stokes solutions compared:
1. **Reference**: Rock-only annulus (r=0.5 to r=1.0), free-slip both boundaries
2. **Air layer**: Full annulus (r=0.5 to r=1.5) with low-viscosity air (eta=1e-3),
   radial velocity penalty on internal boundary

The air layer solve demonstrates the bilateral penalty problem.
"""

# %%
import underworld3 as uw
from underworld3.systems import Stokes
import underworld3.visualisation as vis
import numpy as np
import sympy

if uw.mpi.size == 1:
    import pyvista as pv
    import matplotlib.pyplot as plt

# %% [markdown]
"""
## Parameters
"""

# %%
r_inner = 0.5
r_internal = 1.0
r_outer_full = 1.5
cellsize = 1/16
n = 2
k = 1
eta_air = 1.0e-3

# %% [markdown]
"""
## 1. Rock-only reference solve
"""

# %%
mesh_ref = uw.meshing.Annulus(
    radiusOuter=r_internal, radiusInner=r_inner, cellSize=cellsize,
)

v_ref = uw.discretisation.MeshVariable("V_ref", mesh_ref, mesh_ref.dim, degree=2)
p_ref = uw.discretisation.MeshVariable("P_ref", mesh_ref, 1, degree=1, continuous=True)

unit_rvec_ref = mesh_ref.CoordinateSystem.unit_e_0
r_ref, th_ref = mesh_ref.CoordinateSystem.xR
Gamma_ref = mesh_ref.Gamma
v_theta_ref = r_ref * mesh_ref.CoordinateSystem.rRotN.T * sympy.Matrix((0, 1))

stokes_ref = Stokes(mesh_ref, velocityField=v_ref, pressureField=p_ref)
stokes_ref.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes_ref.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes_ref.saddle_preconditioner = 1.0

rho_ref = ((r_ref / r_internal) ** k) * sympy.cos(n * th_ref)
stokes_ref.bodyforce = rho_ref * (-1.0 * unit_rvec_ref)

stokes_ref.add_natural_bc(1e6 * Gamma_ref.dot(v_ref.sym) * Gamma_ref, "Upper")
stokes_ref.add_natural_bc(1e6 * Gamma_ref.dot(v_ref.sym) * Gamma_ref, "Lower")

stokes_ref.tolerance = 1e-6
stokes_ref.petsc_options["snes_type"] = "newtonls"
stokes_ref.petsc_options["ksp_type"] = "fgmres"
stokes_ref.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes_ref.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

stokes_ref.solve(verbose=False)

# Null space removal
I0 = uw.maths.Integral(mesh_ref, v_theta_ref.dot(v_ref.sym))
norm = I0.evaluate()
I0.fn = v_theta_ref.dot(v_theta_ref)
vnorm = I0.evaluate()
dv = uw.function.evaluate(norm * v_theta_ref, v_ref.coords).reshape(-1, 2) / vnorm
v_ref.data[...] -= dv

v_l2_ref = np.sqrt(uw.maths.Integral(mesh_ref, v_ref.sym.dot(v_ref.sym)).evaluate())
print(f"Reference velocity L2: {v_l2_ref:.6e}")

# %% [markdown]
"""
## 2. Air layer solve
"""

# %%
mesh_air = uw.meshing.AnnulusInternalBoundary(
    radiusOuter=r_outer_full, radiusInternal=r_internal,
    radiusInner=r_inner, cellSize=cellsize,
)

v_air = uw.discretisation.MeshVariable("V_air", mesh_air, mesh_air.dim, degree=2)
p_air = uw.discretisation.MeshVariable("P_air", mesh_air, 1, degree=1, continuous=True)
eta_var = uw.discretisation.MeshVariable("eta", mesh_air, 1, degree=1, continuous=False)
bf_mask = uw.discretisation.MeshVariable("mask", mesh_air, 1, degree=1, continuous=False)

r_at_eta = np.sqrt(eta_var.coords[:, 0]**2 + eta_var.coords[:, 1]**2)
is_rock = r_at_eta < r_internal
eta_var.data[is_rock, 0] = 1.0
eta_var.data[~is_rock, 0] = eta_air
bf_mask.data[is_rock, 0] = 1.0
bf_mask.data[~is_rock, 0] = 0.0

unit_rvec_air = mesh_air.CoordinateSystem.unit_e_0
r_air, th_air = mesh_air.CoordinateSystem.xR
Gamma_air = mesh_air.Gamma
v_theta_air = r_air * mesh_air.CoordinateSystem.rRotN.T * sympy.Matrix((0, 1))

stokes_air = Stokes(mesh_air, velocityField=v_air, pressureField=p_air)
stokes_air.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes_air.constitutive_model.Parameters.shear_viscosity_0 = eta_var.sym[0, 0]
stokes_air.saddle_preconditioner = 1.0 / eta_var.sym[0, 0]

rho_air = ((r_air / r_internal) ** k) * sympy.cos(n * th_air)
stokes_air.bodyforce = bf_mask.sym[0, 0] * rho_air * (-1.0 * unit_rvec_air)

stokes_air.add_natural_bc(1e4 * Gamma_air.dot(v_air.sym) * Gamma_air, "Upper")
stokes_air.add_natural_bc(1e4 * Gamma_air.dot(v_air.sym) * Gamma_air, "Lower")
stokes_air.add_natural_bc(1e4 * v_air.sym.dot(unit_rvec_air) * unit_rvec_air, "Internal")

stokes_air.tolerance = 1e-4
stokes_air.petsc_options["snes_type"] = "newtonls"
stokes_air.petsc_options["ksp_type"] = "fgmres"
stokes_air.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes_air.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

stokes_air.solve(verbose=False)

# Null space removal
I0 = uw.maths.Integral(mesh_air, v_theta_air.dot(v_air.sym))
norm = I0.evaluate()
I0.fn = v_theta_air.dot(v_theta_air)
vnorm = I0.evaluate()
dv = uw.function.evaluate(norm * v_theta_air, v_air.coords).reshape(-1, 2) / vnorm
v_air.data[...] -= dv

v_l2_air_inner = np.sqrt(uw.maths.Integral(
    mesh_air,
    sympy.Piecewise((1.0, r_air < r_internal), (0.0, True)) * v_air.sym.dot(v_air.sym)
).evaluate())
print(f"Air layer inner velocity L2: {v_l2_air_inner:.6e}")
print(f"Relative error vs reference: {abs(v_l2_air_inner - v_l2_ref) / v_l2_ref:.4e}")

# %% [markdown]
"""
## 3. Visualise reference solution
"""

# %%
if uw.mpi.size == 1:
    vis.plot_vector(mesh_ref, v_ref, vector_name="V_ref",
                    clip_angle=0., cpos="xy", show_arrows=False)

# %%
if uw.mpi.size == 1:
    vis.plot_scalar(mesh_ref, p_ref.sym, "P_ref",
                    clip_angle=0., cpos="xy")

# %% [markdown]
"""
## 4. Visualise air layer solution
"""

# %%
if uw.mpi.size == 1:
    vis.plot_vector(mesh_air, v_air, vector_name="V_air",
                    clip_angle=0., cpos="xy", show_arrows=False)

# %%
if uw.mpi.size == 1:
    vis.plot_scalar(mesh_air, p_air.sym, "P_air",
                    clip_angle=0., cpos="xy")

# %%
if uw.mpi.size == 1:
    vis.plot_scalar(mesh_air, eta_var.sym, "viscosity",
                    clip_angle=0., cpos="xy")

# %% [markdown]
"""
## 5. Summary

| Quantity | Reference | Air layer | Relative error |
|----------|-----------|-----------|----------------|
"""

# %%
p_l2_ref = np.sqrt(uw.maths.Integral(mesh_ref, p_ref.sym.dot(p_ref.sym)).evaluate())
p_l2_air_inner = np.sqrt(uw.maths.Integral(
    mesh_air,
    sympy.Piecewise((1.0, r_air < r_internal), (0.0, True)) * p_air.sym.dot(p_air.sym)
).evaluate())

print(f"{'Quantity':<20} {'Reference':>12} {'Air layer':>12} {'Rel. error':>12}")
print("-" * 60)
print(f"{'Velocity L2':<20} {v_l2_ref:>12.4e} {v_l2_air_inner:>12.4e} {abs(v_l2_air_inner - v_l2_ref)/v_l2_ref:>12.4e}")
print(f"{'Pressure L2':<20} {p_l2_ref:>12.4e} {p_l2_air_inner:>12.4e} {abs(p_l2_air_inner - p_l2_ref)/p_l2_ref:>12.4e}")
