"""VEP shear box with embedded fault using Surface interface.

Horizontal fault at y=0.5. Yield stress is low near the fault (gaussian
influence function) and high in the bulk. Strain weakening feedback
via scalar tau_y update each step.

Run: pixi run -e amr-dev python tests/vep_fault_weakening.py
"""

import time
import numpy as np
import sympy
import underworld3 as uw

ETA = 1.0
MU = 1.0
TAU_Y_FAULT = 0.2    # yield stress near fault
TAU_Y_BULK = 2.0     # strong but not rigid in the bulk
FAULT_WIDTH = 0.08   # gaussian half-width
EPS_CRIT = 0.3       # critical plastic strain for full weakening
TAU_Y_RESIDUAL = 0.05
DT = 0.1
NSTEPS = 25
V_TOP = 0.5

t0 = time.time()

mesh = uw.meshing.StructuredQuadBox(
    elementRes=(16, 16), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=2)
v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2, vtype=uw.VarType.VECTOR)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True,
                                    vtype=uw.VarType.SCALAR)

# --- Fault surface ---

fault_points = np.array([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]])
fault = uw.meshing.Surface("fault", mesh, fault_points)
fault.discretize()

print(f"Fault: {fault.n_vertices} vertices")

# Yield stress: gaussian decay from fault value to bulk value
# Uses the smooth gaussian profile — no Piecewise
tau_y_field = fault.influence_function(
    width=FAULT_WIDTH,
    value_near=TAU_Y_FAULT,
    value_far=TAU_Y_BULK,
    profile="gaussian",
)

# --- Solver ---

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
cm = stokes.constitutive_model
cm.Parameters.shear_viscosity_0 = ETA
cm.Parameters.shear_modulus = MU
cm.Parameters.yield_stress = tau_y_field
cm.Parameters.shear_viscosity_min = ETA * 1.0e-2
cm.Parameters.strainrate_inv_II_min = 1.0e-10
stokes.saddle_preconditioner = 1.0
stokes.tolerance = 1.0e-4

stokes.add_essential_bc(sympy.Matrix([V_TOP, 0.0]), "Top")
stokes.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
stokes.add_essential_bc((sympy.oo, 0.0), "Left")
stokes.add_essential_bc((sympy.oo, 0.0), "Right")
stokes.bodyforce = sympy.Matrix([0.0, 0.0])
stokes.petsc_options["ksp_type"] = "fgmres"

print(f"Setup: {time.time()-t0:.1f}s")
print(f"tau_y_fault={TAU_Y_FAULT}, tau_y_bulk={TAU_Y_BULK}, width={FAULT_WIDTH}")
print()

# --- Time stepping ---

times = []
fault_stresses = []
bulk_stresses = []

# Sample points: one near the fault, one in the bulk
fault_sample = np.array([[0.5, 0.5]])
bulk_sample = np.array([[0.5, 0.25]])

print(f"{'step':>4} {'t':>5} {'sigma_fault':>12} {'sigma_bulk':>12}")
print("-" * 40)

for step in range(NSTEPS):
    t1 = time.time()
    stokes.solve(timestep=DT, zero_init_guess=False)
    solve_time = time.time() - t1

    t = (step + 1) * DT
    times.append(t)

    # Read stress from tau — get xy component at fault and bulk locations
    sd = stokes.tau.data
    coords = stokes.tau.coords

    # Find nearest nodes to sample points
    fault_idx = np.argmin(np.sum((coords - fault_sample)**2, axis=1))
    bulk_idx = np.argmin(np.sum((coords - bulk_sample)**2, axis=1))

    sigma_fault = sd[fault_idx, 2]
    sigma_bulk = sd[bulk_idx, 2]
    fault_stresses.append(sigma_fault)
    bulk_stresses.append(sigma_bulk)

    if (step + 1) % 5 == 0 or step < 8:
        print(f"{step+1:4d} {t:5.2f} {sigma_fault:12.4f} {sigma_bulk:12.4f}  ({solve_time:.1f}s)")

print()
print(f"Total: {time.time()-t0:.0f}s")

# --- Plot ---

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(8, 8))

t_anal = np.linspace(0.001, max(times), 200)
t_r = ETA / MU
maxwell = ETA * V_TOP * (1 - np.exp(-t_anal / t_r))

axes[0].plot(t_anal, maxwell, 'k--', linewidth=1, alpha=0.4, label="Maxwell (no yield)")
axes[0].plot(times, fault_stresses, 'r-', linewidth=2, label=r"$\sigma_{xy}$ at fault")
axes[0].plot(times, bulk_stresses, 'b-', linewidth=2, label=r"$\sigma_{xy}$ in bulk")
axes[0].axhline(TAU_Y_FAULT, color='r', linestyle=':', alpha=0.5, label=f"$\\tau_y$ fault={TAU_Y_FAULT}")
axes[0].set_ylabel("Stress")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_title(f"VEP with embedded fault: $\\eta$={ETA}, $\\mu$={MU}")
axes[0].set_xlabel("Time")

# Cross-section of stress at final step
y_coords = stokes.tau.coords[:, 1]
x_coords = stokes.tau.coords[:, 0]
# Get nodes near x=0.5
near_centre = np.abs(x_coords - 0.5) < 0.1
y_profile = y_coords[near_centre]
sigma_profile = sd[near_centre, 2]
sort_idx = y_profile.argsort()

axes[1].plot(y_profile[sort_idx], sigma_profile[sort_idx], 'r-o', markersize=3, linewidth=2)
axes[1].axvline(0.5, color='gray', linestyle=':', alpha=0.5, label="fault location")
axes[1].set_xlabel("y")
axes[1].set_ylabel(r"$\sigma_{xy}$ at final step")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_title("Stress profile across fault")

fig.tight_layout()
out_path = "/Users/lmoresi/+Underworld/underworld3-pixi/.claude/worktrees/solver-unification/vep_fault.png"
fig.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
