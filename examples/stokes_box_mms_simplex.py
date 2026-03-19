"""
Notebook-style manufactured-solution Stokes example on a triangular box.

This example is useful for checking higher-order simplex velocity / pressure
pairs on a straight-edged domain without curved-geometry effects.

Default setup:
    python examples/stokes_box_mms_simplex.py

Useful comparison:
    python examples/stokes_box_mms_simplex.py --vdegree 2 --pdegree 1
    python examples/stokes_box_mms_simplex.py --vdegree 3 --pdegree 2
"""

import argparse
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ["SYMPY_USE_CACHE"] = "no"

import numpy as np
import sympy as sp

import underworld3 as uw
from underworld3.systems import Stokes

# %% [markdown]
# # Stokes Box MMS on a Simplex Mesh
#
# This notebook-style example solves a manufactured Stokes problem on the unit
# square using triangular elements. It is intentionally simple and useful for
# checking higher-order simplex pairs such as `P2/P1` and `P3/P2`.

# %% [markdown]
# ## Runtime parameters
#
# The script still accepts command-line arguments so it can be run directly.

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--cellsize", type=float, default=0.125)
parser.add_argument("--vdegree", type=int, default=3)
parser.add_argument("--pdegree", type=int, default=2)
parser.add_argument("--pcont", action="store_true", default=True)
parser.add_argument("--stokes-tol", type=float, default=1.0e-10)
args = parser.parse_args()

qdegree = max(2 * args.vdegree, args.vdegree + args.pdegree)

# %% [markdown]
# ## Mesh

# %%
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=args.cellsize,
    regular=True,
    qdegree=qdegree,
    filename=f"/tmp/stokes_box_mms_simplex_{args.vdegree}_{args.pdegree}_{args.cellsize}.msh",
)

x, y = mesh.X

# %% [markdown]
# ## Manufactured solution
#
# Stream function:
# `psi = x^2 y^2`
#
# Exact velocity:
# `u = (dpsi/dy, -dpsi/dx)`
#
# Exact pressure:
# `p = x^2 - 1/3`

# %%
psi = x**2 * y**2
v_ana_expr = sp.Matrix([sp.diff(psi, y), -sp.diff(psi, x)])
p_ana_expr = x**2 - sp.Rational(1, 3)

bodyforce = sp.Matrix(
    [
        -(sp.diff(v_ana_expr[0], x, 2) + sp.diff(v_ana_expr[0], y, 2))
        + sp.diff(p_ana_expr, x),
        -(sp.diff(v_ana_expr[1], x, 2) + sp.diff(v_ana_expr[1], y, 2))
        + sp.diff(p_ana_expr, y),
    ]
)

# %% [markdown]
# ## Discretisation

# %%
v_soln = uw.discretisation.MeshVariable(
    varname="Velocity",
    mesh=mesh,
    degree=args.vdegree,
    vtype=uw.VarType.VECTOR,
)

p_soln = uw.discretisation.MeshVariable(
    varname="Pressure",
    mesh=mesh,
    degree=args.pdegree,
    vtype=uw.VarType.SCALAR,
    continuous=args.pcont,
)

# %% [markdown]
# ## Stokes system

# %%
stokes = Stokes(mesh, velocityField=v_soln, pressureField=p_soln)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0
stokes.bodyforce = bodyforce

for boundary_name in (
    mesh.boundaries.Bottom.name,
    mesh.boundaries.Top.name,
    mesh.boundaries.Left.name,
    mesh.boundaries.Right.name,
):
    stokes.add_essential_bc(v_ana_expr, boundary_name)

stokes.tolerance = args.stokes_tol
stokes.petsc_options["snes_type"] = "ksponly"
stokes.petsc_options["ksp_type"] = "fgmres"
stokes.petsc_options["ksp_rtol"] = args.stokes_tol
stokes.petsc_options["ksp_atol"] = 0.0
stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["ksp_monitor_true_residual"] = None
stokes.petsc_options["ksp_converged_reason"] = None

stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")
stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None
stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# %% [markdown]
# ## Solve

# %%
stokes.solve(verbose=False, debug=False)

# %% [markdown]
# ## Pressure gauge

# %%
p_int = uw.maths.Integral(mesh, p_soln.sym[0]).evaluate()
volume = uw.maths.Integral(mesh, 1.0).evaluate()
p_soln.data[:, 0] -= p_int / volume

# %% [markdown]
# ## Relative `L2` errors

# %%
v_err_expr = sp.Matrix(v_soln.sym).T - v_ana_expr
p_err_expr = p_soln.sym[0] - p_ana_expr

v_err_sq_expr = v_err_expr.dot(v_err_expr)
v_ana_sq_expr = v_ana_expr.dot(v_ana_expr)
p_err_sq_expr = p_err_expr * p_err_expr
p_ana_sq_expr = p_ana_expr * p_ana_expr

v_err_l2 = np.sqrt(uw.maths.Integral(mesh, v_err_sq_expr).evaluate()) / np.sqrt(
    uw.maths.Integral(mesh, v_ana_sq_expr).evaluate()
)
p_err_l2 = np.sqrt(uw.maths.Integral(mesh, p_err_sq_expr).evaluate()) / np.sqrt(
    uw.maths.Integral(mesh, p_ana_sq_expr).evaluate()
)

# %% [markdown]
# ## Report

# %%
uw.pprint("cellsize:", args.cellsize)
uw.pprint("vdegree:", args.vdegree)
uw.pprint("pdegree:", args.pdegree)
uw.pprint("Relative velocity L2 error:", v_err_l2)
uw.pprint("Relative pressure L2 error:", p_err_l2)
