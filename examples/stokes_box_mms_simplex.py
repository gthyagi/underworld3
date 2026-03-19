"""
Simple manufactured-solution Stokes example on a triangular box.

This script is useful for checking higher-order simplex velocity / pressure
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cellsize", type=float, default=0.125)
    parser.add_argument("--vdegree", type=int, default=3)
    parser.add_argument("--pdegree", type=int, default=2)
    parser.add_argument("--pcont", action="store_true", default=True)
    parser.add_argument("--stokes-tol", type=float, default=1.0e-10)
    return parser.parse_args()


def subtract_pressure_mean(mesh, pressure_var):
    p_int = uw.maths.Integral(mesh, pressure_var.sym[0]).evaluate()
    volume = uw.maths.Integral(mesh, 1.0).evaluate()
    pressure_var.data[:, 0] -= p_int / volume


def relative_l2_error(mesh, err_expr, ana_expr):
    if isinstance(err_expr, sp.MatrixBase):
        err_expr = err_expr.dot(err_expr)
        ana_expr = ana_expr.dot(ana_expr)
    else:
        err_expr = err_expr * err_expr
        ana_expr = ana_expr * ana_expr

    err_I = uw.maths.Integral(mesh, err_expr)
    ana_I = uw.maths.Integral(mesh, ana_expr)

    return np.sqrt(err_I.evaluate()) / np.sqrt(ana_I.evaluate())


def main():
    args = parse_args()
    qdegree = max(2 * args.vdegree, args.vdegree + args.pdegree)

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=args.cellsize,
        regular=True,
        qdegree=qdegree,
        filename=f"/tmp/stokes_box_mms_simplex_{args.vdegree}_{args.pdegree}_{args.cellsize}.msh",
    )

    x, y = mesh.X

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

    stokes.solve(verbose=False, debug=False)

    subtract_pressure_mean(mesh, p_soln)

    v_err_expr = sp.Matrix(v_soln.sym).T - v_ana_expr
    p_err_expr = p_soln.sym[0] - p_ana_expr

    v_err_l2 = relative_l2_error(mesh, v_err_expr, v_ana_expr)
    p_err_l2 = relative_l2_error(mesh, p_err_expr, p_ana_expr)

    uw.pprint("cellsize:", args.cellsize)
    uw.pprint("vdegree:", args.vdegree)
    uw.pprint("pdegree:", args.pdegree)
    uw.pprint("Relative velocity L2 error:", v_err_l2)
    uw.pprint("Relative pressure L2 error:", p_err_l2)


if __name__ == "__main__":
    main()
