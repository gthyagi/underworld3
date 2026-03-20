#!/usr/bin/env python3

# %%
# Demonstrate essential BC registration on a wrapped imported DMPlex.
#
# Before the fix, this script crashes in stokes.solve() because the imported
# mesh exposes generic Gmsh labels (Face Sets / Cell Sets) rather than named
# boundary labels such as Bottom / Top / Left / Right.
# After the fix, the same solve succeeds because essential BCs are registered
# on the consolidated UW_Boundaries label.

# %%
from enum import Enum
from pathlib import Path
import tempfile

import gmsh
import numpy as np
from petsc4py import PETSc
import sympy as sp

import underworld3 as uw
from underworld3.systems import Stokes


# %%
CELLSIZE = 0.25
MESH_PATH = Path(tempfile.gettempdir()) / "uw3_imported_box_essential_bc.msh"


class boundaries_2D(Enum):
    Bottom = 11
    Top = 12
    Right = 13
    Left = 14
    Elements = 666
    Null_Boundary = 666
    All_Boundaries = 1001


class boundary_normals_2D(Enum):
    Bottom = sp.Matrix([0, 1])
    Top = sp.Matrix([0, 1])
    Right = sp.Matrix([1, 0])
    Left = sp.Matrix([1, 0])


# %%
def build_box_mesh(mesh_path: Path, cellsize: float) -> None:
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("uw3_imported_box_essential_bc")

    p1 = gmsh.model.geo.add_point(0.0, 0.0, 0.0, cellsize)
    p2 = gmsh.model.geo.add_point(1.0, 0.0, 0.0, cellsize)
    p3 = gmsh.model.geo.add_point(1.0, 1.0, 0.0, cellsize)
    p4 = gmsh.model.geo.add_point(0.0, 1.0, 0.0, cellsize)

    l1 = gmsh.model.geo.add_line(p1, p2)
    l2 = gmsh.model.geo.add_line(p2, p3)
    l3 = gmsh.model.geo.add_line(p3, p4)
    l4 = gmsh.model.geo.add_line(p4, p1)

    cl = gmsh.model.geo.add_curve_loop([l1, l2, l3, l4])
    surface = gmsh.model.geo.add_plane_surface([cl])
    gmsh.model.geo.synchronize()

    gmsh.model.add_physical_group(1, [l1], boundaries_2D.Bottom.value, name=boundaries_2D.Bottom.name)
    gmsh.model.add_physical_group(1, [l3], boundaries_2D.Top.value, name=boundaries_2D.Top.name)
    gmsh.model.add_physical_group(1, [l2], boundaries_2D.Right.value, name=boundaries_2D.Right.name)
    gmsh.model.add_physical_group(1, [l4], boundaries_2D.Left.value, name=boundaries_2D.Left.name)
    gmsh.model.add_physical_group(2, [surface], boundaries_2D.Elements.value, name=boundaries_2D.Elements.name)

    gmsh.model.mesh.generate(2)
    gmsh.write(str(mesh_path))
    gmsh.finalize()


# %%
build_box_mesh(MESH_PATH, CELLSIZE)
plex = PETSc.DMPlex().createFromFile(str(MESH_PATH), interpolate=True, comm=PETSc.COMM_WORLD)

print("raw labels:", [plex.getLabelName(i) for i in range(plex.getNumLabels())], flush=True)

mesh = uw.discretisation.Mesh(
    plex,
    degree=1,
    qdegree=4,
    boundaries=boundaries_2D,
    boundary_normals=boundary_normals_2D,
    markVertices=True,
    useMultipleTags=True,
    useRegions=True,
    coordinate_system_type=uw.coordinates.CoordinateSystemType.CARTESIAN,
)

print("has Bottom label?", bool(mesh.dm.getLabel("Bottom")), flush=True)
print("has UW_Boundaries?", bool(mesh.dm.getLabel("UW_Boundaries")), flush=True)


# %%
x, y = mesh.X
v_exact = sp.Matrix([y, 0])

v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2, vtype=uw.VarType.VECTOR)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True, vtype=uw.VarType.SCALAR)

stokes = Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0
stokes.bodyforce = sp.Matrix([0, 0])

for name in ("Bottom", "Top", "Left", "Right"):
    stokes.add_essential_bc(v_exact, name)

stokes.tolerance = 1.0e-10
stokes.petsc_options["snes_type"] = "ksponly"
stokes.petsc_options["ksp_type"] = "preonly"
stokes.petsc_options["pc_type"] = "lu"

print("about to solve", flush=True)
stokes.solve(verbose=False, debug=False)


# %%
vel_err = uw.maths.Integral(
    mesh, fn=((v.sym[0] - v_exact[0]) ** 2 + (v.sym[1] - v_exact[1]) ** 2)
).evaluate()
vel_ref = uw.maths.Integral(mesh, fn=(v_exact[0] ** 2 + v_exact[1] ** 2)).evaluate()
print("relative velocity L2 error", float(np.sqrt(vel_err / vel_ref)), flush=True)
