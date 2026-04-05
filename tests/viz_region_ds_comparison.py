# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Normalised Gamma_N: Rock-Only vs Air-Layer Comparison

Loads checkpointed solutions from `test_normalised_comparison.py`.
Both use normalised `Gamma_N` penalty with penalty=1e4, tol=1e-4.

Run the solve script first:
```
pixi run -e default python tests/test_normalised_comparison.py
```
"""

# %%
import underworld3 as uw
import underworld3.visualisation as vis
from underworld3.cython.petsc_discretisation import petsc_dm_filter_by_label
from underworld3.discretisation import Mesh
from underworld3.coordinates import CoordinateSystemType
import numpy as np
import sympy
from enum import Enum
from scipy.spatial import cKDTree

if uw.mpi.size == 1:
    import pyvista as pv

# %%
r_inner = 0.5
r_internal = 1.0
r_outer_full = 1.5
cellsize = 1/16

# %% [markdown]
"""
## Load rock-only submesh solution
"""

# %%
full_mesh = uw.meshing.AnnulusInternalBoundary(
    radiusOuter=r_outer_full, radiusInternal=r_internal,
    radiusInner=r_inner, cellSize=cellsize,
)

subdm = petsc_dm_filter_by_label(full_mesh.dm, "Inner", 101)
subdm.markBoundaryFaces("All_Boundaries", 1001)

class sub_bd(Enum):
    Lower = 1; Internal = 2

rock_mesh = Mesh(subdm, degree=1, qdegree=2, boundaries=sub_bd,
                 coordinate_system_type=CoordinateSystemType.CYLINDRICAL2D)

v_rock = uw.discretisation.MeshVariable("V_rock", rock_mesh, rock_mesh.dim, degree=2)
p_rock = uw.discretisation.MeshVariable("P_rock", rock_mesh, 1, degree=1, continuous=True)
v_rock.read_timestep("rock", "V", 0, outputPath="../output/normalised_rock/")
p_rock.read_timestep("rock", "P", 0, outputPath="../output/normalised_rock/")
print(f"Rock submesh: {v_rock.data.shape[0]} v-nodes loaded")

# %% [markdown]
"""
## Load air-layer penalty solution
"""

# %%
# DG pressure solve (current checkpoint in normalised_nitsche/)
v_dg = uw.discretisation.MeshVariable("V_dg", full_mesh, full_mesh.dim, degree=2)
v_dg.read_timestep("nitsche", "V", 0, outputPath="../output/normalised_nitsche/")
print(f"Air-layer (DG P): {v_dg.data.shape[0]} v-nodes loaded")

# %% [markdown]
"""
## Load air-layer continuous-P solution
"""

# %%
# Continuous pressure solve (separate checkpoint)
v_air = uw.discretisation.MeshVariable("V_air", full_mesh, full_mesh.dim, degree=2)
p_air = uw.discretisation.MeshVariable("P_air", full_mesh, 1, degree=1, continuous=True)
v_air.read_timestep("cont_p", "V", 0, outputPath="../output/normalised_cont_p/")
p_air.read_timestep("cont_p", "P", 0, outputPath="../output/normalised_cont_p/")
print(f"Air-layer (cont P): {v_air.data.shape[0]} v-nodes loaded")

# %% [markdown]
"""
## Rock-only: velocity
"""

# %%
if uw.mpi.size == 1:
    vmag = np.sqrt(v_rock.data[:, 0]**2 + v_rock.data[:, 1]**2)
    vis.plot_vector(rock_mesh, v_rock, vector_name="V_rock", vfreq=1, vmag=2e1,
                    clip_angle=0., cpos="xy", show_arrows=True,
                    clim=[0., float(vmag.max())], cmap="coolwarm")

# %% [markdown]
"""
## Air-layer: velocity (full mesh)
"""

# %%
if uw.mpi.size == 1:
    vmag_air = np.sqrt(v_air.data[:, 0]**2 + v_air.data[:, 1]**2)
    vis.plot_vector(full_mesh, v_air, vector_name="V_air", vfreq=1, vmag=2e1,
                    clip_angle=0., cpos="xy", show_arrows=True,
                    clim=[0., float(vmag_air.max())], cmap="coolwarm")

# %% [markdown]
"""
## Rock-only: pressure
"""

# %%
if uw.mpi.size == 1:
    pvals = uw.function.evaluate(p_rock.sym[0, 0], p_rock.coords).flatten()
    plim = float(max(abs(pvals.min()), abs(pvals.max())))
    vis.plot_scalar(rock_mesh, p_rock.sym, "P_rock",
                    clip_angle=0., cpos="xy", cmap="RdBu",
                    clim=[-plim, plim])

# %% [markdown]
"""
## Air-layer: pressure
"""

# %%
if uw.mpi.size == 1:
    pvals_air = uw.function.evaluate(p_air.sym[0, 0], p_air.coords).flatten()
    plim_air = float(max(abs(pvals_air.min()), abs(pvals_air.max())))
    vis.plot_scalar(full_mesh, p_air.sym, "P_air",
                    clip_angle=0., cpos="xy", cmap="RdBu",
                    clim=[-plim_air, plim_air])

# %% [markdown]
"""
## Overlay: all three velocity fields (pyvista interactive)

- Blue: rock-only submesh
- Red: air-layer, continuous pressure
- Green: air-layer, discontinuous pressure

Same nodes, same arrow scale. Zoom to compare.
"""

# %%
if uw.mpi.size == 1:
    tree = cKDTree(v_rock.coords)

    # Match continuous-P air-layer nodes to rock nodes
    dists_c, idx_c = tree.query(v_air.coords)
    matched_c = dists_c < 1e-10

    # Match DG-P air-layer nodes to rock nodes
    dists_d, idx_d = tree.query(v_dg.coords)
    matched_d = dists_d < 1e-10

    # Rock submesh
    rock_pts = pv.PolyData(np.column_stack([v_rock.coords, np.zeros(len(v_rock.coords))]))
    rock_pts["vectors"] = np.column_stack([v_rock.data, np.zeros(len(v_rock.data))])

    # Continuous-P at matched nodes
    cont_coords = v_air.coords[matched_c]
    cont_data = v_air.data[matched_c]
    cont_pts = pv.PolyData(np.column_stack([cont_coords, np.zeros(len(cont_coords))]))
    cont_pts["vectors"] = np.column_stack([cont_data, np.zeros(len(cont_data))])

    # DG-P at matched nodes
    dg_coords = v_dg.coords[matched_d]
    dg_data = v_dg.data[matched_d]
    dg_pts = pv.PolyData(np.column_stack([dg_coords, np.zeros(len(dg_coords))]))
    dg_pts["vectors"] = np.column_stack([dg_data, np.zeros(len(dg_data))])

    vmax = max(np.sqrt(v_rock.data[:, 0]**2 + v_rock.data[:, 1]**2).max(),
               np.sqrt(cont_data[:, 0]**2 + cont_data[:, 1]**2).max(),
               np.sqrt(dg_data[:, 0]**2 + dg_data[:, 1]**2).max())
    factor = 0.1 / vmax if vmax > 0 else 1.0

    rock_arrows = rock_pts.glyph(orient="vectors", scale="vectors", factor=factor)
    cont_arrows = cont_pts.glyph(orient="vectors", scale="vectors", factor=factor)
    dg_arrows = dg_pts.glyph(orient="vectors", scale="vectors", factor=factor)

    pl = pv.Plotter()
    pl.add_mesh(rock_arrows, color="blue", opacity=0.7, label="Rock-only submesh")
    pl.add_mesh(cont_arrows, color="red", opacity=0.7, label="Air-layer (cont P)")
    pl.add_mesh(dg_arrows, color="green", opacity=0.7, label="Air-layer (DG P)")

    theta = np.linspace(0, 2*np.pi, 200)
    circle = pv.lines_from_points(np.column_stack([
        r_internal * np.cos(theta), r_internal * np.sin(theta), np.zeros(200)
    ]))
    pl.add_mesh(circle, color="black", line_width=2)

    pl.add_legend()
    pl.camera_position = "xy"
    pl.show()

# %% [markdown]
"""
## Norm comparison at matched nodes
"""

# %%
tree = cKDTree(v_rock.coords)
dists, idx = tree.query(v_air.coords)
matched = dists < 1e-10

v_rock_m = v_rock.data[idx[matched]]
v_air_m = v_air.data[matched]

tree_p = cKDTree(p_rock.coords)
dists_p, idx_p = tree_p.query(p_air.coords)
matched_p = dists_p < 1e-10

p_rock_m = p_rock.data[idx_p[matched_p]]
p_air_m = p_air.data[matched_p]

def l2(a, b):
    return np.sqrt(np.sum((a - b)**2)) / np.sqrt(np.sum(b**2))

def linf(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))

print(f"Matched: {matched.sum()} v-nodes, {matched_p.sum()} p-nodes")
print()
print(f"{'Metric':<22} {'Value':>12}")
print("-" * 36)
print(f"{'Velocity L2 rel':<22} {l2(v_air_m, v_rock_m):>12.4e}")
print(f"{'Velocity Linf rel':<22} {linf(v_air_m, v_rock_m):>12.4e}")
print(f"{'Pressure L2 rel':<22} {l2(p_air_m, p_rock_m):>12.4e}")
print(f"{'Pressure Linf rel':<22} {linf(p_air_m, p_rock_m):>12.4e}")
print()
vmag_r = np.sqrt(v_rock_m[:, 0]**2 + v_rock_m[:, 1]**2)
vmag_a = np.sqrt(v_air_m[:, 0]**2 + v_air_m[:, 1]**2)
print(f"|v| ratio (air/rock): {vmag_a.mean() / vmag_r.mean():.4f}")

# %%
