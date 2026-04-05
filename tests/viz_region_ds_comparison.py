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
# Rock-Only vs Air-Layer (dP1) Comparison

All solves use normalised Gamma_N, penalty=1e4, tol=1e-4, discontinuous pressure.

Three cases:
- Rock-only submesh (extracted via DMPlexFilter)
- Air-layer with eta_air=1e-3
- Air-layer with eta_air=1e-6
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
## Create meshes and load checkpoints
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

# Rock-only
v_rock = uw.discretisation.MeshVariable("V_rock", rock_mesh, rock_mesh.dim, degree=2)
p_rock = uw.discretisation.MeshVariable("P_rock", rock_mesh, 1, degree=1, continuous=True)
v_rock.read_timestep("rock", "V", 0, outputPath="../output/normalised_rock/")
p_rock.read_timestep("rock", "P", 0, outputPath="../output/normalised_rock/")
print(f"Rock submesh: {v_rock.data.shape[0]} v-nodes")

# Air-layer eta=1e-3 (dP1)
v_dg3 = uw.discretisation.MeshVariable("V_dg3", full_mesh, full_mesh.dim, degree=2)
p_dg3 = uw.discretisation.MeshVariable("P_dg3", full_mesh, 1, degree=1, continuous=False)
v_dg3.read_timestep("nitsche", "V", 0, outputPath="../output/normalised_nitsche/")
p_dg3.read_timestep("nitsche", "P", 0, outputPath="../output/normalised_nitsche/")
print(f"Air-layer eta=1e-3 (dP1): {v_dg3.data.shape[0]} v-nodes")

# Air-layer eta=1e-6 (dP1)
v_dg6 = uw.discretisation.MeshVariable("V_dg6", full_mesh, full_mesh.dim, degree=2)
p_dg6 = uw.discretisation.MeshVariable("P_dg6", full_mesh, 1, degree=1, continuous=False)
v_dg6.read_timestep("eta1e6", "V", 0, outputPath="../output/normalised_eta1e6/")
p_dg6.read_timestep("eta1e6", "P", 0, outputPath="../output/normalised_eta1e6/")
print(f"Air-layer eta=1e-6 (dP1): {v_dg6.data.shape[0]} v-nodes")

# %% [markdown]
"""
## Rock-only: velocity and pressure
"""

# %%
if uw.mpi.size == 1:
    vmag = np.sqrt(v_rock.data[:, 0]**2 + v_rock.data[:, 1]**2)
    vis.plot_vector(rock_mesh, v_rock, vector_name="V_rock", vfreq=1, vmag=2e1,
                    clip_angle=0., cpos="xy", show_arrows=True,
                    clim=[0., float(vmag.max())], cmap="coolwarm")

# %%
if uw.mpi.size == 1:
    pvals = uw.function.evaluate(p_rock.sym[0, 0], p_rock.coords).flatten()
    plim = float(max(abs(pvals.min()), abs(pvals.max())))
    vis.plot_scalar(rock_mesh, p_rock.sym, "P_rock",
                    clip_angle=0., cpos="xy", cmap="RdBu",
                    clim=[-plim, plim])

# %% [markdown]
"""
## Air-layer eta=1e-3 (dP1): velocity and pressure
"""

# %%
if uw.mpi.size == 1:
    vmag3 = np.sqrt(v_dg3.data[:, 0]**2 + v_dg3.data[:, 1]**2)
    vis.plot_vector(full_mesh, v_dg3, vector_name="V_dg3", vfreq=1, vmag=2e1,
                    clip_angle=0., cpos="xy", show_arrows=True,
                    clim=[0., float(vmag3.max())], cmap="coolwarm")

# %%
if uw.mpi.size == 1:
    pvals3 = uw.function.evaluate(p_dg3.sym[0, 0], p_dg3.coords).flatten()
    plim3 = float(max(abs(pvals3.min()), abs(pvals3.max())))
    vis.plot_scalar(full_mesh, p_dg3.sym, "P_dg3",
                    clip_angle=0., cpos="xy", cmap="RdBu",
                    clim=[-plim3, plim3])

# %% [markdown]
"""
## Air-layer eta=1e-6 (dP1): velocity and pressure
"""

# %%
if uw.mpi.size == 1:
    vmag6 = np.sqrt(v_dg6.data[:, 0]**2 + v_dg6.data[:, 1]**2)
    vis.plot_vector(full_mesh, v_dg6, vector_name="V_dg6", vfreq=1, vmag=2e1,
                    clip_angle=0., cpos="xy", show_arrows=True,
                    clim=[0., float(vmag6.max())], cmap="coolwarm")

# %%
if uw.mpi.size == 1:
    pvals6 = uw.function.evaluate(p_dg6.sym[0, 0], p_dg6.coords).flatten()
    plim6 = float(max(abs(pvals6.min()), abs(pvals6.max())))
    vis.plot_scalar(full_mesh, p_dg6.sym, "P_dg6",
                    clip_angle=0., cpos="xy", cmap="RdBu",
                    clim=[-plim6, plim6])

# %% [markdown]
"""
## Norm comparison at matched nodes
"""

# %%
tree = cKDTree(v_rock.coords)

dists3, idx3 = tree.query(v_dg3.coords)
matched3 = dists3 < 1e-10

dists6, idx6 = tree.query(v_dg6.coords)
matched6 = dists6 < 1e-10

v_ref = v_rock.data
v3_m = v_dg3.data[matched3]
v6_m = v_dg6.data[matched6]
v_ref3 = v_ref[idx3[matched3]]
v_ref6 = v_ref[idx6[matched6]]

def l2(a, b):
    return np.sqrt(np.sum((a - b)**2)) / np.sqrt(np.sum(b**2))

print(f"Matched: eta=1e-3: {matched3.sum()} nodes, eta=1e-6: {matched6.sum()} nodes")
print()
print(f"{'Metric':<22} {'eta=1e-3':>12} {'eta=1e-6':>12}")
print("-" * 48)
print(f"{'Velocity L2 rel':<22} {l2(v3_m, v_ref3):>12.4e} {l2(v6_m, v_ref6):>12.4e}")

vmag_r3 = np.sqrt(v_ref3[:, 0]**2 + v_ref3[:, 1]**2)
vmag_3 = np.sqrt(v3_m[:, 0]**2 + v3_m[:, 1]**2)
vmag_r6 = np.sqrt(v_ref6[:, 0]**2 + v_ref6[:, 1]**2)
vmag_6 = np.sqrt(v6_m[:, 0]**2 + v6_m[:, 1]**2)
print(f"{'|v| ratio (air/rock)':<22} {vmag_3.mean()/vmag_r3.mean():>12.4f} {vmag_6.mean()/vmag_r6.mean():>12.4f}")

# %%
