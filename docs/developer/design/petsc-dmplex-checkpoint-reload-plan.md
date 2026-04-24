# PETSc DMPlex Checkpoint Reload Plan

## Objective

Implement and validate an exact PETSc DMPlex checkpoint reload path for UW3
mesh variables.

The target workflow is:

```python
mesh = uw.discretisation.Mesh("restart.mesh.0.h5", ...)
v_soln = uw.discretisation.MeshVariable("Velocity", mesh, ...)
p_soln = uw.discretisation.MeshVariable("Pressure", mesh, ...)

v_soln.load_from_checkpoint(
    "restart.checkpoint.00000.h5",
    data_name="Velocity",
)
p_soln.load_from_checkpoint(
    "restart.checkpoint.00000.h5",
    data_name="Pressure",
)
```

This reload path must not use KDTree remapping. It must restore the saved FE
vectors using PETSc DMPlex topology, section, and vector metadata.

## Commit And Test Workflow

Use small, meaningful commits while implementing this work.

Do `git add` and `git commit` after each coherent feature, fix, or debugging
checkpoint so progress is easy to inspect and bisect. Do not wait until the end
to commit everything as one large change, and do not commit after every tiny
edit.

Create or update unit tests whenever they are needed to prove the behavior or
prevent regressions. Prefer adding a failing regression test before the fix
when the failure mode is already known.

Commit at useful milestones, such as:

- adding the first failing regression test
- adding storage-version `3.0.0` write support
- exposing or preserving the checkpoint topology SF
- adding the first working `load_from_checkpoint(...)` helper
- adding MPI same-rank test coverage
- adding different-rank reload coverage or documenting its limitation
- integrating the helper into benchmark-facing workflow tests

Each commit message should describe the specific feature, fix, or debugging
result included in that commit.

## Problem Statement

UW3 currently has two related but different output paths:

- `mesh.write_timestep(...)`
- `mesh.write_checkpoint(...)`

`write_timestep()` is visualization / remap oriented. Its reload path,
`MeshVariable.read_timestep(...)`, reads `/fields/<name>` and
`/fields/coordinates`, builds a KDTree, and maps values onto the current mesh.
That is flexible, but it is expensive and memory-heavy at high MPI counts.

`write_checkpoint()` writes PETSc DMPlex restart metadata, including section
and vector information. This is the correct format for exact restart, but UW3
does not yet expose a clean, validated helper for loading mesh variables from
this format.

The benchmark scripts should not manually guess PETSc section/SF details. The
fix belongs in UW3 checkpoint I/O.

## PETSc Requirements

PETSc's DMPlex HDF5 restart workflow requires these steps:

1. Load topology with `DMPlexTopologyLoad(...)`.
2. Keep the `PetscSF` returned by topology load.
3. Load coordinates with `DMPlexCoordinatesLoad(...)`.
4. Load labels with `DMPlexLabelsLoad(...)`.
5. If the mesh is redistributed after load, compose the topology-load SF with
   the redistribution SF.
6. Load the saved section with `DMPlexSectionLoad(...)`.
7. Load the saved vector with `DMPlexGlobalVectorLoad(...)`.
8. Scatter the global vector into UW3's local mesh-variable storage.

The critical object identity requirements are:

- The topology DM name used during reload must match the saved topology group.
- The section DM name must match the saved variable/subDM group.
- The vector name must match the saved vector group.
- The `PetscSF` passed to `DMPlexSectionLoad(...)` must describe the mapping
  from the saved topology points to the current distributed topology points.

## Required File Format Changes

### 1. Write DMPlex HDF5 Storage Version 3.0.0

Checkpoint mesh and field files should be written with PETSc DMPlex HDF5
storage version `3.0.0`.

Expected implementation direction:

- Set the HDF5 viewer / PETSc option used by DMPlex output so the saved files
  contain `dmplex_storage_version = "3.0.0"`.
- Ensure this is used by `Mesh.write_checkpoint(...)`.
- Prefer a local option scope or explicit viewer setting if available, rather
  than mutating global PETSc options permanently.

Expected outcome:

- New checkpoint files report storage version `3.0.0`.
- Files remain readable by PETSc `3.25.x`.
- Existing `write_timestep()` behavior is not unintentionally changed.

### 2. Preserve Or Expose The Correct Topology SF

When loading a mesh from a PETSc DMPlex HDF5 file, UW3 currently calls the
PETSc topology load path internally. The reload implementation must preserve
or expose the exact `PetscSF` returned by `DMPlexTopologyLoad(...)`.

Expected implementation direction:

- Verify that `_from_plexh5(..., return_sf=True)` stores the topology-load SF
  on the UW3 mesh object.
- Confirm whether any subsequent `DM.distribute()` call changes the topology.
- If redistribution happens, capture the redistribution SF and compose it with
  the topology-load SF as PETSc documents.

Expected outcome:

- The mesh object has a reliable SF suitable for `DMPlexSectionLoad(...)`.
- This SF is not guessed from `mesh.sf`, `mesh.sf0`, or `dm.getDefaultSF()`
  unless those names are explicitly verified to mean the PETSc-required SF.

### 3. Add A Supported MeshVariable Reload Helper

Add a public helper on mesh variables, for example:

```python
mesh_var.load_from_checkpoint(
    filename,
    data_name=None,
)
```

Expected behavior:

- `filename` points to a `write_checkpoint()` checkpoint file.
- `data_name` defaults to `mesh_var.clean_name`.
- The helper loads section metadata and vector data exactly.
- The helper updates both global and local PETSc vectors.
- The helper works under MPI.
- The helper does not use KDTree, coordinate remapping, or interpolation.

Expected internal sequence:

```python
if mesh_var._lvec is None:
    mesh_var._set_vec(available=True)

indexset, subdm = mesh_var.mesh.dm.createSubDM(mesh_var.field_id)
sectiondm = subdm.clone()

mesh_var.mesh.dm.setName("uw_mesh")
subdm.setName(data_name)
sectiondm.setName(data_name)
mesh_var._gvec.setName(data_name)

gsf, lsf = mesh_var.mesh.dm.sectionLoad(
    viewer,
    sectiondm,
    mesh_var.mesh.<checkpoint_topology_sf>,
)

subdm.setSection(sectiondm.getSection())
mesh_var.mesh.dm.globalVectorLoad(viewer, subdm, gsf, mesh_var._gvec)
subdm.globalToLocal(mesh_var._gvec, mesh_var._lvec, addv=False)
```

This sketch is not final API code. The important requirement is that UW3 owns
the PETSc SF and section semantics.

## Current Failure To Reproduce

A temporary Mac test was run with:

- spherical Thieulot benchmark
- `8` MPI ranks
- `uw_cellsize = 1/8`
- PETSc `3.25.0`

The baseline `read_timestep(...)` reload reproduced expected metrics:

```text
v_l2_norm               = 0.0059010703925659195
p_l2_norm               = 0.26146895222007555
sigma_rr_l2_norm_lower  = 0.1900342889700883
```

A manual PETSc metadata reload first failed because the runtime mesh DM name
was `plex`, while the checkpoint file stored data under `uw_mesh`:

```text
Object (dataset) "order" not stored in group /topologies/plex/dms/Velocity
```

After setting the runtime mesh DM name to `uw_mesh`, the reload reached
`DMPlexSectionLoad(...)` but failed with:

```text
Nonconforming object sizes
SF roots 6421 < pEnd 47112
```

The same class of failure occurred with:

- `mesh.sf`
- `mesh.sf0`
- `mesh.dm.getDefaultSF()`

This strongly indicates that the correct PETSc topology-load SF is not being
used or not being composed correctly after distribution.

## Step-By-Step Implementation Plan

### Step 1: Add Minimal Unit-Level Checkpoint Reload Test

Create a small test that does not depend on the full spherical benchmark.

Test shape:

1. Create a small mesh under MPI.
2. Create one continuous scalar variable.
3. Fill it with deterministic values.
4. Write `mesh.write_checkpoint(...)`.
5. Reload mesh from the written checkpoint mesh file.
6. Create the same mesh variable.
7. Load the variable with the new checkpoint helper.
8. Compare values against the original field.

Expected outcome:

- Test passes with `mpirun -np 1`.
- Test passes with `mpirun -np 2`.
- No KDTree path is used.

### Step 2: Add Vector Variable Coverage

Extend the test to include a vector variable.

Expected outcome:

- Component ordering is preserved.
- Local and global vectors are valid after reload.
- `mesh_var.data` matches expected values.

### Step 3: Add Continuous And Discontinuous Coverage

Test both:

- continuous variables
- discontinuous variables

Expected outcome:

- Continuous variables reload exactly.
- Discontinuous variables reload exactly.
- No coordinate ambiguity appears for discontinuous fields.

### Step 4: Test Same-Rank Reload

Run write and reload with the same MPI rank count.

Examples:

```bash
mpirun -np 1 pytest tests/parallel/test_checkpoint_reload.py
mpirun -np 2 pytest tests/parallel/test_checkpoint_reload.py
mpirun -np 4 pytest tests/parallel/test_checkpoint_reload.py
```

Expected outcome:

- Same-rank reload works for scalar and vector variables.
- Global vector sizes and section sizes match.

### Step 5: Test Different-Rank Reload

Run write and reload with different MPI rank counts.

Examples:

```bash
mpirun -np 1 python write_checkpoint_case.py
mpirun -np 2 python reload_checkpoint_case.py

mpirun -np 2 python write_checkpoint_case.py
mpirun -np 4 python reload_checkpoint_case.py
```

Expected outcome:

- Reload works when PETSc can redistribute correctly.
- If PETSc cannot support a specific rank-change path, document the limitation
  explicitly.

### Step 6: Validate Storage Version 3.0.0

After writing checkpoint files, inspect them:

```bash
h5dump -A restart.mesh.0.h5 | grep dmplex_storage_version
h5dump -A restart.checkpoint.00000.h5 | grep dmplex_storage_version
```

Expected outcome:

```text
dmplex_storage_version = "3.0.0"
```

### Step 7: Validate Against Spherical Benchmark

After the unit tests pass, validate with a small spherical benchmark case.

Test setup:

```bash
mpirun -np 8 python ex_stokes_thieulot.py -uw_cellsize 1/8
mpirun -np 8 python ex_stokes_thieulot.py -uw_cellsize 1/8 -uw_metrics_from_checkpoint_only true
```

Expected outcome:

- Checkpoint helper reloads velocity and pressure.
- Metrics match the known `read_timestep(...)` baseline.
- No KDTree construction occurs during field reload.

### Step 8: Validate At Larger Mesh Size

Run progressively larger spherical cases:

- `1/16`
- `1/32`
- `1/64`

Expected outcome:

- Memory usage during reload is lower than the KDTree path.
- Reload time is stable.
- Metrics match the existing baseline.

### Step 9: Validate High-Rank Failure Case

The motivating case is spherical `1/128` on `1000+` ranks.

Expected outcome:

- Reload does not exceed requested memory.
- Velocity and pressure reload complete.
- Boundary metric computation can proceed without KDTree memory blow-up.

## Debugging Checklist

When reload fails, inspect these items first:

- `dmplex_storage_version` in mesh and checkpoint files.
- PETSc version used to write and read the files.
- Runtime mesh DM name.
- Saved topology group name.
- Saved section DM group name.
- Saved vector name.
- Size of saved `order` dataset.
- Size of saved `atlasDof` and `atlasOff`.
- Size of current mesh chart, especially `pEnd`.
- Root count in the SF passed to `DMPlexSectionLoad(...)`.
- Whether mesh distribution occurred after topology load.
- Whether the topology-load SF was composed with the distribution SF.

Useful HDF5 inspection commands:

```bash
h5ls -r restart.mesh.0.h5
h5ls -r restart.checkpoint.00000.h5
h5dump -A restart.mesh.0.h5
h5dump -A restart.checkpoint.00000.h5
```

Useful PETSc-level checks:

```python
mesh.dm.getName()
mesh.dm.getChart()
mesh.sf0.getGraph()
mesh.sf.getGraph()
mesh.dm.getDefaultSF().getGraph()
```

## Acceptance Criteria

The implementation is complete only when:

- `write_checkpoint()` writes PETSc DMPlex HDF5 storage version `3.0.0`.
- UW3 exposes a public checkpoint reload helper for mesh variables.
- The helper uses PETSc section/vector metadata, not KDTree remapping.
- Scalar and vector variables reload correctly.
- Continuous and discontinuous variables reload correctly.
- MPI reload works at least for same-rank runs.
- Different-rank support is either working or clearly documented.
- Spherical benchmark metrics match the previous baseline.
- The benchmark scripts no longer need manual PETSc reload logic.

## Expected Final Outcome

After this work, benchmark postprocessing should be able to do:

```python
mesh = uw.discretisation.Mesh("restart.mesh.0.h5", ...)
v_soln.load_from_checkpoint("restart.checkpoint.00000.h5", "Velocity")
p_soln.load_from_checkpoint("restart.checkpoint.00000.h5", "Pressure")
```

This should provide exact FE restart behavior, avoid high-memory KDTree reload,
and make large spherical benchmark postprocessing practical on high-rank jobs.
