# PETSc int64 compatibility workflow

Work on `feature/petsc-int64-compatible-uw3`. This branch contains the verified
integer-width fixes originally developed on `bugfix/petsc-int64-boundary-types`.
The redundant bugfix branch has been deleted; its commits remain on this branch.
Do not rebuild the shared native Pixi installation during container debugging.

## Baseline and failure triage

The original Gadi runtime images use PETSc 3.25.4, NumPy 2.4.6, Cython 3.2.3 and
parallel HDF5. Their UW3 source is `ecdec1d023f6b6d0f758a4f8791dab47820bebb8`.

| Validation | Result |
|---|---|
| Original int32 Level 1, job 180501381 | 2013 passed, 13 failed |
| Original int64 Level 1, job 180502398 | 1888 passed, 138 failed, 4 collection errors |
| Corrected int64 failed-test rerun, job 180503500 | 125 passed, 13 failed, no collection errors |
| Integer-width regression, job 180502919 | Five analytic tests passed for each index width; four-rank Stokes/HDF5 passed for both |
| Corrected int64 two-node validation, job 180502950 | Passed; maximum velocity difference from native: 0 |

The 13 remaining failures in the failed-test rerun are also present in the
original int32 run. Twelve require the missing `trame-pyvista` backend plugin.
The other is an outcropping-sheet fixture whose cavity reaches the right wall.

### Integer-width defect

A 32-bit array `[0, 1]`, passed as a 64-bit `PetscInt *`, is read as the invalid
component `4294967296`. Boundary label buffers and generated callbacks also
assumed a 32-bit integer width. The fixes use PETSc header declarations,
`PETSc.IntType` arrays and matching Cython memory views. Regression tests compare
2D and 3D Couette solutions with `u=(y, 0, ...)`, including inferred and legacy
explicit components and natural traction conditions.

### Dependency defect

PyVista 0.49 requires `trame-pyvista` for UW3's selected Trame backend. Install the
plugin through the container recipe so pip resolves compatible Trame versions.
For experiments, use a separate dependency overlay; do not install into the SIF
or native environment. The tested plugin is 0.1.7; its metadata selects Trame
3.13.2, client 3.13.2, common 1.2.4, server 3.12.5, VTK widgets 2.11.16 and
Vuetify widgets 3.2.5. Installing only the plugin with `--no-deps` against the
original Trame 4 packages does not provide its declared compatible stack.

### Mesh fixture defect

The sheet cavity includes whole vertex stars and shell growth. On the Linux
mesh, the original strike half-extent 0.22 consumes a boundary triangle on
`x=1`, away from the top outcrop. A half-extent of 0.15 clears the side wall.
The test retains the manifold, top boundary-label and P2 polynomial exactness
assertions. The production refusal of unintended wall contact remains intact.

## Development loop

1. Preserve the failed job's logs and JUnit XML. Extract exact failed test node
   IDs, including parameter suffixes. Record collection errors separately.
2. Use detached source worktrees and separate installed-package directories for
   int32 and int64. Their PETSc paths inside the images are identical, so they
   must not share a compiled UW3 build cache.
3. Build UW3 inside its matching SIF with `pip install --no-index
   --no-build-isolation --no-deps --target <isolated-package-directory> .`.
   Prepare Python dependencies on a login node; compute nodes require no network.
4. Verify the imported `uw.__file__`, PETSc version and index width before testing.
   Put the dependency overlay and matching UW3 package before the image packages
   on `PYTHONPATH`. Use the existing host-MPI injection wrapper for compute jobs.
5. Rerun the failed tests, then investigate one representative test per remaining
   cause. Short, lightweight serial checks may run on the login node. Use a small
   interactive or batch `express` allocation for longer solves and compilation.
   The TCP-only MPI settings used for login diagnostics are not production settings.
6. Keep exact numerical assertions and production guards. Check every fix with
   both integer widths using the same source commit. Record each focused fix in
   its own commit on the feature branch.
7. Run both complete Level 1 suites sequentially, followed by regression and
   single-node/two-node MPI checks. Investigate unexpected failures before publishing.

The maintained selection is:

```bash
python3 -m pytest -c tests/pytest.ini tests \
  -m 'level_1 and not level_2 and not level_3 and not mpi' \
  --timeout=120 --dist loadfile -n 4 --junitxml=level1.xml
```

Capture the source commit, installed-package commit, dependency versions, image
reference, job ID, exit status and complete test counts with each run. A Level 1
pass does not replace MPI validation.

## Rebuild the UW3 runtime without rebuilding PETSc

The Gadi image workflow accepts an optional `petsc_image` input. Leave it empty
to build PETSc, or reuse the existing dependency image for source-only UW3 fixes:

```bash
gh workflow run gadi-container-images.yml --repo gthyagi/underworld3 \
  --ref feature/petsc-int64-compatible-uw3 \
  -f variant=int64 \
  -f petsc_image=ghcr.io/gthyagi/uw3-petsc:petsc3.25.4-int64-ecdec1d023f6b6d0f758a4f8791dab47820bebb8
```

Repeat with `int32` and its matching dependency image after the int64 build is
verified. The runtime receives a new tag containing the tested UW3 commit; the
PETSc image is reused. CI checks imports, versions, index width, parallel HDF5,
analytic 2D/3D Stokes solutions, the visualisation backend and retained build
tools before publishing. Pull new SIF files with new filenames, then validate
those exact files on Gadi before using them for benchmark jobs.

Mac ARM64 images remain a separate architecture. Neither the amd64 PETSc image
nor its runtime can replace the Mac image.
