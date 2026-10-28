# Native Mac Docker-matched development environment

The `amr-dev` environment is an Apple Silicon profile for debugging UW3 without
Docker's virtual-machine memory ceiling. Its numerical Python package versions
are pinned in `feature.mac-docker` in `pixi.toml`; Gmsh uses the official stable
4.15.2 wheel. Other environment definitions remain unchanged.

PETSc/petsc4py 3.25.4 use 32-bit indices, real double precision and the AMR,
partitioning and direct-solver libraries enabled in the Mac Docker image.
Native Open MPI 5.x, MPI-enabled HDF5 1.14.6 and h5py 3.16.0 come from one
Conda stack. MPI is deliberately newer than Docker's Open MPI 4.1.1.
Native compilers and BLAS builds differ from Linux; matching package versions
does not imply identical binaries or floating-point results.

From the local UW3 checkout:

```bash
pixi install -e amr-dev
pixi run -e amr-dev stack-build
pixi run -e amr-dev build
```

The stack build requires Apple Command Line Tools and the existing Homebrew
`gfortran`. It uses the installed macOS SDK consistently across compiler wrappers.
PETSc installs inside `.pixi/envs/amr-dev/opt/petsc`; its source and object cache
are under `.pixi/build/mac-petsc-3.25.4`. It does not modify the older PETSc
checkout shared by other worktrees. JIT extensions use an environment-local cache,
so binaries from an older PETSc prefix are not reused. The exact Python binding requirement is in
`petsc-custom/mac-docker-bindings.txt`.

The build applies `petsc-custom/patches/petsc-3.25.4-darwin-commons.patch` to
PETSc's macOS linker configuration. On this native Open MPI stack, the automatically
added `-Wl,-commons,use_dylibs` flag caused MUMPS's distributed-RHS solve to return
zeros despite a positive KSP convergence reason. An independent C reproducer
passed with the same static libraries and with a shared library omitting that
flag; enabling the flag reproduced the failure. HDF5-first imports additionally
required `-Wl,-flat_namespace`, matching this Open MPI package's Fortran wrapper.
The patch replaces the COMMON-resolution flag with that namespace flag and
leaves solver options unchanged. The build finishes with serial and two-rank
manufactured-solution checks for MUMPS and SuperLU_DIST, including true residuals
and repeated solves. To repeat them with eight ranks:

```bash
pixi run -e amr-dev mpiexec -n 8 python petsc-custom/check-mac-docker-stack.py
```

The reinstall was verified on local `development` source `7222caf34c1e` with
serial and eight-rank manufactured solves, UW3 unit-square area integration,
parallel HDF5 writes/reads, and the core import tests. These checks validate
the installation; they do not establish convergence of a benchmark model.

Rebuild UW3 after changing source branches. Use the matching MPI launcher from
this environment, and verify serial/MPI imports, parallel HDF5 and small numerical
checks before interpreting benchmark results. Native processes share the Mac's
physical RAM, memory compression and swap with other applications.
