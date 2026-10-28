#!/usr/bin/env bash
# Use one installed SDK and compiler family for PETSc, UW3 and runtime JIT code.
export SDKROOT="$(xcrun --show-sdk-path)"
export MACOSX_DEPLOYMENT_TARGET=12.0
export OMPI_CC=/usr/bin/clang
export OMPI_CXX=/usr/bin/clang++
export OMPI_FC=/opt/homebrew/bin/gfortran
export OMPI_CFLAGS="--sysroot=${SDKROOT} -mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET}"
export OMPI_CXXFLAGS="${OMPI_CFLAGS}"
export OMPI_FCFLAGS="${OMPI_CFLAGS} -I${CONDA_PREFIX}/include"
export CC="${CONDA_PREFIX}/bin/mpicc"
export CXX="${CONDA_PREFIX}/bin/mpicxx"
export FC="${CONDA_PREFIX}/bin/mpifort"
# Package-specific versioning tolerates non-version publication tags in the repo.
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_UNDERWORLD3="0.0.0+g$(git -C "${PIXI_PROJECT_ROOT}" rev-parse --short=12 HEAD)"
