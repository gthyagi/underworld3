#!/usr/bin/env bash
# Build the Docker PETSc version against native Pixi MPI/HDF5, independently
# of the older PETSc checkout shared by other worktrees.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
env_prefix="${project_root}/.pixi/envs/amr-dev"
petsc_prefix="${env_prefix}/opt/petsc"
build_root="${project_root}/.pixi/build/mac-petsc-3.25.4"
petsc_source="${build_root}/petsc-3.25.4"
scotch_archive="${project_root}/petsc-custom/patches/scotch-7.0.10-c23-fix.tar.gz"

[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]]
[[ "${CONDA_PREFIX:?Activate amr-dev first}" == "${env_prefix}" ]]
[[ -x "${env_prefix}/bin/mpicc" && -x /opt/homebrew/bin/gfortran ]]
[[ -f "${scotch_archive}" ]]
mkdir -p "${build_root}"
source "${project_root}/petsc-custom/activate-mac-docker.sh"

if [[ ! -f "${petsc_prefix}/include/petscversion.h" || ! -f "${petsc_prefix}/.darwin-mpi-common-fixed" ]]; then
    if [[ ! -f "${petsc_source}/configure" ]]; then
        curl --fail --location --retry 3 --retry-delay 2 \
            https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.25.4.tar.gz \
            --output "${build_root}/petsc-3.25.4.tar.gz"
        tar -xzf "${build_root}/petsc-3.25.4.tar.gz" -C "${build_root}"
        shasum -a 256 "${build_root}/petsc-3.25.4.tar.gz" > "${build_root}/source.sha256"
    fi

    if grep -q -- "'-Wl,-commons,use_dylibs'" "${petsc_source}/config/BuildSystem/config/setCompilers.py"; then
        patch -d "${petsc_source}" -p1 < "${project_root}/petsc-custom/patches/petsc-3.25.4-darwin-commons.patch"
    fi
    export PETSC_DIR="${petsc_source}" PETSC_ARCH=arch-mac-docker-int32
    export PKG_CONFIG_PATH="${env_prefix}/lib/pkgconfig"
    cd "${petsc_source}"
    if [[ ! -f "${petsc_prefix}/include/petscversion.h" ]]; then
      "${env_prefix}/bin/python" ./configure \
        --prefix="${petsc_prefix}" \
        --with-cc="${env_prefix}/bin/mpicc" \
        --with-cxx="${env_prefix}/bin/mpicxx" \
        --with-fc="${env_prefix}/bin/mpifort" \
        --with-64-bit-indices=0 --with-64-bit-blas-indices=0 \
        --with-scalar-type=real --with-precision=double \
        --with-debugging=0 --with-shared-libraries=1 --with-cxx-dialect=C++11 \
        --COPTFLAGS='-g -O3' --CXXOPTFLAGS='-g -O3' --FOPTFLAGS='-g -O3' \
        --useThreads=0 --with-x=0 --with-make-np=4 \
        --with-hdf5=1 --with-hdf5-dir="${env_prefix}" --download-hdf5=0 \
        --with-petsc4py=0 --with-slepc4py=0 \
        --download-pragmatic=1 --with-pragmatic=1 --download-parmmg=1 \
        --download-superlu=1 --download-eigen=1 --download-metis=1 \
        --download-parmetis=1 --download-mumps=1 --download-scalapack=1 \
        --download-hypre=1 --download-superlu_dist=1 --download-mmg=1 \
        '--download-mmg-cmake-arguments=-DMMG_INSTALL_PRIVATE_HEADERS=ON -DUSE_SCOTCH=OFF' \
        --download-ptscotch="${scotch_archive}" --download-slepc=1 \
        --download-fblaslapack=1 --download-zlib=1 --download-ctetgen=1 --download-triangle=1
    else
        # Repair an existing build by relinking, preserving compiled dependencies.
        "${env_prefix}/bin/python" - "${petsc_source}/${PETSC_ARCH}" <<'PY'
from pathlib import Path
import sys
arch = Path(sys.argv[1])
variables = arch / "lib/petsc/conf/petscvariables"
text = variables.read_text().replace("-Wl,-commons,use_dylibs", "")
lines = text.splitlines()
for i, line in enumerate(lines):
    if line.startswith(("PCC_LINKER_FLAGS =", "CXX_LINKER_FLAGS =", "FC_LINKER_FLAGS =")) and "-Wl,-flat_namespace" not in line:
        lines[i] = line + " -Wl,-flat_namespace"
variables.write_text("\n".join(lines) + "\n")
library = arch / "lib/libpetsc.3.25.4.dylib"
if library.exists():
    library.rename(library.with_suffix(".dylib.before-commons-fix"))
PY
    fi
    make -j4 all
    make install
    touch "${petsc_prefix}/.darwin-mpi-common-fixed"
fi

export PETSC_DIR="${petsc_prefix}" PETSC_ARCH=''
cd "${project_root}"
"${env_prefix}/bin/python" -m pip install --no-deps --no-build-isolation --no-cache-dir \
    --no-binary=petsc4py -r petsc-custom/mac-docker-bindings.txt
"${env_prefix}/bin/python" -c 'from petsc4py import PETSc; import numpy as np; assert PETSc.Sys.getVersion()==(3,25,4); assert np.dtype(PETSc.IntType).itemsize==4; print("PETSC_STACK_READY", PETSc.Sys.getVersion())'
"${env_prefix}/bin/python" petsc-custom/check-mac-docker-stack.py
"${env_prefix}/bin/mpiexec" -n 2 "${env_prefix}/bin/python" petsc-custom/check-mac-docker-stack.py
