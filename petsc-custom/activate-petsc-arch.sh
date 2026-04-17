#!/usr/bin/env bash
# activate-petsc-arch.sh — sourced by pixi on environment activation.
#
# Dynamically sets PETSC_ARCH for AMR environments (custom PETSc builds)
# by combining the active pixi environment name with the PETSc version
# pinned in petsc-custom/.petsc-version.
#
# This replaces hardcoded PETSC_ARCH entries in pixi.toml so that
# switching PETSc versions (via ./uw petsc switch <tag>) does not require
# editing tracked configuration.

_uw_mpi=""
_uw_suffix=""
case "${PIXI_ENVIRONMENT_NAME:-}" in
    amr|amr-runtime|amr-dev)
        case "$(uname -s)" in
            Darwin) _uw_mpi="openmpi" ;;
            *)      _uw_mpi="mpich" ;;
        esac ;;
    amr-mpich|amr-mpich-dev)     _uw_mpi="mpich" ;;
    amr-openmpi|amr-openmpi-dev) _uw_mpi="openmpi" ;;
    amr-debug)                   _uw_mpi="openmpi"; _uw_suffix="-debug" ;;
esac

if [ -n "$_uw_mpi" ]; then
    _uw_ver="4"
    _uw_ver_file="${PIXI_PROJECT_ROOT:-.}/petsc-custom/.petsc-version"
    [ -f "$_uw_ver_file" ] && _uw_ver=$(cat "$_uw_ver_file" 2>/dev/null || echo "4")

    export PETSC_ARCH="petsc-${_uw_ver}-uw-${_uw_mpi}${_uw_suffix}"
fi

unset _uw_mpi _uw_suffix _uw_ver _uw_ver_file
