"""Unsupported native solver communicators/charts fail before solver setup."""
from types import SimpleNamespace
from mpi4py import MPI
from petsc4py import PETSc
import pytest
import underworld3 as uw
from underworld3.utilities._spherical_solver_contract import require_native_solver_mesh
from underworld3.postprocessing.spherical_potential import solve_spherical_potential


pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

def test_world_mesh_with_one_empty_chart_is_rejected_collectively():
    dm=PETSc.DMPlex().create(comm=uw.mpi.comm)
    try:
        dm.setChart(0,0 if uw.mpi.rank==uw.mpi.size-1 else 1)
        with pytest.raises(ValueError,match='nonempty mesh charts'):
            require_native_solver_mesh(SimpleNamespace(dm=dm))
    finally:dm.destroy()


def test_duplicate_world_communicator_is_accepted():
    comm=uw.mpi.comm.Dup()
    dm=PETSc.DMPlex().create(comm=comm)
    try:
        dm.setChart(0,1)
        require_native_solver_mesh(SimpleNamespace(dm=dm))
    finally:
        dm.destroy();comm.Free()


@pytest.mark.skipif(uw.mpi.size==1,reason='Requires a proper subcommunicator')
def test_subcommunicators_rejected_before_world_collectives():
    comm=uw.mpi.comm.Split(uw.mpi.rank%2,uw.mpi.rank)
    dm=PETSc.DMPlex().create(comm=comm)
    try:
        dm.setChart(0,1)
        with pytest.raises(ValueError,match='communicator congruent'):
            require_native_solver_mesh(SimpleNamespace(dm=dm))
    finally:
        dm.destroy();comm.Free()


def test_mixed_degree_poisson_rejected_before_field_allocation():
    dm=PETSc.DMPlex().create(comm=uw.mpi.comm)
    try:
        dm.setChart(0,1)
        with pytest.raises(ValueError,match='must share harmonic_degree'):
            solve_spherical_potential(mesh=SimpleNamespace(dm=dm),field_name='Unused',
                harmonic_degree=2,sheet_modes=[dict(harmonic_degree=4,sheet_densities={'Internal':1.})],
                radius_inner=.55,radius_outer=1.,radius_exterior=1.5,gravitational_constant=.01)
    finally:dm.destroy()
