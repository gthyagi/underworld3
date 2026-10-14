"""Collective restrictions of the current native spherical solver adapters."""
from mpi4py import MPI
import underworld3 as uw


def require_native_solver_mesh(mesh):
    """Fail before native world-communicator solver construction or empty charts.

    The trace integrators themselves use the mesh communicator. Native UW3
    solver setup still constructs SNES on COMM_WORLD, and mesh/field setup is
    not empty-chart safe. Do not imply wider support from local integrators.
    """
    comm = mesh.dm.getComm().tompi4py()
    if MPI.Comm.Compare(comm, uw.mpi.comm) not in (MPI.IDENT, MPI.CONGRUENT):
        raise ValueError('Native spherical solver adapters require a mesh communicator congruent to UW3 COMM_WORLD.')
    start, end = mesh.dm.getChart()
    if comm.allreduce(end == start, op=MPI.LOR):
        raise ValueError('Native spherical solver adapters require nonempty mesh charts on every rank; empty interface/source strata are supported.')
