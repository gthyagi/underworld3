"""Native Poisson reuse must preserve source linearity and the unique zero state.

A small Cartesian component fixture supplies genuinely interior triangular
facets. This tests the real sheet assembly/native DM/SNES/KSP path without a
spherical benchmark, gravity mesh or analytical spherical accuracy claim.
"""

from enum import Enum
import json
import os

from mpi4py import MPI
import numpy as np
import pytest
import underworld3 as uw


pytestmark = pytest.mark.level_1


@pytest.mark.tier_b
@pytest.mark.parametrize("snes_type", ("newtonls", "ksponly"))
@pytest.mark.parametrize("boundary_condition", ("dirichlet", "robin"))
def test_repeated_sheet_sources_initialize_operator_and_preserve_linearity(monkeypatch, boundary_condition, snes_type):
    from underworld3.utilities._spherical_interface_load import _vertex_coordinates

    root_pid = uw.mpi.comm.bcast(os.getpid() if uw.mpi.rank == 0 else None, root=0)
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(1., 1., 1.), maxCoords=(2., 2., 2.), cellSize=.25,
        regular=True, qdegree=2, filename=f"/tmp/uw-native-sheet-reuse-{boundary_condition}-{snes_type}-{root_pid}.msh")
    facets = []
    first, last = mesh.dm.getHeightStratum(1)
    for facet in range(first, last):
        vertices = _vertex_coordinates(mesh, facet)
        # All vertices strictly inside the convex box guarantee an interior
        # facet. Local support counts can omit a neighbour after partitioning.
        if (np.mean(vertices[:, 0]) < 1.6 and np.mean(vertices[:, 2]) > 1.5
                and np.min(vertices) > 1.+1e-12 and np.max(vertices) < 2.-1e-12):
            facets.append(facet)
    assert uw.mpi.comm.allreduce(len(facets), op=MPI.SUM) > 0
    original_facets = mesh._boundary_facets
    monkeypatch.setattr(mesh, "_boundary_facets", lambda name:
        facets if name == "ReuseSource" else original_facets(name))
    labels = {name: value.value for name, value in mesh.boundaries.__members__.items()}
    labels["ReuseSource"] = 91234
    monkeypatch.setattr(mesh, "boundaries", Enum("ReuseBoundaries", labels))

    potential = uw.discretisation.MeshVariable("NativeReusePotential", mesh, 1, degree=2)
    load = uw.discretisation.MeshVariable("NativeReuseLoad", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=potential)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1
    poisson.f = 0
    poisson.tolerance = 1e-10
    options = ({"ksp_type": "preonly", "pc_type": "lu"} if uw.mpi.size == 1
               else {"ksp_type": "fgmres", "pc_type": "gamg"})
    # A linear base solve must assemble its Jacobian even when f=0 and u=0.
    # Exercise both the public default and the explicit linear driver setting.
    poisson.petsc_options["snes_type"] = snes_type
    for key, value in options.items():
        poisson.petsc_options[key] = value
    for wall in ("Top", "Bottom", "Left", "Right", "Front", "Back"):
        if boundary_condition == "dirichlet":
            poisson.add_dirichlet_bc([0.], wall)
        else:
            # Positive Robin coefficient gives a unique homogeneous zero base;
            # this tests lifecycle/linearity, not spherical exterior accuracy.
            poisson.add_natural_bc([2*potential.sym[0]], wall)
    registered_fields = len(mesh.vars)
    fields, identities = [], []
    for amplitude in (1., -.7, 1., 0.):
        result = poisson.solve_linear_spherical_sheet_sources(
            load_variable=load, harmonic_degree=2,
            harmonic_normalization="real_orthonormal",
            sheet_densities={"ReuseSource": amplitude},
            gravitational_constant=1/(4*np.pi), tolerance=1e-10,
            solver_options=options)
        assert result["solver_reason"] > 0
        assert len(mesh.vars) == registered_fields
        fields.append(np.asarray(potential.array[:, 0, 0]).copy())
        operator, preconditioner, _ = poisson.snes.getJacobian()
        identities.append((poisson.dm.handle, poisson.snes.handle,
                           operator.handle, preconditioner.handle))
    assert all(identity == identities[0] for identity in identities)
    norm = float(uw.mpi.comm.allreduce(np.max(np.abs(fields[0]), initial=0), op=MPI.MAX))
    assert norm > 1e-8, "Nonzero interior load must produce a measurable potential"
    absolute = 2e-8*norm + 2e-11
    errors = []
    for actual, expected in ((fields[1], -.7*fields[0]),
                             (fields[2], fields[0]), (fields[3], np.zeros_like(fields[3]))):
        error = float(uw.mpi.comm.allreduce(np.max(np.abs(actual-expected), initial=0), op=MPI.MAX))
        errors.append(error)
        assert error <= absolute, (error, absolute, norm)
    if uw.mpi.rank == 0:
        print("native_sheet_reuse=" + json.dumps(dict(boundary_condition=boundary_condition,
            snes_type=snes_type, mpi_size=uw.mpi.size, field_count=registered_fields, native_handles_stable=True,
            baseline_max_abs=norm, maximum_absolute_errors=errors, absolute_tolerance=absolute)))
