r"""Spherical harmonic loads and projections on continuous P2 triangular traces.

This explicitly restricted API requires a 3D simplicial mesh with straight
triangular interface facets and a continuous P2 scalar space. Owned facets
contribute once; shared DOF additions use PETSc ADD_VALUES. It is not an
element-generic trace API. Mesh construction and output remain caller-owned.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from contextlib import ExitStack

from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import underworld3 as uw

from underworld3.maths.spherical_harmonics import (
    AXISYMMETRIC_UNNORMALIZED,
    harmonic_norm,
    real_harmonic_values,
    validate_harmonic,
)


@dataclass(frozen=True)
class InterfaceLoadDiagnostics:
    interface: str
    harmonic_degree: int
    harmonic_order: int
    harmonic_component: str
    harmonic_normalization: str
    density_coefficient: float
    source_factor: float
    owned_facets: int
    ghost_facets: int
    global_nonzero_dofs: int
    global_l2_norm: float
    uncorrected_integrated_source: float
    removed_mean_density: float
    integrated_source: float
    harmonic_moment: float
    coordinate_value_sha256: str


def _vertex_coordinates(mesh, facet: int) -> np.ndarray:
    vertex_start, vertex_end = mesh.dm.getDepthStratum(0)
    section = mesh.dm.getCoordinateSection()
    coordinates = np.asarray(mesh.dm.getCoordinatesLocal().array).reshape(
        -1, mesh.cdim
    )
    vertices = []
    for point in mesh.dm.getTransitiveClosure(facet)[0]:
        point = int(point)
        if vertex_start <= point < vertex_end:
            offset = section.getOffset(point) // mesh.cdim
            vertices.append(coordinates[offset])
    if len(vertices) != 3:
        raise RuntimeError(f"Interface facet {facet} has {len(vertices)} vertices.")
    return np.asarray(vertices, dtype=float)


def _triangle_p2_load(
    vertices: np.ndarray,
    *,
    harmonic_degree: int,
    harmonic_order: int,
    harmonic_component: str,
    harmonic_normalization: str,
    source_amplitude: float,
    quadrature_points: int,
) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    cross_norm = np.linalg.norm(
        np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    )
    contribution = np.zeros(6, dtype=float)
    for u, weight_u in zip(unit_nodes, unit_weights, strict=True):
        for v, weight_v in zip(unit_nodes, unit_weights, strict=True):
            barycentric = np.array(
                [(1.0 - u) * (1.0 - v), u, (1.0 - u) * v],
                dtype=float,
            )
            position = barycentric @ vertices
            radius = np.linalg.norm(position)
            harmonic = (
                1.0
                if harmonic_degree == 0
                else float(
                    real_harmonic_values(
                        position[None, :],
                        degree=harmonic_degree,
                        order=harmonic_order,
                        component=harmonic_component,
                        normalization=harmonic_normalization,
                    )[0]
                )
            )
            l0, l1, l2 = barycentric
            basis = np.array(
                [
                    l0 * (2.0 * l0 - 1.0),
                    l1 * (2.0 * l1 - 1.0),
                    l2 * (2.0 * l2 - 1.0),
                    4.0 * l0 * l1,
                    4.0 * l1 * l2,
                    4.0 * l2 * l0,
                ]
            )
            contribution += (
                source_amplitude
                * harmonic
                * basis
                * cross_norm
                * (1.0 - u)
                * weight_u
                * weight_v
            )
    return contribution


def _triangle_p2_weighted_moment(
    vertices: np.ndarray,
    nodal_values: np.ndarray,
    *,
    harmonic_degree: int,
    harmonic_order: int,
    harmonic_component: str,
    harmonic_normalization: str,
    quadrature_points: int,
) -> float:
    """Integrate a P2 trace field times one real harmonic on a facet."""

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    cross_norm = np.linalg.norm(
        np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    )
    moment = 0.0
    for u, weight_u in zip(unit_nodes, unit_weights, strict=True):
        for v, weight_v in zip(unit_nodes, unit_weights, strict=True):
            barycentric = np.array(
                [(1.0 - u) * (1.0 - v), u, (1.0 - u) * v],
                dtype=float,
            )
            position = barycentric @ vertices
            radius = np.linalg.norm(position)
            harmonic = float(
                real_harmonic_values(
                    position[None, :],
                    degree=harmonic_degree,
                    order=harmonic_order,
                    component=harmonic_component,
                    normalization=harmonic_normalization,
                )[0]
            )
            l0, l1, l2 = barycentric
            basis = np.array(
                [
                    l0 * (2.0 * l0 - 1.0),
                    l1 * (2.0 * l1 - 1.0),
                    l2 * (2.0 * l2 - 1.0),
                    4.0 * l0 * l1,
                    4.0 * l1 * l2,
                    4.0 * l2 * l0,
                ]
            )
            moment += (
                float(np.dot(basis, nodal_values))
                * harmonic
                * cross_norm
                * (1.0 - u)
                * weight_u
                * weight_v
            )
    return moment


def _coordinate_value_payload(variable) -> tuple[np.ndarray, np.ndarray]:
    comm = variable.mesh.dm.getComm().tompi4py()
    coordinates = np.asarray(variable.coords, dtype=float).reshape(-1, 3)
    values = np.asarray(variable.data[:, 0], dtype=float)
    gathered = comm.gather((coordinates, values), root=0)
    error = None
    if comm.rank == 0:
        merged: dict[tuple[float, ...], list[float]] = {}
        for rank_coordinates, rank_values in gathered:
            for coordinate, value in zip(
                rank_coordinates, rank_values, strict=True
            ):
                key = tuple(np.round(coordinate, 14))
                merged.setdefault(key, []).append(float(value))
        ordered = sorted(merged)
        unique_coordinates = np.asarray(ordered, dtype=np.float64).reshape(-1, 3)
        unique_values = np.asarray(
            [np.mean(merged[key]) for key in ordered], dtype=np.float64
        )
        maximum_duplicate_disagreement = max(
            (
                max(values) - min(values)
                for values in merged.values()
                if len(values) > 1
            ),
            default=0.0,
        )
        if maximum_duplicate_disagreement > 5.0e-13:
            error = f"Shared interface-load DOFs disagree by {maximum_duplicate_disagreement:.3e}."
    else:
        unique_coordinates = None
        unique_values = None
    error = comm.bcast(error, root=0)
    if error:
        raise AssertionError(error)
    unique_coordinates = comm.bcast(unique_coordinates, root=0)
    unique_values = comm.bcast(unique_values, root=0)
    return unique_coordinates, unique_values


def assemble_spherical_harmonic_sheet_load(
    *,
    mesh,
    load_variable,
    interface: str,
    harmonic_degree: int,
    harmonic_order: int = 0,
    harmonic_component: str = "cos",
    harmonic_normalization: str = AXISYMMETRIC_UNNORMALIZED,
    density_coefficient: float,
    gravitational_constant: float = 1.0 / (4.0 * np.pi),
    quadrature_points: int = 8,
) -> tuple[PETSc.Vec, InterfaceLoadDiagnostics]:
    """Assemble one real spherical-harmonic density-sheet load vector."""

    if mesh.dim != 3:
        raise ValueError("Spherical trace operations require a 3D mesh.")
    comm = mesh.dm.getComm().tompi4py()
    validate_harmonic(
        harmonic_degree,
        harmonic_order,
        harmonic_component,
        harmonic_normalization,
    )
    if load_variable.degree != 2 or not load_variable.continuous:
        raise ValueError("load_variable must use continuous P2 elements.")
    if load_variable.mesh is not mesh or load_variable.num_components != 1:
        raise ValueError("load_variable must be a scalar on the supplied mesh.")
    if not np.all(np.isfinite((density_coefficient, gravitational_constant))):
        raise ValueError("Density and gravitational constant must be finite.")
    if quadrature_points < 2:
        raise ValueError("quadrature_points must be at least two.")
    if interface not in mesh.boundaries.__members__:
        raise ValueError(f"Unknown interface {interface!r}.")

    ilocal = None
    if comm.size > 1:
        _nroots, ilocal, _iremote = mesh.dm.getPointSF().getGraph()
    ghost_points = set() if ilocal is None else {int(point) for point in ilocal}
    facets = [int(facet) for facet in mesh._boundary_facets(interface)]
    owned_facets = [facet for facet in facets if facet not in ghost_points]
    ghost_facets = [facet for facet in facets if facet in ghost_points]

    load_variable._set_vec(available=True)
    local_values = load_variable.vec.array.reshape(-1, 1)
    local_values[:, 0] = 0.0
    with ExitStack() as cleanup:
        constant_local = load_variable.vec.duplicate()
        cleanup.callback(constant_local.destroy)
        constant_local.set(0.0)
        local_constant_values = constant_local.array.reshape(-1, 1)
        tree = load_variable._get_kdtree() if owned_facets else None
        source_factor = 4.0 * np.pi * float(gravitational_constant)
        source_amplitude = source_factor * float(density_coefficient)

        for facet in owned_facets:
            vertices = _vertex_coordinates(mesh, facet)
            node_coordinates = np.vstack(
                [
                    vertices,
                    0.5 * (vertices[0] + vertices[1]),
                    0.5 * (vertices[1] + vertices[2]),
                    0.5 * (vertices[2] + vertices[0]),
                ]
            )
            distances, rows = tree.query(node_coordinates, sqr_dists=False)
            if np.max(distances) > 1.0e-10:
                raise RuntimeError(
                    f"Could not map P2 trace nodes for facet {facet}: "
                    f"maximum distance={np.max(distances):.3e}."
                )
            contribution = _triangle_p2_load(
                vertices,
                harmonic_degree=harmonic_degree,
                harmonic_order=harmonic_order,
                harmonic_component=harmonic_component,
                harmonic_normalization=harmonic_normalization,
                source_amplitude=source_amplitude,
                quadrature_points=quadrature_points,
            )
            np.add.at(local_values[:, 0], rows, contribution)
            constant_contribution = _triangle_p2_load(
                vertices,
                harmonic_degree=0,
                harmonic_order=0,
                harmonic_component="cos",
                harmonic_normalization=AXISYMMETRIC_UNNORMALIZED,
                source_amplitude=1.0,
                quadrature_points=quadrature_points,
            )
            np.add.at(local_constant_values[:, 0], rows, constant_contribution)

        index_set, subdm = mesh.dm.createSubDM(load_variable.field_id)
        cleanup.callback(index_set.destroy)
        cleanup.callback(subdm.destroy)
        global_load = load_variable._gvec
        constant_global = global_load.duplicate()
        cleanup.callback(constant_global.destroy)
        global_load.set(0.0)
        constant_global.set(0.0)
        subdm.localToGlobal(
            load_variable.vec,
            global_load,
            addv=PETSc.InsertMode.ADD_VALUES,
        )
        subdm.localToGlobal(
            constant_local,
            constant_global,
            addv=PETSc.InsertMode.ADD_VALUES,
        )
        uncorrected_integrated_source = float(global_load.sum())
        discrete_area = float(constant_global.sum())
        if discrete_area <= 0.0:
            raise RuntimeError(f"Interface {interface!r} has non-positive discrete area.")
        removed_mean_density = uncorrected_integrated_source / discrete_area
        global_load.axpy(-removed_mean_density, constant_global)
        subdm.globalToLocal(
            global_load,
            load_variable.vec,
            addv=PETSc.InsertMode.INSERT_VALUES,
        )

        coordinates, values = _coordinate_value_payload(load_variable)
        nonzero = np.abs(values) > 1.0e-15 * max(np.max(np.abs(values), initial=0.0), 1.0)
        nonzero_coordinates = np.ascontiguousarray(coordinates[nonzero], dtype=np.float64)
        nonzero_values = np.ascontiguousarray(values[nonzero], dtype=np.float64)
        digest = hashlib.sha256()
        digest.update(nonzero_coordinates.tobytes())
        # Shared-DOF additions may occur in a different order after partitioning.
        # Quantize well below the 1e-12 acceptance threshold so the digest records
        # a numerically identical vector rather than irrelevant last-bit ordering.
        digest.update(np.round(nonzero_values, 14).tobytes())
        radii = np.linalg.norm(nonzero_coordinates, axis=1)
        harmonic_values = real_harmonic_values(
            nonzero_coordinates,
            degree=harmonic_degree,
            order=harmonic_order,
            component=harmonic_component,
            normalization=harmonic_normalization,
        )
        diagnostics = InterfaceLoadDiagnostics(
            interface=interface,
            harmonic_degree=int(harmonic_degree),
            harmonic_order=int(harmonic_order),
            harmonic_component=harmonic_component,
            harmonic_normalization=harmonic_normalization,
            density_coefficient=float(density_coefficient),
            source_factor=source_factor,
            owned_facets=int(comm.allreduce(len(owned_facets), op=MPI.SUM)),
            ghost_facets=int(comm.allreduce(len(ghost_facets), op=MPI.SUM)),
            global_nonzero_dofs=int(np.count_nonzero(nonzero)),
            global_l2_norm=float(global_load.norm()),
            uncorrected_integrated_source=uncorrected_integrated_source,
            removed_mean_density=removed_mean_density,
            integrated_source=float(np.sum(values)),
            harmonic_moment=float(np.dot(nonzero_values, harmonic_values)),
            coordinate_value_sha256=digest.hexdigest(),
        )
        return global_load.copy(), diagnostics


def project_spherical_harmonic_trace_coefficient(
    *,
    mesh,
    variable,
    interface: str,
    interface_radius: float,
    harmonic_degree: int,
    harmonic_order: int = 0,
    harmonic_component: str = "cos",
    harmonic_normalization: str = AXISYMMETRIC_UNNORMALIZED,
    harmonic_norm_value: float | None = None,
    quadrature_points: int = 8,
) -> float:
    """Project a parent P2 trace with explicit, uniquely owned facets.

    UW3's generic boundary-integral path is retained elsewhere as a useful
    diagnostic, but this routine gives the interface-sheet benchmark an
    explicitly partition-independent coefficient recovery path matching the
    ownership rule used by :func:`assemble_axisymmetric_sheet_load`.
    """

    if variable.degree != 2 or not variable.continuous:
        raise ValueError("variable must use continuous P2 elements.")
    if variable.mesh is not mesh or variable.num_components != 1:
        raise ValueError("variable must be a scalar on the supplied mesh.")
    if mesh.dim != 3:
        raise ValueError("Spherical trace operations require a 3D mesh.")
    comm = mesh.dm.getComm().tompi4py()
    validate_harmonic(
        harmonic_degree,
        harmonic_order,
        harmonic_component,
        harmonic_normalization,
    )
    if harmonic_norm_value is None:
        harmonic_norm_value = harmonic_norm(
            harmonic_degree,
            harmonic_order,
            harmonic_component,
            harmonic_normalization,
        )
    if interface_radius <= 0.0 or harmonic_norm_value <= 0.0:
        raise ValueError("interface_radius and harmonic_norm must be positive.")
    if interface not in mesh.boundaries.__members__:
        raise ValueError(f"Unknown interface {interface!r}.")

    ilocal = None
    if comm.size > 1:
        _nroots, ilocal, _iremote = mesh.dm.getPointSF().getGraph()
    ghost_points = set() if ilocal is None else {int(point) for point in ilocal}
    facets = [int(facet) for facet in mesh._boundary_facets(interface)]
    owned_facets = [facet for facet in facets if facet not in ghost_points]
    tree = variable._get_kdtree() if owned_facets else None
    local_moment = 0.0
    for facet in owned_facets:
        vertices = _vertex_coordinates(mesh, facet)
        node_coordinates = np.vstack(
            [
                vertices,
                0.5 * (vertices[0] + vertices[1]),
                0.5 * (vertices[1] + vertices[2]),
                0.5 * (vertices[2] + vertices[0]),
            ]
        )
        distances, rows = tree.query(node_coordinates, sqr_dists=False)
        if np.max(distances) > 1.0e-10:
            raise RuntimeError(
                f"Could not map P2 trace nodes for facet {facet}: "
                f"maximum distance={np.max(distances):.3e}."
            )
        nodal_values = np.asarray(variable.data[rows, 0], dtype=float)
        local_moment += _triangle_p2_weighted_moment(
            vertices,
            nodal_values,
            harmonic_degree=harmonic_degree,
            harmonic_order=harmonic_order,
            harmonic_component=harmonic_component,
            harmonic_normalization=harmonic_normalization,
            quadrature_points=quadrature_points,
        )
    global_moment = float(comm.allreduce(local_moment, op=MPI.SUM))
    return global_moment / (
        float(interface_radius) ** 2 * float(harmonic_norm_value)
    )


def assemble_axisymmetric_sheet_load(**kwargs):
    """Backward-compatible wrapper for the legacy unnormalised zonal basis."""

    return assemble_spherical_harmonic_sheet_load(
        **kwargs,
        harmonic_order=0,
        harmonic_component="cos",
        harmonic_normalization=AXISYMMETRIC_UNNORMALIZED,
    )


def project_axisymmetric_trace_coefficient(*, harmonic_norm: float, **kwargs):
    """Backward-compatible wrapper for the legacy unnormalised zonal basis."""

    return project_spherical_harmonic_trace_coefficient(
        **kwargs,
        harmonic_order=0,
        harmonic_component="cos",
        harmonic_normalization=AXISYMMETRIC_UNNORMALIZED,
        harmonic_norm_value=harmonic_norm,
    )


__all__ = [
    "InterfaceLoadDiagnostics",
    "assemble_axisymmetric_sheet_load",
    "assemble_spherical_harmonic_sheet_load",
    "project_axisymmetric_trace_coefficient",
    "project_spherical_harmonic_trace_coefficient",
]


def install_linear_spherical_stokes_sheet_load(
    *, stokes, interface, harmonic_degree, density_coefficient,
    harmonic_order=0, harmonic_component='cos',
    harmonic_normalization=AXISYMMETRIC_UNNORMALIZED, quadrature_points=8,
    modes=None,
):
    """Install a radial sheet load in a constant linear P2 Stokes system.

    Call after viscosity, wall conditions and solver policy are configured,
    instead of adding the same sheet as a natural BC. Only straight triangular
    facets, continuous P2 velocity, zero bulk force and constant linear viscosity
    are supported. The exact linear residual uses the native assembled Jacobian
    plus the owned-facet P2 weak load, avoiding generic facet-trace interpolation.
    Each facet load has exactly zero moment against all rigid rotations.
    Reinstall after any model/mesh setup change. The SNES callback owns its RHS
    vector for the lifetime of this solve setup; reinstalling releases its old
    RHS. ``modes`` optionally replaces the single mode with a nonempty sequence
    of dictionaries containing harmonic_degree, density_coefficient and the
    optional harmonic_order/component/normalization keys. Different degrees
    can share this Stokes operator. Native solver adapters currently require
    the world communicator and nonempty mesh charts on every rank.
    """
    import sympy as sp
    from ._spherical_solver_contract import require_native_solver_mesh
    mesh=stokes.mesh
    require_native_solver_mesh(mesh)
    comm=mesh.dm.getComm().tompi4py()
    specifications=[dict(harmonic_degree=harmonic_degree, harmonic_order=harmonic_order,
        harmonic_component=harmonic_component, harmonic_normalization=harmonic_normalization,
        density_coefficient=density_coefficient)] if modes is None else list(modes)
    if not specifications:raise ValueError('modes must contain at least one harmonic.')
    validated=[]
    for spec in specifications:
        degree,order,component,normalization=validate_harmonic(spec['harmonic_degree'],
            spec.get('harmonic_order',0),spec.get('harmonic_component','cos'),
            spec.get('harmonic_normalization',harmonic_normalization))
        amplitude=float(spec['density_coefficient'])
        if not np.isfinite(amplitude):raise ValueError('Invalid mode density.')
        validated.append((degree,order,component,normalization,amplitude))
    if mesh.dim!=3 or stokes.u.degree!=2 or not stokes.u.continuous:
        raise ValueError('This load needs a 3D continuous P2 velocity space.')
    if interface not in mesh.boundaries.__members__:raise ValueError(f'Unknown interface {interface!r}.')
    if quadrature_points<2:raise ValueError('Invalid quadrature.')
    if type(stokes.constitutive_model) is not uw.constitutive_models.ViscousFlowModel:
        raise ValueError('Only the constant linear ViscousFlowModel is supported.')
    viscosity=stokes.constitutive_model.Parameters.shear_viscosity_0
    viscosity=uw.function.unwrap(viscosity)
    force=uw.function.unwrap(stokes.bodyforce)
    if not sp.sympify(viscosity).is_number or force.is_zero_matrix is not True:
        raise ValueError('This restricted API requires constant viscosity and zero bulk force.')
    if stokes.natural_bcs:
        raise ValueError('Install this sheet instead of natural load data, not in addition to them.')
    for bc in stokes.essential_bcs:
        if any(sp.sympify(value).is_zero is not True for value in bc.fn if value is not None):
            raise ValueError('Only homogeneous essential data are supported by this linear load API.')
    # Build the unchanged native constant linear operator before installing the
    # RHS. It is also the operator used by the rotated constraint solve.
    stokes._build(False,False,None)
    mesh.update_lvec();stokes.dm.setAuxiliaryVec(mesh.lvec,None)
    stokes._update_constants(record=True)
    # A direct computeJacobian requires SNES setup. Rotational-nullspace
    # setup happened to do this for free-slip callers; other callers can
    # otherwise receive null Jacobian handles and crash inside PETSc.
    stokes.snes.setUp()
    zero=stokes.dm.createGlobalVec();zero.set(0.)
    J,Jp=stokes.snes.getJacobian()[:2]
    try:stokes.snes.computeJacobian(zero,J,Jp)
    finally:zero.destroy()
    stokes.u._set_vec(available=True)
    ilocal=None
    if comm.size>1:_nroots,ilocal,_iremote=mesh.dm.getPointSF().getGraph()
    ghosts=set() if ilocal is None else set(map(int,ilocal))
    nodes,weights=np.polynomial.legendre.leggauss(quadrature_points)
    nodes=(nodes+1)/2;weights=weights/2
    with ExitStack() as cleanup:
        local=stokes.u.vec.duplicate();cleanup.callback(local.destroy);local.set(0.)
        values=local.array.reshape(-1,3)
        owned_facets=[int(facet) for facet in mesh._boundary_facets(interface) if int(facet) not in ghosts]
        tree=stokes.u._get_kdtree() if owned_facets else None
        for facet in owned_facets:
            vertices=_vertex_coordinates(mesh,int(facet))
            trace=np.vstack((vertices,(vertices[0]+vertices[1])/2,
                             (vertices[1]+vertices[2])/2,(vertices[2]+vertices[0])/2))
            distances,rows=tree.query(trace,sqr_dists=False)
            if np.max(distances)>1e-10:raise ValueError('Could not map triangular P2 velocity trace.')
            cross=np.linalg.norm(np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0]))
            contribution=np.zeros((6,3))
            for i,u in enumerate(nodes):
                for j,v in enumerate(nodes):
                    bary=np.array([1-u,u*(1-v),u*v])
                    point=bary@vertices
                    l0,l1,l2=bary
                    basis=np.array([l0*(2*l0-1),l1*(2*l1-1),l2*(2*l2-1),4*l0*l1,4*l1*l2,4*l2*l0])
                    density=sum(amplitude*real_harmonic_values(point,degree=degree,order=order,
                        component=component,normalization=normalization)
                        for degree,order,component,normalization,amplitude in validated)
                    direction=point/np.linalg.norm(point)
                    contribution+=np.outer(basis,density*direction)*cross*u*weights[i]*weights[j]
            np.add.at(values,rows,contribution)
        rhs=stokes.dm.createGlobalVec();rhs.set(0.)
        velocity_is,velocity_dm=stokes._subdict['velocity']
        try:
            sub=rhs.getSubVector(velocity_is)
            try:velocity_dm.localToGlobal(local,sub,addv=PETSc.InsertMode.ADD_VALUES)
            finally:rhs.restoreSubVector(velocity_is,sub)
        except Exception:
            rhs.destroy()
            raise
    def residual(snes,vector,out):
        J.mult(vector,out)
        out.axpy(1.,rhs)
    try:stokes.snes.setFunction(residual)
    except Exception:
        rhs.destroy()
        raise
    previous=getattr(stokes,'_linear_spherical_sheet_rhs',None)
    stokes._linear_spherical_sheet_rhs=rhs
    if previous is not None:previous.destroy()
    return rhs


__all__.append('install_linear_spherical_stokes_sheet_load')
