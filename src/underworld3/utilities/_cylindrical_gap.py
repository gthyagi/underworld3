r"""Coupled signed-domain Stokes integration for affine P2/P1 annuli.

Solution and test polynomials extend into outer circular caps; inner caps
are removed. Both viscous and incompressibility blocks use this same domain.
A continuous P2-in-angle multiplier enforces the physical-circle normal trace
weakly and solves lambda=-sigma_nn in the coupled system. This differs from
nodal rotated free slip and from lagged stress-derivative wall transfer.

References
----------
Collins, Li, Lozinski & Scovazzi (2025), "Gap-SBM: A New Conceptualization of
the Shifted Boundary Method with Optimal Convergence for the Neumann and
Dirichlet Problems", https://arxiv.org/abs/2508.09613: adopt the principle of
extending both trial and test functions and including gap virtual work.
Urquiza, Garon & Farinas (2014), "Weak imposition of the slip boundary condition
on curved boundaries for Stokes flow", JCP 256, 748-767,
https://doi.org/10.1016/j.jcp.2013.08.045: normal-velocity multiplier formulation.
Burman, Hansbo & Larson (2020), "Dirichlet boundary value correction using
Lagrange multipliers", BIT 60, 235-260,
https://doi.org/10.1007/s10543-019-00773-4: related multiplier boundary correction.

The signed circular-cap geometry, affine P2/P1 volume spaces, P2-in-angle
normal multiplier and paired pressure/reaction gauge below are this UW3
adaptation. It is not a verbatim implementation of any one paper, and their
convergence/stability theorems do not establish those properties for this space.
No analytical Stokes solution or harmonic coefficients enter the solve.
"""

from typing import NamedTuple

import numpy as np
import sympy as sp
import underworld3 as uw
from mpi4py import MPI
from petsc4py import PETSc

from underworld3.utilities._cylindrical_dirichlet import _labelled_edge_keys, _edge_key


class _GapResult(NamedTuple):
    normal_reaction: object
    checks: dict


def _p2_basis(
    vertices: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inverse = np.linalg.inv(
        np.stack(
            (vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=-1
        )
    )
    xi = np.einsum("eij,eqj->eqi", inverse, points - vertices[:, None, 0])
    bary = np.concatenate((1 - xi.sum(axis=-1, keepdims=True), xi), axis=-1)
    grad = np.stack((-inverse.sum(axis=1), inverse[:, 0], inverse[:, 1]), axis=1)
    shape = np.stack(
        [bary[..., i] * (2 * bary[..., i] - 1) for i in range(3)]
        + [4 * bary[..., i] * bary[..., j] for i, j in ((0, 1), (1, 2), (2, 0))],
        axis=-1,
    )
    derivative = np.stack(
        [(4 * bary[..., i, None] - 1) * grad[:, None, i] for i in range(3)]
        + [
            4
            * (
                bary[..., i, None] * grad[:, None, j]
                + bary[..., j, None] * grad[:, None, i]
            )
            for i, j in ((0, 1), (1, 2), (2, 0))
        ],
        axis=2,
    )
    return (shape, derivative, bary)


def _cap_quadrature(
    edges: np.ndarray, radius: float, angular_order: int, radial_order: int
) -> dict:
    """Exact circular cap parametrization, positive weights, independent of edge order."""
    delta = np.arctan2(
        edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0],
        np.sum(edges[:, 0] * edges[:, 1], axis=1),
    )
    if np.any(abs(delta) >= np.pi) or np.any(abs(delta) < 1e-14):
        raise ValueError("Cap rule requires nonzero minor arcs")
    a, wa = np.polynomial.legendre.leggauss(angular_order)
    b, wb = np.polynomial.legendre.leggauss(radial_order)
    angle0 = np.arctan2(edges[:, 0, 1], edges[:, 0, 0])
    angle = angle0[:, None] + delta[:, None] * (a + 1) / 2
    direction = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    chord_radius = (
        radius
        * np.cos(delta[:, None] / 2)
        / np.cos(angle - angle0[:, None] - delta[:, None] / 2)
    )
    gap = radius - chord_radius
    if np.any(gap < -1e-12):
        raise ValueError("Invalid circle/chord cap geometry")
    radial = chord_radius[..., None] + gap[..., None] * (b + 1) / 2
    points = radial[..., None] * direction[:, :, None, :]
    weight = (
        abs(delta[:, None, None])
        * wa[None, :, None]
        / 2
        * gap[..., None]
        * wb[None, None, :]
        / 2
        * radial
    )
    return dict(
        circle_points=radius * direction,
        circle_weight=radius * abs(delta[:, None]) * wa[None, :] / 2,
        cap_points=points.reshape(len(edges), angular_order * radial_order, 2),
        cap_weight=weight.reshape(len(edges), angular_order * radial_order),
    )


def _triangle_rule(vertices: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Positive tensor-product Duffy rule on each affine triangle."""
    x, w = np.polynomial.legendre.leggauss(order)
    x, w = ((x + 1) / 2, w / 2)
    a, b = np.meshgrid(x, x, indexing="ij")
    bary = np.stack((1 - a, a * (1 - b), a * b), axis=-1).reshape(-1, 3)
    determinant = abs(
        np.linalg.det(
            np.stack(
                (vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
                axis=-1,
            )
        )
    )
    return (
        np.einsum("qj,ejd->eqd", bary, vertices),
        determinant[:, None] * (a * w[:, None] * w[None, :]).ravel(),
    )


def _volume_blocks(
    vertices: np.ndarray, points: np.ndarray, weight: np.ndarray
) -> dict:
    """Viscosity-one element matrices; interleaved x/y velocity components."""
    shape, grad, bary = _p2_basis(vertices, points)
    stiffness = np.zeros((len(vertices), 12, 12))
    divergence = np.zeros((len(vertices), 3, 12))
    dot = np.einsum("eq,eqid,eqjd->eij", weight, grad, grad)
    for a in range(2):
        divergence[:, :, a::2] = -np.einsum(
            "eq,eql,eqi->eli", weight, bary, grad[..., a]
        )
        for b in range(2):
            stiffness[:, a::2, b::2] = (a == b) * dot + np.einsum(
                "eq,eqi,eqj->eij", weight, grad[..., b], grad[..., a]
            )
    rotation = np.stack((-points[..., 1], points[..., 0]), axis=-1)
    return dict(
        A=stiffness,
        B=divergence,
        pressure_mean=np.einsum("eq,eql->el", weight, bary),
        rotation=np.einsum("eq,eqi,eqd->eid", weight, shape, rotation),
        unit_force=np.einsum("eq,eqi->ei", weight, shape),
    )


def _layout(field):
    """Independent field global numbering, including local ghost-node indices."""
    iset, dm = field.mesh.dm.createSubDM(field.field_id)
    global_vector = dm.createGlobalVec()
    mapping = dm.getLGMap()
    indices = mapping.getIndices().copy()
    size = global_vector.getSize()
    global_vector.destroy()
    mapping.destroy()
    dm.destroy()
    iset.destroy()
    return indices, size


def _ordered_rows(mesh, field, vertices):
    rows = mesh._cell_node_indices(field.degree, field.continuous)
    standard = vertices
    if field.degree == 2:
        standard = np.concatenate(
            (vertices, (vertices[:, [0, 1, 2]] + vertices[:, [1, 2, 0]]) / 2), axis=1
        )
    distance = np.linalg.norm(
        np.asarray(field.coords)[rows][:, :, None] - standard[:, None], axis=-1
    )
    error = float(np.max(distance.min(axis=1), initial=0.0))
    if uw.mpi.comm.allreduce(error, op=MPI.MAX) > 1e-10:
        raise ValueError(
            "Cylindrical gap integration requires affine P2/P1 coordinates"
        )
    return np.take_along_axis(rows, distance.argmin(axis=1), axis=1)


def _coordinate_values(expression, points, components, mesh):
    expression = uw.function.expressions._unwrap_for_compilation(
        expression, keep_constants=False, return_self=False
    )
    if any(hasattr(type(atom), "meshvar") for atom in expression.atoms(sp.Function)):
        raise ValueError(
            "Cap sources and pressure lifts must depend on coordinates only"
        )
    if not expression.free_symbols and not expression.atoms(sp.Function):
        return np.broadcast_to(
            np.asarray(expression, dtype=float).reshape(components),
            (*points.shape[:-1], components),
        ).copy()
    values = uw.function.evaluate(
        expression, np.ascontiguousarray(points.reshape(-1, 2)), coord_sys=mesh.N
    )
    return np.asarray(values).reshape(*points.shape[:-1], components)


def _insert_blocks(matrix, velocity_ids, pressure_ids, blocks, viscosity, sign):
    for u, p, a, b in zip(
        velocity_ids, pressure_ids, blocks["A"], blocks["B"], strict=True
    ):
        matrix.setValues(u, u, sign * viscosity * a, addv=PETSc.InsertMode.ADD_VALUES)
        matrix.setValues(p, u, sign * b, addv=PETSc.InsertMode.ADD_VALUES)
        matrix.setValues(u, p, sign * b.T, addv=PETSc.InsertMode.ADD_VALUES)


def _scatter_values(vector, indices):
    """Fetch local/ghost field values without replicating the volume solution."""
    iset = PETSc.IS().createGeneral(
        np.asarray(indices, dtype=PETSc.IntType), comm=PETSc.COMM_SELF
    )
    local = PETSc.Vec().createSeq(len(indices), comm=PETSc.COMM_SELF)
    scatter = PETSc.Scatter().create(vector, iset, local, None)
    scatter.scatter(
        vector,
        local,
        addv=PETSc.InsertMode.INSERT_VALUES,
        mode=PETSc.ScatterMode.FORWARD,
    )
    values = local.array.copy()
    scatter.destroy()
    local.destroy()
    iset.destroy()
    return values


def _solve_cylindrical_gap(
    stokes,
    inner_boundary,
    outer_boundary,
    radius_inner,
    radius_outer,
    viscosity,
    pressure_jump,
    pressure_jump_integral,
    cap_bodyforce,
    quadrature,
    linear_solver_options,
):
    mesh, velocity, pressure = stokes.mesh, stokes.u, stokes.p
    if (
        mesh.dim != 2
        or mesh.cdim != 2
        or velocity.degree != 2
        or not velocity.continuous
        or pressure.degree != 1
        or not pressure.continuous
    ):
        raise ValueError(
            "Cylindrical gap Stokes requires affine 2D continuous P2/P1 fields"
        )
    if (
        uw.is_nondimensional_scaling_active()
        or uw.get_default_model().has_units_active()
    ):
        raise ValueError("Cylindrical gap Stokes supports nondimensional data only")
    if (
        not 0 < radius_inner < radius_outer
        or not np.isfinite(viscosity)
        or viscosity <= 0
        or int(quadrature) != quadrature
        or quadrature < 3
    ):
        raise ValueError(
            "Require positive ordered radii, positive constant viscosity and integer quadrature >= 3"
        )
    if (
        stokes.essential_bcs
        or stokes.natural_bcs
        or stokes._rotated_freeslip_bcs
        or getattr(stokes, "_block_constraint_bcs", [])
    ):
        raise ValueError(
            "Cylindrical gap Stokes replaces ordinary wall registration on both complete circles"
        )
    mu = uw.function.expressions._unwrap_for_compilation(
        stokes.constitutive_model.viscosity, keep_constants=False, return_self=False
    )
    if mu.free_symbols or float(mu) != float(viscosity):
        raise ValueError(
            "The constitutive viscosity must equal the supplied positive constant viscosity"
        )
    if not hasattr(stokes, "_cylindrical_gap_reaction"):
        stokes._cylindrical_gap_reaction = uw.discretisation.MeshVariable(
            f"CylindricalNormalReaction_{stokes.instance_number}",
            mesh,
            1,
            degree=2,
            continuous=True,
        )
    reaction = stokes._cylindrical_gap_reaction
    vertex_rows = mesh._cell_node_indices(1, True)
    vertices = np.asarray(mesh._get_coords_for_basis(1, True))[vertex_rows]
    urows, prows = _ordered_rows(mesh, velocity, vertices), _ordered_rows(
        mesh, pressure, vertices
    )
    umap, nu = _layout(velocity)
    pmap, npres = _layout(pressure)
    uid = umap.reshape(-1, 2)[urows].reshape(len(vertices), 12)
    pid = pmap[prows] + nu
    owned = np.flatnonzero(mesh._get_owned_cells_mask())
    walls = []
    boundary_uids = []
    for boundary, radius, sign in (
        (inner_boundary, radius_inner, -1),
        (outer_boundary, radius_outer, 1),
    ):
        keys = _labelled_edge_keys(mesh, boundary, radius, radius * 1e-10)
        owners, edge_rows, edges = [], [], []
        for i, j, mid in ((0, 1, 3), (1, 2, 4), (2, 0, 5)):
            selected = owned[
                np.all(
                    abs(np.linalg.norm(vertices[owned][:, [i, j]], axis=-1) - radius)
                    < 1e-10,
                    axis=1,
                )
            ]
            for cell in selected:
                edge = vertices[cell, [i, j]]
                if _edge_key(edge, radius * 1e-10) in keys:
                    owners.append(cell)
                    edge_rows.append(urows[cell, [i, j, mid]])
                    edges.append(edge)
        owners = np.asarray(owners, dtype=int)
        edges = np.asarray(edges, dtype=float).reshape(-1, 2, 2)
        edge_rows = np.asarray(edge_rows, dtype=int).reshape(-1, 3)
        found = set().union(
            *uw.mpi.comm.allgather({_edge_key(edge, radius * 1e-10) for edge in edges})
        )
        if not keys or found != keys:
            raise ValueError(
                "Owned cells do not cover the requested physical circular wall"
            )
        boundary_uids.extend(umap.reshape(-1, 2)[edge_rows, 0].ravel().tolist())
        walls.append(
            dict(
                owners=owners,
                edge_rows=edge_rows,
                radius=radius,
                sign=sign,
                rule=_cap_quadrature(edges, radius, quadrature, 4),
            )
        )
    boundary_uids = np.unique(
        np.concatenate(
            uw.mpi.comm.allgather(np.asarray(boundary_uids, dtype=PETSc.IntType))
        )
    )
    size = nu + npres + len(boundary_uids)
    matrix = PETSc.Mat().createAIJ(size=(size, size), nnz=100, comm=mesh.dm.comm)
    matrix.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    rhs = matrix.createVecRight()
    pmean = rhs.duplicate()
    rotation_weights = rhs.duplicate()
    for vector in (rhs, pmean, rotation_weights):
        vector.set(0.0)
    points, weights = _triangle_rule(vertices[owned], 4)
    blocks = _volume_blocks(vertices[owned], points, weights)
    _insert_blocks(matrix, uid[owned], pid[owned], blocks, viscosity, 1)
    bodyforce = uw.function.expressions._unwrap_for_compilation(
        stokes.bodyforce, keep_constants=False, return_self=False
    )
    variables = {
        type(atom).meshvar()
        for atom in bodyforce.atoms(sp.Function)
        if hasattr(type(atom), "meshvar")
    }
    if variables:
        if len(variables) != 1:
            raise ValueError(
                "Use one P1/P2 vector MeshVariable or a coordinate-only volume body force"
            )
        field = variables.pop()
        if (
            field.mesh is not mesh
            or field.degree not in (1, 2)
            or field.num_components != 2
            or bodyforce != field.sym
        ):
            raise ValueError(
                "Use the unmodified P1/P2 vector source field on the Stokes mesh"
            )
        source_rows = _ordered_rows(mesh, field, vertices)[owned]
        shape, _, bary = _p2_basis(vertices[owned], points)
        source_shape = shape if field.degree == 2 else bary
        force = np.einsum(
            "eqi,eid->eqd", source_shape, np.asarray(field.array)[source_rows, 0, :]
        )
    else:
        force = _coordinate_values(bodyforce, points, 2, mesh)
    shape, _, _ = _p2_basis(vertices[owned], points)
    load = np.einsum("eq,eqi,eqd->eid", weights, shape, force)
    for ids, value in zip(uid[owned], load, strict=True):
        rhs.setValues(ids, value.ravel(), addv=PETSc.InsertMode.ADD_VALUES)
    for ids, mass in zip(pid[owned], blocks["pressure_mean"], strict=True):
        pmean.setValues(ids, mass, addv=PETSc.InsertMode.ADD_VALUES)
    for ids, rot in zip(uid[owned], blocks["rotation"], strict=True):
        rotation_weights.setValues(ids, rot.ravel(), addv=PETSc.InsertMode.ADD_VALUES)
    lift_cap_integral = 0.0
    for wall in walls:
        owners, sign, rule = wall["owners"], wall["sign"], wall["rule"]
        cap = _volume_blocks(vertices[owners], rule["cap_points"], rule["cap_weight"])
        _insert_blocks(matrix, uid[owners], pid[owners], cap, viscosity, sign)
        shape, _, _ = _p2_basis(vertices[owners], rule["cap_points"])
        force = _coordinate_values(cap_bodyforce, rule["cap_points"], 2, mesh)
        load = sign * np.einsum("eq,eqi,eqd->eid", rule["cap_weight"], shape, force)
        for ids, value, rot in zip(uid[owners], load, cap["rotation"], strict=True):
            rhs.setValues(ids, value.ravel(), addv=PETSc.InsertMode.ADD_VALUES)
            rotation_weights.setValues(
                ids, sign * rot.ravel(), addv=PETSc.InsertMode.ADD_VALUES
            )
        for ids, value in zip(pid[owners], cap["pressure_mean"], strict=True):
            pmean.setValues(ids, sign * value, addv=PETSc.InsertMode.ADD_VALUES)
        lift_cap_integral += sign * np.sum(
            rule["cap_weight"]
            * _coordinate_values(pressure_jump, rule["cap_points"], 1, mesh)[..., 0]
        )
        shape, _, _ = _p2_basis(vertices[owners], rule["circle_points"])
        t = (np.polynomial.legendre.leggauss(quadrature)[0] + 1) / 2
        multiplier = np.array(
            [(1 - t) * (1 - 2 * t), t * (2 * t - 1), 4 * t * (1 - t)]
        ).T
        c = sign * np.einsum(
            "eq,ql,eqi,eqd->elid",
            rule["circle_weight"],
            multiplier,
            shape,
            rule["circle_points"] / wall["radius"],
        )
        edge_uids = umap.reshape(-1, 2)[wall["edge_rows"], 0]
        lid = nu + npres + np.searchsorted(boundary_uids, edge_uids)
        for u, l, value in zip(uid[owners], lid, c.reshape(-1, 3, 12), strict=True):
            matrix.setValues(l, u, value, addv=PETSc.InsertMode.ADD_VALUES)
            matrix.setValues(u, l, value.T, addv=PETSc.InsertMode.ADD_VALUES)
    matrix.assemble()
    for vector in (rhs, pmean, rotation_weights):
        vector.assemble()
    pressure_mode = rhs.duplicate()
    rotation_mode = rhs.duplicate()
    pressure_mode.set(0.0)
    rotation_mode.set(0.0)
    lo, hi = pressure_mode.getOwnershipRange()
    indices = np.arange(max(lo, nu), hi, dtype=PETSc.IntType)
    pressure_mode.setValues(indices, np.ones(len(indices)))
    pressure_mode.assemble()
    rotation = np.column_stack(
        (-np.asarray(velocity.coords)[:, 1], np.asarray(velocity.coords)[:, 0])
    ).ravel()
    rotation_mode.setValues(umap, rotation)
    rotation_mode.assemble()
    work = rhs.duplicate()
    matrix.mult(pressure_mode, work)
    pn = work.norm() / (matrix.norm() * pressure_mode.norm())
    matrix.mult(rotation_mode, work)
    rn = work.norm() / (matrix.norm() * rotation_mode.norm())
    if max(pn, rn) > 1e-12:
        raise ArithmeticError(
            f"Cylindrical gap null-mode assembly failed: pressure={pn}, rotation={rn}"
        )
    compatibility = max(
        abs(rhs.dot(pressure_mode)) / pressure_mode.norm(),
        abs(rhs.dot(rotation_mode)) / rotation_mode.norm(),
    )
    if compatibility > 1e-10 * max(rhs.norm(), 1.0):
        raise ValueError(
            "The prescribed volume/cap forces have incompatible net rotation torque"
        )
    # Two algebraic null-mode pins; physical volume gauges are restored below.
    best = float(np.max(abs(rotation), initial=0.0))
    maximum = uw.mpi.comm.allreduce(best, op=MPI.MAX)
    candidate = int(
        np.min(umap[np.isclose(abs(rotation), maximum, rtol=0, atol=1e-13)], initial=nu)
    )
    rotation_pin = uw.mpi.comm.allreduce(candidate, op=MPI.MIN)
    pins = np.array([rotation_pin, nu], dtype=PETSc.IntType)
    pinned = matrix.copy()
    pinned.zeroRowsColumns(pins, diag=1.0)
    pinned_rhs = rhs.copy()
    lo, hi = pinned_rhs.getOwnershipRange()
    local_pins = pins[(pins >= lo) & (pins < hi)]
    pinned_rhs.setValues(local_pins, np.zeros(len(local_pins)))
    pinned_rhs.assemble()
    ksp = PETSc.KSP().create(comm=mesh.dm.comm)
    ksp.setOperators(pinned)
    prefix = f"cylindrical_gap_{stokes.instance_number}_"
    ksp.setOptionsPrefix(prefix)
    options = PETSc.Options(prefix)
    for key, value in linear_solver_options.items():
        options[key] = value
    ksp.setFromOptions()
    solution = rhs.duplicate()
    ksp.solve(pinned_rhs, solution)
    if ksp.getConvergedReason() <= 0:
        raise ArithmeticError(
            f"Cylindrical gap KSP failed: reason={ksp.getConvergedReason()}"
        )
    lift_integral = float(pressure_jump_integral) + uw.mpi.comm.allreduce(
        float(lift_cap_integral), op=MPI.SUM
    )
    solution.axpy(-(pmean.dot(solution) + lift_integral) / pmean.sum(), pressure_mode)
    solution.axpy(
        -rotation_weights.dot(solution) / rotation_weights.dot(rotation_mode),
        rotation_mode,
    )
    matrix.mult(solution, work)
    work.axpy(-1.0, rhs)
    relative = work.norm() / max(rhs.norm(), np.finfo(float).tiny)
    checks = dict(
        relative_residual=relative,
        pressure_null_relative_residual=pn,
        rotation_null_relative_residual=rn,
        physical_area=pmean.sum(),
        physical_pressure_integral=pmean.dot(solution) + lift_integral,
        rotation_integral=rotation_weights.dot(solution),
        unknowns=size,
        ksp_reason=int(ksp.getConvergedReason()),
        ranks=uw.mpi.size,
    )
    if relative > 1e-9:
        raise ArithmeticError(f"Cylindrical gap unpinned residual failed: {checks}")
    velocity.array[:, 0, :] = _scatter_values(solution, umap).reshape(-1, 2)
    pressure.array[:, 0, 0] = _scatter_values(solution, pmap + nu)
    reaction.array[:, 0, 0] = 0.0
    global_node_ids = umap.reshape(-1, 2)[:, 0]
    selected = np.isin(global_node_ids, boundary_uids)
    reaction.array[selected, 0, 0] = _scatter_values(
        solution, nu + npres + np.searchsorted(boundary_uids, global_node_ids[selected])
    )
    for field in (velocity, pressure, reaction):
        field._sync_lvec_to_gvec()
    mesh._stale_lvec = True
    for key in linear_solver_options:
        del options[key]
    for obj in (
        ksp,
        solution,
        pinned_rhs,
        pinned,
        work,
        pressure_mode,
        rotation_mode,
        rhs,
        pmean,
        rotation_weights,
        matrix,
    ):
        obj.destroy()
    return _GapResult(reaction, checks)
