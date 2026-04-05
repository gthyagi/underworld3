# Submesh Solver Architecture: Multi-Domain Equation Systems

## Context

Underworld3 needs to support solving different equations on different subsets of a mesh while maintaining a unified field representation. Use cases include:

- **Air/rock**: Stokes on rock only, full mesh for temperature/gravity
- **Surface evolution**: deforming air mesh coupled to rock Stokes
- **Gravity**: Poisson on full domain, density source from rock only
- **Multi-physics**: different equations on different subdomains (Stokes, Darcy, etc.)

### What we've established (2026-04-05)

1. **`DMPlexFilter`** extracts a submesh with exact shared nodes. The submesh carries a subpoint IS mapping back to the parent via `getSubpointIS()`.

2. **PETSc Region DS** (`DMSetRegionDS`) segfaults during assembly — no examples exist in PETSc, likely incomplete infrastructure. Dead end for now.

3. **Solver `part` parameter** in PETSc boundary assembly (`support[key.part]`) — controls which cell's closure is used for internal boundary integrals. Useful for one-sided boundary assembly but doesn't address the core problem of restricting volume assembly to a subdomain.

4. **Low-viscosity air layer** with discontinuous pressure works reasonably but the air's incompressibility constraint acts as an unintended physical boundary condition. Not equivalent to solving on rock alone.

5. **Normalised `Gamma_N`** (merged) — `mesh.Gamma_N` now returns a unit normal. Penalty and Nitsche BCs are mesh-independent.

## Design Principles

### 1. One field, multiple solvers

MeshVariables live on the **parent (full) mesh**. They are the single source of truth. A solver on a submesh reads from and writes to the parent-mesh variable — it only modifies DOFs it owns (the submesh region). The user never creates submesh-local variables.

```python
v = MeshVariable("v", full_mesh, ...)
p = MeshVariable("p", full_mesh, ...)

stokes = Stokes(rock_mesh, velocityField=v, pressureField=p)
stokes.solve()  # updates v, p at rock DOFs only
```

### 2. Meshes know their lineage

Every mesh has a `parent` attribute and a `subpoint_is` mapping. Top-level meshes have `parent=None` and `subpoint_is=None`. Submeshes reference their parent and carry the IS.

```python
full_mesh.parent       # None
full_mesh.subpoint_is  # None

rock_mesh = full_mesh.extract_region("Inner")
rock_mesh.parent       # full_mesh
rock_mesh.subpoint_is  # IS mapping submesh points -> parent points
```

### 3. Restrict/prolongate as mesh operations

```python
mesh.restrict(var)    # parent -> submesh DOFs (no-op if parent is None)
mesh.prolongate(var)  # submesh DOFs -> parent (no-op if parent is None)
```

Solvers call these uniformly. On a top-level mesh they're no-ops. On a submesh they gather/scatter via the subpoint IS. The solver code doesn't branch.

### 4. Boundary mapping is automatic

When `extract_region("Inner")` creates a submesh, boundaries are remapped:
- Full mesh "Lower" (r=r_inner) → submesh "Lower"
- Full mesh "Internal" (r=r_internal) → submesh outer boundary
- Full mesh "Upper" (r=r_outer) → not present on submesh

The label names are preserved from the parent (they survive `DMPlexFilter`). The user refers to boundaries by the same names.

## PETSc Infrastructure Available

| API | What it does | Status |
|-----|-------------|--------|
| `DMPlexFilter(dm, label, value, ...)` | Extract cells by label → new DMPlex | **Works**, tested |
| `DMPlex.getSubpointIS()` | IS mapping submesh → parent points | Available in petsc4py |
| `DMSetRegionDS(dm, label, fields, ds, dsIn)` | Per-region discrete system | **Segfaults**, no examples |
| `DMGetCellDS(dm, point, &ds, &dsIn)` | Per-cell DS dispatch in assembly | Works but requires Region DS |
| `DMPlexCreateSubmesh(dm, label, value, ...)` | Co-dimension 1 submesh (boundaries) | Works but wrong dimension |
| `VecScatter` / `PetscSF` | Parallel data transfer | Standard PETSc |

### PETSc Alternatives Investigated (2026-04-05)

**DMComposite** — packs multiple DMs into one composite. Tested 2026-04-05.

- Accepts DMPlex sub-DMs from DMPlexFilter. Scatter/gather works correctly.
- Interface nodes appear in both sub-DMs (102 shared vertices + 102 shared edges confirmed).
- Composite Vec concatenates sub-DM DOFs — interface DOFs are **duplicated**, not shared. Synchronisation after each solve is still required.
- **Verdict**: Designed for **combining** separate problems (fluid + structure), not **subdividing** one mesh. Doesn't simplify our use case — the core challenge (interface DOF ownership, restrict/prolongate) remains the same either way. The direct subpoint IS approach is simpler and more natural.

**PCFIELDSPLIT with spatial IS** — split by region, not field.

- `PCFieldSplitSetIS()` accepts arbitrary IS — confirmed no restriction to field-based splits.
- Supports Schur complement strategies between spatial blocks.
- **Problem**: This is a preconditioner, not an assembly strategy. Both blocks still assemble from the same DS. Doesn't let you have different equations per region.
- **Verdict**: Useful for preconditioning variable-viscosity systems, but doesn't solve the core problem.

**DMCreateDomainDecomposition** — PETSc's native spatial decomposition.

- `DMCreateDomainDecomposition_Plex()` returns inner/outer IS with configurable overlap.
- `DMCreateDomainDecompositionScatters_Plex()` creates VecScatter for restrict/prolongate.
- **Problem**: Designed for PCASM/PCGASM where the *same* equations are solved on each subdomain. Not for different physics per region.
- **Verdict**: Scatter infrastructure is useful but intent doesn't match multi-physics.

### Assessment

None of the PETSc mechanisms directly solve "different equations on different subsets of the same mesh with shared fields." They each address adjacent problems:

| Mechanism | Different equations? | Shared fields? | Fits? |
|-----------|---------------------|----------------|-------|
| DMComposite | Yes | No (different vector layout) | Partial |
| PCFIELDSPLIT | No (same assembly) | Yes | No |
| DomainDecomp | No (same equations) | Yes | No |
| Region DS | Yes (in theory) | Yes | Segfaults |

The **DMPlexFilter + subpoint IS + UW3-level restrict/prolongate** approach remains the best fit. PETSc provides the building blocks (mesh filtering, IS mapping, parallel SF), UW3 handles the multi-physics orchestration.

## Open Questions

1. **DM lifecycle**: The solver currently clones DMs freely (`clone_dm_hierarchy`). If the submesh also clones, DMs proliferate with no clear ownership. Need a cleanup strategy.

2. **Mesh adaptation**: If the full mesh adapts (refinement, coarsening, surface deformation), the submesh must be re-extracted and the IS rebuilt. All in-flight MeshVariables need re-projection. How does this interact with the existing `refinement_callback` infrastructure?

3. **Parallel decomposition**: `DMPlexFilter` builds a new SF for the submesh. If the partition differs from the parent, restrict/prolongate need MPI communication. How expensive is this? Does it matter for the target use cases?

4. **Coupled solves**: If two solvers on different submeshes need to iterate (e.g., rock Stokes + air transport), the restrict/prolongate happens every outer iteration. Is the data copy overhead acceptable, or do we need shared vectors?

5. **Pressure space**: Discontinuous pressure (dP1) is required for viscosity contrasts at internal boundaries. Should this be the default for submesh solvers, or should the user choose?

## Implementation Plan

### Immediate: `Mesh.extract_region()`

The minimum viable feature. Everything else follows from existing UW3 patterns.

```python
rock_mesh = full_mesh.extract_region("Inner")
```

Wraps `DMPlexFilter`, returns a new `Mesh` with:
- `parent` reference to the full mesh
- `subpoint_is` from `getSubpointIS()` (stored for future optimisation)
- Boundaries inherited from parent labels (they survive DMPlexFilter)
- Coordinate system inherited from parent

The extracted mesh is fully independent — users create their own MeshVariables on it, set up solvers normally, and use the existing `uw.function.evaluate(expr, coords)` path for mesh-to-mesh data transfer:

```python
# Separate variables on separate meshes
v_rock = MeshVariable("v", rock_mesh, ...)
rho_full = MeshVariable("rho", full_mesh, ...)

# Stokes on rock submesh — standard solver, nothing special
stokes = Stokes(rock_mesh, velocityField=v_rock, ...)
stokes.add_natural_bc(penalty * Gamma_N.dot(v_rock.sym) * Gamma_N, "Internal")
stokes.solve()

# Transfer rock solution to full mesh via evaluate (existing infrastructure)
v_full.data[:] = uw.function.evaluate(v_rock.sym, v_full.coords)

# Gravity on full mesh using transferred data
gravity = Poisson(full_mesh, ...)
gravity.solve()
```

This works today with the `petsc_dm_filter_by_label()` function we already built. `extract_region` just packages it with the parent reference and boundary setup.

### Future: Optimised restrict/prolongate

When `evaluate` becomes a bottleneck (large meshes, frequent transfers), add IS-based restrict/prolongate that skips the kd-tree:

```python
mesh.restrict(parent_var, sub_var)    # gather via subpoint IS
mesh.prolongate(sub_var, parent_var)  # scatter via subpoint IS
```

No-op when `parent is None`. The subpoint IS is already stored from `extract_region`.

### Future: Solver auto-detection

If desired, the solver could accept parent-mesh variables and handle restrict/prolongate internally. But this adds complexity and hides data flow. The explicit two-mesh pattern above is clearer for users and works now.

### Other items

- **Boundary remapping**: Document which parent labels map to submesh boundaries. DMPlexFilter preserves labels; "Internal" on the parent becomes an exterior boundary on the submesh.
- **DM lifecycle**: Audit clone/destroy patterns, ensure submesh DMs are cleaned up.
- **Parallel**: `DMPlexFilter` builds a new SF. Test in MPI before relying on it.

## Additional Findings

### Discontinuous pressure required for viscosity contrasts

Continuous P1 pressure cannot represent the pressure jump at a viscosity discontinuity (scales with viscosity ratio). With eta_rock/eta_air = 1000, the pressure smears across interface elements and corrupts velocity direction up to 177 degrees. Discontinuous P1 handles each side independently — velocity direction error drops to <5 degrees.

### Normalised boundary normal (Gamma_N)

`mesh.Gamma_N` now returns `Gamma / |Gamma|` — a unit normal regardless of element size. The raw `mesh.Gamma` magnitude scales with edge length (2D) / face area (3D). This affects penalty scaling: `penalty * Gamma.dot(v) * Gamma` has effective penalty ~ penalty * h², while `penalty * Gamma_N.dot(v) * Gamma_N` is mesh-independent. Nitsche's `gamma * mu / h` term now has correct 1/h scaling with normalised normals.
