"""Circular weak loads for affine-triangle Poisson potentials.

Assemble b_i=4*pi*G*(int_Omega rho*N_i + sum int_Gamma sigma*N_i)
using the physical annulus and circles, not polygon tags. The native diffusion
and Robin operator is unchanged. Feedback adds -4*pi*G*int alpha*N_i*N_j
on the physical circle. With Phi=u+L and sigma=sigma0+alpha*Phi,
the regular-field equation also needs +4*pi*G*int alpha*L*N_i on its RHS.
Only the prescribed jump is extracted; feedback remains in the linear operator.
No harmonic representation of a density is required.
The PDE/sign convention follows Simons (1996), MIT thesis, Appendix B;
geometry-resolving quadrature follows the principle of Saye (2015),
doi:10.1137/140966290, using a circle-specific polar clipping construction.
"""

import time

from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import sympy as sp
import underworld3 as uw

from underworld3.function.expressions import mesh_vars_in_expression
from underworld3.utilities._circular_source_geometry import annulus_rule, circle_rule
from underworld3.utilities._field_layout import _layout


def _polynomials(points, vertices, powers):
    local = np.linalg.solve((vertices[1:]-vertices[0]).T, (points-vertices[0]).T).T
    return np.column_stack([local[:, 0]**i*local[:, 1]**j for i, j in powers])


class _CellField:
    """Exact polynomial in each actual FE cell; no nearest-point evaluation."""

    def __init__(self, field, vertices):
        self.field, self.vertices = field, vertices
        # Read all fields on every rank before cell-dependent work. This also
        # keeps the frozen density nondimensional and avoids lazy vector setup
        # or unit conversion inside unequal per-rank quadrature loops.
        self.nodal_values = np.asarray(field.data).copy()
        self.rows = field.mesh._cell_node_indices(field.degree, field.continuous)
        self.powers = [(i, degree-i) for degree in range(field.degree+1) for i in range(degree+1)]
        self.inverse = [np.linalg.inv(_polynomials(np.asarray(field.coords)[row], triangle, self.powers))
                        for row, triangle in zip(self.rows, vertices)]

    def basis(self, cell, points):
        return _polynomials(points, self.vertices[cell], self.powers)@self.inverse[cell]

    def basis_gradients(self, cell, points):
        triangle = self.vertices[cell]
        inverse = np.linalg.inv((triangle[1:]-triangle[0]).T)
        local = (points-triangle[0])@inverse.T
        derivatives = []
        for axis in (0, 1):
            terms = []
            for i, j in self.powers:
                power = (i, j)[axis]
                terms.append(np.zeros(len(points)) if power == 0 else
                    power*local[:, 0]**(i-(axis == 0))*local[:, 1]**(j-(axis == 1)))
            derivatives.append(np.column_stack(terms)@self.inverse[cell])
        return np.stack(derivatives, axis=-1)@inverse

    def values(self, cell, points, component):
        return self.basis(cell, points)@self.nodal_values[self.rows[cell], component]

    def gradients(self, cell, points, component):
        return np.einsum("qid,i->qd", self.basis_gradients(cell, points),
                         self.nodal_values[self.rows[cell], component])


def _density(expression, mesh, vertices):
    """Compile coordinates and same-mesh FE values without collective point search."""
    expression = uw.function.expressions._unwrap_for_compilation(
        sp.sympify(expression), keep_constants=False, return_self=False)
    _, functions, derivatives = mesh_vars_in_expression(expression)
    if derivatives:
        raise ValueError("Circular source densities must use field values, not field derivatives")
    functions = sorted(functions, key=str)
    sources = []
    symbols = sp.symbols(f"density_0:{len(functions)}", real=True)
    for function in functions:
        field = function.meshvar()
        if field.mesh is not mesh:
            raise ValueError("Circular density fields must belong to the gravity mesh")
        sources.append((_CellField(field, vertices), function.component))
    # Spatial inputs are real. Without this assumption Abs(r-R) gradients
    # retain unevaluated derivatives of re()/im() and cannot compile to NumPy.
    coordinates = sp.symbols("source_x source_y", real=True)
    replacements = dict(zip(functions, symbols))
    replacements.update(zip((mesh.N.x, mesh.N.y), coordinates))
    for coordinate, symbol in zip(mesh.X, coordinates):
        replacements[uw.function.expressions._unwrap_for_compilation(
            coordinate, keep_constants=False, return_self=False)] = symbol
    expression = expression.xreplace(replacements)
    unknown = expression.free_symbols-set((*coordinates, *symbols))
    if unknown:
        raise ValueError(f"Circular density contains unresolved symbols: {unknown}")
    evaluate = sp.lambdify((*coordinates, *symbols), expression, modules="numpy", cse=True)

    return _SpatialDensity(evaluate, expression, coordinates, symbols, sources)


class _SpatialDensity:
    """Frozen values and chain-rule gradients of a coordinate/FE expression."""

    def __init__(self, evaluate, expression, coordinates, symbols, sources):
        self.evaluate, self.sources = evaluate, sources
        self.derivatives = sp.lambdify((*coordinates, *symbols),
            [sp.diff(expression, v) for v in (*coordinates, *symbols)], modules="numpy", cse=True)

    def __call__(self, cell, points):
        extra = [field.values(cell, points, component) for field, component in self.sources]
        return np.broadcast_to(np.asarray(self.evaluate(points[:, 0], points[:, 1], *extra), dtype=float), (len(points),))

    def gradient(self, cell, points):
        extra = [field.values(cell, points, component) for field, component in self.sources]
        partials = [np.broadcast_to(np.asarray(v, dtype=float), (len(points),))
                    for v in self.derivatives(points[:, 0], points[:, 1], *extra)]
        gradient = np.column_stack(partials[:2]).copy()
        for partial, (field, component) in zip(partials[2:], self.sources):
            gradient += partial[:, None]*field.gradients(cell, points, component)
        return gradient


def install_circular_sources(*, poisson, radii, volume_density, sheets,
                             gravitational_constant, quadrature_order, potential_lifting=None):
    """Install a frozen linear residual with distributed, geometry-aware loads.

    Owned triangles alone contribute. Ghost-node loads are added by PETSc's
    normal distributed assembly. Every quadrature point uses its containing
    element's basis and density. Recreate the solve after changing the mesh,
    sources, diffusivity or Robin data; this API installs a fixed linear system.
    """
    start = time.perf_counter()
    mesh, phi = poisson.mesh, poisson.u
    if mesh.dim != 2 or mesh.degree != 1 or not phi.continuous:
        raise ValueError("Circular source integration requires affine triangles and continuous potential")
    vertex_rows = mesh._cell_node_indices(1, True)
    vertices = np.asarray(mesh._get_coords_for_basis(1, True))[vertex_rows]
    test = _CellField(phi, vertices)
    density = _density(volume_density, mesh, vertices)
    sheet_fields = [(radius, _density(sigma, mesh, vertices), float(alpha)) for radius, sigma, alpha in sheets]
    if potential_lifting is not None:
        potential_lifting.prepare(mesh, vertices)
    feedback_lift = (_density(potential_lifting.expression, mesh, vertices)
        if potential_lifting is not None and any(alpha for _, _, alpha in sheets) else None)
    poisson._build(False, False, None)
    mesh.update_lvec()
    poisson.dm.setAuxiliaryVec(mesh.lvec, None)
    poisson._update_constants(record=True)
    poisson.snes.setUp()
    zero = poisson.dm.createGlobalVec()
    zero.set(0.)
    jacobian, preconditioner = poisson.snes.getJacobian()[:2]
    poisson.snes.computeJacobian(zero, jacobian, preconditioner)
    # The native residual at zero contains beta*Phi_J on the unchanged
    # polygonal Robin facets. Subtract it, with native boundary quadrature.
    boundary_lift = zero.duplicate()
    if potential_lifting is not None:
        poisson.snes.computeFunction(zero, boundary_lift)
    else:
        boundary_lift.set(0.)
    zero.destroy()
    mapping, size = _layout(phi)
    rhs = poisson.dm.createGlobalVec()
    if size != rhs.getSize():
        rhs.destroy()
        raise ValueError("Circular load numbering differs from the native scalar operator")
    rhs.set(0.)
    rhs.axpy(-1., boundary_lift)
    boundary_lift.destroy()
    scale = 4*np.pi*gravitational_constant
    measures = np.zeros(1+len(sheets))
    load_integrals = np.zeros_like(measures)
    volume_active = sp.sympify(volume_density) != 0
    for cell in np.flatnonzero(mesh._get_owned_cells_mask()):
        indices = np.asarray(mapping[test.rows[cell]], dtype=PETSc.IntType)
        if volume_active:
            points, weights = annulus_rule(vertices[cell], radii, quadrature_order)
            if len(weights):
                values, basis = density(cell, points), test.basis(cell, points)
                rhs.setValues(indices, scale*(weights*values)@basis, addv=PETSc.InsertMode.ADD_VALUES)
                measures[0] += weights.sum()
                load_integrals[0] += weights@values
        if potential_lifting is not None:
            # Integrate over the complete gravity element, including buffers;
            # split at each physical sheet where grad(Phi_J) jumps.
            limit = radii[-1]+float(np.linalg.norm(vertices[cell], axis=1).max())+1.
            cuts = sorted(set((*radii,*getattr(potential_lifting,"break_radii",()))))
            points, weights = annulus_rule(vertices[cell], (0., *cuts, limit), quadrature_order)
            gradient = potential_lifting.gradient(cell, points)
            rhs.setValues(indices, -np.einsum("q,qd,qid->i", weights, gradient,
                test.basis_gradients(cell, points)), addv=PETSc.InsertMode.ADD_VALUES)
        for k, (radius, values_at, alpha) in enumerate(sheet_fields, start=1):
            points, weights = circle_rule(vertices[cell], radius, quadrature_order)
            if not len(weights):
                continue
            values, basis = values_at(cell, points), test.basis(cell, points)
            rhs.setValues(indices, scale*(weights*values)@basis, addv=PETSc.InsertMode.ADD_VALUES)
            if alpha:
                if feedback_lift is not None:
                    # sigma=sigma0+alpha*(u+L): the known alpha*L belongs on the RHS.
                    rhs.setValues(indices, scale*alpha*(weights*feedback_lift(cell, points))@basis,
                                  addv=PETSc.InsertMode.ADD_VALUES)
                jacobian.setValues(indices, indices, -scale*alpha*(basis.T@(weights[:, None]*basis)),
                                   addv=PETSc.InsertMode.ADD_VALUES)
            measures[k] += weights.sum()
            load_integrals[k] += weights@values
    rhs.assemble()
    jacobian.assemble()
    measures = uw.mpi.comm.allreduce(measures, op=MPI.SUM)
    load_integrals = uw.mpi.comm.allreduce(load_integrals, op=MPI.SUM)
    expected = np.array([np.pi*(radii[-1]**2-radii[0]**2), *[2*np.pi*s[0] for s in sheets]])
    checked = np.ones(len(expected), dtype=bool)
    checked[0] = bool(volume_active)
    coverage = float(np.max(abs(measures[checked]/expected[checked]-1), initial=0.))
    if coverage > 1e-9:
        rhs.destroy()
        raise ValueError(f"Physical circular sources are not covered exactly once by owned cells: {coverage=}")

    def residual(snes, vector, out):
        jacobian.mult(vector, out)
        out.axpy(-1., rhs)

    def fixed_jacobian(snes, vector, matrix, pc_matrix):
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

    poisson.snes.setFunction(residual)
    poisson.snes.setJacobian(fixed_jacobian, jacobian, jacobian)
    poisson._linear_assembled_rhs = rhs
    poisson.u._set_vec(available=True)
    poisson.circular_source_checks = dict(
        integration="circular", quadrature_order=quadrature_order,
        normal_derivative_jump=potential_lifting is not None,
        lifting_radial_sides=getattr(potential_lifting, "sides", None),
        lifting_half_widths=getattr(potential_lifting, "widths", None),
        lifting_break_radii=getattr(potential_lifting, "break_radii", ()),
        potential_representation="regular_plus_jump_lifting" if potential_lifting is not None else "ordinary_fe",
        robin_lifting="native polygonal boundary residual" if potential_lifting is not None else None,
        measured_area_and_lengths=measures.tolist(), expected_area_and_lengths=expected.tolist(),
        signed_density_integrals=load_integrals.tolist(), coverage_relative_error=coverage,
        assembly_wall_seconds=uw.mpi.comm.allreduce(time.perf_counter()-start, op=MPI.MAX),
        operator_geometry="unchanged affine diffusion and polygonal Robin boundaries")
