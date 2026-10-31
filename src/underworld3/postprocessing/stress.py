"""Discontinuous tensor recovery for velocity-derived physical stress diagnostics."""

import underworld3 as uw


def project_deviatoric_stress(mesh, expression, name: str, tolerance: float):
    """L2-project deviatoric stress into a symmetric DG P1 tensor without smoothing.

    Use a matching DG P1 scalar work field and block Jacobi in Tensor_Projection.
    Compose Cauchy stress afterwards with the original physical Stokes pressure,
    including its prescribed jump and common gauge. This diagnostic does not
    replace boundary reactions or alter the Stokes solution.
    """
    field = uw.discretisation.MeshVariable(
        name, mesh, (mesh.dim, mesh.dim), vtype=uw.VarType.SYM_TENSOR,
        degree=1, continuous=False,
    )
    work = uw.discretisation.MeshVariable(name+"_work", mesh, 1, degree=1, continuous=False)
    projector = uw.systems.Tensor_Projection(mesh, tensor_Field=field, scalar_Field=work, degree=1)
    projector.uw_function = expression
    projector.smoothing = 0.
    projector.tolerance = tolerance
    projector.linear_solver(pc="bjacobi", rtol=tolerance)
    projector.solve()
    return field
