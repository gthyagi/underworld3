# Coupled-circle annulus topography and gravity

The opt-in cylindrical gap solve returns physical normal reaction
`lambda = -sigma_nn`, in the same physical-pressure gauge as `p = q + p_J`.
`postprocessing.cylindrical_gap.cylindrical_gap_topography` reads that reaction
without modifying Stokes fields. The multiplier is P2 in arc angle, not a
Cartesian P2 field evaluated off its chord. Its trace supplies owned-arc cosine
integration and direct spatial evaluation; only boundary data are gathered.

For radial-outward displacement and density contrasts defined as smaller-radius
side minus larger-radius side, use `h_outer = lambda/(Delta_rho*g)` and
`h_inner = -lambda/(Delta_rho*g)`. Gravity magnitudes are positive. Keep
zero-slip tangential reaction separate. The analytical cylindrical kernels in
`analytic.cylindrical.gravity` use `laplacian(Phi) = -4*pi*G*rho` and `N = Phi/g`.
They retain the references and extension limits in their docstrings.

The direct-field potential API accepts spatial sheet and volume expressions on
a conforming gravity mesh. Its mode number specifies Robin continuation and
post-solve diagnostics; it never constructs numerical sources from coefficients.
That continuation is exact for the specified mode only. Poisson source/facet
geometry is not corrected by the separate Stokes circular-gap method.
The coefficient-fed sheet API delegates to the same potential assembly.

## Integration validation

Numerical checkpoint `4e73aa72f792` is built against the existing Linux ARM64
PETSc int32 container. Serial and each of eight MPI ranks pass 72 focused
analytical, coupled Stokes, angle-trace, potential-contract, point-location and
Robin regressions. Twelve optional assess comparisons skip; assess is not needed.
Module origins, version, int32 and parallel HDF5 are verified.

The port includes only the missing exact-cell evaluation fix from `ef62a45b`
and scalar Robin tangent fix from `a6e0b9ec`. Before the latter, an independent
P2-exact Robin problem has maximum field error 1.2857142857145443 because the
shared PETSc J/P matrix receives both boundary tangent registrations. Registering
that tangent once restores the solution and residual-derivative checks.
The coupled Stokes matrices and their preallocation are unchanged.

Complete annulus gravity acceptance, production-driver migration, feedback and
Gadi convergence remain separate gates in the benchmark repository's integration
TODO. These regression results do not establish arbitrary-mesh or spherical
coupled-gap support.
