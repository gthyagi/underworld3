---
title: Coupled-boundary topography and gravity
families: [Stokes, Poisson]
---

# Coupled-boundary topography and gravity

Use `Stokes.cylindrical_gap_topography` after a coupled annulus solve, or
`Stokes.spherical_gap_topography` after a coupled spherical solve. Pass that
solve's result, both physical radii, boundary labels, signed density contrasts
and positive reference-gravity magnitudes explicitly. The adapters preserve
the solver's pressure/reaction gauge and leave its fields unchanged.

For normal reaction $\lambda=-\sigma_{nn}$ and radial-outward displacement,

$$h_o=\lambda_o/(\Delta\rho_o g_o),\qquad
h_i=-\lambda_i/(\Delta\rho_i g_i).$$

Contrasts mean density at smaller radius minus density at larger radius.
Do not independently remove the two wall means. Projected volume stress is a
separate diagnostic and is not the topography source.

The annulus trace is quadratic in arc angle. Spherical free-slip traces use
the original scalar P2 facet pullback; zero-slip traces reconstruct the native
mapped Cartesian multiplier before taking its physical normal component.
`evaluate` retains the complete spatial trace. Spherical point routing uses
exact radial-cone containment and bounded sparse exchanges, including empty
surface ranks; the complete surface or volume mesh is not replicated.

## Independent gravity methods

Spectral gravity integrates numerical topography coefficients and applies the
canonical kernels under `analytic.cylindrical.gravity` or
`analytic.spherical.gravity`. Analytical Stokes references are independent of
numerical reaction recovery. Numerical sources and references use the same
declared angular normalization, signs and physical/nondimensional units.

Direct-field Poisson samples the spatial trace into source fields on a
conforming gravity mesh, then calls `solve_cylindrical_spatial_potential` or
`solve_spherical_spatial_potential`. Returned potential coefficients are
post-solve diagnostics. Coefficient-fed Poisson reconstructs a source from
saved coefficients and remains an explicitly selected legacy comparison.

The common assembly solves $-\nabla^2\Phi=4\pi G\rho$, with interface density
sources and optional linear density feedback $\sigma=\sigma_0+\alpha\Phi$.
Geoid is $N=\Phi/g$. Feedback keeps Stokes traction fixed; it does not implement
a new coupled Stokes/deformation evolution problem.

Exterior/interior Robin coefficients are $n/r,n/r$ for an annulus and
$(\ell+1)/r,\ell/r$ for a spherical shell. They are exact for the selected
mode and approximate for other numerical modes. A full-ball mesh instead
retains centre regularity. Stokes physical-gap corrections do not automatically
correct Poisson facet/source geometry. Assess that error independently.

## Scope and validation

This common branch starts from spherical Stokes `5d49bae4` and selectively
ports annulus integration `4528372a`. Existing accepted campaigns retain their
original runtime identities. Validation of an analytical gravity kernel alone
does not validate the spatial Poisson calculation or a full Stokes-to-geoid
workflow. Component, serial/MPI, isolated-source and full-workflow evidence is
tracked separately in the benchmark integration plan. Coupled multiplier
stability remains subject to the base formulation's documented limitations.

Initial Mac Linux ARM64/int32 gates: 120 serial tests (27 optional-oracle
skips), 93 tests on every MPI8 rank, and two serial physical spherical
reaction-to-topography checks pass. Complete spherical gravity workflows and
Gadi validation are separate pending gates; the earlier unprepared h8 serial
hydrostatic gate reached its 180-second test timeout and is not counted.
