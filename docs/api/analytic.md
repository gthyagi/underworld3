# Analytic Solutions

```{eval-rst}
.. automodule:: underworld3.analytic
   :members:
   :show-inheritance:
```

## The contract

Every solution satisfies the same contract, so a validation run reads the same way
whichever one you use.

```{eval-rst}
.. automodule:: underworld3.analytic._base
   :members:
   :show-inheritance:
```

## Kramer cylindrical fields

The single `underworld3.analytic.kramer_cylindrical` module owns coefficients,
symbolic fields and cached numerical evaluators. `analytic.kramer` preserves
legacy imports without a second implementation.

`uw.analytic.CylindricalStokes` supplies native symbolic velocity, pressure,
strain rate and total/radial/shear stress, without requiring `assess`. Named-field
evaluation uses cached vectorised NumPy evaluators with CSE; symbolic expressions
remain usable by the UW3 JIT. `compile_evaluators("velocity", "pressure")` can
prepare the evaluators before repeated measurements. The optional
`AssessCylindricalStokes` class retains the former numerical reference wrapper.

The point-evaluation comparison is reproducible with
`scripts/benchmark_kramer_evaluators.py --output-json <common-output-path>`.
Construction, function generation, first evaluation and seven repeated warmed
measurements are reported separately. These are evaluator timings, not solver
speedups.

## Kramer spherical fields

`uw.analytic.SphericalStokes` combines radial coefficients, symbolic velocity,
pressure, strain rate, total/radial/shear stress and cached array evaluators in
`analytic.kramer_spherical`. Select `ell`, `m`, `component` and `normalization`
explicitly for non-axisymmetric work. The real orthonormal basis includes the
sqrt(2) factor for nonzero m; assess's real complex-harmonic oracle does not.
Delta and smooth forcing support free/zero slip with positive viscosity and
explicit forcing amplitude. Exceptional ell=0/1 and resonant smooth powers
are rejected. Cartesian solid harmonics give finite exact pole values.

Use `apply_forcing` once to apply the conforming delta interface jump or smooth
bulk force. Positive g is inward physical force and an outward normal-stress
jump; sample bulk equations away from that interface. Radial shear is returned
as a Cartesian tangential traction vector, not an angular scalar.

`scripts/benchmark_kramer_spherical_evaluators.py --output-json <path>` records
setup, first-call and seven-repeat warmed timings for three batch sizes, plus
nonzero-m harmonic comparisons including both poles. These are analytical
point-evaluator measurements, not solver speedups.

## Zhong spherical-shell response oracle

`Zhong2008` is a mesh-independent propagator-matrix oracle. It returns the
boundary response coefficients used by the Zhong et al. (2008) benchmark; it
does not implement the symbolic mesh-field contract above.

```{eval-rst}
.. automodule:: underworld3.analytic.zhong2008
   :members:
   :show-inheritance:
```

## Radial gravitational-potential kernels

The mesh-independent cylindrical and spherical kernels use
`laplacian(Phi) = -4*pi*G*delta_rho`. They include sheet jumps, regular/decaying
continuations and compact power-law volume sources. Cylindrical coefficient
responses remain available through `uw.postprocessing.geoid`; the analytical
implementation is shared rather than copied into each benchmark. Spherical
normalised response coefficients are distinct from physical potential values.

```{eval-rst}
.. automodule:: underworld3.analytic.cylindrical_gravity
   :members:

.. automodule:: underworld3.analytic.spherical_gravity
   :members:
```

Real spherical-harmonic conventions and batched evaluators are available in
`uw.maths.spherical_harmonics`.

## See also

- {doc}`solvers` — the solvers these solutions validate.
- `docs/developer/subsystems/analytic-solutions.md` — the implementation form,
  the validation protocol every transcription must pass, and the provenance of
  each vendored reference kernel.

## Physical spherical responses and restricted trace support

`analytic.spherical_gravity.spherical_physical_shell_response` returns physical
potential and geoid with explicit G, densities and inner/outer gravity. It
supports sheet/volume forcing and linear interface feedback, recording matrix
conditioning and residuals. This API is distinct from the normalized Zhong
response.

`postprocessing.spherical_interface_load` supports only continuous P2 scalar
traces on straight triangular facets of 3D meshes. Facets contribute on their
owning rank and shared DOFs are added collectively on the mesh communicator.
Temporary PETSc objects are released on exceptions.

The restricted `install_linear_spherical_stokes_sheet_load` applies radial
sheet forcing to constant linear ViscousFlowModel/P2 velocity systems after
homogeneous wall conditions and solver policy have been configured. It uses the
native assembled Jacobian and an explicit owned-facet weak RHS, whose moment
against rigid rotations is zero. It rejects nonlinear models, other natural
loads and nonzero essential data. Reinstall after any model/mesh setup change;
it is not a general nonlinear residual hook. The generic natural-load route
remains available for diagnostics, but the benchmark records a same-mesh MPI
compatibility discrepancy for that route.

`postprocessing.spherical_potential.solve_spherical_potential` keeps geometry,
labels and solver options caller-owned. Its full-ball continuation defaults to
the single-mode exterior Robin condition; an explicit analytic Dirichlet
comparison is also supported. It preserves an owned copy of the base solution
before borrowing DM work vectors, including for volume-plus-sheet sources.
The source radius must be supplied for each Dirichlet-comparison sheet.
