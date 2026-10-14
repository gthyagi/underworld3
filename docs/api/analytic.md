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
