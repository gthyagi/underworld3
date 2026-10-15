# Analytic Solutions

```{eval-rst}
.. automodule:: underworld3.analytic
   :members:
   :show-inheritance:
```

## The contract

Symbolic field solutions share this contract. The Zhong response propagator
exposes a separate coefficient-response contract, described below.

```{eval-rst}
.. automodule:: underworld3.analytic._base
   :members:
   :show-inheritance:
```

## Geometry packages and case selection

```text
analytic/
├── cylindrical/
│   ├── __init__.py
│   ├── gravity.py
│   ├── kramer.py
│   └── thieulot.py
├── spherical/
│   ├── __init__.py
│   ├── gravity.py
│   ├── kramer.py
│   ├── thieulot.py
│   └── zhong.py
├── _base.py
├── _thieulot_fields.py
├── _kramer_geoid.py
└── stokes.py
```

```python
sol = uw.analytic.StokesSolution(mesh, geometry="cylindrical", case="kramer", n=2)
reference = uw.analytic.StokesSolution(geometry="spherical", case="zhong", harmonic_degree=5)
```

Each case module owns its radial formulas and solution/evaluation class.
The private `_thieulot_fields.py` helper shares Cartesian field assembly and cached
array evaluation between the two Thieulot geometries. Each geometry owns its
gravity kernels; the common field/error contract remains in `_base.py`. Geometry packages expose the same
class objects as the root namespace. Case and gravity imports use the
geometry packages directly; the flat case/gravity modules have been removed.
The removed year-suffixed Zhong aliases/module remain absent.

Concrete shell case classes consistently use `<Case><Geometry>Stokes`:

| Geometry | Kramer | Thieulot | Zhong |
| --- | --- | --- | --- |
| Cylindrical | `KramerCylindricalStokes` | `ThieulotCylindricalStokes` | — |
| Spherical | `KramerSphericalStokes` | `ThieulotSphericalStokes` | `ZhongSphericalStokes` |

The optional numeric oracle is `AssessKramerCylindricalStokes`.
`ZhongSphericalStokes.response()` returns `ZhongSphericalStokesResponse`;
it retains its mesh-free coefficient-response contract. The short `Zhong` and
`ZhongResponse` names are removed. `CylindricalStokes`, `SphericalStokes`, and
`AssessCylindricalStokes` remain aliases to the canonical Kramer classes;
Discovery distinguishes the contracts explicitly:

```python
uw.analytic.available()                  # field references, including optional assess
uw.analytic.available(kind="responses")  # ["ZhongSphericalStokes"]
uw.analytic.available(kind="all")        # both kinds, using canonical names
uw.analytic.describe("ZhongSphericalStokes")
uw.analytic.is_available("ZhongSphericalStokes")
```

The default field list preserves existing conformance sweeps. Response result
dataclasses are public types, rather than constructible references in this list.
`installed_only=True` filters missing optional dependencies for any kind.
`describe()` and `is_available()` accept both kinds and the historical Kramer
names; unknown reference names raise `ValueError` consistently.

`_thieulot_fields.py` is private field-assembly/evaluation machinery currently
shared by the two Thieulot classes. Their radial formulas stay in the case
modules. Kramer handles its distinct density branches inside each geometry
module. `_kramer_geoid.py` shares its stress-to-topography/geoid adapter across
both geometries. Geometry-specific kernels are in `cylindrical.gravity` and
`spherical.gravity`.

The native and assess cylindrical Kramer classes are siblings sharing the
private `_KramerCylindricalBase` wall/branch policy. The native class has no
assess dependency or inherited assess-specific evaluator. `_KramerGeoidMixin`
and `_ThieulotStokesBase` are implementation helpers. Each case module declares
`__all__` so wildcard imports retain supported classes/formula helpers without
exposing imported libraries or private bases. `AnalyticSolution` remains public
through the root namespace even though its implementation file is private.

Pure normalized spherical geoid/self-gravity functions and their
`GeoidResponse`/`SelfGravityResponse` types are owned by `spherical.gravity`.
Their existing `uw.postprocessing.geoid` names re-export the same objects.
Zhong depends directly on the pure gravity module; finite-element reaction
adapters remain in postprocessing.

## Parameter conventions

The cylindrical and spherical case constructors, exported formula helpers,
evaluators and gravity kernels describe their parameters and defaults in
NumPy-style docstrings. Shared public methods carry their contracts at their
owning base or mixin. Use one consistent physical or nondimensional scaling;
these references do not convert units. In spherical Kramer, ``m`` is harmonic
order; in spherical Thieulot, ``m`` sets the radial viscosity exponent.
Signed stress, source density and load coefficients have distinct meanings.

## Kramer cylindrical fields

The single `underworld3.analytic.cylindrical.kramer` module owns coefficients,
symbolic fields and cached numerical evaluators. Import its helpers from
`analytic.cylindrical.kramer`; there is no flat compatibility module.

`uw.analytic.KramerCylindricalStokes` supplies native symbolic velocity, pressure,
strain rate and total/radial/shear stress, without requiring `assess`. Named-field
evaluation uses cached vectorised NumPy evaluators with CSE; symbolic expressions
remain usable by the UW3 JIT. `compile_evaluators("velocity", "pressure")` can
prepare the evaluators before repeated measurements. The optional
`AssessKramerCylindricalStokes` class retains the former numerical reference wrapper.

The point-evaluation comparison is reproducible with
`scripts/benchmark_kramer_evaluators.py --output-json <common-output-path>`.
Construction, function generation, first evaluation and seven repeated warmed
measurements are reported separately. These are evaluator timings, not solver
speedups.

## Kramer spherical fields

`uw.analytic.KramerSphericalStokes` combines radial coefficients, symbolic velocity,
pressure, strain rate, total/radial/shear stress and cached array evaluators in
`analytic.spherical.kramer`. Select `ell`, `m`, `component` and `normalization`
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

## Thieulot fields and geoid scope

`ThieulotCylindricalStokes` and `ThieulotSphericalStokes` provide exact velocity,
pressure, viscosity, density, strain rate and stress with prescribed wall
velocities. They expose no `geoid_response()` adapter.

The scope decision of 2026-10-14 is to keep Thieulot geoid functionality out of
the current implementation and benchmark work. For future reference, distinguish
gravity generated by the density anomaly alone from gravity that also includes
boundary topography inferred from dynamic stress. The latter additionally needs
an explicit physical interface interpretation, density contrasts and reference
gravity. Both calculations would retain the existing UW3 potential/geoid sign
conventions and require consistent dimensional or nondimensional scales.

See the [future Thieulot gravity notes](../developer/subsystems/analytic-solutions.md#thieulot-gravity-future-reference).

## Zhong spherical-shell response oracle

`ZhongSphericalStokes` is a mesh-independent propagator-matrix oracle. It returns the
boundary response coefficients used by the Zhong et al. (2008) benchmark; it
does not implement the symbolic mesh-field contract above.

```{eval-rst}
.. automodule:: underworld3.analytic.spherical.zhong
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
.. automodule:: underworld3.analytic.cylindrical.gravity
   :members:

.. automodule:: underworld3.analytic.spherical.gravity
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

`analytic.spherical.gravity.spherical_physical_shell_response` returns physical
potential and geoid with explicit G, densities and inner/outer gravity. It
supports sheet/volume forcing and linear interface feedback, recording matrix
conditioning and residuals. This API is distinct from the normalized Zhong
response.

`postprocessing.spherical_interface_load` supports only continuous P2 scalar
traces on straight triangular facets of 3D meshes. Facets contribute on their
owning rank and shared DOFs are added collectively on the mesh communicator.
Temporary PETSc objects are released on exceptions.

The restricted `install_linear_spherical_stokes_sheet_load` applies radial
sheet forcing to linear ViscousFlowModel/P2 velocity systems with fixed constant or spatial viscosity after
homogeneous wall conditions and solver policy have been configured. It uses the
native assembled Jacobian and an explicit owned-facet weak RHS, whose moment
against rigid rotations is zero. It rejects nonlinear models, other natural
loads and nonzero essential data. Reinstall after any model/mesh setup change;
it is not a general nonlinear residual hook. This is pre-solve load installation;
its current postprocessing module location is an ownership issue for a separate
Stokes API change. The generic natural-load route
remains available for diagnostics, but the benchmark records a same-mesh MPI
compatibility discrepancy for that route.

`postprocessing.spherical_potential.solve_spherical_potential` keeps geometry,
labels and solver options caller-owned. Its full-ball continuation defaults to
the single-mode exterior Robin condition; an explicit analytic Dirichlet
comparison is also supported. It preserves an owned copy of the base solution
before borrowing DM work vectors, including for volume-plus-sheet sources.
The source radius must be supplied for each Dirichlet-comparison sheet.
