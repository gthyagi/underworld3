# Analytical geometry parameter documentation — 15 October 2026

Audited the cylindrical Kramer/Thieulot/gravity modules and spherical
Kramer/Thieulot/Zhong/gravity modules. Completed missing NumPy-style parameter
contracts for constructors, exported radial/symbolic helpers, evaluators,
forcing and wall methods, gravity kernels and inherited public methods.
Zhong's response payload now describes each of its nine fields. Existing
complete documentation was retained and checked alongside the additions.

The geometry-module audit covers 62 callable/constructor parameter contracts
and 350 named arguments. Including shared base/mixin owners, the runtime
inspection covers 82 contracts and 400 arguments. These are coverage counts,
not counts of newly added APIs or newly written docstrings.

Descriptions distinguish spherical harmonic order from Thieulot's viscosity
exponent, physical from normalized gravity quantities, surface from volume
density, fluid stress from wall reactions, and residual from force signs.
They specify defaults, supported branches, units/scaling and result shapes.
The optional assess constructor's malformed table was replaced by prose.
Geometry-specific module reference headings avoid duplicate Sphinx labels.

## Validation

- All ten changed numerical Python files have identical executable ASTs after
  removing docstrings: formulas, signatures, defaults and solver behavior are
  unchanged.
- Runtime inspection verifies every audited argument has a nonempty parameter
  description and that Napoleon parses the NumPy contracts.
- A fresh isolated Mac wheel/package verifies Python/native origins. Later
  docstring-only corrections were copied into that isolated package after AST
  verification; no shared runtime was overwritten.
- The focused geometry API HTML builds with Sphinx warnings treated as errors,
  with zero warnings. The full API HTML has eight unrelated pre-existing
  diagnostics in mesh, surface and advection docs and references outside the
  API-only source tree. No changed analytical contract contributes a diagnostic.
- Logs, source audit, runtime payload and rendered HTML are retained in the
  benchmark common output tree under
  `consolidation_2026-10-14/analytic_geometry_docs_2026-10-15/`.

## Sheet-load API ownership

`install_linear_spherical_stokes_sheet_load` installs an internal-interface
Stokes residual before solving; it is solver setup, not postprocessing.
Its current location shares interface assembly utilities with potential
recovery. The recommended future ownership is a public Stokes method backed
by solver-load assembly utilities. This documentation pass records that issue
in the analytic/postprocessing API pages without relocating or renaming the API.

## Gadi update policy

The user explicitly requests batched Gadi rebuilding. Fetch this validated
source milestone only. Documentation, comments and formatting do not require
recompilation; rebuild after substantial numerical/native changes or before a
model/test needs changed runtime behavior. Preserve existing int32 installs,
PETSc and SIF. No Gadi build/model job or int64 job is part of this change.
