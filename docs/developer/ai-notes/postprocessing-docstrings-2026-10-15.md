# Postprocessing API docstrings — 15 October 2026

Completed NumPy/Sphinx parameter, return and exception documentation for all
20 functions exposed by the four public postprocessing modules, including the
eight gravity kernels re-exported from their geometry owners. All 200 named
arguments (including the two compatibility wrappers' keyword mappings) are
covered. Eight response/diagnostic dataclasses now describe every returned
field. The private native spherical mesh-contract helper is documented too.

The descriptions distinguish physical potential from normalized spherical
response coefficients, explicit signed buoyancy scales, harmonic normalization,
source support, solver options and collective-call restrictions. Poisson sheet
vectors are caller-owned copies and have their discrete mean removed. Stokes
sheet vectors belong to the callback and enter the residual as
`F(x)=Jx+b_sheet`, so the corresponding algebraic RHS is `-b_sheet`.

`docs/api/postprocessing.md` now renders all four modules and the re-exported
gravity functions. Both Sphinx configurations use `napoleon_use_ivar=True` so
NumPy dataclass attributes are not registered twice by autodoc.

## Validation

- Removing docstrings gives identical executable ASTs in all seven modified
  numerical source files; signatures, defaults, formulas and solver settings
  are unchanged.
- A fresh noneditable wheel/package was built in an isolated Mac output prefix.
  Installed Python and native extension origins are verified, and the seven
  installed Python files match their source copies. Existing runtimes are preserved.
- Runtime inspection checks every exported function argument against its
  `Parameters` section and every dataclass field against `Attributes`.
  Napoleon parses the function docs into parameter/return fields successfully.
- The complete postprocessing page builds with Sphinx warnings treated as
  errors (`-W`), with zero warnings. Full API HTML builds with nine remaining
  unrelated diagnostics: pre-existing malformed Kramer/advection tables, mesh
  and surface docstrings, and references to documents outside the API-only tree.
  None originate in the postprocessing page or changed gravity docstrings.
- Rendered pages, build logs, wheel and runtime audit are retained under the
  benchmark repository's common output tree in
  `consolidation_2026-10-14/postprocessing_docs_2026-10-15/`.

The validated source is to be fetched on Gadi and rebuilt in a fresh int32
prefix inside the existing SIF, preserving older prefixes and PETSc. No
benchmark-model rerun or int64 job is required by this documentation change.

## Scope notes

Legacy flat `.data` accesses and free-function solver adapters remain in the
existing numerical implementation. They differ from the current style charter
and are noted for the maintainer; this documentation task does not refactor
those APIs or alter their behavior. The unrelated full-API diagnostics above
also remain outside this change.
