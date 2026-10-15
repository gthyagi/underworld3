# Spherical sheet ownership migration — 15 October 2026

Implemented the agreed solver-method/private-utility structure. Stokes exposes
`install_linear_spherical_sheet_load`; Poisson exposes
`solve_linear_spherical_sheet_sources`. Both lazily import their implementation.

Three private utility modules own shared interface assembly, restricted Poisson
correction/DG0 support and the existing native mesh contract. Postprocessing
retains scientific potential workflows, public trace projection, response
conversion and legacy public delegates. The private postprocessing mesh-contract
module is removed; internal consumers use its utility owner. The existing Cython
cleanup hooks and `dm_labels.py` leaf module are unchanged. See the
[architecture decision](../design/spherical-sheet-solver-ownership.md).

## Numerical and API boundaries

The seven relocated interface algorithms match their originals after normalizing
private names and the standard scalar array access shape. No force sign,
quadrature default, mean-correction policy, Jacobian, preconditioner or iteration
cap was changed. The native mesh-contract implementation is byte-identical.
DG0 construction now uses one bulk array write and reuses safe label reads;
independent constant/linear cell moments and empty-rank checks validate it.

The new Poisson method explicitly rejects unsupported operators: unit-diffusivity
DiffusionModel, distinct scalar continuous-P2 load field and P2 solution are
required. The existing potential workflow already supplies these conditions.
It performs the same base solve plus sheet correction and preserves the pooled
vector alias regression. The old public wrapper signatures are unchanged and
InterfaceLoadDiagnostics is re-exported as the same class object.

## Validation

- Fresh isolated noneditable Mac wheel/package; Python/native origins and six
  relevant installed source files verified. Local PETSc integer width is 32.
- 138 serial component tests pass, one subcommunicator-only test skips.
- 27 component tests pass on each of eight MPI ranks, including force moments,
  lifecycle release/reinstallation, volume regions, pool aliasing, mesh contracts,
  public method input guards and exact quadratic base-solution recovery.
- Focused Sphinx HTML for both methods and relocated public delegates builds with
  `-W`, zero warnings. All 18 method arguments have parameter documentation.
- Test coverage scanner reaches all 370 files; scripts/test.sh now includes the
  relevant 108x band. Changed implementation files pass deprecated-pattern checks.
- Benchmark call/import migrations parse successfully. The early runtime guard
  accepts the new installed package and rejects the earlier documentation package
  before mesh setup. Source and runtime metadata are kept distinct.

Artifacts are retained in the common benchmark output tree at
`consolidation_2026-10-14/interface_structure_2026-10-15/`, with the final installed
package under `final/python/`. No shared Mac runtime was overwritten.

## Remaining scientific/deployment work

The Poisson mean-removal accuracy/refinement audit and layered-model velocity
accuracy remain open. This migration provides no new production accuracy result.
Fetch the validated feature source on Gadi, preserve the home checkout and old
installs, and batch a fresh int32 install before a migrated benchmark needs the
new runtime methods. No Gadi rebuild/model or int64 job is submitted here.
