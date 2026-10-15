# Spherical sheet solver ownership

The public configuration entry point is
`Stokes.install_linear_spherical_sheet_load(...)`. It lazily imports the
implementation from `utilities/_spherical_interface_load.py`. The retained
vector is the callback-owned residual load in `F(x)=Jx+b_sheet`; its algebraic
RHS is `-b_sheet`. Existing Cython reset and DM-rebuild hooks release the vector
only after destroying its owning SNES. Reinstallation replaces the old load.
The old postprocessing free function delegates to the method without duplicating
assembly or changing its signature.

`Poisson.solve_linear_spherical_sheet_sources(...)` delegates its base solve and
sheet correction to `utilities/_spherical_potential.py`. It is restricted to the
linear, unit-diffusivity, scalar continuous-P2 gravity operator. The distinct
caller-created P2 load field must exist before solver setup. Scientific volume
forcing and exterior conditions are configured by the caller/workflow. The
correction preserves a copy of the base solution before borrowing DM work
vectors, because pooled storage can alias the SNES solution.

```text
systems/solvers.py
  Stokes.install_linear_spherical_sheet_load
  Poisson.solve_linear_spherical_sheet_sources
utilities/_spherical_interface_load.py
  facet/P2 integration, distributed scalar/vector loads, trace integration
utilities/_spherical_potential.py
  Poisson correction, KSP/DM-vector lifetime, DG0 volume-region support
utilities/_spherical_solver_contract.py
  communicator and nonempty-local-chart restrictions
utilities/dm_labels.py
  existing dependency-free label reads
postprocessing/spherical_interface_load.py
  public trace projection, diagnostics and compatibility delegates
postprocessing/spherical_potential.py
  physical source/exterior workflow and result recovery
postprocessing/cylindrical_potential.py
  existing workflow using public Poisson methods
```

Private utility implementations do not import postprocessing. Public trace
projection stays in postprocessing and shares the same facet machinery.
InterfaceLoadDiagnostics is re-exported as the same class object. The old private
postprocessing mesh-contract module is removed; internal tests use its utility
owner. No speculative interface-load package or general nonlinear load API is
introduced. The geometry-specific names retain their actual restrictions.

The scalar sheet's uniform-mean correction and Stokes vector ownership are
preserved. Whether the mean correction is justified by an accuracy/refinement
budget remains a separate scientific audit. Relocation does not answer that
question or close layered-model velocity accuracy.

The boundary-condition capability guide documents the new entry points.
Nonlinear/free-surface/adaptation recipes retain their existing solver policies;
this relocation does not expand the restricted adapter to those formulations.
