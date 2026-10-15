# Post-processing

```{eval-rst}
.. automodule:: underworld3.postprocessing
   :members:
   :show-inheritance:
```

## Geoid and self-gravity responses

The geoid module provides coefficient-only spherical-shell and
cylindrical-annulus gravity operators plus adapters for completed
rotated-free-slip Stokes solves.

The spherical shell response adapters return **normalized coefficients** and
use SI constants only for the optional dimensionless self-gravity feedback.
The cylindrical physical-potential helpers accept a consistent caller-selected
unit system and return potential coefficients; dividing by reference gravity
gives geoid heights. No helper performs an implicit unit conversion.

Each public function documents its source signs, normalization, defaults,
return values and collective-call requirements in the sections below.

```{eval-rst}
.. automodule:: underworld3.postprocessing.geoid
   :members:
   :imported-members:
   :show-inheritance:
```

## Physical cylindrical potential

```{eval-rst}
.. automodule:: underworld3.postprocessing.cylindrical_potential
   :members:
   :show-inheritance:
```

## Spherical sheet loads and trace projection

The scalar Poisson load routine returns a caller-owned vector and removes its
discrete mean source. The Stokes sheet installer returns a callback-owned
**residual** load: it installs $F(x)=Jx+b_{\mathrm{sheet}}$, so the corresponding
linear equation is $Jx=-b_{\mathrm{sheet}}$. These signs and ownership rules
are part of the API contract.

Despite its current module path, `install_linear_spherical_stokes_sheet_load`
is **solver setup**: call it before `stokes.solve()`. It installs a conforming
internal-interface forcing term rather than deriving a quantity from a completed
solution. Its placement alongside the potential trace integrators is an
ownership issue; a future solver API should expose this load through a Stokes
method, with assembly support under solver utilities. The current path remains
available while that separate API change is reviewed.

```{eval-rst}
.. automodule:: underworld3.postprocessing.spherical_interface_load
   :members:
   :show-inheritance:
```

## Physical spherical potential

```{eval-rst}
.. automodule:: underworld3.postprocessing.spherical_potential
   :members:
   :show-inheritance:
```
