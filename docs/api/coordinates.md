# Coordinate Systems

## CoordinateSystem

The base class for coordinate system handling on meshes.

Each mesh has a distinct symbolic Cartesian system (`mesh.N`). Its internal
name includes the mesh instance number, so SymPy cannot reuse another mesh's
cached coordinate expressions. Use `mesh.X` or `mesh.N` when constructing
expressions; transferring a field between meshes still requires an explicit
field transfer. Generated coordinate kernels continue to use `petsc_x`.

```{eval-rst}
.. autoclass:: underworld3.coordinates.CoordinateSystem
   :members:
   :show-inheritance:
```

## CoordinateSystemType

Enumeration of supported coordinate system types.

```{eval-rst}
.. autoclass:: underworld3.coordinates.CoordinateSystemType
   :members:
   :show-inheritance:
```
