from . import tensors as tensor
from .vector_calculus import mesh_vector_calculus as vector_calculus
from .vector_calculus import (
    mesh_vector_calculus_cylindrical as vector_calculus_cylindrical,
)

from .functions import delta as delta_function
from .functions import L2_norm as L2_norm

# from .vector_calculus import (
#     mesh_vector_calculus_spherical_lonlat as vector_calculus_spherical_lonlat,
# )

from .vector_calculus import (
    mesh_vector_calculus_spherical as vector_calculus_spherical,
)
from .vector_calculus import (
    mesh_vector_calculus_spherical_surface2D_lonlat as vector_calculus_spherical_surface2D_lonlat,
)

# These could be wrapped so that they can be documented along with the math module
from underworld3.cython.petsc_maths import Integral
from underworld3.cython.petsc_maths import CellWiseIntegral
