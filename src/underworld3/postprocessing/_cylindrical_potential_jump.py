r"""Prescribed normal-derivative jumps for cylindrical Poisson potentials.

For -Delta Phi=4*pi*G*(rho+sigma*delta_Gamma), let n point radially
outward and [a]=a_out-a_in. Then [Phi]=0 and [d_n Phi]=-4*pi*G*sigma.
We use Phi=Phi_regular+L with L=-2*pi*G*sum sigma_j(x)*abs(r-R_j).
For a wall at R, the buffer-side variant uses abs(r-R)+s*(r-R), with
s=-1 at the inner wall and s=+1 at the outer wall. It vanishes identically
in the mantle, retaining the same derivative jump without altering its trial
space there. Internal sheets retain the two-sided lifting.
The given spatial density extends each jump off its circle; it is not a
harmonic reconstruction or the analytical gravity solution. Solving
 a(Phi_regular,v)=b(v)-a(L,v)
removes the derivative jump from the unknown. Both volume and Robin parts
of a(L,v) must be included. Return/evaluate the total field without projecting
L into an ordinary polynomial FE space.

Reference: Guzman, Sanchez & Sarkis (2016), Higher-order finite element methods
for elliptic problems with interfaces, doi:10.1051/m2an/2015093. Their curved
interface correction functions motivate this jump-aware decomposition. This
distance liftings and polygonal Robin adaptation is a UW3 construction;
their convergence theorem is not asserted for it. See also LeVeque & Li (1994),
doi:10.1137/0731054, for elliptic interface conditions with singular sources.
"""

import numpy as np
import sympy as sp

from ._circular_density_potential import _density


class NormalDerivativeJump:
    """Known spatial lifting for an annular domain and prescribed mass sheets."""

    def __init__(self, mesh, sheets, gravitational_constant, radial_sides=None):
        self.sources = [(float(radius), sp.sympify(sigma)) for radius, sigma, _ in sheets if sp.sympify(sigma) != 0]
        # side=-1 confines a wall lift to r<R; side=+1 to r>R.
        # side=0 retains the original two-sided distance lift.
        self.sides = {radius: 0 for radius, _ in self.sources} if radial_sides is None else radial_sides
        if any(self.sides.get(radius) not in (-1, 0, 1) for radius, _ in self.sources):
            raise ValueError("Each active sheet needs a radial side of -1, 0 or 1")
        self.scale = -2*np.pi*float(gravitational_constant)
        radius = sp.sqrt(mesh.N.x**2+mesh.N.y**2)
        self.expression = -2*sp.pi*sp.nsimplify(gravitational_constant, rational=True)*sum(
            sigma*(sp.Abs(radius-sp.Float(r))+self.sides[r]*(radius-sp.Float(r))) for r, sigma in self.sources)
        self.densities = []

    def prepare(self, mesh, vertices):
        self.densities = [(radius, _density(sigma, mesh, vertices)) for radius, sigma in self.sources]
        for _, density in self.densities:
            if any(not field.field.continuous for field, _ in density.sources):
                raise ValueError("Jump lifting needs continuous spatial density fields")

    def gradient(self, cell, points):
        radius = np.linalg.norm(points, axis=1)
        normal = np.divide(points, radius[:, None], out=np.zeros_like(points), where=radius[:, None] > 0)
        gradient = np.zeros_like(points)
        for source_radius, density in self.densities:
            distance = radius-source_radius
            side = self.sides[source_radius]
            value = abs(distance)+side*distance
            slope = np.sign(distance)+side
            gradient += self.scale*(density.gradient(cell, points)*value[:, None]
                +density(cell, points)[:, None]*slope[:, None]*normal)
        return gradient
