r"""Prescribed normal-derivative jumps for cylindrical Poisson potentials.

For -Delta Phi=4*pi*G*(rho+sigma*delta_Gamma), let n point radially
outward and [a]=a_out-a_in. Then [Phi]=0 and [d_n Phi]=-4*pi*G*sigma.
We use Phi=Phi_regular+L with L=-2*pi*G*sum sigma_j(x)*abs(r-R_j).
For a wall at R, the buffer-side variant uses abs(r-R)+s*(r-R), with
s=-1 at the inner wall and s=+1 at the outer wall. It vanishes identically
in the mantle, retaining the same derivative jump without altering its trial
space there. Internal sheets retain the two-sided lifting unless a compact half-width w
is supplied. Then multiply |r-R| by chi(t), t=|r-R|/w, with
chi=(1-t)^3*(1+3*t+6*t^2) for t<1 and zero otherwise. chi(0)=1,
and chi, chi', chi'' vanish at t=1. Include the cutoff derivative in grad(L)
and split quadrature at R-w and R+w. This removes the artificial extension
through weak far-field regions without changing the prescribed jump.
The given spatial density extends each jump off its circle; it is not a
harmonic reconstruction or the analytical gravity solution. Solving
 a(Phi_regular,v)=b(v)-a(L,v)
removes the derivative jump from the unknown. Both volume and Robin parts
of a(L,v) must be included. For linear interface feedback
sigma=sigma0+alpha*Phi, construct L from sigma0 alone and solve
 a(u,v)-4*pi*G*sum int_Gamma alpha*u*v
   = b0(v)-a(L,v)+4*pi*G*sum int_Gamma alpha*L*v.
This is an algebraic substitution, not an extraction of the unknown feedback
jump. The cylindrical feedback convention follows Simons (1996), Appendix B;
the known-lift feedback RHS here is a UW3 extension. Return/evaluate the total field without projecting
L into an ordinary polynomial FE space.

Reference: Guzman, Sanchez & Sarkis (2016), Higher-order finite element methods
for elliptic problems with interfaces, doi:10.1051/m2an/2015093. Their curved
interface correction functions motivate this jump-aware decomposition. These
distance liftings and polygonal Robin adaptations are UW3 constructions;
their convergence theorem is not asserted for it. See also LeVeque & Li (1994),
doi:10.1137/0731054, for elliptic interface conditions with singular sources.
"""

import numpy as np
import sympy as sp

from ._circular_density_potential import _density


class NormalDerivativeJump:
    """Known spatial lifting for an annular domain and prescribed mass sheets."""

    def __init__(self, mesh, sheets, gravitational_constant, radial_sides=None, localized_widths=None):
        self.sources = [(float(radius), sp.sympify(sigma)) for radius, sigma, _ in sheets if sp.sympify(sigma) != 0]
        # side=-1 confines a wall lift to r<R; side=+1 to r>R.
        # side=0 retains the original two-sided distance lift.
        self.sides = {radius: 0 for radius, _ in self.sources} if radial_sides is None else radial_sides
        if any(self.sides.get(radius) not in (-1, 0, 1) for radius, _ in self.sources):
            raise ValueError("Each active sheet needs a radial side of -1, 0 or 1")
        self.widths = {} if localized_widths is None else localized_widths
        if any(not np.isfinite(w) or w <= 0 or self.sides.get(r,0) != 0 for r,w in self.widths.items()):
            raise ValueError("Compact jump lifts need positive finite widths and two-sided radial support")
        self.break_radii = tuple(sorted({edge for r,_ in self.sources if r in self.widths
                                        for edge in (r-self.widths[r],r+self.widths[r])}))
        self.scale = -2*np.pi*float(gravitational_constant)
        radius = sp.sqrt(mesh.N.x**2+mesh.N.y**2)
        terms = []
        for r,sigma in self.sources:
            distance = sp.Abs(radius-sp.Float(r))
            radial = distance+self.sides[r]*(radius-sp.Float(r))
            if r in self.widths:
                t = distance/sp.Float(self.widths[r])
                radial = sp.Piecewise((distance*(1-t)**3*(1+3*t+6*t*t),t<1),(0,True))
            terms.append(sigma*radial)
        self.expression = -2*sp.pi*sp.nsimplify(gravitational_constant, rational=True)*sum(terms)
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
            if source_radius in self.widths:
                t = abs(distance)/self.widths[source_radius]
                inside = t < 1
                u = np.minimum(t,1.)
                cutoff = (1-u)**3*(1+3*u+6*u*u)
                derivative = -30*u*u*(1-u)**2
                value = np.where(inside,abs(distance)*cutoff,0.)
                slope = np.where(inside,np.sign(distance)*(cutoff+u*derivative),0.)
            gradient += self.scale*(density.gradient(cell, points)*value[:, None]
                +density(cell, points)[:, None]*slope[:, None]*normal)
        return gradient
