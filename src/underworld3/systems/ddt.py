import sympy
from sympy import sympify
import numpy as np

from typing import Optional, Callable, Union

import underworld3 as uw
from underworld3 import VarType

# from underworld3.swarm import NodalPointSwarm, Swarm
import underworld3.timing as timing
from underworld3.utilities._api_tools import uw_object


## We need a pure Eulerian one of these too

# class Eulerian(uw_object):
# etc etc...


class SemiLagrangian(uw_object):
    r"""Nodal-Swarm  Lagrangian History Manager:
    This manages the update of a Lagrangian variable, $\psi$ on the swarm across timesteps.
    $$\quad \psi_p^{t-n\Delta t} \leftarrow \psi_p^{t-(n-1)\Delta t}\quad$$
    $$\quad \psi_p^{t-(n-1)\Delta t} \leftarrow \psi_p^{t-(n-2)\Delta t} \cdots\quad$$
    $$\quad \psi_p^{t-\Delta t} \leftarrow \psi_p^{t}$$
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        psi_fn: sympy.Function,
        V_fn: sympy.Function,
        vtype: uw.VarType,
        degree: int,
        continuous: bool,
        varsymbol: Optional[str] = r"u",
        verbose: Optional[bool] = False,
        bcs=[],
        order=1,
        smoothing=0.0,
        under_relaxation=0.0,
        bc_mask_fn=1,
    ):
        super().__init__()

        self.mesh = mesh
        self.bcs = bcs
        self.verbose = verbose
        self.degree = degree

        # meshVariables are required for:
        #
        # u(t) - evaluation of u_fn at the current time
        # u*(t) - u_* evaluated from

        # psi is evaluated/stored at `order` timesteps. We can't
        # be sure if psi is a meshVariable or a function to be evaluated
        # psi_star is reaching back through each evaluation and has to be a
        # meshVariable (storage)

        self._psi_fn = psi_fn
        self.V_fn = V_fn
        self.order = order
        self.bc_mask_fn = bc_mask_fn

        psi_star = []
        self.psi_star = psi_star

        for i in range(order):
            self.psi_star.append(
                uw.discretisation.MeshVariable(
                    f"psi_star_sl_{self.instance_number}_{i}",
                    self.mesh,
                    vtype=vtype,
                    degree=degree,
                    continuous=continuous,
                    varsymbol=rf"{{ {varsymbol}^{{ {'*'*(i+1)} }} }}",
                )
            )

        # We just need one swarm since this is inherently a sequential operation
        nswarm = uw.swarm.NodalPointSwarm(self.psi_star[0])
        self._nswarm_psi = nswarm

        # The projection operator for mapping swarm values to the mesh - needs to be different for
        # each variable type, unfortunately ...

        if vtype == uw.VarType.SCALAR:
            self._psi_star_projection_solver = uw.systems.solvers.SNES_Projection(
                self.mesh, self.psi_star[0], verbose=False
            )
        elif vtype == uw.VarType.VECTOR:
            self._psi_star_projection_solver = (
                uw.systems.solvers.SNES_Vector_Projection(
                    self.mesh, self.psi_star[0], verbose=False
                )
            )
        elif vtype == uw.VarType.SYM_TENSOR or vtype == uw.VarType.TENSOR:
            self._WorkVar = uw.discretisation.MeshVariable(
                f"W_star_slcn_{self.instance_number}",
                self.mesh,
                vtype=uw.VarType.SCALAR,
                degree=degree,
                continuous=continuous,
                varsymbol=r"W^{*}",
            )
            self._psi_star_projection_solver = (
                uw.systems.solvers.SNES_Tensor_Projection(
                    self.mesh, self.psi_star[0], self._WorkVar, verbose=False
                )
            )

        self._psi_star_projection_solver.uw_function = self.psi_fn
        self._psi_star_projection_solver.bcs = bcs
        self._psi_star_projection_solver.smoothing = smoothing

        return

    @property
    def psi_fn(self):
        return self._psi_fn

    @psi_fn.setter
    def psi_fn(self, new_fn):
        self._psi_fn = new_fn
        self._psi_star_projection_solver.uw_function = self._psi_fn
        return

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        # display(Latex(r"$\quad\psi = $ " + self.psi._repr_latex_()))
        # display(
        #     Latex(
        #         r"$\quad\Delta t_{\textrm{phys}} = $ "
        #         + sympy.sympify(self.dt_physical)._repr_latex_()
        #     )
        # )
        display(Latex(rf"$\quad$History steps = {self.order}"))

    def update(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
        dt_physical: Optional = None,
    ):
        self.update_pre_solve(dt, evalf, verbose, dt_physical)
        return

    def update_post_solve(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
        dt_physical: Optional[float] = None,
    ):
        return

    def update_pre_solve(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
        dt_physical: Optional[float] = None,
    ):

        ## Progress from the oldest part of the history
        # 1. Copy the stored values down the chain

        if dt_physical is not None:
            phi = min(1, dt / dt_physical)
        else:
            phi = sympy.sympify(1)

        for i in range(self.order - 1, 0, -1):
            with self.mesh.access(self.psi_star[i]):
                self.psi_star[i].data[...] = (
                    phi * self.psi_star[i - 1].data[...]
                    + (1 - phi) * self.psi_star[i].data[...]
                )

        # 2. Compute the upstream values

        # We use the u_star variable as a working value here so we have to work backwards
        for i in range(self.order - 1, -1, -1):
            with self._nswarm_psi.access(self._nswarm_psi._X0):
                self._nswarm_psi._X0.data[...] = self._nswarm_psi.data[...]

            # march nodes backwards along characteristics
            self._nswarm_psi.advection(
                self.V_fn,
                -dt,
                order=2,
                corrector=False,
                restore_points_to_domain_func=self.mesh.return_coords_to_bounds,
                evalf=evalf,
            )

            if i == 0:
                # Recalculate u_star from u_fn
                self._psi_star_projection_solver.uw_function = self.psi_fn
                self._psi_star_projection_solver.solve(verbose=verbose)

            if evalf:
                with self._nswarm_psi.access(self._nswarm_psi.swarmVariable):
                    for d in range(self.psi_star[i].shape[1]):
                        self._nswarm_psi.swarmVariable.data[:, d] = uw.function.evalf(
                            self.psi_star[i].sym[d], self._nswarm_psi.data
                        )
            else:
                with self._nswarm_psi.access(self._nswarm_psi.swarmVariable):
                    for d in range(self.psi_star[i].shape[1]):
                        self._nswarm_psi.swarmVariable.data[:, d] = (
                            uw.function.evaluate(
                                self.psi_star[i].sym[d], self._nswarm_psi.data
                            )
                        )

            # with self.mesh.access():
            #     print("1:", self.psi_star[0].data, flush=True)

            # with self._nswarm_psi.access():
            #     print("1S:", self._nswarm_psi.swarmVariable.data, flush=True)

            # restore coords (will call dm.migrate after context manager releases)
            with self._nswarm_psi.access(self._nswarm_psi.particle_coordinates):
                self._nswarm_psi.data[...] = self._nswarm_psi._nX0.data[...]

            # Now project to the mesh using bc's to obtain u_star

            self._psi_star_projection_solver.uw_function = (
                self._nswarm_psi.swarmVariable.sym
            )

            self._psi_star_projection_solver.solve()

            # Copy data from the projection operator if required
            if i != 0:
                with self.mesh.access(self.psi_star[i]):
                    self.psi_star[i].data[...] = self.psi_star[0].data[...]

        return

    def bdf(self, order=None):
        r"""Backwards differentiation form for calculating DuDt
        Note that you will need `bdf` / $\delta t$ in computing derivatives"""

        if order is None:
            order = self.order
        else:
            order = max(1, min(self.order, order))

        with sympy.core.evaluate(True):
            if order == 1:
                bdf0 = self.psi_fn - self.psi_star[0].sym

            elif order == 2:
                bdf0 = (
                    3 * self.psi_fn / 2
                    - 2 * self.psi_star[0].sym
                    + self.psi_star[1].sym / 2
                )

            elif order == 3:
                bdf0 = (
                    11 * self.psi_fn / 6
                    - 3 * self.psi_star[0].sym
                    + 3 * self.psi_star[1].sym / 2
                    - self.psi_star[2].sym / 3
                )

        return bdf0

    def adams_moulton_flux(self, order=None):
        if order is None:
            order = self.order
        else:
            order = max(1, min(self.order, order))

        with sympy.core.evaluate(True):
            if order == 1:
                am = (self.psi_fn + self.psi_star[0].sym) / 2

            elif order == 2:
                am = (
                    5 * self.psi_fn + 8 * self.psi_star[0].sym - self.psi_star[1].sym
                ) / 12

            elif order == 3:
                am = (
                    9 * self.psi_fn
                    + 19 * self.psi_star[0].sym
                    - 5 * self.psi_star[1].sym
                    + self.psi_star[2].sym
                ) / 24

        return am


## Consider Deprecating this one - it is the same as the Lagrangian_Swarm but
## sets up the swarm for itself. This does not have a practical use-case - the swarm version
## is slower, more cumbersome, and less stable / accurate. The only reason to use
## it is if there is an existing swarm that we can re-purpose.


class Lagrangian(uw_object):
    r"""Swarm-based Lagrangian History Manager:

    This manages the update of a Lagrangian variable, $\psi$ on the swarm across timesteps.

    $\quad \psi_p^{t-n\Delta t} \leftarrow \psi_p^{t-(n-1)\Delta t}\quad$

    $\quad \psi_p^{t-(n-1)\Delta t} \leftarrow \psi_p^{t-(n-2)\Delta t} \cdots\quad$

    $\quad \psi_p^{t-\Delta t} \leftarrow \psi_p^{t}$
    """

    instances = 0  # count how many of these there are in order to create unique private mesh variable ids

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        psi_fn: sympy.Function,
        V_fn: sympy.Function,
        vtype: uw.VarType,
        degree: int,
        continuous: bool,
        varsymbol: Optional[str] = r"u",
        verbose: Optional[bool] = False,
        bcs=[],
        order=1,
        smoothing=0.0,
        fill_param=3,
    ):
        super().__init__()

        # create a new swarm to manage here
        dudt_swarm = uw.swarm.Swarm(mesh)

        self.mesh = mesh
        self.swarm = dudt_swarm
        self.psi_fn = psi_fn
        self.V_fn = V_fn
        self.verbose = verbose
        self.order = order

        psi_star = []
        self.psi_star = psi_star

        for i in range(order):
            print(f"Creating psi_star[{i}]")
            self.psi_star.append(
                uw.swarm.SwarmVariable(
                    f"psi_star_sw_{self.instance_number}_{i}",
                    self.swarm,
                    vtype=vtype,
                    proxy_degree=degree,
                    proxy_continuous=continuous,
                    varsymbol=rf"{varsymbol}^{{ {'*'*(i+1)} }}",
                )
            )

        dudt_swarm.populate(fill_param)

        return

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(Latex(r"$\quad\psi = $ " + self.psi._repr_latex_()))
        display(
            Latex(
                r"$\quad\Delta t_{\textrm{phys}} = $ "
                + sympy.sympify(self.dt_physical)._repr_latex_()
            )
        )
        display(Latex(rf"$\quad$History steps = {self.order}"))

    ## Note: We may be able to eliminate this
    ## The SL updater and the Lag updater have
    ## different sequencing because of the way they
    ## update the history. It makes more sense for the
    ## full Lagrangian swarm to be updated after the solve
    ## and this means we have to grab the history values first.

    def update(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        self.update_post_solve(dt, evalf, verbose)
        return

    def update_pre_solve(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        return

    def update_post_solve(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        for h in range(self.order - 1):
            i = self.order - (h + 1)

            # copy the information down the chain
            print(f"Lagrange order = {self.order}")
            print(f"Lagrange copying {i-1} to {i}")

            with self.swarm.access(self.psi_star[i]):
                self.psi_star[i].data[...] = self.psi_star[i - 1].data[...]

        # Now update the swarm variable

        if evalf:
            psi_star_0 = self.psi_star[0]
            with self.swarm.access(psi_star_0):
                for i in range(psi_star_0.shape[0]):
                    for j in range(psi_star_0.shape[1]):
                        updated_psi = uw.function.evalf(
                            self.psi_fn[i, j], self.swarm.data
                        )
                        psi_star_0[i, j].data[:] = updated_psi

        else:
            psi_star_0 = self.psi_star[0]
            with self.swarm.access(psi_star_0):
                for i in range(psi_star_0.shape[0]):
                    for j in range(psi_star_0.shape[1]):
                        updated_psi = uw.function.evaluate(
                            self.psi_fn[i, j], self.swarm.data
                        )
                        psi_star_0[i, j].data[:] = updated_psi

        # Now update the swarm locations

        self.swarm.advection(
            self.V_fn,
            delta_t=dt,
            restore_points_to_domain_func=self.mesh.return_coords_to_bounds,
        )

    def bdf(self, order=None):
        r"""Backwards differentiation form for calculating DuDt
        Note that you will need `bdf` / $\delta t$ in computing derivatives"""

        if order is None:
            order = self.order

        with sympy.core.evaluate(True):
            if order == 0:  # special case - no history term (catch )
                bdf0 = sympy.simpify[0]

            if order == 1:
                bdf0 = self.psi_fn - self.psi_star[0].sym

            elif order == 2:
                bdf0 = (
                    3 * self.psi_fn / 2
                    - 2 * self.psi_star[0].sym
                    + self.psi_star[1].sym / 2
                )

            elif order == 3:
                bdf0 = (
                    11 * self.psi_fn / 6
                    - 3 * self.psi_star[0].sym
                    + 3 * self.psi_star[1].sym / 2
                    - self.psi_star[2].sym / 3
                )

        return bdf0

    def adams_moulton_flux(self, order=None):
        if order is None:
            order = self.order

        with sympy.core.evaluate(True):
            if order == 0:  # Special case - no history term
                am = self.psi_fn

            elif order == 1:
                am = (self.psi_fn + self.psi_star[0].sym) / 2

            elif order == 2:
                am = (
                    5 * self.psi_fn + 8 * self.psi_star[0].sym - self.psi_star[1].sym
                ) / 12

            elif order == 3:
                am = (
                    9 * self.psi_fn
                    + 19 * self.psi_star[0].sym
                    - 5 * self.psi_star[1].sym
                    + self.psi_star[2].sym
                ) / 24

        return am


class Lagrangian_Swarm(uw_object):
    r"""Swarm-based Lagrangian History Manager:
    This manages the update of a Lagrangian variable, $\psi$ on the swarm across timesteps.

    $\quad \psi_p^{t-n\Delta t} \leftarrow \psi_p^{t-(n-1)\Delta t}\quad$

    $\quad \psi_p^{t-(n-1)\Delta t} \leftarrow \psi_p^{t-(n-2)\Delta t} \cdots\quad$

    $\quad \psi_p^{t-\Delta t} \leftarrow \psi_p^{t}$
    """

    instances = 0  # count how many of these there are in order to create unique private mesh variable ids

    @timing.routine_timer_decorator
    def __init__(
        self,
        swarm: uw.swarm.Swarm,
        psi_fn: sympy.Function,
        vtype: uw.VarType,
        degree: int,
        continuous: bool,
        varsymbol: Optional[str] = r"u",
        verbose: Optional[bool] = False,
        bcs=[],
        order=1,
        smoothing=0.0,
        step_averaging=2,
    ):
        super().__init__()

        self.mesh = swarm.mesh
        self.swarm = swarm
        self.psi_fn = psi_fn
        self.verbose = verbose
        self.order = order
        self.step_averaging = step_averaging

        psi_star = []
        self.psi_star = psi_star

        for i in range(order):
            print(f"Creating psi_star[{i}]")
            self.psi_star.append(
                uw.swarm.SwarmVariable(
                    f"psi_star_sw_{self.instance_number}_{i}",
                    self.swarm,
                    vtype=vtype,
                    proxy_degree=degree,
                    proxy_continuous=continuous,
                    varsymbol=rf"{varsymbol}^{{ {'*'*(i+1)} }}",
                )
            )

        return

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        super()._object_viewer()

        ## feedback on this instance
        display(Latex(r"$\quad\psi = $ " + self.psi._repr_latex_()))
        display(
            Latex(
                r"$\quad\Delta t_{\textrm{phys}} = $ "
                + sympy.sympify(self.dt_physical)._repr_latex_()
            )
        )
        display(Latex(rf"$\quad$History steps = {self.order}"))

    ## Note: We may be able to eliminate this
    ## The SL updater and the Lag updater have
    ## different sequencing because of the way they
    ## update the history. It makes more sense for the
    ## full Lagrangian swarm to be updated after the solve
    ## and this means we have to grab the history values first.

    def update(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        self.update_post_solve(dt, evalf, verbose)
        return

    def update_pre_solve(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        return

    def update_post_solve(
        self,
        dt: float,
        evalf: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        for h in range(self.order - 1):
            i = self.order - (h + 1)

            # copy the information down the chain
            if verbose:
                print(f"Lagrange swarm order = {self.order}", flush=True)
                print(
                    f"Mesh interpolant order = {self.psi_star[0]._meshVar.degree}",
                    flush=True,
                )
                print(f"Lagrange swarm copying {i-1} to {i}", flush=True)

            with self.swarm.access(self.psi_star[i]):
                self.psi_star[i].data[...] = self.psi_star[i - 1].data[...]

        phi = 1 / self.step_averaging

        # Now update the swarm variable
        if evalf:
            psi_star_0 = self.psi_star[0]
            with self.swarm.access(psi_star_0):
                for i in range(psi_star_0.shape[0]):
                    for j in range(psi_star_0.shape[1]):
                        updated_psi = uw.function.evalf(
                            self.psi_fn[i, j], self.swarm.data
                        )
                        psi_star_0[i, j].data[:] = (
                            phi * updated_psi + (1 - phi) * psi_star_0[i, j].data[:]
                        )
        else:
            psi_star_0 = self.psi_star[0]
            with self.swarm.access(psi_star_0):
                for i in range(psi_star_0.shape[0]):
                    for j in range(psi_star_0.shape[1]):
                        updated_psi = uw.function.evaluate(
                            self.psi_fn[i, j], self.swarm.data
                        )
                        psi_star_0[i, j].data[:] = (
                            phi * updated_psi + (1 - phi) * psi_star_0[i, j].data[:]
                        )

        return

    def bdf(self, order=None):
        r"""Backwards differentiation form for calculating DuDt
        Note that you will need `bdf` / $\delta t$ in computing derivatives"""

        if order is None:
            order = self.order
        else:
            order = max(1, min(self.order, order))

        with sympy.core.evaluate(False):
            if order <= 1:
                bdf0 = self.psi_fn - self.psi_star[0].sym

            elif order == 2:
                bdf0 = (
                    3 * self.psi_fn / 2
                    - 2 * self.psi_star[0].sym
                    + self.psi_star[1].sym / 2
                )

            elif order == 3:
                bdf0 = (
                    11 * self.psi_fn / 6
                    - 3 * self.psi_star[0].sym
                    + 3 * self.psi_star[1].sym / 2
                    - self.psi_star[2].sym / 3
                )

            bdf0 /= self.step_averaging

        # This is actually calculated over several steps so scaling is required
        return bdf0

    def adams_moulton_flux(self, order=None):
        if order is None:
            order = self.order
        else:
            order = max(1, min(self.order, order))

        with sympy.core.evaluate(False):
            if order == 1:
                am = (self.psi_fn + self.psi_star[0].sym) / 2

            elif order == 2:
                am = (
                    5 * self.psi_fn + 8 * self.psi_star[0].sym - self.psi_star[1].sym
                ) / 12

            elif order == 3:
                am = (
                    9 * self.psi_fn
                    + 19 * self.psi_star[0].sym
                    - 5 * self.psi_star[1].sym
                    + self.psi_star[2].sym
                ) / 24

        return am
