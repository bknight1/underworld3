import sympy
from sympy import sympify
import numpy as np

from typing import Optional, Callable, Union

import underworld3 as uw
import underworld3.timing as timing
from underworld3.systems import SNES_Scalar


from .ddt import SemiLagrangian as SemiLagrangian_DDt
from .ddt import Lagrangian as Lagrangian_DDt


class SNES_Diffusion_SLCN(SNES_Scalar):
    r"""
    This class provides functionality for a discrete representation
    of the Poisson equation

    $$
    \nabla \cdot
            \color{Blue}{\underbrace{\Bigl[ \boldsymbol\kappa \nabla u \Bigr]}_{\mathbf{F}}} =
            \color{Maroon}{\underbrace{\Bigl[ f \Bigl] }_{\mathbf{f}}}
    $$

    The term $\mathbf{F}$ relates the flux to gradients in the unknown $u$

    ## Properties

      - The unknown is $u$

      - The diffusivity tensor, $\kappa$ is provided by setting the `constitutive_model` property to
    one of the scalar `uw.constitutive_models` classes and populating the parameters.
    It is usually a constant or a function of position / time and may also be non-linear
    or anisotropic.

      - $f$ is a volumetric source term
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable = None,
        # u_Star: uw.discretisation.MeshVariable = None,
        solver_name: str = "",
        verbose=False,
        degree=2,
    ):
        ## Keep track

        ## Parent class will set up default values etc
        super().__init__(
            mesh,
            u_Field,
            None,
            None,
            degree,
            solver_name,
            verbose,
        )

        if solver_name == "":
            solver_name = "Poisson_{}_".format(self.instance_number)

        # Register the problem setup function
        self._setup_problem_description = self.diffusion_problem_description

        # default values for source term
        self.f = sympy.Matrix.zeros(1, 1)

        self.theta = sympy.Number(0)

        self._constitutive_model = None

        self.u_Field = u_Field
        self.u_Star = uw.discretisation.MeshVariable(f"{self.u_Field.name}_1", mesh, 1, degree=self.u_Field.degree, continuous=self.u_Field.continuous)

        self.flux = None
        self.flux_star = None
        # self.Lambda = None
        self.delta_t = None

    @timing.routine_timer_decorator
    def diffusion_problem_description(self):

        self.bdf = (self.u_Field.sym - self.u_Star.sym)
        
        ### f0 
        self._f0 = (self.F0 + self.bdf - self.f) / self.delta_t


        amf = (self.theta*self.flux.T) + ((1-self.theta) * self.flux_star.T)


        self._f1 = self.F1 + amf

        return

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self.is_setup = False
        self._f = sympy.Matrix((value,))

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        self.is_setup = False
        self._theta = sympy.Number(value)

    @timing.routine_timer_decorator
    def solve(
        self,
        zero_init_guess: bool = True,
        timestep: float = None,
        _force_setup: bool = False,
        verbose=False,
    ):
        """
        Generates solution to constructed system.

        Params
        ------
        zero_init_guess:
            If `True`, a zero initial guess will be used for the
            system solution. Otherwise, the current values of `self.u` will be used.
        """

        self.delta_t = timestep 

        if _force_setup:
            self.is_setup = False

        ### initial setup of flux terms
        if self.flux == None:
            self.flux = self.constitutive_model.flux.copy()

        self.flux_star = self.constitutive_model.flux.copy()

        if not self.constitutive_model._solver_is_setup:
            self.is_setup = False
            with self.mesh.access(self.u_Field, self.u_Star):
                self.u_Star.data[:,0] = np.copy(self.u_Field.data[:,0])  

            

        if not self.is_setup:
            self._setup_pointwise_functions(verbose)
            self._setup_discretisation(verbose)
            self._setup_solver(verbose)


        super().solve(zero_init_guess, _force_setup)

        ### update history terms
        self.flux_star = self.constitutive_model.flux.copy()

        ### post solve update
        # u_lost_fn =  self.u_Field.sym[0] - (self.u_Field.sym[0] * np.exp(-self.Lambda*self.delta_t))
        with self.mesh.access(self.u_Field, self.u_Star):
            ### calculate the amount lost due to decay
            # self.u_Field.data[:,0] = uw.function.evalf(u_lost_fn, self.u_Field.coords)
            self.u_Star.data[:,0]  = np.copy(self.u_Field.data[:,0])
        



        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

        return
    
