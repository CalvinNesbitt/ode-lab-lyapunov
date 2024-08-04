"""
Code for integrating the tangent linear model (TLM) of a dynamical system.
"""

from typing import Callable

import numpy as np
from ode_lab.base.integrate import Integrator


class TangentIntegrator(Integrator):
    """
    Integrate a system of ordinary diffferential equations dx/dt = f(x).
    This is a light wrapper of scipy.integrate.solve_ivp [1].

    ...

    Attributes
    ----------
    rhs : function
        The rhs of our ODE system.
    jacobian : function
        The jacobian of our ODE system.
    ic : np.array
        The ic for our IVP.
    tangent_ic : np.array
        The ic for the tangent linear model.
    parameters: dict
        Parameters of our ode.
        {'foo' : bar} will be passed as rhs(foo=bar).
    method: string
        The scheme we use to integrate our IVP.
        See scipy.integrate.solve_ivp for more information.

    Methods
    -------
    run(t)
        Integrate the ODEs for a length of time t.

    References
    -------
    """

    def __init__(
        self,
        rhs: Callable,
        jacobian: Callable,
        ic: np.ndarray | list,
        tangent_ic: np.ndarray | list,
        parameters: dict | None = None,
        method: str = "DOP853",
    ):
        tlm_rhs = self._get_tlm(rhs, jacobian)
        if isinstance(ic, list):
            ic = np.array(ic)
        if isinstance(tangent_ic, list):
            tangent_ic = np.array(tangent_ic)
        tlm_ic = np.append(ic, tangent_ic)
        super().__init__(tlm_rhs, tlm_ic, parameters, method)

    def _get_tlm(self, rhs: Callable, jacobian: Callable) -> Callable:
        def tlm_rhs(t, state):
            trajectory = state[: int(self.ndim / 2)]
            perturbation = state[int(self.ndim / 2) :]
            trajectory_rhs = rhs(trajectory, **self.parameters)
            tangent_rhs_dt = jacobian(trajectory, **self.parameters).dot(perturbation)
            return np.append(trajectory_rhs, tangent_rhs_dt)

        return tlm_rhs

    @property
    def model_state(self):
        return self.state[: int(self.ndim / 2)]

    @property
    def tangent_state(self):
        return self.state[int(self.ndim / 2) :]
