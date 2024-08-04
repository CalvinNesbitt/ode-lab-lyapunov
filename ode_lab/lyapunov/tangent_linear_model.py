"""
Code for integrating the tangent linear model (TLM) of a dynamical system.
"""

from typing import Callable

import numpy as np
from ode_lab.base.integrate import Integrator
from ode_lab.base.observe import BaseObserver


class TangentIntegrator(Integrator):
    """
    Class for integrating the tangent linear model (TLM) of a dynamical system.
    The TLM is a linear approximation of the dynamical system around a trajectory.

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
        def tlm_rhs(state, **kwargs):
            trajectory = state[: int(self.ndim / 2)]
            perturbation = state[int(self.ndim / 2) :]
            trajectory_rhs = rhs(trajectory, **kwargs)
            tangent_rhs_dt = jacobian(trajectory, **kwargs).dot(perturbation)
            return np.append(trajectory_rhs, tangent_rhs_dt)

        return tlm_rhs

    @property
    def model_state(self):
        return self.state[: int(self.ndim / 2)]

    @property
    def tangent_state(self):
        return self.state[int(self.ndim / 2) :]


class TangentObserver(BaseObserver):
    def __init__(
        self,
        rhs: Callable,
        jacobian: Callable,
        ic: np.ndarray | list,
        tangent_ic: np.ndarray | list,
        parameters: dict | None = None,
        method: str = "DOP853",
        observable_names: list["str"] | None = None,
        log_level: str = "INFO",
        log_file: str | None = None,
    ):
        if observable_names is None:
            observable_names = [f"X_{i}" for i in range(len(ic))] + [
                f"dX_{i}" for i in range(len(ic))
            ]
        tangent_integrator = TangentIntegrator(
            rhs, jacobian, ic, tangent_ic, parameters, method
        )

        super().__init__(
            integrator=tangent_integrator,
            observable_names=observable_names,
            log_level=log_level,
            log_file=log_file,
        )

    def observing_function(self, tangent_integrator: TangentIntegrator) -> np.ndarray:
        return tangent_integrator.state
