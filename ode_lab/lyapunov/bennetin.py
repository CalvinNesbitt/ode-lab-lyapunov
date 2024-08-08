# """
# Implementation of the Bennetin algorithm for the computation of the Lyapunov
# exponents of a dynamical system.
# See [1] for more information.

# References
# ----------
# [1] TO DO: Add reference.
# """

# from pathlib import Path

# import numpy as np
# import xarray as xr
# from ode_lab.base.logger import logger
# from ode_lab.base.observe import Observer
# from tqdm import tqdm

# from ode_lab.lyapunov.tangent_linear_model import TangentIntegrator


# def posQR(M):
#     """Returns QR decomposition of a matrix with positive diagonals on R.
#     Parameter, M: Array that is being decomposed
#     """
#     Q, R = np.linalg.qr(M)
#     signs = np.diag(np.sign(np.diagonal(R)))
#     Q, R = np.dot(Q, signs), np.dot(signs, R)
#     return Q, R


# class BennetinAlgorithmIntegrator:
#     """
#     Class for integrating the Bennetin algorithm.
#     """

#     def __init__(
#         self, rhs, trajectory_ic, jacobian, Q_ic, tau=0.01, parameters={},
#     ):
#         # Algorithm parameters
#         self.rhs = rhs
#         self.jacobian = jacobian
#         self.tau = tau
#         self.parameters = parameters
#         self.method = method
#         self.Q_ic = Q_ic
#         self.trajectory_ic = trajectory_ic
#         self.alg_params = {
#             "tau": tau,
#             "Q_ic": Q_ic,
#             "trajectory_ic": trajectory_ic,
#             "method": method,
#         }
#         for key, value in parameters.items():
#             self.alg_params[key] = value

#         # Algorithm state
#         self.time = 0
#         self._trajectory_state = trajectory_ic
#         self.Q = Q_ic
#         self.R = np.zeros(Q_ic.shape)

#     def step(self):
#         # Trajectory state prior to pusing matrix forward
#         trajectory_state = self._trajectory_state
#         P = np.zeros(self.Q.shape)  # dummy stretched matrix

#         # Integrate perturbation matrix forward
#         for i, column in enumerate(self.Q.T):
#             self._trajectory_state = trajectory_state

#             # Use a tangent integrator to integrate the perturbation matrix
#             # forward in time
#             tangent_integrator = TangentIntegrator(
#                 rhs=self.rhs,
#                 jacobian=self.jacobian,
#                 ic=self._trajectory_state,
#                 tangent_ic=column,
#                 parameters=self.parameters,
#                 method=self.method,
#             )
#             tangent_integrator.run(self.tau)
#             P.T[i] = tangent_integrator.tangent_state

#         # Update Q and R
#         self.Q, self.R = posQR(P)
#         return

#     def run(self, n: int):
#         for i in range(n):
#             self.step()


# # Aspects of the Bennetin algorithm that we want to observe


# class BLVObserver(Observer):
#     """
#     Class for observing the backward Lyapunov vectors (BLVs).
#     """

#     def __init__(
#         self,
#         number_of_blvs: int,
#     ):
#         self.number_of_blvs = number_of_blvs
#         self._blv_obs = []
#         super().__init__(observable_names=[f"blv_{i}" for i in range(number_of_blvs)])

#     def look(self, integrator: BennetinAlgorithmIntegrator):
#         self._time_obs.append(integrator.time)
#         self._blv_obs.append(integrator.Q)
#         self.ndim = integrator.Q.shape[1]
#         return

#     @property
#     def observations(self):
#         """
#         Package the observations into an xarray dataset.

#         Returns
#         -------
#         xr.Dataset
#             An xarray dataset containing the observations.
#         """
#         np.arange(1, 1 + self.number_of_blvs)
#         np.arange(1, 1 + self.ndim)
#         self._time_obs

#         return xr.Dataset(data_var_dict, attrs=self.attrs)


# class FTBLEObserver(Observer):
#     """
#     Class for observing the forward-time backward Lyapunov exponents (FTBLEs).
#     """

#     def __init__(
#         self,
#         number_of_ftbles: int,
#     ):
#         self.number_of_ftbles = number_of_ftbles
#         self._ftble_obs = []
#         super().__init__(
#             observable_names=[f"ftble_{i}" for i in range(number_of_ftbles)]
#         )

#     def look(self, integrator: BennetinAlgorithmIntegrator):
#         # Compute the FTBLEs
#         ftble = np.log(np.diag(integrator.R)) / integrator.tau
#         self._ftble_obs.append(ftble)
#         return

#     @property
#     def observations(self):
#         """
#         Package the observations into an xarray dataset.

#         Returns
#         -------
#         xr.Dataset
#             An xarray dataset containing the observations.
#         """

#         data_var_dict = {}
#         for i, obs_name in enumerate(self.observable_names):
#             data_var_dict[obs_name] = xr.DataArray(
#                 np.stack(self._ftble_obs)[:, i],
#                 dims=["time"],
#                 coords={"time": self._time_obs},
#             )
#         return xr.Dataset(data_var_dict, attrs=self.attrs)


# class BennetinAlgorithm:
#     """
#     High-level class for running the Bennetin algorithm and making observations.
#     """

#     def __init__(
#         self,
#         rhs,
#         ic,
#         jacobian,
#         Q_ic,
#         tau=0.01,
#         parameters={},
#         method="RK45",
#         number_of_blvs=3,
#         number_of_ftbles=3,
#         observe_trajectory=False,
#     ):
#         self.integrator = BennetinAlgorithmIntegrator(
#             rhs, ic, jacobian, Q_ic, tau, parameters, method
#         )
#         observer_list = []
#         if number_of_blvs > 0:
#             observer_list.append(BLVObserver(number_of_blvs))
#         if number_of_ftbles > 0:
#             observer_list.append(FTBLEObserver(number_of_ftbles))

#     def make_observations(
#         self, number: int, frequency: float, transient: float = 0, timer: bool = False
#     ) -> None:
#         if isinstance(number, float):
#             number = int(number)

#         # Determine if we need to run a transient
#         if self.have_I_run_a_transient and transient > 0:
#             logger.warning(
#                 "I've already run a transient! I'm going to ignore this one."
#             )
#             transient = 0
#         if transient > 0:
#             self.integrator.run(transient)  # No observations for transient
#             self.integrator.time = 0  # Reset time
#             self.have_I_run_a_transient = True

#         # Make observations
#         for observer in self.observer_list:
#             observer.look(self.integrator)
#         logger.info(f"Making {number} observations with frequency {frequency}.")
#         for _ in tqdm(range(number), disable=not timer):
#             self.integrator.run(frequency)
#             for observer in self.observer_list:
#                 observer.look(self.integrator)
#         return

#     def save_observations(self, path: str | Path):
#         for observer in self.observer_list:
#             observer.save(path)
#         return
