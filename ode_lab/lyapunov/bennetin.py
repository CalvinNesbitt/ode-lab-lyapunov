"""
Implementation of the Bennetin algorithm for the computation of the Lyapunov
exponents of a dynamical system.
See [1] for more information.

References
----------
[1] TO DO: Add reference.
"""

import numpy as np


def posQR(M):
    """Returns QR decomposition of a matrix with positive diagonals on R.
    Parameter, M: Array that is being decomposed
    """
    Q, R = np.linalg.qr(M)
    signs = np.diag(np.sign(np.diagonal(R)))
    Q, R = np.dot(Q, signs), np.dot(signs, R)
    return Q, R


# class BennetinAlgorithm:
#     def __init_(self,
# rhs, ic, jacobian, Q_ic, tau=0.01, parameters={}, method="RK45"):
#         self.rhs = rhs
#         self.ic = ic
#         self.jacobian = jacobian
#         self.Q_ic = Q_ic
#         self.tau = tau
#         self.parameters = parameters
#         self.method = method

#         self.tangent_integrator = TangentIntegrator(
#             rhs, jacobian, ic, Q_ic, parameters, method
#         )
