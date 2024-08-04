import numpy as np


def l63_jacobian(state, sigma=10, rho=28, beta=8 / 3):
    x, y, z = state
    grad_f0 = np.array([-sigma, sigma, 0])
    grad_f1 = np.array([-z + rho, -1, -x])
    grad_f2 = np.array([y, x, -beta])
    return np.array([grad_f0, grad_f1, grad_f2])
