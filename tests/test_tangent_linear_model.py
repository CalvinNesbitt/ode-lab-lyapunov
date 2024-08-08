from ode_lab.base.examples import l63

from ode_lab.lyapunov.examples import l63_jacobian
from ode_lab.lyapunov.tangent_linear_model import TangentObserver


def test_tangent_observer():
    l63_tlm = TangentObserver(l63, l63_jacobian, [1, 1, 1], [0.1, 0.1, 0.1])
    l63_tlm.make_observations(10, 1)
