import math

from ode_lab.base.examples import l63

from ode_lab.lyapunov.examples import l63_jacobian
from ode_lab.lyapunov.tangent_linear_model import TLMObserver


def test_tangent_observer():
    observables = ["x", "y", "z", "dx", "dy", "dz"]
    ic = [1, 2, 3, 0.1, 0.2, 0.3]
    number_of_obs = 500
    obs_freq = 0.01
    l63_tlm = TLMObserver(
        l63,
        l63_jacobian,
        [1, 2, 3],
        [0.1, 0.2, 0.3],
        observable_names=observables,
    )
    l63_tlm.make_observations(number_of_obs, obs_freq)
    assert hasattr(l63_tlm, "observations")
    observations = l63_tlm.observations
    params = observations.attrs
    # check keys
    assert "sigma" in params
    assert "rho" in params
    assert "beta" in params

    # check values
    for val, obs in zip(ic, observables):
        assert obs in observations
        assert observations[obs].shape == (number_of_obs + 1,)
        assert observations[obs].values[0] == val

    # check time
    assert "time" in observations
    assert observations["time"].shape == (number_of_obs + 1,)
    assert observations["time"].values[0] == 0.0
    # check time gaps
    assert all(
        [
            math.isclose(
                observations["time"].values[i + 1] - observations["time"].values[i],
                obs_freq,
                abs_tol=1e-15,
            )
            for i in range(number_of_obs)
        ]
    )
