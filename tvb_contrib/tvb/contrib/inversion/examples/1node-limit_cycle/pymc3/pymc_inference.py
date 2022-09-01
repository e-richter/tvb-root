import tvb.simulator.models
import tvb.simulator.integrators
from tvb.contrib.inversion.pymcInference import NonCenteredModel

import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import pymc3 as pm
import scipy
import theano.tensor as tt
import theano
import math
from tqdm import tqdm
import pickle


# Simulation parameters
with open("limit-cycle_simulation.pkl", "rb") as f:
    simulation_params = pickle.load(f)

# Model
oscillator_model = getattr(tvb.simulator.models, simulation_params["model"])(
    a=np.asarray([simulation_params["a_sim"]]),
    b=np.asarray([simulation_params["b_sim"]]),
    c=np.asarray([simulation_params["c_sim"]]),
    d=np.asarray([simulation_params["d_sim"]]),
    I=np.asarray([simulation_params["I_sim"]]),
)
oscillator_model.configure()

# Integrator
integrator = getattr(tvb.simulator.integrators, simulation_params["integrator"])(dt=simulation_params["dt"])
integrator.noise.nsig = np.array([simulation_params["nsig"]])
integrator.noise.configure()
integrator.noise.configure_white(dt=integrator.dt)
integrator.set_random_state(random_state=None)
integrator.configure()
integrator.configure_boundaries(oscillator_model)

X = simulation_params["simulation"]

# global inference parameters
shape = X.shape
draws = 500
tune = 500
num_cores = 2


if __name__ == "__main__":
    ncModel = NonCenteredModel(oscillator_model)

    with ncModel.pymc_model:
        a_star = pm.Normal(name="a_star", mu=0.0, sd=1.0)
        a = pm.Deterministic(name="a", var=2.0 + a_star)

        priors = {
            "a": a,
            "b": np.array([simulation_params["b_sim"]]),
            "c": np.array([simulation_params["c_sim"]]),
            "d": np.array([simulation_params["d_sim"]]),
            "I": np.array([simulation_params["I_sim"]]),
            "tau": np.array([1.0]),
            "e": np.array([3.0]),
            "f": np.array([1.0]),
            "g": np.array([0.0]),
            "alpha": np.array([1.0]),
            "beta": np.array([1.0]),
            "gamma": np.array([1.0])
        }

        consts = {
            "coupling": np.zeros([2, 1, 1]),
            "local_coupling": 0.0
        }

        x_init = theano.shared(X[0], name="x_init")

        BoundedNormal = pm.Bound(pm.Normal, lower=0.0)

        noise_star = BoundedNormal(name="noise_star", mu=0.0, sd=1.0)
        noise = pm.Deterministic(name="noise", var=0.05 + 0.1 * noise_star)

        x_t_star = pm.Normal(name="x_t_star", mu=0.0, sd=1.0, shape=tuple(shape))
        x_t = pm.Deterministic(name="x_t", var=noise * x_t_star)

        amplitude_star = pm.Normal(name="amplitude_star", mu=0.0, sd=1.0)
        amplitude = pm.Deterministic(name="amplitude", var=0.0 + amplitude_star)

        offset_star = pm.Normal(name="offset_star", mu=0.0, sd=1.0)
        offset = pm.Deterministic(name="offset", var=0.0 + offset_star)

        epsilon = BoundedNormal(name="epsilon", mu=0.0, sd=1.0)

        ncModel.prior_stats = {
            "a": {"mean": 2.0, "sd": 1.0},
            "noise": {"mean": 0.05, "sd": 0.1},
            "epsilon": {"mean": 0.0, "sd": 1.0}
        }

    ncModel.set_model(
        priors=priors,
        consts=consts,
        obs=X,
        time_step=simulation_params["dt"],
        x_init=x_init,
        time_series=x_t,
        amplitude=amplitude,
        offset=offset,
        obs_noise=epsilon
    )

    nc_data = ncModel.run_inference(
        draws=draws,
        tune=tune,
        cores=num_cores,
        target_accept=0.9,
        max_treedepth=20,
        save=True
    )

    ncModel.save(simulation_params=simulation_params.copy())
