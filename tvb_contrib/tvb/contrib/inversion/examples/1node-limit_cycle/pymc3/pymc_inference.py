import tvb.simulator.models
import tvb.simulator.integrators
from tvb.contrib.inversion.pymcInference import pymcModel1node

import numpy as np
import pymc3 as pm
import theano
import pickle

# Simulation parameters
with open("../limit-cycle_simulation.pkl", "rb") as f:
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
draws = 400
tune = 400
num_cores = 4

if __name__ == "__main__":
    pymc_model = pymcModel1node(oscillator_model)

    with pymc_model.stat_model:
        model_a_star = pm.Normal(name="model_a_star", mu=0.0, sd=1.0)
        model_a = pm.Deterministic(name="model_a", var=1.5 + model_a_star)

        model_b_star = pm.Normal(name="model_b_star", mu=0.0, sd=1.0)
        model_b = pm.Deterministic(name="model_b", var=-11.0 + 5.0 * model_b_star)

        model_c_star = pm.Normal(name="model_c_star", mu=0.0, sd=1.0)
        model_c = pm.Deterministic(name="model_c", var=0.1 + 0.1 * model_c_star)

        model_I_star = pm.Normal(name="model_I_star", mu=0.0, sd=1.0)
        model_I = pm.Deterministic(name="model_I", var=0.1 + 0.1 * model_I_star)

        # model_d_star = pm.Normal(name="model_d_star", mu=0.0, sd=1.0)
        # model_d = pm.Deterministic(name="model_d", var=0.02 + 0.01 * model_d_star)

        # model_tau_star = pm.Normal(name="model_tau_star", mu=0.0, sd=1.0)
        # model_tau = pm.Deterministic(name="model_tau", var=1.0 + 0.5 * model_tau_star)

        x_init = theano.shared(X[0], name="x_init")

        BoundedNormal = pm.Bound(pm.Normal, lower=0.0)

        noise_gfun_star = BoundedNormal(name="noise_gfun_star", mu=0.0, sd=1.0)
        noise_gfun = pm.Deterministic(name="noise_gfun", var=0.05 + 0.1 * noise_gfun_star)

        noise_star = pm.Normal(name="noise_star", mu=0.0, sd=1.0, shape=tuple(shape))
        dynamic_noise = pm.Deterministic(name="dynamic_noise", var=noise_gfun * noise_star)

        global_noise = BoundedNormal(name="global_noise", mu=0.0, sd=1.0)

        priors = {
            "model_a": model_a,
            "model_b": model_b,
            "model_c": model_c,
            "model_d": np.array([0.02]),
            "model_I": model_I,
            "model_tau": np.array([1.0]),
            "model_e": np.array([3.0]),
            "model_f": np.array([1.0]),
            "model_g": np.array([0.0]),
            "model_alpha": np.array([1.0]),
            "model_beta": np.array([1.0]),
            "model_gamma": np.array([1.0]),
            "x_init": x_init,
            "dynamic_noise": dynamic_noise,
            "global_noise": global_noise,
            "node_coupling": np.zeros([2, 1, 1]),
            "local_coupling": 0.0
        }

        pymc_model.prior_stats = {
            "model_a": {"mean": 1.5, "sd": 1.0},
            "model_b": {"mean": -11.0, "sd": 5.0},
            "model_c": {"mean": 0.1, "sd": 0.1},
            # "model_d": {"mean": 0.02, "sd": 0.01},
            "model_I": {"mean": 0.1, "sd": 0.1},
            # "model_tau": {"mean": 1.0, "sd": 0.5},
            "noise_gfun": {"mean": 0.05, "sd": 0.1},
            "global_noise": {"mean": 0.0, "sd": 1.0}
        }

    pymc_model.set_model(
        priors=priors,
        obs=X,
        time_step=simulation_params["dt"]
    )

    inference_data = pymc_model.run_inference(
        draws=draws,
        tune=tune,
        cores=num_cores,
        target_accept=0.9,
        max_treedepth=35,
        step_scale=0.25,
        save=True
    )

    pymc_model.save(simulation_params=simulation_params.copy())
