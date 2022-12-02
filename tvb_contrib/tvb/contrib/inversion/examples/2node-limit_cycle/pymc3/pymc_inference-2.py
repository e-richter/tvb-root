from tvb.simulator.simulator import Simulator
from tvb.datatypes.connectivity import Connectivity
from tvb.contrib.inversion.pymcInference import pymcModel

import tvb.simulator.models
import tvb.simulator.integrators
import tvb.simulator.coupling
import tvb.simulator.monitors

import numpy as np
import pymc3 as pm
import pickle

# Simulation parameters
with open("../limit-cycle_simulation.pkl", "rb") as f:
    simulation_params = pickle.load(f)

# Connectivity
connectivity = Connectivity()
connectivity.weights = np.array([[0., 2.], [2., 0.]])
connectivity.region_labels = np.array(["R1", "R2"])
connectivity.centres = np.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]])
connectivity.tract_lengths = np.array([[0., 2.5], [2.5, 0.]])
connectivity.configure()

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
integrator.configure()

# Global coupling
coupling = getattr(tvb.simulator.coupling, simulation_params["coupling"])()

# Monitor
monitor = getattr(tvb.simulator.monitors, simulation_params["monitor"])()

# Simulator
sim = Simulator(
    model=oscillator_model,
    connectivity=connectivity,
    coupling=coupling,
    integrator=integrator,
    monitors=(monitor,),
    simulation_length=simulation_params["simulation_length"]
)

sim.configure()

X = simulation_params["simulation"]
x0 = simulation_params["x0"]

# global inference parameters
shape = X.shape
draws = 250
tune = 250
num_cores = 4

Nt = int(sim.simulation_length)
Nsv = len(sim.model.state_variables)
Nr = sim.connectivity.number_of_regions


if __name__ == "__main__":
    pymc_model = pymcModel(sim)

    with pymc_model.stat_model:
        model_a_star = pm.Normal(name="model_a_star", mu=0.0, sd=1.0)
        model_a = pm.Deterministic(name="model_a", var=2.0 + model_a_star)
        
        model_b_star = pm.Normal(name="model_b_star", mu=0.0, sd=1.0)
        model_b = pm.Deterministic(name="model_b", var=-10.0 + 5.0 * model_b_star)

        coupling_a_star = pm.Normal(name="coupling_a_star", mu=0.0, sd=1.0)
        coupling_a = pm.Deterministic(name="coupling_a", var=0.1 + 0.1 * coupling_a_star)

        BoundedNormal = pm.Bound(pm.Normal, lower=0.0)

        noise_gfun_star = BoundedNormal(name="noise_gfun_star", mu=0.0, sd=1.0)
        noise_gfun = pm.Deterministic(name="noise_gfun", var=0.05 + 0.1 * noise_gfun_star)

        noise_star = pm.Normal(name="noise_star", mu=0.0, sd=1.0, shape=(Nt, Nsv, Nr, 1))
        dynamic_noise = pm.Deterministic(name="dynamic_noise", var=noise_gfun * noise_star)

        global_noise = BoundedNormal(name="global_noise", mu=0.0, sd=1.0)

        # Passing the prior distributions as dictionary. Also including fixed model parameters.
        priors = {
            "model_a": model_a,
            "model_b": model_b,
            "model_c": np.array([simulation_params["c_sim"]]),
            "model_d": np.array([simulation_params["d_sim"]]),
            "model_I": np.array([simulation_params["I_sim"]]),
            "model_tau": np.array([1.0]),
            "model_e": np.array([3.0]),
            "model_f": np.array([1.0]),
            "model_g": np.array([0.0]),
            "model_alpha": np.array([1.0]),
            "model_beta": np.array([1.0]),
            "model_gamma": np.array([1.0]),
            "coupling_a": coupling_a,
            "dynamic_noise": dynamic_noise,
            "global_noise": global_noise,
            "local_coupling": 0.0
        }

        pymc_model.prior_stats = {
            "model_a": {"mean": 2.0, "sd": 1.0},
            "model_b": {"mean": -10.0, "sd": 5.0},
            "coupling_a": {"mean": 0.1, "sd": 0.1},
            "noise_gfun": {"mean": 0.05, "sd": 0.1},
            "global_noise": {"mean": 0.0, "sd": 1.0}
        }

    pymc_model.set_model(
        priors=priors,
        obs=X,
        time_step=simulation_params["dt"],
        x0=x0
    )

    inference_data = pymc_model.run_inference(
        draws=draws,
        tune=tune,
        cores=num_cores,
        target_accept=0.9,
        max_treedepth=30,
        step_scale=0.25,
        save=True
    )

    pymc_model.save(simulation_params=simulation_params.copy())
