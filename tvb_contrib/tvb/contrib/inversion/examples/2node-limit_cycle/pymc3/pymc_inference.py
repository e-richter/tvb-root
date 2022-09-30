from tvb.simulator.simulator import Simulator
from tvb.datatypes.connectivity import Connectivity
from tvb.contrib.inversion.pymcInference import pymcModel

import tvb.simulator.models
import tvb.simulator.integrators
import tvb.simulator.coupling
import tvb.simulator.monitors

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

# global inference parameters
shape = X.shape
draws = 500
tune = 500
num_cores = 2

Nt = int(sim.simulation_length)
Nsv = len(sim.model.state_variables)
Nr = sim.connectivity.number_of_regions


if __name__ == "__main__":
    pymc_model = pymcModel(sim)

    with pymc_model.stat_model:
        #a_star = pm.Normal(name="a_star", mu=0.0, sd=1.0)
        #a = pm.Deterministic(name="a", var=2.0 + a_star)

        a_coupling_star = pm.Normal(name="a_coupling_star", mu=0.0, sd=1.0)
        a_coupling = pm.Deterministic(name="a_coupling", var=0.1 + 0.05 * a_coupling_star)

        BoundedNormal = pm.Bound(pm.Normal, lower=0.0)

        noise_gfun_star = BoundedNormal(name="noise_gfun_star", mu=0.0, sd=1.0)
        noise_gfun = pm.Deterministic(name="noise_gfun", var=0.05 + 0.1 * noise_gfun_star)

        noise_star = pm.Normal(name="noise_star", mu=0.0, sd=1.0, shape=(Nt, Nsv, Nr, 1))
        noise = pm.Deterministic(name="noise", var=noise_gfun * noise_star)

        epsilon = BoundedNormal(name="epsilon", mu=0.0, sd=1.0)

        # Passing the prior distributions as dictionary. Also including fixed model parameters.
        priors = {
            "a": np.array([simulation_params["a_sim"]]),
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
            "gamma": np.array([1.0]),
            "coupling.a": a_coupling,
            "integrator.noise": noise,
            "global.noise": epsilon,
            "local_coupling": 0.0
        }

        pymc_model.prior_stats = {
            "coupling.a": {"mean": 0.1, "sd": 0.05},
            "noise_gfun": {"mean": 0.05, "sd": 0.1},
            "global.epsilon": {"mean": 0.0, "sd": 1.0}
        }

    pymc_model.set_model(
        priors=priors,
        obs=X,
        time_step=simulation_params["dt"],
    )

    inference_data = pymc_model.run_inference(
        draws=draws,
        tune=tune,
        cores=num_cores,
        target_accept=0.9,
        max_treedepth=20,
        step_scale=0.5,
        save=True
    )

    pymc_model.save(simulation_params=simulation_params.copy())
