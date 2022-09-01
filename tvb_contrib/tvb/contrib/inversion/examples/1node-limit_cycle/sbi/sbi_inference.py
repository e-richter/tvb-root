#!/usr/bin/python3

import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
import pickle

from tvb.simulator.simulator import Simulator
from tvb.datatypes.connectivity import Connectivity
from tvb.contrib.inversion.sbiInference import sbiModel

import tvb.simulator.models
import tvb.simulator.integrators
import tvb.simulator.coupling
import tvb.simulator.monitors

with open('../limit-cycle_simulation.pkl', 'rb') as f:
    simulation_params = pickle.load(f)

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

prior_vars = {
    "model": {
        "a": [2.0, 1.0]
    },
    "integrator.noise": {
        "nsig": [0.003, 0.002]
    },
    "global": {
        "epsilon": [0.0, 1.0]
    },
}


def job(i):
    snpe_model = sbiModel(
        method="SNPE",
        obs=X,
        model_instance=deepcopy(oscillator_model),
        integrator_instance=deepcopy(integrator)
    )

    snpe_model.run_inference(
        prior_vars=prior_vars,
        prior_dist="Normal",
        num_simulations=5000,
        num_workers=4,
        num_samples=2000,
    )

    _ = snpe_model.to_arviz_data(num_workers=4)

    snpe_model.save(simulation_params=simulation_params.copy())


if __name__ == "__main__":
    num_inferences = 1
    _ = Parallel(n_jobs=num_inferences)(delayed(job)(i) for i in range(num_inferences))
