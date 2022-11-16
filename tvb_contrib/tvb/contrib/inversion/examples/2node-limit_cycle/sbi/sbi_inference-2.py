#!/usr/bin/python3

import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
import pickle

from tvb.simulator.simulator import Simulator
from tvb.datatypes.connectivity import Connectivity
from tvb.contrib.inversion.sbiInference import sbiModel, sbiPrior

import tvb.simulator.models
import tvb.simulator.integrators
import tvb.simulator.coupling
import tvb.simulator.monitors


with open('../limit-cycle_simulation.pkl', 'rb') as f:
    simulation_params = pickle.load(f)

X = simulation_params["simulation"]

# Connectivity
connectivity = Connectivity()
connectivity.weights = np.array([[0., 2.], [2., 0.]])
connectivity.region_labels = np.array(["R1", "R2"])
connectivity.centres = np.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]])
connectivity.tract_lengths = np.array([[0., 2.5], [2.5, 0.]])
# connectivity.configure()

# Model
oscillator_model = getattr(tvb.simulator.models, simulation_params["model"])(
    a=np.asarray([simulation_params["a_sim"]]),
    b=np.asarray([simulation_params["b_sim"]]),
    c=np.asarray([simulation_params["c_sim"]]),
    d=np.asarray([simulation_params["d_sim"]]),
    I=np.asarray([simulation_params["I_sim"]]),
)
# oscillator_model.configure()

# Integrator
integrator = getattr(tvb.simulator.integrators, simulation_params["integrator"])(dt=simulation_params["dt"])
integrator.noise.nsig = np.array([simulation_params["nsig"]])
# integrator.configure()

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

prior = sbiPrior()
prior.append("a", "model", "Normal", 2.0, 1.0)
prior.append("b", "model", "Normal", -10.0, 5.0)
prior.append("c", "model", "Normal", 0.0, 0.1)
prior.append("I", "model", "Normal", 0.0, 0.1)
prior.append("a", "coupling", "Normal", 0.1, 0.1)
prior.append("nsig", "integrator.noise", "LogNormal", 0.003, 0.001)
prior.append("noise", "global", "HalfNormal", 0.0, 0.1)


def job(i):
    snpe_model = sbiModel(
        method="SNPE",
        obs=X,
        simulator_instance=deepcopy(sim)
    )

    snpe_model.run_inference(
        prior=prior,
        num_simulations=75000,
        num_workers=10,
        num_samples=2000
    )

    _ = snpe_model.to_arviz_data(num_workers=10)

    snpe_model.save(simulation_params=simulation_params.copy())


if __name__ == "__main__":
    num_inferences = 4
    _ = Parallel(n_jobs=num_inferences)(delayed(job)(i) for i in range(num_inferences))
