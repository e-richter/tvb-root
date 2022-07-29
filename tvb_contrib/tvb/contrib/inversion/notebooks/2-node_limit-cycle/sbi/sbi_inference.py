#!/usr/bin/python3

import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed

from tvb.simulator.models.oscillator import Generic2dOscillator
from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.simulator import Simulator
from tvb.simulator.coupling import Linear
from tvb.simulator.monitors import Raw, TemporalAverage
from tvb.datatypes.connectivity import Connectivity
from tvb.contrib.inversion.sbiInference import sbiModel

# Simulation parameters
a_sim = 2.0
b_sim = -10.0
c_sim = 0.0
d_sim = 0.02
I_sim = 0.0
nsig = 0.003
dt = 1.0
simulation_length = 1000

# Connectivity
connectivity = Connectivity()
connectivity.weights = np.array([[0., 2/3], [2/3, 0.]])
connectivity.region_labels = np.array(["R1", "R2"])
connectivity.centres = np.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]])
connectivity.tract_lengths = np.array([[0., 0.1], [0.1, 0.]])
connectivity.configure()

# Model
oscillator_model = Generic2dOscillator(
    a=np.asarray([a_sim]),
    b=np.asarray([b_sim]),
    c=np.asarray([c_sim]),
    d=np.asarray([d_sim]),
    I=np.asarray([I_sim]),
)
oscillator_model.configure()

# Integrator
integrator = HeunStochastic(dt=dt)
integrator.noise.nsig = np.array([nsig])
integrator.configure()

# Global coupling
coupling = Linear()

# Monitor
monitor = TemporalAverage()

# Simulator
sim = Simulator(
    model=oscillator_model,
    connectivity=connectivity,
    coupling=coupling,
    integrator=integrator,
    monitors=(monitor,),
    simulation_length=simulation_length
)

sim.configure()

X = np.load("limit-cycle_simulation.npy")

priors = {
    "a": [2.0, 0.1, False],
    "b": [-10, 0.1, False],
    "c": [0.0, 0.05, False],
    "d": [0.02, 0.005, False],
    "I": [0.0, 0.05, False],
    "epsilon": [0.0, 0.01, False]
}


def job():
    snpe_model = sbiModel(
        simulator_instance=sim,
        method="SNPE",
        obs=X
    )

    snpe_model.run_inference(
        prior_vars=priors,
        prior_dist="Normal",
        num_simulations=1200,
        num_workers=4,
        num_samples=2000,
        neural_net="mdn"
    )

    _ = snpe_model.to_arviz_data(num_workers=4, save=True)

    snpe_model.save()


if __name__ == "__main__":
    num_inferences = 4
    _ = Parallel(n_jobs=4)(delayed(job) for _ in range(num_inferences))
