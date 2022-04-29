import numpy as np
import torch
import math
from typing import Dict, List

from sbi import utils as sbi_utils
from sbi import analysis as sbi_analysis
from sbi.inference.base import infer

from tvb.simulator.simulator import Simulator
from tvb.simulator.integrators import Integrator
from tvb.simulator.models.base import Model


class SNPEModel:
    def __init__(
            self,
            # simulator_instance: Simulator,
            integrator_instance: Integrator,
            model_instance: Model,
            obs: Dict,
            priors: Dict[str, List]
    ):
        # self.simulator_instance = simulator_instance
        self.integrator_instance = integrator_instance
        self.model_instance = model_instance
        self.obs = obs

        prior_min = [value[0] for _, value in priors.items()]
        prior_max = [value[1] for _, value in priors.items()]

        self.priors = sbi_utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min),
                                                      high=torch.as_tensor(prior_max))

    def _simulation_wrapper(self, params):

        # for i, param in enumerate(params):
        #     self.simulator_instance.model.__dict__[self.simulator_instance.model.parameter_names[i]] = np.asarray(param)
        #
        # self.simulator_instance.configure()
        # X = self.simulator_instance.run()
        #
        # return torch.as_tensor(X)

        for i, param in enumerate(params):
            self.model_instance.__dict__[self.model_instance.parameter_names[i]] = np.asarray(param)

        self.model_instance.configure()
        self.integrator_instance.noise.configure()
        self.integrator_instance.noise.configure_white(dt=self.integrator_instance.dt)
        self.integrator_instance.set_random_state(random_state=None)
        self.integrator_instance.configure()
        self.integrator_instance.configure_boundaries(self.model_instance)

        simulation_length = 100
        stimulus = 0.0
        local_coupling = 0.0
        current_state = np.random.uniform(low=-1.0, high=1.0, size=[1, 1, 1])
        state = current_state
        current_step = 0
        number_of_nodes = 1
        start_step = current_step + 1
        node_coupling = np.zeros([1, 1, 1])
        n_steps = int(math.ceil(simulation_length / self.integrator_instance.dt))

        X = [current_state.copy()]
        for step in range(start_step, start_step + n_steps):
            state = self.integrator_instance.integrate(state, self.model_instance, node_coupling, local_coupling, stimulus)
            X.append(state.copy())

        X = np.squeeze(np.asarray(X))
        t = np.linspace(0, simulation_length, n_steps + 1)

        return torch.as_tensor(X)

    def run_inference(self, num_simulations, num_workers, num_samples):
        posterior = infer(self._simulation_wrapper, prior=self.priors, method="SNPE", num_simulations=num_simulations, num_workers=num_workers)
        samples = posterior.sample((num_samples, ), x=torch.as_tensor(np.squeeze(self.obs["xs"])))
        # log_prob = posterior.log_prob(samples, x=torch.as_tensor(np.squeeze(self.obs["xs"])))
        return posterior, samples


