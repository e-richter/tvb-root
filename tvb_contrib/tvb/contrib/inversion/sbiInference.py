import numpy as np
import torch
import math
from typing import Dict, List, Callable
from tqdm import tqdm

import sbi.inference
from sbi import utils as sbi_utils
from sbi import analysis as sbi_analysis
from sbi.inference.base import infer, simulate_for_sbi
from sbi.utils.user_input_checks import prepare_for_sbi
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from torch.distributions import Distribution

from tvb.simulator.simulator import Simulator
from tvb.simulator.integrators import Integrator
from tvb.simulator.models.base import Model


class sbiModel:
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

        self.posterior = None
        self.posterior_samples = None
        self.simulations = None
        self.simulation_params = None
        self.density_estimator = None
        self.map_estimator = None
        self.log_prob = None

    def simulation_wrapper(self, params):

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

    def run_inference(
            self,
            method: str,
            num_simulations: int,
            num_workers: int,
            num_samples: int
    ):
        self.posterior, self.simulations, self.simulation_params, self.density_estimator = infer_main(
            simulator=self.simulation_wrapper,
            prior=self.priors,
            method=method,
            num_simulations=num_simulations,
            num_workers=num_workers
        )
        self.posterior.set_default_x(torch.as_tensor(np.squeeze(self.obs["x_obs"])))
        self.posterior_samples = self.posterior.sample(sample_shape=(num_samples,))
        self.log_prob = self.posterior.log_prob(self.posterior_samples)

        # return self.posterior, self.posterior_samples, self.simulations, self.simulation_params, self.density_estimator

    def simulations_from_samples(self, n):
        X_pp = []
        for sample in tqdm(self.posterior_samples[::n]):
            sample = np.asarray(sample)
            X = self.simulation_wrapper(params=sample)
            X_pp.append(np.asarray(X))

        X_pp = np.asarray(X_pp)
        return X_pp

    def get_sample(self):
        return self.posterior.sample((1,)).numpy()

    def get_map_estimator(self):
        self.map_estimator = self.posterior.map(show_progress_bars=False)
        return self.map_estimator


def infer_main(
        simulator: Callable,
        prior: Distribution,
        method: str,
        num_simulations: int,
        num_workers: int = 1
):
    try:
        method_fun: Callable = getattr(sbi.inference, method.upper())
    except AttributeError:
        raise NameError(
            "Method not available. `method` must be one of 'SNPE', 'SNLE', 'SNRE'."
        )

    simulator, prior = prepare_for_sbi(simulator, prior)

    inference = method_fun(prior=prior)
    theta, x = simulate_for_sbi(
        simulator=simulator,
        proposal=prior,
        num_simulations=num_simulations,
        num_workers=num_workers,
    )
    density_estimator = inference.append_simulations(theta, x).train()
    if method == "SNPE":
        posterior = inference.build_posterior()
    else:
        posterior = inference.build_posterior(mcmc_method="nuts")

    return posterior, x, theta, density_estimator
