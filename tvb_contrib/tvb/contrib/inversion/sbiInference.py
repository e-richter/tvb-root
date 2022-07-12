import os
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import arviz as az
from typing import Dict, List, Callable, Tuple, Union
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import pickle

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
            method: str,
            obs: np.ndarray,
            priors: Dict[str, List],
            obs_shape: Union[Tuple, List]
    ):
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H%M")
        # self.simulator_instance = simulator_instance
        self.integrator_instance = integrator_instance
        self.model_instance = model_instance
        self.method = method
        self.obs = obs
        self.shape = tuple(obs_shape)
        self.priors = self._set_priors(priors=priors)

        self.posterior = None
        self.posterior_samples = None
        # self.simulations = None
        # self.simulation_params = None
        self.density_estimator = None
        self.map_estimator = None
        self.inference_data = None

    def simulation_wrapper(self, params, return_sim=False):

        for (i, key, _) in self.prior_keys:
            if key in ("noise", "epsilon", "x_t"):
                continue
            self.model_instance.__dict__[key] = np.asarray(params[i])

        # noise = params[[i for (i, key, _) in self.prior_keys if key == "noise"][0]]
        epsilon = params[[i for (i, key, _) in self.prior_keys if key == "epsilon"][0]]
        # x_t = params[[i for (i, key, _) in self.prior_keys if key == "x_t"][0]:]

        self.model_instance.configure()
        self.integrator_instance.noise.nsig = np.array([0.003])
        self.integrator_instance.noise.configure()
        self.integrator_instance.noise.configure_white(dt=self.integrator_instance.dt)
        self.integrator_instance.set_random_state(random_state=None)
        self.integrator_instance.configure()
        self.integrator_instance.configure_boundaries(self.model_instance)

        simulation_length = 300
        stimulus = 0.0
        local_coupling = 0.0
        current_state = np.random.uniform(low=-2.0, high=2.0, size=self.shape[1:])
        state = current_state
        current_step = 0
        number_of_nodes = 1
        start_step = current_step + 1
        node_coupling = np.zeros(self.shape[1:])
        n_steps = int(math.ceil(simulation_length / self.integrator_instance.dt))

        X = [current_state.copy()]
        for step in range(start_step, start_step + n_steps):
            state = self.integrator_instance.integrate(state, self.model_instance, node_coupling, local_coupling, stimulus)
            X.append(state.copy())

        x_sim = np.asarray(X)  # [int(simulation_length / self.integrator_instance.dt * 0.2):]
        # reshape output to be used by sbi TODO: eventually adjust when multiple nodes are simulated (shape[2]>1)
        x_sim = torch.as_tensor(x_sim.reshape(x_sim.size, order="F"))

        x_obs = x_sim + torch.distributions.MultivariateNormal(
            loc=torch.as_tensor(np.zeros(x_sim.numpy().size)),
            scale_tril=torch.diag(torch.as_tensor(epsilon * np.ones(x_sim.numpy().size)))
        ).sample()

        if return_sim:
            return x_obs, x_sim
        else:
            return x_obs

    def run_inference(
            self,
            num_simulations: int,
            num_workers: int,
            num_samples: int
    ):
        self.posterior, self.density_estimator = infer_main(
            simulator=self.simulation_wrapper,
            prior=self.priors,
            method=self.method,
            num_simulations=num_simulations,
            num_workers=num_workers
        )
        self.posterior.set_default_x(torch.as_tensor(self.obs.reshape(self.obs.size, order="F")))
        self.posterior_samples = self.posterior.sample(sample_shape=(num_samples,))

    def simulations_from_samples(self, n):
        X_posterior_predictive = []
        X_simulated = []
        for sample in tqdm(self.posterior_samples[::n]):
            sample = np.asarray(sample)
            X_obs, X_sim = self.simulation_wrapper(params=sample, return_sim=True)

            X_posterior_predictive.append(X_obs)
            X_simulated.append(X_sim)

        # X_posterior_predictive = np.asarray(X_posterior_predictive)
        # X_simulated = np.asarray(X_simulated)
        return torch.stack(X_posterior_predictive), torch.stack(X_simulated)

    def get_sample(self):
        return self.posterior.sample((1,))

    def get_map_estimator(self):
        self.map_estimator = self.posterior.map(show_progress_bars=False)
        return self.map_estimator

    def to_arviz_data(self, save: bool = False):
        X_posterior_predictive, X_simulated = self.simulations_from_samples(n=1)

        epsilon = self.posterior_samples[:, [i for (i, key, _) in self.prior_keys if key == "epsilon"][0]]
        log_probability = self.log_probability(X_posterior_predictive, X_simulated, sigma=epsilon)

        self.inference_data = az.from_dict(
            posterior=dict(zip([key for i, key, _ in self.prior_keys], np.asarray(self.posterior_samples.T))),
            posterior_predictive={"x_obs": X_posterior_predictive.numpy().reshape((1, len(self.posterior_samples), *self.shape), order="F")},
            log_likelihood={"x_obs": log_probability.numpy().reshape((1, len(self.posterior_samples), *self.shape), order="F")},
            observed_data={"x_obs": self.obs.reshape(self.shape, order="F")}
        )

        if save:
            self.inference_data.to_netcdf(filename=f"sbi_data/inference_data/{self.run_id}_inference_data.nc", compress=False)

        return self.inference_data

    def log_probability(self, X_pp: torch.Tensor, X_sim: torch.Tensor, sigma: Union[float, torch.Tensor]):

        if X_pp.ndim > 1 and X_sim.ndim > 1:
            logp = []
            for i, (x_sim, x_pp) in enumerate(zip(X_sim, X_pp)):
                mu = x_sim
                sig = sigma if isinstance(sigma, float) else sigma[i]
                logp_ = -math.log(sig * math.sqrt(2. * math.pi)) - ((x_pp - mu) / sig) ** 2 / 2.
                logp_ = torch.as_tensor(logp_.numpy().reshape(self.shape, order="F"))

                logp.append(logp_)

            logp = torch.stack(logp)

        else:
            mu = X_sim
            assert type(sigma) == float
            sig = sigma

            logp = -math.log(sig * math.sqrt(2. * math.pi)) - ((X_pp - mu) / sig) ** 2 / 2.
            logp = torch.as_tensor(logp.numpy().reshape(self.shape, order="F"))

        return logp

    def information_criteria(self):
        if self.inference_data is None:
            self.inference_data = self.to_arviz_data()

        waic = az.waic(self.inference_data, scale="deviance")
        loo = az.loo(self.inference_data, scale="deviance")

        return {"WAIC": waic.waic, "LOO": loo.loo}

    def plot_posterior_samples(self, init_params: Dict[str, float], save: bool = False):
        if self.inference_data is None:
            self.inference_data = self.to_arviz_data()

        num_params = len(init_params)
        ncols = int(np.ceil(np.sqrt(num_params)))
        nrows = int(np.ceil(num_params / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 16))
        for i, (key, value) in enumerate(init_params.items()):
            posterior_ = self.inference_data.posterior[key].values.reshape((self.inference_data.posterior[key].values.size,))
            ax = axes.reshape(-1)[i]
            ax.hist(posterior_, bins=100)
            ax.axvline(init_params[key], color="r")
            ax.set_title(key)

        if save:
            plt.savefig(f"sbi_data/figures/{self.run_id}_posterior_samples.png", dpi=600, bbox_inches=None)

    def save(self):
        with open(f"sbi_data/inference_data/{self.run_id}_instance.pkl", "wb") as out:
            pickle.dump(self.__dict__, out, pickle.HIGHEST_PROTOCOL)

    def load(self, pkl_file):
        with open(f"sbi_data/inference_data/{pkl_file}", "rb") as out:
            tmp = pickle.load(out)
            self.__dict__.update(tmp)

    def _set_priors(self, priors):

        # if dist == "Normal":
        #     prior_mean = [value[0] for _, value in priors.items()]
        #     prior_sd = [value[1] for _, value in priors.items()]
        #     self.prior_keys = [(i, key, value[-1]) for i, (key, value) in enumerate(priors.items())]
        #
        #     prior_loc = []
        #     prior_scale = []
        #     for mean, sd, key in zip(prior_mean, prior_sd, self.prior_keys):
        #         if key[-1]:
        #             prior_loc.append(torch.as_tensor(mean * np.ones(self.obs.size)))
        #             prior_scale.append(torch.as_tensor(sd * np.ones(self.obs.size)))
        #         else:
        #             prior_loc.append(torch.as_tensor(np.array([mean])))
        #             prior_scale.append(torch.as_tensor(np.array([sd])))
        #
        #     prior_loc = torch.cat(prior_loc)
        #     prior_scale = torch.diag(torch.cat(prior_scale))
        #
        #     return torch.distributions.MultivariateNormal(loc=prior_loc, scale_tril=prior_scale)

        prior_min = [value[0] for _, value in priors.items()]
        prior_max = [value[1] for _, value in priors.items()]
        self.prior_keys = [(i, key, value[-1]) for i, (key, value) in enumerate(priors.items())]

        return sbi_utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))


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

    return posterior, density_estimator
