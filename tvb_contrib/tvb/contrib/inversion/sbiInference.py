import os
import time

import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import arviz as az
from typing import Dict, List, Callable, Tuple, Union, Literal
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import pickle
from copy import deepcopy
from joblib import Parallel, delayed
import itertools
import operator

import sbi.inference
from sbi import utils as sbi_utils
from sbi import analysis as sbi_analysis
from sbi.inference.base import infer, simulate_for_sbi
from sbi.simulators.simutils import tqdm_joblib
from sbi.utils.user_input_checks import prepare_for_sbi
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from torch.distributions import Distribution

from tvb.simulator.simulator import Simulator
from tvb.simulator.integrators import Integrator
from tvb.simulator.models.base import Model


class sbiModel:
    def __init__(
            self,
            method: Literal["SNPE", "SNLE", "SNRE"],
            obs: np.ndarray,
            simulator_instance: Simulator = None,
            integrator_instance: Integrator = None,
            model_instance: Model = None
    ):

        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%-M-%S-%f")
        self.simulator_instance = simulator_instance
        self.integrator_instance = integrator_instance
        self.model_instance = model_instance
        if self.simulator_instance is None:
            self.simulation_wrapper = self.simulation_wrapper_1node
        else:
            self.simulation_wrapper = self.simulation_wrapper_nnodes

        self.method = method
        self.obs = obs
        self.shape = self.obs.shape
        self.neural_net = None
        self.priors = None

        self.posterior = None
        self.posterior_samples = None
        self.num_simulations = None
        self.simulations = None
        self.simulation_params = None
        self.density_estimator = None
        self.map_estimator = None
        self.inference_data = None

    def simulation_wrapper_nnodes(self, params, return_sim=False):

        sim_ = deepcopy(self.simulator_instance)

        for (i, key, target) in self.prior_keys:
            if target == "global":
                continue
            if "noise" in target:
                operator.attrgetter(target)(sim_).__dict__[key] = np.abs(np.array([float(params[i])]))
            else:
                operator.attrgetter(target)(sim_).__dict__[key] = np.array([float(params[i])])
            # getattr(sim_, target).__dict__[key] = np.asarray(params[i])

        sim_.configure()

        # amplitude = params[[i for (i, key, _) in self.prior_keys if key == "amplitude"][0]]
        # offset = params[[i for (i, key, _) in self.prior_keys if key == "offset"][0]]
        epsilon = torch.abs(params[[i for (i, key, _) in self.prior_keys if key == "epsilon"][0]])

        (t, X), = sim_.run()

        x_sim = torch.as_tensor(X.reshape(X.size, order="F"))

        x_obs = x_sim + torch.distributions.MultivariateNormal(
            loc=torch.as_tensor(np.zeros(x_sim.numpy().size)),
            scale_tril=torch.diag(torch.as_tensor(epsilon * np.ones(x_sim.numpy().size)))
        ).sample()

        del sim_

        if return_sim:
            return torch.stack((x_obs, x_sim), dim=0)
        else:
            return x_obs

    def simulation_wrapper_1node(self, params, return_sim=False):

        model_ = deepcopy(self.model_instance)
        integrator_ = deepcopy(self.integrator_instance)

        for (i, key, target) in self.prior_keys:
            if target == "global":
                continue
            if "model" in target:
                model_.__dict__[key] = np.array([float(params[i])])
            elif "integrator" in target:
                integrator_.__dict__[key] = np.abs(np.array([float(params[i])]))

        epsilon = torch.abs(params[[i for (i, key, _) in self.prior_keys if key == "epsilon"][0]])

        model_.configure()
        integrator_.noise.configure()
        integrator_.noise.configure_white(dt=integrator_.dt)
        integrator_.set_random_state(random_state=None)
        integrator_.configure()
        integrator_.configure_boundaries(model_)

        simulation_length = int(len(self.obs) * integrator_.dt - 1)
        stimulus = 0.0
        local_coupling = 0.0
        current_state = np.random.uniform(low=-2.0, high=2.0, size=self.shape[1:])
        state = current_state
        current_step = 0
        number_of_nodes = 1
        start_step = current_step + 1
        node_coupling = np.zeros(self.shape[1:])
        n_steps = int(math.ceil(simulation_length / integrator_.dt))

        X = [current_state.copy()]
        for step in range(start_step, start_step + n_steps):
            state = integrator_.integrate(state, model_, node_coupling, local_coupling, stimulus)
            X.append(state.copy())

        x_sim = np.asarray(X)
        x_sim = torch.as_tensor(x_sim.reshape(x_sim.size, order="F"))

        x_obs = x_sim + torch.distributions.MultivariateNormal(
            loc=torch.as_tensor(np.zeros(x_sim.numpy().size)),
            scale_tril=torch.diag(torch.as_tensor(epsilon * np.ones(x_sim.numpy().size)))
        ).sample()

        if return_sim:
            return torch.stack((x_obs, x_sim), dim=0)
        else:
            return x_obs

    def run_inference(
            self,
            prior_vars: Dict[str, List],
            prior_dist: Literal["Normal", "Uniform"],
            num_simulations: int,
            num_workers: int,
            num_samples: int,
            neural_net: str = "maf"
    ):

        self.neural_net = neural_net
        self.priors = self._set_priors(prior_vars=prior_vars, prior_dist=prior_dist)

        self.num_simulations = num_simulations
        self.posterior, self.density_estimator, self.simulation_params = infer_main(
            simulator=self.simulation_wrapper,
            prior=self.priors,
            method=self.method,
            neural_net=self.neural_net,
            num_simulations=num_simulations,
            num_workers=num_workers
        )
        self.posterior.set_default_x(torch.as_tensor(self.obs.reshape(self.obs.size, order="F")))
        self.posterior_samples = self.posterior.sample(sample_shape=(num_samples,))

    def simulations_from_samples(self, num_workers: int, n: int = 1):
        theta = self.posterior_samples[::n]

        X_posterior_predictive, X_simulated = parallel_simulations(
            simulator=self.simulation_wrapper,
            theta=theta,
            num_workers=num_workers
        )

        return X_posterior_predictive, X_simulated

        # X_posterior_predictive = []
        # X_simulated = []
        # for sample in tqdm(self.posterior_samples[::n]):
        #     sample = np.asarray(sample)
        #     X_obs, X_sim = self.simulation_wrapper(params=sample, return_sim=True)
        #
        #     X_posterior_predictive.append(X_obs)
        #     X_simulated.append(X_sim)
        #
        # return torch.stack(X_posterior_predictive), torch.stack(X_simulated)

    def get_sample(self):
        return self.posterior.sample((1,))

    def get_map_estimator(self):
        self.map_estimator = self.posterior.map(show_progress_bars=False)
        return self.map_estimator

    def to_arviz_data(self, num_workers: int, save: bool = False):
        X_posterior_predictive, X_simulated = self.simulations_from_samples(num_workers=num_workers)

        epsilon = self.posterior_samples[:, [i for (i, key, _) in self.prior_keys if key == "epsilon"][0]]
        log_probability = self.log_probability(X_posterior_predictive, X_simulated, sigma=epsilon)

        self.inference_data = az.from_dict(
            posterior=dict(zip(["_".join([key, target]) for i, key, target in self.prior_keys], np.asarray(self.posterior_samples.T))),
            posterior_predictive={"x_obs": X_posterior_predictive.numpy().reshape((1, len(self.posterior_samples), *self.shape), order="F")},
            log_likelihood={"x_obs": log_probability.numpy().reshape((1, len(self.posterior_samples), *self.shape), order="F")},
            observed_data={"x_obs": self.obs.reshape(self.shape, order="F")}
        )

        if save:
            self.inference_data.to_netcdf(filename=f"sbi_data/inference_data/{self.run_id}_inference_data.nc", compress=False)

        return self.inference_data

    def posterior_zscore(self, init_params: Dict[str, float]):
        z = torch.empty(len(init_params))
        for i, (key, value) in enumerate(init_params.items()):
            posterior_ = self.inference_data.posterior[key].values.reshape((self.inference_data.posterior[key].values.size,))
            z_ = np.abs(value - posterior_.mean()) / posterior_.std()
            z[i] = z_
        return z

    def posterior_shrinkage(self):
        s = 1 - (self.posterior_samples.std(dim=0)**2 / torch.diag(self.priors.scale_tril)**2)
        return s

    def log_probability(self, X_pp: torch.Tensor, X_sim: torch.Tensor, sigma: Union[float, torch.Tensor]):

        if X_pp.ndim > 1 and X_sim.ndim > 1:
            logp = []
            for i, (x_sim, x_pp) in enumerate(zip(X_sim, X_pp)):
                mu = x_sim
                sig = sigma if isinstance(sigma, float) else sigma[i]
                sig = torch.abs(sig)
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

        waic = az.waic(self.inference_data)
        loo = az.loo(self.inference_data)

        return {"WAIC": waic.waic, "LOO": loo.loo}

    def plot_posterior_samples(self, init_params: Dict[str, float], bins: int = 100, save: bool = False):
        if self.inference_data is None:
            self.inference_data = self.to_arviz_data()

        num_params = len(init_params)
        ncols = int(np.ceil(np.sqrt(num_params)))
        nrows = int(np.ceil(num_params / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 16))
        for i, (key, value) in enumerate(init_params.items()):
            posterior_ = self.inference_data.posterior[key].values.reshape((self.inference_data.posterior[key].values.size,))
            ax = axes.reshape(-1)[i]
            ax.hist(posterior_, bins=bins)
            ax.axvline(init_params[key], color="r", label="simulation parameter")
            ax.set_title(key, fontsize=18)
            ax.tick_params(axis="both", labelsize=16)
        axes[0, 0].legend(fontsize=18)

        if save:
            plt.savefig(f"sbi_data/figures/{self.run_id}_posterior_samples.png", dpi=600, bbox_inches=None)

    def save(self, simulation_params: dict):
        with open(f"sbi_data/inference_data/{self.run_id}_instance.pkl", "wb") as out:
            tmp = self.__dict__.copy()
            del tmp["simulator_instance"]
            del tmp["model_instance"]
            del tmp["integrator_instance"]
            del tmp["simulation_wrapper"]
            del simulation_params["simulation"]
            pickle.dump({**tmp, "simulation_params": simulation_params}, out, pickle.HIGHEST_PROTOCOL)
            out.close()

    def load(self, pkl_file):
        with open(f"sbi_data/inference_data/{pkl_file}", "rb") as out:
            tmp = pickle.load(out)
            self.__dict__.update(tmp)

    def _set_priors(self, prior_vars, prior_dist):

        if prior_dist == "Normal":
            prior_mean = [v[0] for v in list(itertools.chain(*[list(w.values()) for _, w in prior_vars.items()]))]
            prior_sd = [v[1] for v in list(itertools.chain(*[list(w.values()) for _, w in prior_vars.items()]))]
            
            # self.prior_keys = [(i, key, value["for"]) for i, (key, value) in enumerate(prior_vars.items())]
            targets = []
            keys = []
            for target, v in prior_vars.items():
                targets.append([target] * len(list(v.keys())))
                keys.append(list(v.keys()))
            targets = list(itertools.chain(*targets))
            keys = list(itertools.chain(*keys))
            self.prior_keys = []
            for i, (key, target) in enumerate(zip(keys, targets)):
                self.prior_keys.append((i, key, target))

            prior_loc = []
            prior_scale = []
            for mean, sd in zip(prior_mean, prior_sd):
                prior_loc.append(torch.as_tensor(np.array([mean])))
                prior_scale.append(torch.as_tensor(np.array([sd])))

            prior_loc = torch.cat(prior_loc)
            prior_scale = torch.diag(torch.cat(prior_scale))

            return torch.distributions.MultivariateNormal(loc=prior_loc, scale_tril=prior_scale)

        elif prior_dist == "Uniform":
            prior_min = [(v[0] - v[1]) for v in list(itertools.chain(*[list(w.values()) for _, w in prior_vars.items()]))]
            prior_max = [(v[0] + v[1]) for v in list(itertools.chain(*[list(w.values()) for _, w in prior_vars.items()]))]
            
            # self.prior_keys = [(i, key, value["for"]) for i, (key, value) in enumerate(prior_vars.items())]
            targets = []
            keys = []
            for target, v in prior_vars.items():
                targets.append([target] * len(list(v.keys())))
                keys.append(list(v.keys()))
            targets = list(itertools.chain(*targets))
            keys = list(itertools.chain(*keys))
            self.prior_keys = []
            for i, (key, target) in enumerate(zip(keys, targets)):
                self.prior_keys.append((i, key, target))

            return sbi_utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))


def infer_main(
        simulator: Callable,
        prior: Distribution,
        method: str,
        neural_net: str,
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

    inference = method_fun(prior=prior, density_estimator=neural_net)
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

    return posterior, density_estimator, theta


def parallel_simulations(
    simulator: Callable,
    theta: torch.Tensor,
    sim_batch_size: int = 1,
    num_workers: int = 1,
    return_sim: bool = True,
    show_progress_bars: bool = True,
) -> torch.Tensor:

    num_sims, *_ = theta.shape

    if num_sims == 0:
        x = torch.tensor([])
    elif sim_batch_size is not None and sim_batch_size < num_sims:
        batches = torch.split(theta, sim_batch_size, dim=0)

        if num_workers > 1:
            with tqdm_joblib(
                tqdm(
                    batches,
                    disable=not show_progress_bars,
                    desc=f"Running {num_sims} simulations in {len(batches)} batches.",
                    total=len(batches),
                )
            ) as progress_bar:
                simulation_outputs = Parallel(n_jobs=num_workers)(
                    delayed(simulator)(batch[0], return_sim) for batch in batches
                )
        else:
            pbar = tqdm(
                total=num_sims,
                disable=not show_progress_bars,
                desc=f"Running {num_sims} simulations.",
            )

            with pbar:
                simulation_outputs = []
                for batch in batches:
                    simulation_outputs.append(simulator(batch[0], return_sim))
                    pbar.update(sim_batch_size)

        x = torch.stack(simulation_outputs, dim=1)
    else:
        x = simulator(theta, return_sim)

    return x
