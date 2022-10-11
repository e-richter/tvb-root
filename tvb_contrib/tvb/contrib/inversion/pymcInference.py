from tvb.simulator.models.base import Model
from tvb.simulator.simulator import Simulator
from typing import Dict, Tuple, List, Union
from pymc3.model import FreeRV, TransformedRV, DeterministicWrapper
import pymc3 as pm
import theano
import theano.tensor as tt
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from copy import deepcopy, copy


class pymcModel1node:
    def __init__(self, tvb_model: Model):
        self.tvb_model = tvb_model
        self.stat_model = pm.Model()

        self.run_id = datetime.now().strftime("%Y-%m-%d_%H%M")

        self.priors = None
        self.prior_stats = None
        self.obs = None
        self.shape = None
        self.dt = None

        self.trace = None
        self.inference_data = None
        self.summary = None

    def set_model(
            self,
            priors: Dict[str, Union[FreeRV, TransformedRV, DeterministicWrapper]],
            obs: np.ndarray,
            time_step: float,
    ):
        self.priors = priors
        self.obs = obs
        self.shape = tuple(self.obs.shape)
        with self.stat_model:
            self.dt = theano.shared(time_step, name="dt")

            x_sim, updates = theano.scan(fn=self.scheme, sequences=[self.priors["dynamic_noise"]], outputs_info=[self.priors["x_init"]], n_steps=self.shape[0])

            amplitude_star = pm.Normal(name="amplitude_star", mu=0.0, sd=1.0)
            amplitude = pm.Deterministic(name="amplitude", var=0.0 + amplitude_star)

            offset_star = pm.Normal(name="offset_star", mu=0.0, sd=1.0)
            offset = pm.Deterministic(name="offset", var=0.0 + offset_star)

            x_hat = pm.Deterministic(name="x_hat", var=amplitude * x_sim + offset)

            x_obs = pm.Normal(name="x_obs", mu=x_hat, sd=self.priors["global_noise"], shape=self.shape, observed=self.obs)

    def scheme(self, x_eta, x_prev):
        x_next = x_prev + self.dt * self.tvb_model.dfun_tensor(x_prev, self.priors, self.priors["node_coupling"]) + x_eta  # * self.noise * tt.sqrt(self.dt)
        return x_next

    def run_inference(self, draws: int, tune: int, cores: int, target_accept: float, max_treedepth: int, step_scale: float, save: bool = False):
        with self.stat_model:
            self.trace = pm.sample(draws=draws, tune=tune, cores=cores, target_accept=target_accept, max_treedepth=max_treedepth, step_scale=step_scale)
            posterior_predictive = pm.sample_posterior_predictive(trace=self.trace)
            self.inference_data = az.from_pymc3(trace=self.trace, posterior_predictive=posterior_predictive)
            self.summary = az.summary(self.inference_data)

            if save:
                self.inference_data.to_netcdf(filename=f"pymc_data/inference_data/{self.run_id}_inference_data.nc", compress=False)

        return self.inference_data

    def model_criteria(self):
        waic = az.waic(self.inference_data)
        loo = az.loo(self.inference_data)

        return {"WAIC": waic.waic, "LOO": loo.loo}

    def posterior_zscore(self, init_params: Dict[str, float]):
        z = np.empty(len(init_params))
        for i, (key, value) in enumerate(init_params.items()):
            posterior_ = self.inference_data.posterior[key].values.reshape((self.inference_data.posterior[key].values.size,))
            z_ = np.abs(value - posterior_.mean()) / posterior_.std()
            z[i] = z_
        return z

    def posterior_shrinkage(self):
        s = np.empty(len(self.prior_stats))
        for i, (key, value) in enumerate(self.prior_stats.items()):
            posterior_ = self.inference_data.posterior[key].values.reshape((self.inference_data.posterior[key].values.size,))
            s_ = 1 - (posterior_.std()**2 / value["sd"]**2)
            s[i] = s_
        return s

    def plot_posterior_samples(self, init_params: Dict[str, float], save: bool = False):
        num_params = len(init_params)
        nrows = int(np.ceil(np.sqrt(num_params)))
        ncols = int(np.ceil(num_params / nrows))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
        for i, (key, value) in enumerate(init_params.items()):
            if isinstance(value, np.ndarray):
                continue

            posterior_ = self.inference_data.posterior[key].values.reshape((self.inference_data.posterior[key].values.size,))
            ax = axes.reshape(-1)[i]
            ax.hist(posterior_, bins=100)
            ax.axvline(init_params[key], color="r", label="simulation parameter")
            ax.axvline(self.prior_stats[key]["mean"], color="r", linestyle="-.", label="prior mean")
            ax.set_xlim(
                xmin=self.prior_stats[key]["mean"] - 2 * self.prior_stats[key]["sd"],
                xmax=self.prior_stats[key]["mean"] + 2 * self.prior_stats[key]["sd"]
            )
            ax.set_title(key, fontsize=18)
            ax.tick_params(axis="both", labelsize=16)
        try:
            axes[0, 0].legend(fontsize=18)
        except IndexError:
            axes[0].legend(fontsize=18)

        if save:
            plt.savefig(f"pymc_data/figures/{self.run_id}_posterior_samples.png", dpi=600, bbox_inches=None)

    def save(self, simulation_params: dict):
        with open(f"pymc_data/inference_data/{self.run_id}_instance.pkl", "wb") as out:
            tmp = self.__dict__.copy()
            del tmp["tvb_model"]
            del simulation_params["simulation"]
            pickle.dump({**tmp, "simulation_params": simulation_params}, out, pickle.HIGHEST_PROTOCOL)
            out.close()

    def load(self, pkl_file):
        with open(f"pymc_data/inference_data/{pkl_file}", "rb") as out:
            tmp = pickle.load(out)
            self.__dict__.update(tmp)


class pymcModel:
    def __init__(self, tvb_simulator: Simulator):
        self.tvb_simulator = tvb_simulator
        self.stat_model = pm.Model()

        self.run_id = datetime.now().strftime("%Y-%m-%d_%H%M")

        self.priors = None
        self.prior_stats = None
        self.obs = None
        self.shape = None
        self.dt = None

        self.trace = None
        self.inference_data = None
        self.summary = None

    def set_model(
            self,
            priors: Dict[str, Union[FreeRV, TransformedRV, DeterministicWrapper, np.ndarray, float]],
            obs: np.ndarray,
            time_step: float,
            observation_model="Raw",
    ):
        self.priors = priors
        self.obs = obs
        self.shape = tuple(self.obs.shape)
        with self.stat_model:
            self.dt = theano.shared(time_step, name="dt")

            Nsv = len(self.tvb_simulator.model.state_variables)
            Nr = self.tvb_simulator.connectivity.number_of_regions
            idmax = self.tvb_simulator.connectivity.idelays.max()

            # x0_init = pm.Normal(name="x0_init", mu=0.0, sd=1.0, shape=(Nsv, Nr, 1))
            x0_init = np.zeros((Nsv, Nr, 1))
            for i, (_, value) in enumerate(self.tvb_simulator.model.state_variable_range.items()):
                loc = (value[0] + value[1]) / 2
                scale = (value[1] - value[0]) / 2
                x0_init[i, :, :] = np.random.normal(loc=loc, scale=scale, size=(1, Nr, 1))

            x_init = np.zeros((idmax + 1, Nsv, Nr, 1))
            x_init = theano.shared(x_init, name="x_init")
            x_init = tt.set_subtensor(x_init[-1], x0_init)

            # history_init = pm.Normal(name="history_init", mu=0.0, sd=5.0, shape=(idmax, Nsv, Nr, 1))
            # nc_init = np.zeros([Nt, Nc, Nr, 1])
            # nc_init = theano.shared(nc_init, name="nc_init")
            #
            # X_init = np.empty([Nt, Nsv, Nr, 1])
            # X_init = theano.shared(X_init, name="X_init")
            # nc, _ = theano.scan(
            #     fn=self.compute_node_coupling,
            #     sequences=[tt.as_tensor_variable(np.arange(Nt))],
            #     non_sequences=[X_init, x_init, history_init],
            #     outputs_info=[nc_init],
            #     n_steps=self.shape[0]
            # )
            # self.priors["node_coupling"] = nc

            taps = list(-1 * np.arange(np.unique(self.tvb_simulator.history.nnz_idelays).max() + 1) - 1)
            x_sim, updates = theano.scan(
                fn=self.scheme,
                sequences=[self.priors["dynamic_noise"]],
                outputs_info=[dict(initial=x_init, taps=taps)],
                n_steps=self.shape[0]
            )

            if observation_model == "Raw":
                amplitude_star = pm.Normal(name="amplitude_star", mu=0.0, sd=1.0)
                amplitude = pm.Deterministic(name="amplitude", var=0.0 + amplitude_star)

                offset_star = pm.Normal(name="offset_star", mu=0.0, sd=1.0)
                offset = pm.Deterministic(name="offset", var=0.0 + offset_star)

                x_hat = pm.Deterministic(name="x_hat", var=amplitude * x_sim[:, self.tvb_simulator.model.cvar, :, :] + offset)

                x_obs = pm.Normal(name="x_obs", mu=x_hat, sd=self.priors["global_noise"], shape=self.shape, observed=self.obs)

    def scheme(self, x_eta, *args):
        Nr = self.tvb_simulator.connectivity.number_of_regions
        Ncv = self.tvb_simulator.history.n_cvar

        x_prev = args[-1]

        x_i = x_prev[self.tvb_simulator.model.cvar, :, :]
        x_i = x_i[:, self.tvb_simulator.history.nnz_row_el_idx]

        x_j = tt.stack(args, axis=0)
        x_j = x_j[:, self.tvb_simulator.model.cvar, :, :]
        x_j = x_j[-1 * self.tvb_simulator.history.nnz_idelays - 1]
        x_j = x_j[np.arange(self.tvb_simulator.history.n_nnzw), :, self.tvb_simulator.history.nnz_col_el_idx, :].reshape([Ncv, self.tvb_simulator.history.n_nnzw, 1])

        pre = self.tvb_simulator.coupling.pre(x_i, x_j)

        weights_col = self.tvb_simulator.history.nnz_weights.reshape((self.tvb_simulator.history.n_nnzw, 1))
        sum_ = np.zeros((Ncv, Nr, 1))
        lri, nzr = self.tvb_simulator.coupling._lri(self.tvb_simulator.history.nnz_row_el_idx)
        try:
            sum_[:, nzr] = np.add.reduceat(weights_col * pre, lri, axis=1)
            node_coupling = self.tvb_simulator.coupling.post_tensor(sum_, self.priors)
        except:
            node_coupling = self.tvb_simulator.coupling.post_tensor(sum_, self.priors)

        m_dx_tn = self.tvb_simulator.model.dfun_tensor(x_prev, self.priors, node_coupling)
        inter = x_prev + self.dt * m_dx_tn + x_eta
        x_next = x_prev + (m_dx_tn + self.tvb_simulator.model.dfun_tensor(inter, self.priors, node_coupling)) * self.dt / 2.0 + x_eta
        return x_next

    def compute_node_coupling(self, it, nc, X_init, x_init, history_init):

        Nt = int(self.tvb_simulator.simulation_length)
        Nsv = len(self.tvb_simulator.model.state_variables)
        Nr = self.tvb_simulator.connectivity.number_of_regions
        Ncv = self.tvb_simulator.history.n_cvar
        Nc = 1
        idmax = self.tvb_simulator.connectivity.idelays.max()
        cvars = self.tvb_simulator.history.cvars

        delayed_indices = (it - self.tvb_simulator.connectivity.idelays).flatten()

        X_init_bundle = copy(X_init)
        X_init_bundle = tt.set_subtensor(X_init_bundle[0], x_init)
        X_init_bundle = tt.set_subtensor(X_init_bundle[-idmax:], history_init)

        X_delayed = X_init_bundle[delayed_indices, cvars, np.repeat(np.arange(Nr), Nr), :].reshape(
            (Nr, Ncv, Nr, 1))
        X_current = X_init[it, cvars, :, :]

        nc = tt.set_subtensor(nc[it, :, :, :], (self.tvb_simulator.history.nnz_weights[np.newaxis, :].T * self.tvb_simulator.coupling.pre(
            X_current, X_delayed)).sum(axis=2).reshape([Ncv, Nr, 1]))

        nc = tt.set_subtensor(nc[it, :, :, :], self.tvb_simulator.coupling.post(nc[it, :, :, :]))

        return nc

    def run_inference(self, draws: int, tune: int, cores: int, target_accept: float, max_treedepth: int, step_scale: float, save: bool = False):
        with self.stat_model:
            self.trace = pm.sample(draws=draws, tune=tune, cores=cores, target_accept=target_accept, max_treedepth=max_treedepth, step_scale=step_scale)
            posterior_predictive = pm.sample_posterior_predictive(trace=self.trace)
            self.inference_data = az.from_pymc3(trace=self.trace, posterior_predictive=posterior_predictive)
            self.summary = az.summary(self.inference_data)

            if save:
                self.inference_data.to_netcdf(filename=f"pymc_data/inference_data/{self.run_id}_inference_data.nc", compress=False)

        return self.inference_data

    def model_criteria(self):
        waic = az.waic(self.inference_data)
        loo = az.loo(self.inference_data)

        return {"WAIC": waic.waic, "LOO": loo.loo}

    def posterior_zscore(self, init_params: Dict[str, float]):
        z = np.empty(len(init_params))
        for i, (key, value) in enumerate(init_params.items()):
            posterior_ = self.inference_data.posterior[key].values.reshape((self.inference_data.posterior[key].values.size,))
            z_ = np.abs(value - posterior_.mean()) / posterior_.std()
            z[i] = z_
        return z

    def posterior_shrinkage(self):
        s = np.empty(len(self.prior_stats))
        for i, (key, value) in enumerate(self.prior_stats.items()):
            posterior_ = self.inference_data.posterior[key].values.reshape((self.inference_data.posterior[key].values.size,))
            s_ = 1 - (posterior_.std()**2 / value["sd"]**2)
            s[i] = s_
        return s

    def plot_posterior_samples(self, init_params: Dict[str, float], save: bool = False):
        num_params = len(init_params)
        nrows = int(np.ceil(np.sqrt(num_params)))
        ncols = int(np.ceil(num_params / nrows))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
        for i, (key, value) in enumerate(init_params.items()):
            if isinstance(value, np.ndarray):
                continue

            posterior_ = self.inference_data.posterior[key].values.reshape((self.inference_data.posterior[key].values.size,))
            ax = axes.reshape(-1)[i]
            ax.hist(posterior_, bins=100)
            ax.axvline(init_params[key], color="r", label="simulation parameter")
            ax.axvline(self.prior_stats[key]["mean"], color="r", linestyle="-.", label="prior mean")
            ax.set_xlim(
                xmin=self.prior_stats[key]["mean"] - 2 * self.prior_stats[key]["sd"],
                xmax=self.prior_stats[key]["mean"] + 2 * self.prior_stats[key]["sd"]
            )
            ax.set_title(key, fontsize=18)
            ax.tick_params(axis="both", labelsize=16)
        try:
            axes[0, 0].legend(fontsize=18)
        except IndexError:
            axes[0].legend(fontsize=18)

        if save:
            plt.savefig(f"pymc_data/figures/{self.run_id}_posterior_samples.png", dpi=600, bbox_inches=None)

    def save(self, simulation_params: dict):
        with open(f"pymc_data/inference_data/{self.run_id}_instance.pkl", "wb") as out:
            tmp = self.__dict__.copy()
            del tmp["tvb_simulator"]
            del simulation_params["simulation"]
            pickle.dump({**tmp, "simulation_params": simulation_params}, out, pickle.HIGHEST_PROTOCOL)
            out.close()

    def load(self, pkl_file):
        with open(f"pymc_data/inference_data/{pkl_file}", "rb") as out:
            tmp = pickle.load(out)
            self.__dict__.update(tmp)


class CenteredModel:
    def __init__(self, model_instance: Model):
        self.model_instance = model_instance
        self.pymc_model = pm.Model()

        self.priors = None
        self.consts = None
        self.obs = None
        self.dt = None
        self.noise = None

        self.trace = None
        self.inference_data = None
        self.summary = None

    def set_model(
            self,
            priors: Dict[str, Union[FreeRV, TransformedRV, DeterministicWrapper]],
            consts: Dict[str, float],
            obs: Dict,
            time_step: float,
            x_init: Union[FreeRV, TransformedRV],
            noise: Union[FreeRV, TransformedRV],
            amplitude: Union[FreeRV, TransformedRV],
            offset: Union[FreeRV, TransformedRV],
            epsilon: Union[FreeRV, TransformedRV],
            shape: Union[Tuple, List]
    ):
        self.priors = priors
        self.consts = consts
        self.obs = obs
        with self.pymc_model:
            self.dt = theano.shared(time_step, name="dt")
            self.noise = noise

            x_sim, updates = theano.scan(fn=self.scheme, outputs_info=[x_init], n_steps=shape[0])

            x_t = pm.Normal(name="x_t", mu=x_sim, sd=tt.sqrt(time_step) * self.noise, shape=tuple(shape))
            x_hat = pm.Deterministic(name="x_hat", var=amplitude * x_t + offset)

            x_obs = pm.Normal(name="x_obs", mu=x_hat, sd=epsilon, shape=tuple(shape), observed=self.obs["x_obs"])

    def scheme(self, x_prev):
        x_next = x_prev + self.dt * self.model_instance.pymc_dfun(x_prev, {**self.priors, **self.consts})
        return x_next

    def run_inference(self, draws, tune, cores, target_accept):
        with self.pymc_model:
            self.trace = pm.sample(draws=draws, tune=tune, cores=cores, target_accept=target_accept)
            posterior_predictive = pm.sample_posterior_predictive(trace=self.trace)
            self.inference_data = az.from_pymc3(trace=self.trace, posterior_predictive=posterior_predictive)
            self.summary = az.summary(self.inference_data)

        return self.inference_data

    def model_criteria(self, criteria: List[str]):
        out = dict()
        if "WAIC" in criteria:
            waic = az.waic(self.inference_data, scale="deviance")
            out["WAIC"] = waic.waic
        if "LOO" in criteria:
            loo = az.loo(self.inference_data, scale="deviance")
            out["LOO"] = loo.loo

        map_estimate = None
        if "AIC" in criteria:
            with self.pymc_model:
                map_estimate = pm.find_MAP()
            aic = -2 * self.pymc_model.logp(map_estimate) + 2 * len(map_estimate)
            out["AIC"] = aic
        if "BIC" in criteria:
            if map_estimate:
                bic = -2 * self.pymc_model.logp(map_estimate) + len(map_estimate) * np.log(len(self.obs["xs"]))
                out["BIC"] = bic
            else:
                with self.pymc_model:
                    map_estimate = pm.find_MAP()
                bic = -2 * self.pymc_model.logp(map_estimate) + len(map_estimate) * np.log(len(self.obs["xs"]))
                out["BIC"] = bic

        return out

    def plot_posterior(self, init_params: Dict[str, float]):
        num_params = len([key for key, value in self.priors.items() if not isinstance(value, np.ndarray)])
        nrows = int(np.ceil(np.sqrt(num_params)))
        ncols = int(np.ceil(num_params / nrows))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
        for i, (key, value) in enumerate(self.priors.items()):
            if isinstance(value, np.ndarray):
                continue

            posterior_ = self.inference_data.posterior[key].values.reshape((self.inference_data.posterior[key].values.size,))
            ax = axes.reshape(-1)[i]
            ax.hist(posterior_, bins=100)
            ax.axvline(init_params[key], color="r")
            ax.set_title(key)
