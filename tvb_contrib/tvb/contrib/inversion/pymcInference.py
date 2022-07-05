from tvb.simulator.models.base import Model
from typing import Dict, Tuple, List, Union
from pymc3.model import FreeRV, TransformedRV, DeterministicWrapper
from pymc3.distributions.timeseries import EulerMaruyama
import pymc3 as pm
import theano
import theano.tensor as tt
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class NonCenteredModel:
    def __init__(self, model_instance: Model):
        self.model_instance = model_instance
        self.pymc_model = pm.Model()
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H%M")

        self.priors = None
        self.consts = None
        self.params = None
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
        self.params = {**self.priors, **self.consts}
        self.obs = obs
        with self.pymc_model:

            self.dt = theano.shared(time_step, name="dt")
            self.noise = noise

            x_t = pm.Normal(name="x_t", mu=0.0, sd=1.0, shape=tuple(shape))
            x_sim, updates = theano.scan(fn=self.scheme, sequences=[x_t], outputs_info=[x_init], n_steps=shape[0])

            x_hat = pm.Deterministic(name="x_hat", var=amplitude * x_sim + offset)

            x_obs = pm.Normal(name="x_obs", mu=x_hat, sd=epsilon, shape=tuple(shape), observed=self.obs["x_obs"])

    def scheme(self, x_eta, x_prev):
        x_next = x_prev + self.dt * self.model_instance.pymc_dfun(x_prev, self.params) + tt.sqrt(self.dt) * x_eta * self.noise
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
            ax.axvline(init_params[key], color="r")
            ax.set_title(key)


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

            x_t = pm.Normal(name="x_t", mu=x_sim, sd=tt.sqrt(time_step)*self.noise, shape=tuple(shape))
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
            waic = az.waic(self.inference_data)
            out["WAIC"] = waic.waic
        if "LOO" in criteria:
            loo = az.loo(self.inference_data)
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


class EulerMaruyamaModel:
    def __init__(self, model_instance: Model):
        self.model_instance = model_instance
        self.pymc_model = pm.Model()

        self.priors = None
        self.consts = None
        self.obs = None
        self.dt = None
        self.noise = None

        self.inference_data = None

    def set_model(
            self,
            priors: Dict[str, Union[FreeRV, TransformedRV, DeterministicWrapper]],
            consts: Dict[str, float],
            obs: Dict,
            time_step: float,
            noise: float,
            epsilon: Union[FreeRV, TransformedRV],
            shape: Union[Tuple, List]
    ):
        self.priors = priors
        self.consts = consts
        self.obs = obs
        self.dt = time_step
        self.noise = noise
        with self.pymc_model:
            priors_tuple = tuple([value for _, value in self.priors.items()])

            xhat = EulerMaruyama(name="xhat", dt=self.dt, sde_fn=self.sde_fn, sde_pars=priors_tuple, shape=shape[0], testval=self.obs["xs"])
            xs = pm.Normal(name="xs", mu=xhat, sd=epsilon, shape=tuple(shape), observed=self.obs["xs"])

    def sde_fn(self, x, *args):
        return self.model_instance.theano_dfun(x, {**self.priors, **self.consts}), self.noise

    def run_inference(self, draws, tune, cores):
        with self.pymc_model:
            trace = pm.sample(draws=draws, tune=tune, cores=cores)
            posterior_predictive = pm.sample_posterior_predictive(trace=trace)
            self.inference_data = az.from_pymc3(trace=trace, posterior_predictive=posterior_predictive)

        return self.inference_data
