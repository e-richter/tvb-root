from tvb.simulator.models.base import Model
from typing import Dict, Tuple, List, Union
from pymc3.model import FreeRV, TransformedRV, DeterministicWrapper
from pymc3.distributions.timeseries import EulerMaruyama
import pymc3 as pm
import theano
import theano.tensor as tt
import arviz as az
import numpy as np


class NonCenteredModel:
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
            x_eta = pm.Normal(name="x_eta", mu=0.0, sd=1.0, shape=tuple(shape))

            self.dt = theano.shared(time_step, name="dt")
            self.noise = noise

            output, updates = theano.scan(fn=self.sde_scheme, sequences=[x_eta], outputs_info=[x_init], n_steps=shape[0])

            x_sym = output

            xhat = pm.Deterministic(name="xhat", var=amplitude * x_sym + offset)

            xs = pm.Normal(name="xs", mu=xhat, sd=epsilon, shape=tuple(shape), observed=self.obs["xs"])

    def sde_scheme(self, x_eta, x_prev):
        x_next = x_prev + self.dt * self.model_instance.theano_dfun(x_prev, {**self.priors, **self.consts}) + tt.sqrt(self.dt) * x_eta * self.noise
        return x_next

    def run_inference(self, draws, tune, cores):
        with self.pymc_model:
            trace = pm.sample(draws=draws, tune=tune, cores=cores)
            posterior_predictive = pm.sample_posterior_predictive(trace=trace)
            self.inference_data = az.from_pymc3(trace=trace, posterior_predictive=posterior_predictive)

        return self.inference_data


class CenteredModel:
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
            x_eta = pm.Normal(name="x_eta", mu=0.0, sd=1.0, shape=tuple(shape))

            self.dt = theano.shared(time_step, name="dt")
            self.noise = noise

            output, updates = theano.scan(fn=self.sde_scheme, sequences=[x_eta], outputs_info=[x_init], n_steps=shape[0])

            x_sym = output

            xhat = pm.Deterministic(name="xhat", var=amplitude * x_sym + offset)

            xs = pm.Normal(name="xs", mu=xhat, sd=epsilon, shape=tuple(shape), observed=self.obs["xs"])

    def sde_scheme(self, x_eta, x_prev):
        x_next = x_prev + self.dt * self.model_instance.theano_dfun(x_prev, {**self.priors, **self.consts}) + tt.sqrt(self.dt) * x_eta * self.noise
        return x_next

    def run_inference(self, draws, tune, cores):
        with self.pymc_model:
            trace = pm.sample(draws=draws, tune=tune, cores=cores)
            posterior_predictive = pm.sample_posterior_predictive(trace=trace)
            self.inference_data = az.from_pymc3(trace=trace, posterior_predictive=posterior_predictive)

        return self.inference_data



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


