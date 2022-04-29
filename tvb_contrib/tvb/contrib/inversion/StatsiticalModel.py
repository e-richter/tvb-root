from tvb.simulator.simulator import Simulator
from tvb.contrib.inversion.pymcInference import NonCenteredModel, EulerMarayumaModel
from typing import Literal, Dict, Union
from pymc3.model import FreeRV, TransformedRV, DeterministicWrapper

class StatisticalModel:
    def __init__(
            self,
            simulator_instance: Simulator
    ):
        self.simulator_instance = simulator_instance
        self.statistical_model = None

    def sbi_inference(self):
        pass

    def pymc_inference(
            self,
            method: Literal["NonCentered", "EulerMaruyama"],
    ):
        if method == "NonCentered":
            self.statistical_model = NonCenteredModel(self.simulator_instance.model)
        elif method == "EulerMaruyama":
            self.statistical_model = EulerMarayumaModel(self.simulator_instance.model)

