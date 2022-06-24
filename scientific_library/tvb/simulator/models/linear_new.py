# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)

"""
Generic linear model.

"""

import numpy
import theano.tensor as tt
from typing import Union, Type, ClassVar
from .base import Model
from tvb.basic.neotraits.api import NArray, Final, List, Range
from tvb.basic.neotraits.ex import TraitTypeError


class Linear(Model):
    gamma = NArray(
        label=r":math:`\gamma`",
        default=numpy.array([-10.0]),
        domain=Range(lo=-100.0, hi=0.0, step=1.0),
        doc="The damping coefficient specifies how quickly the node's activity relaxes, must be larger"
            " than the node's in-degree in order to remain stable.")

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"x": numpy.array([-1, 1])},
        doc="Range used for state variable initialization and visualization.")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("x",),
        default=("x",), )

    coupling_terms = Final(
        label="Coupling terms",
        # how to unpack coupling array
        default=["c"]
    )

    state_variable_dfuns = Final(
        label="Drift functions",
        default={
            "x": "gamma * x + c",
        }
    )

    parameter_names = List(
        of=str,
        label="List of parameters for this model",
        default=tuple('gamma'.split()))

    state_variables = ('x',)
    _nvar = 1
    cvar = numpy.array([0], dtype=numpy.int32)
    statistical_model = None

    # def to_StatsModel(self, **kwargs):
    #     self.statistical_model = StatisticalModel(self, **kwargs)

    def dfun(self, state, coupling, local_coupling=0.0):
        """
        .. math::
            x = a{\gamma} + b
        """
        x, = state
        c, = coupling
        dx = self.gamma * x + c + local_coupling * x
        return numpy.array([dx])

    @staticmethod
    def pymc_dfun(state, params: dict):
        return params["gamma"] * state + params["coupling"] + params["local_coupling"] * state

    # @staticmethod
    # def tensor_dfun(state, coupling, local_coupling, params):
    #     dx = params[0] * state + coupling + local_coupling * state
    #     return dx
    #
    # @staticmethod
    # def tensor_dfun_sde(state, *args):
    #     x = state
    #     # c, = coupling
    #     # dx = kwargs["gamma"] * x + c + local_coupling * x
    #     return args[0] * x  # + args[1] + args[2] * x
    #     # return numpy.array([dx.eval()])
    #
    # @staticmethod
    # def tensor_dfun_ode(state, time, params):
    #     return params[0] * state[0]  # + params[1] + params[2] * x
