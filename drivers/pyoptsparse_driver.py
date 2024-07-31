#  Copyright 2019-2023, FADO Contributors (cf. AUTHORS.md)
#
#  This file is part of FADO.
#
#  FADO is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  FADO is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with FADO.  If not, see <https://www.gnu.org/licenses/>.

import os
import time
import numpy as np
import argparse
from pyoptsparse import SLSQP, Optimization
from drivers.constrained_optim_driver import ConstrainedOptimizationDriver


class PyoptsparseDriver(ConstrainedOptimizationDriver):
    """
    Driver to use with the Ipopt optimizer via pyoptsparse.
    """

    def __init__(self):
        ConstrainedOptimizationDriver.__init__(self)

        # Sparse indices of the constraint gradient, for now assumed to be dense
        self._sparseIndices = None

        # The optimization problem
        self._opt_prob = None
        self._opt_grad = None
        self.fail = False
        self.funcs = {}
    
        
    def _dictfun(self):
        funcs = {}
        funcs['obj'] = self._eval_f
        funcs['cons'] = self._eval_g
        fail = False
        return funcs, fail
    
    def getNLP(self):
        """
        Prepares and returns the optimization problem for Ipopt (an instance of pyoptsparse.OptimizationProblem).
        For convenience also does other preprocessing, must be called after all functions are set.
        Do not destroy the driver after obtaining the problem.
        """
        ConstrainedOptimizationDriver.preprocess(self)

        conLowerBound = np.zeros([self._nCon, ])
        conUpperBound = np.zeros([self._nCon, ])

        i = len(self._constraintsEQ)
        conUpperBound[i:(i + len(self._constraintsGT))] = 1e20

        # Assume row-major storage for gradient sparsity
        rg = range(self._nVar * self._nCon)
        self._sparseIndices = (
            np.array([i // self._nVar for i in rg], dtype=int),
            np.array([i % self._nVar for i in rg], dtype=int)
        )

        # Create the optimization problem
        
        self._opt_prob = Optimization("OptimizationName",_dictfun)
        self._opt_prob.addVarGroup('x', self._nVar, lower=self.getLowerBound(), upper=self.getUpperBound())
        self._opt_prob.addConGroup('cons', self._nCon, lower=conLowerBound, upper=conUpperBound)
        self._opt_prob.addObj('obj')

        return self._opt_prob
    

    
    def getGrad(self):
        grads = {}
        grads['obj'] = {'x':self._eval_grad_f}
        grads['cons'] = {'x':self._eval_jac_g}
        self._opt_grad = grads
        fail = False
        return self._opt_grad,fail
    
    def _eval_f(self, xdict):
        """
        Method to evaluate the objective function, computes all functions if necessary.
        """
        x = xdict['x']

        self._evaluateFunctions(x)

        return self._ofval.sum()

    def _eval_grad_f(self, xdict, out):
        """
        Method to evaluate the gradient of the objective function, evaluates gradients and
        functions if necessary, otherwise it simply combines and scales the results.
        """
        x = xdict['x']

        self._evaluateGradients(x)

        out[()] = 0.0
        for obj in self._objectives:
            out += obj.function.getGradient(self._variableStartMask) * obj.scale

        out /= self._varScales

        return out

    def _eval_g(self, xdict, out):
        """
        Method to expose the constraint vector, see also "_eval_f".
        """
        x = xdict['x']

        self._evaluateFunctions(x)

        i = 0
        out[i:(i + len(self._constraintsEQ))] = self._eqval

        i += len(self._constraintsEQ)
        out[i:(i + len(self._constraintsGT))] = self._gtval

        return out

    def _eval_jac_g(self, xdict, out):
        """
        Method to expose the constraint Jacobian, see also "_eval_grad_f".
        """
        x = xdict['x']

        self._evaluateGradients(x)

        i = 0
        mask = self._variableStartMask

        for con in self._constraintsEQ:
            out[i:(i + self._nVar)] = con.function.getGradient(mask) * con.scale / self._varScales
            i += self._nVar

        for (con, f) in zip(self._constraintsGT, self._gtval):
            if f < 0.0 or not self._asNeeded:
                out[i:(i + self._nVar)] = con.function.getGradient(mask) * con.scale / self._varScales
            else:
                out[i:(i + self._nVar)] = 0.0

            i += self._nVar

        return out
        
