#  Copyright 2019-2020, Pedro Gomes.
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
import ipyopt as opt
import subprocess as sp
from drivers.parallel_eval_driver import ParallelEvalDriver


# Wrapper to use with the Ipopt optimizer via IPyOpt.
class IpoptDriver(ParallelEvalDriver):
    def __init__(self):
        ParallelEvalDriver.__init__(self)

        # counters, flags, sizes...
        self._funEval = 0
        self._jacEval = 0
        self._nCon = 0

        # sparse indices of the constraint gradient, for now assumed to be dense
        self._sparseIndices = None

        # the optimization problem
        self._nlp = None

        # old values of the gradients, as fallback in case of evaluation failure
        self._old_grad_f = None
        self._old_jac_g = None
    #end

    # Update the problem parameters (triggers new evaluations).
    def update(self):
        for par in self._parameters: par.increment()

        self._x[()] = 1e20
        self._funReady = False
        self._jacReady = False
        self._resetAllValueEvaluations()
        self._resetAllGradientEvaluations()

        if self._hisObj is not None:
            self._hisObj.write("Parameter update.\n")
    #end

    # Prepares and returns the optimization problem for Ipopt.
    # For convenience also does other preprocessing, all functions must be set before calling this method.
    # Do not destroy the driver after obtaining the problem.
    def getNLP(self):
        self.preprocessVariables()

        self._ofval = np.zeros((len(self._objectives),))
        self._eqval = np.zeros((len(self._constraintsEQ),))
        self._gtval = np.zeros((len(self._constraintsGT),))

        # write the header for the history file
        if self._hisObj is not None:
            header = "ITER"+self._hisDelim
            for obj in self._objectives:
                header += obj.function.getName()+self._hisDelim
            for obj in self._constraintsEQ:
                header += obj.function.getName()+self._hisDelim
            for obj in self._constraintsGT:
                header += obj.function.getName()+self._hisDelim
            header = header.strip(self._hisDelim)+"\n"
            self._hisObj.write(header)
        #end

        # prepare constraint information, the bounds are based on the shifting and scaling
        self._nCon = len(self._constraintsEQ) + len(self._constraintsGT)

        conLowerBound = np.zeros([self._nCon,])
        conUpperBound = np.zeros([self._nCon,])

        i = len(self._constraintsEQ)
        conUpperBound[i:(i+len(self._constraintsGT))] = 1e20

        # assume row major storage for gradient sparsity
        rg = range(self._nVar * self._nCon)
        self._sparseIndices = (np.array([i // self._nVar for i in rg], dtype=int),
                               np.array([i % self._nVar for i in rg], dtype=int))

        # create the optimization problem
        self._nlp = opt.Problem(self._nVar, self.getLowerBound(), self.getUpperBound(),
                                self._nCon, conLowerBound, conUpperBound, self._sparseIndices, 0,
                                self._eval_f, self._eval_grad_f, self._eval_g, self._eval_jac_g)
        return self._nlp
    #end

    # Method passed to Ipopt to get the objective value,
    # evaluates all functions if necessary.
    def _eval_f(self, x):
        self._evaluateFunctions(x)
        return self._ofval.sum()
    #end

    # Method passed to Ipopt to get the objective gradient, evaluates gradients and
    # functions if necessary, otherwise it simply combines and scales the results.
    def _eval_grad_f(self, x, out):
        assert out.size >= self._nVar, "Wrong size of gradient vector (\"out\")."

        self._jacTime -= time.time()

        try:
            self._evaluateGradients(x)

            os.chdir(self._workDir)

            out[()] = 0.0
            for obj in self._objectives:
                out += obj.function.getGradient(self._variableStartMask) * obj.scale
            out /= self._varScales

            # keep reference to result to use as fallback on next iteration if needed
            self._old_grad_f = out
        except:
            if self._failureMode is "HARD": raise
            if self._old_grad_f is None: out[()] = 0.0
            else: out[()] = self._old_grad_f
        #end

        self._jacTime += time.time()
        os.chdir(self._userDir)

        return out
    #end

    # Method passed to Ipopt to expose the constraint vector, see also "_eval_f"
    def _eval_g(self, x, out):
        assert out.size >= self._nCon, "Wrong size of constraint vector (\"out\")."

        self._evaluateFunctions(x)

        i = 0
        out[i:(i+len(self._constraintsEQ))] = self._eqval

        i += len(self._constraintsEQ)
        out[i:(i+len(self._constraintsGT))] = self._gtval

        return out
    #end

    # Method passed to Ipopt to expose the constraint Jacobian, see also "_eval_grad_f".
    def _eval_jac_g(self, x, out):
        assert out.size >= self._nCon*self._nVar, "Wrong size of constraint Jacobian vector (\"out\")."

        self._jacTime -= time.time()

        try:
            self._evaluateGradients(x)

            os.chdir(self._workDir)

            i = 0
            mask = self._variableStartMask

            for conType in [self._constraintsEQ, self._constraintsGT]:
                for con in conType:
                    out[i:(i+self._nVar)] = con.function.getGradient(mask) * con.scale / self._varScales
                    i += self._nVar
                #end
            #end

            # keep reference to result to use as fallback on next iteration if needed
            self._old_jac_g = out
        except:
            if self._failureMode is "HARD": raise
            if self._old_jac_g is None: out[()] = 0.0
            else: out[()] = self._old_jac_g
        #end

        self._jacTime += time.time()
        os.chdir(self._userDir)

        return out
    #end

    # Evaluate all functions (objectives and constraints), imediately
    # retrieves and stores the results after shifting and scaling.
    def _evaluateFunctions(self, x):
        self._handleVariableChange(x)

        # lazy evaluation
        if self._funReady: return

        if self._userPreProcessFun:
            os.chdir(self._userDir)
            sp.call(self._userPreProcessFun,shell=True)
        #end

        self._evalAndRetrieveFunctionValues()
    #end

    # Evaluates all gradients in parallel execution mode, otherwise
    # it only runs the user preprocessing and the execution takes place
    # when the results are read in "_eval_grad_f" or in "_eval_jac_g".
    def _evaluateGradients(self, x):
        # we assume that evaluating the gradients requires the functions
        self._evaluateFunctions(x)        

        # lazy evaluation
        if self._jacReady: return

        if self._userPreProcessGrad:
            os.chdir(self._userDir)
            sp.call(self._userPreProcessGrad,shell=True)
        #end

        os.chdir(self._workDir)

        # evaluate everything, either in parallel or sequentially,
        # in the latter case the evaluations occur when retrieving the values
        if self._parallelEval: self._evalJacInParallel()

        os.chdir(self._userDir)

        self._jacEval += 1
        self._jacReady = True
    #end
#end

