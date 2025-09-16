import numpy as np

from do_mpc.model import Model
from do_mpc.controller import MPC
from do_mpc.simulator import Simulator

from nn_policy import NNRegressor
from config import Config
from data_collector import run_trajectory

from typing import Any, Callable
from numpy.typing import NDArray
import warnings

class Evaluator():
    def __init__(self) -> None:
        pass

    def evaluate(self,
                 model: Model,
                 controller: MPC | NNRegressor,
                 simulator: Simulator,
                 cfgs: Config,
                 sampler: Sampler,
                 M: int = 100,) -> None:
        
        steps = cfgs.N        
        M = 100
        for _ in range(M):
            x0 = sampler.sample()
            x, u = run_trajectory(x0, steps, simulator, controller)
        

class Sampler:
    def __init__(self, X0: NDArray | None = None, f: Callable | None = None):
        # The sampling index
        assert (X0 is None) ^ (f is None), "Provide exactly one of X0 or f"
        self.i = 0

        if X0 is not None:
            # Will sample from a grid, indexed by rows in X0.
            self.X0 = X0
            self.n = X0.shape[0]
            self._sampler = self._sample_from_grid
        else:
            self.f = f
            self._sampler = self._sample_from_function

    def _sample_from_grid(self):
        x = self.X0[self.i]
        self.i = (self.i + 1) % self.n
        if self.i == 0:
            warnings.warn("All elements have been sampled once, repeating samples now...")
        return x

    def _sample_from_function(self):
        return self.f() # type: ignore

    def sample(self):
        return self._sampler()



