"""
Generalization of the Rosenbrock test function to arbitrary dimensions from
https://arxiv.org/abs/1110.2997 (via https://arxiv.org/abs/2306.16923).

Difficulty: pronounced curving degeneracies.

"""

import numpy as np

from ..pdf import PDF


class Rosenbrock(PDF):
    """
    Generalized Rosenbrock function from https://arxiv.org/abs/1110.2997
    """

    dim = None
    dim_min = 2
    multimodal = False
    nongaussian = True

    def __init__(self, dim):
        super().__init__(dim)
        self.bounds = np.array(dim * [[-5, 5]])

    def logp(self, *params):
        params = np.array(params)
        return -np.sum((1 - params[:-1])**2 + 100 * (params[1:] - params[:-1]**2)**2)

    def samples(self, n=None):
        pass
