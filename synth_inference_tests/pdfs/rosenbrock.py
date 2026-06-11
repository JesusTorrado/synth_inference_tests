"""
Generalization of the Rosenbrock test function to arbitrary dimensions from
https://arxiv.org/abs/1110.2997 (via https://arxiv.org/abs/2306.16923).

Difficulty: pronounced curving degeneracies.

"""

import os
from warnings import warn
import numpy as np

from ..pdf import PDF
from ..io import path_data


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

    def logp(self, params):
        params = np.atleast_2d(params)
        return -np.sum(
            (1 - params[:, :-1])**2 + 100 * (params[:, 1:] - params[:, :-1]**2)**2,
            axis=-1)

    def samples(self, n=None):
        try:
            samples = np.load(os.path.join(path_data, self.NameDim + ".npy"))
        except FileNotFoundError:
            warn(f"Samples not precomputed for Rosenbrock with dim {self.dim}.")
            return None
        if n is not None:
            warn("Ignoring the number of samples requested, since it is a precomputed "
                 "weighted sample.")
        warn(f"Returning samples weights as first column of the table.")
        return samples
