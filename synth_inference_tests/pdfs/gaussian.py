"""
Random Gaussian pdf's from GPry ("Fast and robust Bayesian Inference using Gaussian Processes with GPry", J. El Gammal, N. Schöneberg, J. Torrado, C. Fidler, https://arxiv.org/abs/2211.02045).
"""

import numpy as np
from numpy.random import default_rng
from scipy.stats import multivariate_normal, random_correlation

from ..pdf import PDF
from ..mpi import is_main_process, multiple_processes, mpi_comm

_prior_size_in_std = 5
_default_random_mean_in_std = 5


class Gaussian(PDF):
    """
    Randomly correlated Gaussians.
    """

    dim = None
    multimodal = False
    nongaussian = 0
    random = True

    def __init__(self, dim, rng=None, prior_size_in_std=_prior_size_in_std,
                 random_mean_in_std=_default_random_mean_in_std):
        super().__init__(dim)
        self.prior_size_in_std = prior_size_in_std
        self.random_mean_in_std = random_mean_in_std
        self.draw(rng=rng)
        self.bounds = np.array([-self.prior_size_in_std * self.std,
                                self.prior_size_in_std * self.std]).T

    def draw(self, rng=None):
        if is_main_process:
            self.mean, self.cov, self.std = self.draw_mean_cov(
                self.dim, rng=rng, random_mean_in_std=_default_random_mean_in_std)
        if multiple_processes:
            for attr in ["mean", "cov", "std"]:
                setattr(self, attr, mpi_comm.bcast(getattr(self, attr, None)))
        self.rv = multivariate_normal(self.mean, self.cov)

    @staticmethod
    def draw_mean_cov(dim, rng=None, random_mean_in_std=5):
        """
        Draws a random mean and covmat for a multi-variate Gaussian.
        """
        if rng is None:
            rng = default_rng()
        stds = rng.uniform(size=dim)
        eigs = rng.uniform(size=dim)
        eigs = eigs / np.sum(eigs) * dim
        corr = random_correlation.rvs(eigs) if dim > 1 else [[1]]
        cov = np.multiply(np.outer(stds, stds), corr)
        mean = rng.uniform(low=-1., size=dim) * stds * random_mean_in_std
        return mean, cov, stds

    def logp(self, params):
        return self.rv.logpdf(params)

    def samples(self, n=None):
        if n is None:
            n = 100000  # seems to work OK for dim <= 20
        samples = self.rv.rvs(n)
        i_not_too_low = np.logical_and.reduce((samples > self.bounds[:, 0]).T)
        i_not_too_high = np.logical_and.reduce((samples < self.bounds[:, 1]).T)
        samples = samples[np.logical_and(i_not_too_low, i_not_too_high)]
        return samples
