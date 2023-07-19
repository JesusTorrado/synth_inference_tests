"""
Multimodal Gaussian mixture, simplified version of the LogGamma-Normal distribution from
https://arxiv.org/abs/1304.7808.

Difficulty: multimodality with different correlations for different modes.

"""

import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.special import gamma
from scipy.integrate import quad
from scipy import interpolate
from functools import partial

from ..pdf import PDF
from ..utils import invCDFinterp


class GaussianMix(PDF):
    """
    Multimodal Gaussian mixture, simplified version of the LogGamma-Normal distribution
    from https://arxiv.org/abs/1304.7808.
    """

    dim = None
    dim_min = 2
    multimodal = True
    nongaussian = False

    def __init__(self, dim):
        super().__init__(dim)
        self.bounds = np.array(dim * [[-5, 5]])
        self.mean_1_1 = 1 / 3
        self.mean_1_2 = 2 / 3
        self.mean_2_1 = 1 / 3
        self.mean_2_2 = 2 / 3
        self.std = 1 / 25
        self.cov_a = self.std**2 * np.array([[1, 0.9], [0.9, 1]])
        self.cov_b = self.std**2 * np.array([[1, -0.9], [-0.9, 1]])
        self.norm_ul = multivariate_normal(
            mean=(self.mean_1_1, self.mean_2_1), cov=self.cov_a)
        self.norm_ur = multivariate_normal(
            mean=(self.mean_1_2, self.mean_2_1), cov=self.cov_b)
        self.norm_bl = multivariate_normal(
            mean=(self.mean_1_1, self.mean_2_2), cov=self.cov_b)
        self.norm_br = multivariate_normal(
            mean=(self.mean_1_2, self.mean_2_2), cov=self.cov_a)
        self.norm_ge_2 = norm(loc=self.mean_1_2, scale=self.std)

    def logp(self, *params):
        params = np.array(params)
        logp = np.log(0.25) + np.log(
            self.norm_ul.pdf(params[:2]) + self.norm_ur.pdf(params[:2]) +
            self.norm_bl.pdf(params[:2]) + self.norm_br.pdf(params[:2]))
        if self.dim > 2:
            logp += np.sum(self.norm_ge_2.logpdf(params[2:]))
        return logp

    def samples(self, n=None):
        if n is None:
            n = 100000  # seems to work OK for dim <= 20
        # TODO: not sure if enough for KL stability
        # ALSO check that n_samples_cdf is enough for KL
        sample = np.full(shape=(n, self.dim), fill_value=np.nan)
        mix_param = np.random.random(n)
        i_ul = np.where(mix_param < 0.25)[0]
        i_ur = np.where(np.logical_and(0.25 < mix_param, mix_param < 0.50))[0]
        i_bl = np.where(np.logical_and(0.50 < mix_param, mix_param < 0.75))[0]
        i_br = np.where(0.75 < mix_param)[0]
        sample[i_ul, :2] = self.norm_ul.rvs(len(i_ul))
        sample[i_ur, :2] = self.norm_ur.rvs(len(i_ur))
        sample[i_bl, :2] = self.norm_bl.rvs(len(i_bl))
        sample[i_br, :2] = self.norm_br.rvs(len(i_br))
        if self.dim > 2:
            sample[:, 2:] = \
                self.norm_ge_2.rvs(n * (self.dim - 2)).reshape((n, self.dim - 2))
        return sample

    @property
    def logZ(self):
        logZ = 0
        logZ += -np.sum(np.log(self.bounds.T[1] - self.bounds.T[0]))
        return logZ
