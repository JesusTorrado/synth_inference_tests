"""
Multimodal LogGamma-Normal distribution from https://arxiv.org/abs/1304.7808, with
parameter values as in https://arxiv.org/abs/1407.5459 (via
https://arxiv.org/abs/2306.16923).

Difficulty: multimodality and heavy tails.

"""

import numpy as np
from scipy.stats import norm, loggamma  # type: ignore

from ..pdf import PDF


class LogGammaMix(PDF):
    """
    Multimodal LogGamma-Normal distribution from https://arxiv.org/abs/1304.7808, with
    default parameter values as in https://arxiv.org/abs/1407.5459
    """

    dim = None
    dim_min = 2
    multimodal = True
    nongaussian = True

    def __init__(self, dim):
        super().__init__(dim)
        self.bounds = np.array(dim * [[-5, 5]])
        # NB: mind the 0-based indexing of dimensionality when comparing to sources
        self.loggamma_0_1 = loggamma(c=1, loc=1 / 3, scale=1 / 30)
        self.loggamma_0_2 = loggamma(c=1, loc=2 / 3, scale=1 / 30)
        self.logp_loggamma_0 = lambda x: (
            np.log(0.5)
            + np.logaddexp(self.loggamma_0_1.logpdf(x), self.loggamma_0_2.logpdf(x))
        )
        self.norm_1_1 = norm(loc=1 / 3, scale=1 / 30)
        self.norm_1_2 = norm(loc=2 / 3, scale=1 / 30)
        self.logp_norm_1 = lambda x: (
            np.log(0.5) + np.logaddexp(self.norm_1_1.logpdf(x), self.norm_1_2.logpdf(x))
        )
        self.last_loggamma_index = int(np.floor(self.dim / 2))
        self.dims_loggamma = list(range(2, 1 + self.last_loggamma_index))
        self.dims_norm = list(range(1 + self.last_loggamma_index, self.dim))

    def logp(self, params):
        params = np.atleast_2d(params)
        logp = self.logp_loggamma_0(params[:, 0]) + self.logp_norm_1(params[:, 1])
        if self.dims_loggamma:
            logp += np.sum(
                self.loggamma_0_2.logpdf(params[:, self.dims_loggamma]), axis=-1
            )
        if self.dims_norm:
            logp += np.sum(self.norm_1_2.logpdf(params[:, self.dims_norm]), axis=-1)
        # We subtract the prior density here so that the evidence is 1
        return logp - self.logprior_density

    def samples(self, n=None):
        if n is None:
            n = 100000  # seems to work OK for dim <= 20
        sample = np.full(shape=(n, self.dim), fill_value=np.nan)
        sample[:, 0] = np.where(
            np.random.random(n) > 0.5,
            self.loggamma_0_1.rvs(n),
            self.loggamma_0_2.rvs(n),
        )
        sample[:, 1] = np.where(
            np.random.random(n) > 0.5, self.norm_1_1.rvs(n), self.norm_1_2.rvs(n)
        )
        if self.dims_loggamma:
            # despite .rvs being able to take a shape, the .reshape is needed to
            # avoid the corner case len(self.dims_X) == 1
            sample[:, self.dims_loggamma] = self.loggamma_0_2.rvs(
                n * len(self.dims_loggamma)
            ).reshape(n, len(self.dims_loggamma))
        if self.dims_norm:
            sample[:, self.dims_norm] = self.norm_1_2.rvs(
                n * len(self.dims_norm)
            ).reshape(n, len(self.dims_loggamma))
        return sample

    @property
    def logZ(self):
        return 0
