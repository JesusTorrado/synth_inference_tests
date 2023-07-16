"""
Multimodal LogGamma-Normal distribution from https://arxiv.org/abs/1304.7808, with
parameter values as in https://arxiv.org/abs/1407.5459 (via
https://arxiv.org/abs/2306.16923).

Difficulty: multimodality and heavy tails.

"""

import numpy as np
from scipy.stats import norm
from scipy.special import gamma

from ..pdf import PDF


def logp_loggamma(x, alpha, mu, sigma):
    """
    log-PDF of the LogGamma distribution from https://arxiv.org/abs/1005.3274.
    """
    return - np.log(np.abs(sigma) * gamma(alpha)) + (
        alpha * (x - mu) / sigma - np.exp((x - mu) / sigma))


class LogGamma(PDF):
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
        self.pdf_2_1 = norm(loc=1 / 3, scale=1 / 30)
        self.pdf_2_2 = norm(loc=2 / 3, scale=1 / 30)
        self.last_loggamma_index = int(np.floor(self.dim / 2))
        self.dims_loggamma = list(range(2, 1 + self.last_loggamma_index))
        self.dims_norm = list(range(1 + self.last_loggamma_index, self.dim))

    def logp(self, *params):
        params = np.array(params)
        logp = np.log(0.5) + np.log(np.exp(logp_loggamma(params[0], 1, 1 / 3, 1 / 30)) +
                                      np.exp(logp_loggamma(params[0], 1, 2 / 3, 1 / 30)))
        logp += np.log(0.5) + np.log(self.pdf_2_1.pdf(params[1]) +
                                      self.pdf_2_2.pdf(params[1]))
        if self.dims_loggamma:
            logp += np.sum(logp_loggamma(params[self.dims_loggamma], 1, 2 / 3, 1 / 30))
        if self.dims_norm:
            logp += np.sum(self.pdf_2_2.logpdf(params[self.dims_norm]))
        return logp
