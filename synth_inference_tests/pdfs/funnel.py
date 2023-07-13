"""
"Correlated funnel" pdf in arbitrary dimensions from https://arxiv.org/abs/2002.06212
(via https://arxiv.org/abs/2306.16923).

From https://arxiv.org/abs/2306.16923:

> This likelihood problem is challenging because at theta_1 < 0 the posterior covers a
> small volume with high likelihood and whereas it covers a large volume with low
> likelihood for theta_1 > 0. This leads to large auto-correlation times for MCMC samplers
> (Karamanis & Beutler 2020). Similarly, this problem presents a challenge for NS
> algorithms with region sampling since the proposal volume likely misses parts of the
> low-theta_1 iso-likelihood surface due to the large volume differences along the first
> dimension.

"""

from warnings import warn

import numpy as np
from scipy.stats import norm, multivariate_normal

from ..pdf import PDF


class Funnel(PDF):
    """
    Correlated funnel from https://arxiv.org/abs/2002.06212
    """

    dim = None
    dim_min = 2
    multimodal = False
    nongaussian = True

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        if dim == 1:
            warn("A correlated funnel with dim=1 is simply a standard normal")
        self.bounds = np.array(dim * [[-10, 10]])
        self.rv_1 = norm(loc=0, scale=1)
        self.mean_rest = np.zeros(dim - 1)
        self.corr_rest = np.full(shape=(dim - 1, dim - 1), fill_value=0.95)
        np.fill_diagonal(self.corr_rest, 1)

    def logp(self, *params):
        cov = np.exp(params[0]) * self.corr_rest
        return (self.dim * np.log(20) + self.rv_1.logpdf(params[0]) +
                multivariate_normal.logpdf(params[1:], mean=self.mean_rest, cov=cov))

    def samples(self, n=None):
        if n is None:
            n = 100000  # seems to work OK for dim <= 20
        sample = np.full(shape=(n, self.dim), fill_value=np.nan)
        sample[:, 0] = self.rv_1.rvs(n)
        # Trick: remove exp(x_1) factor by transforming back to Norm(0, corr_rest),
        #        and then multiply. Easier to vectorize.
        corr_samples = multivariate_normal.rvs(
            size=n, mean=self.mean_rest, cov=self.corr_rest)
        transf_factor = np.exp(0.5 * sample[:, 0])
        if self.dim == 2:
            sample[:, 1] = corr_samples * transf_factor
        else:
            sample[:, 1:] = np.multiply(corr_samples, transf_factor[:, np.newaxis])
        return sample
