"""
Multimodal LogGamma-Normal distribution from https://arxiv.org/abs/1304.7808, with
parameter values as in https://arxiv.org/abs/1407.5459 (via
https://arxiv.org/abs/2306.16923).

Difficulty: multimodality and heavy tails.

"""

import os
import json
import numpy as np
from scipy.stats import norm
from scipy.special import gamma
from scipy.integrate import quad
from scipy import interpolate
from functools import partial

from ..pdf import PDF
from ..utils import invCDFinterp

path_loggamma_interp_data = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data", "loggamma.json")


def logp_loggamma(x, alpha, mu, sigma):
    """
    log-PDF of the LogGamma distribution from https://arxiv.org/abs/1005.3274.
    """
    return - np.log(np.abs(sigma) * gamma(alpha)) + (
        alpha * (x - mu) / sigma - np.exp((x - mu) / sigma))


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
        self.norm_2_1 = norm(loc=1 / 3, scale=1 / 30)
        self.norm_2_2 = norm(loc=2 / 3, scale=1 / 30)
        self.args_loggamma_1_1 = {"alpha": 1, "mu": 1 / 3, "sigma": 1 / 30}
        self.args_loggamma_1_2 = {"alpha": 1, "mu": 2 / 3, "sigma": 1 / 30}
        self.logp_loggamma_1_1 = partial(logp_loggamma, **self.args_loggamma_1_1)
        self.logp_loggamma_1_2 = partial(logp_loggamma, **self.args_loggamma_1_2)
        self.logp_loggamma_1 = lambda x: (np.log(0.5) + np.logaddexp(
            self.logp_loggamma_1_1(x), self.logp_loggamma_1_2(x)))
        self.last_loggamma_index = int(np.floor(self.dim / 2))
        self.dims_loggamma = list(range(2, 1 + self.last_loggamma_index))
        self.dims_norm = list(range(1 + self.last_loggamma_index, self.dim))

    def logp(self, params):
        params = np.atleast_2d(params)
        logp = self.logp_loggamma_1(params[:, 0])
        logp += np.log(0.5) + np.log(self.norm_2_1.pdf(params[:, 1]) +
                                     self.norm_2_2.pdf(params[:, 1]))
        if self.dims_loggamma:
            logp += np.sum(self.logp_loggamma_1_2(params[:, self.dims_loggamma]), axis=-1)
        if self.dims_norm:
            logp += np.sum(self.norm_2_2.logpdf(params[:, self.dims_norm]), axis=-1)
        return logp

    def samples(self, n=None):
        if not hasattr(self, "_loggamma_interp_1"):
            self._load_or_cache_inv_cdf()
        if n is None:
            n = 100000  # seems to work OK for dim <= 20
        sample = np.full(shape=(n, self.dim), fill_value=np.nan)
        sample[:, 0] = interpolate.splev(np.random.random(n), self._loggamma_interp_1)
        sample[:, 1] = np.where(
            np.random.random(n) > 0.5, self.norm_2_1.rvs(n), self.norm_2_2.rvs(n))
        if self.dims_loggamma:
            sample[:, self.dims_loggamma] = interpolate.splev(
                np.random.random(n * len(self.dims_loggamma)).reshape(
                    (n, len(self.dims_loggamma))),
                self._loggamma_interp_1_2)
        if self.dims_norm:
            sample[:, self.dims_norm] = self.norm_2_2.rvs(
                n * len(self.dims_norm)).reshape(
                    (n, len(self.dims_norm)))
        return sample

    def _load_or_cache_inv_cdf(self):
        """
        Loads the interpolator for fast Inverse Transform Sampling of the LogGammas,
        or pre-computes it and caches it if the logGamma args have changed.
        """
        splrep_kwargs = {}
        try:
            with open(path_loggamma_interp_data, "r") as f:
                data = json.loads("\n".join(f.readlines()))
            max_dim_test = min(len(self.bounds), 4)
            assert np.allclose(data["bounds"][:max_dim_test], self.bounds[:max_dim_test])
            assert all(np.isclose(data["args_loggamma_1_1"][k], v) for k, v in
                       self.args_loggamma_1_1.items())
            assert all(np.isclose(data["args_loggamma_1_2"][k], v) for k, v in
                       self.args_loggamma_1_2.items())
            # Sampling `n_samples_cdf` not tested because the function that prepares
            # the interpolator may drop some samples at the borders
        except (
                FileNotFoundError,  # interpolator not precomputed
                json.decoder.JSONDecodeError,  # interpolator not correctly saved
                KeyError,  # interpolator incorrectly precomputed (older version)
                AssertionError,  # parameters changed since interpolator saved
        ):
            # We need to calculate it
            n_samples_cdf = 5000
            xs = np.linspace(self.bounds[0][0] + 1e-5, self.bounds[0][1], n_samples_cdf)
            CDF_1, xs_1, self._loggamma_interp_1 = invCDFinterp(
                xs, lambda x: np.exp(self.logp_loggamma_1(x)),
                splrep_kwargs=splrep_kwargs)
            if self.dims_loggamma:
                CDF_1_2, xs_1_2, self._loggamma_interp_1_2 = invCDFinterp(
                    xs, lambda x: np.exp(self.logp_loggamma_1_2(x)),
                    splrep_kwargs=splrep_kwargs)
            # Uncomment only in development, to save the interp. data into the package
            # Run with dim >=4 only.
            save_data = {"bounds": np.array(self.bounds).tolist(),
                         "args_loggamma_1_1": self.args_loggamma_1_1,
                         "args_loggamma_1_2": self.args_loggamma_1_2,
                         "CDF_1": CDF_1.tolist(), "xs_1": xs_1.tolist(),
                         "CDF_1_2": CDF_1_2.tolist(), "xs_1_2": xs_1_2.tolist()}
            with open(path_loggamma_interp_data, "w") as f:
                f.write(json.dumps(save_data))
        else:
            # Re-build the interpolators from the saved data
            self._loggamma_interp_1 = interpolate.splrep(
                data["CDF_1"], data["xs_1"], **splrep_kwargs)
            if self.dims_loggamma:
                self._loggamma_interp_1_2 = interpolate.splrep(
                    data["CDF_1_2"], data["xs_1_2"], **splrep_kwargs)

    @property
    def logZ(self):
        return self.logprior_density
