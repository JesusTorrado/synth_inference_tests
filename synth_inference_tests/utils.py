import numpy as np
from numpy.linalg import det
from scipy import integrate
from scipy import interpolate


def kl_sym(sample_1, logp_sample_1, logp_2_sample_1,
           sample_2, logp_sample_2, logp_1_sample_2, weights_1=None, weights_2=None):
    """
    Computes the Jeffrey's divergence between two distributions given samples of each and
    their log-posterior functions.
    """
    P_to_Q = kl(sample_1, logp_sample_1, logp_2_sample_1, weights_P=weights_1)
    Q_to_P = kl(sample_2, logp_sample_2, logp_1_sample_2, weights_P=weights_2)
    return P_to_Q + Q_to_P


# Originally from GPry
def kl(sample_P, logp_sample_P, logp_Q_sample_P, weights_P=None):
    """
    Computes a Monte Carlo estimate of the Kullback-Leibler divergence ``KL(P|Q)`` given
    a sample from ``P`` and the log-posterior ``Q`` for that sample.
    """
    if weights_P is None:
        weights_P = np.ones(len(sample_P))
    else:
        # Numerical stability: make the highest weight 1
        weights_P /= max(weights_P)
    kl = np.sum(weights_P * (logp_sample_P - logp_Q_sample_P)) / np.sum(weights_P)
    return kl


def kl_norm_sym(mean_0, cov_0, mean_1, cov_1):
    return kl_norm(mean_0, cov_0, mean_1, cov_1) + kl_norm(mean_1, cov_1, mean_0, cov_0)


def kl_norm(mean_0, cov_0, mean_1, cov_1):
    """
    Computes the KL divergence between two normal distributions defined by their means
    and covariance matrices.

    May raise ``numpy.linalg.LinAlgError``.
    """
    cov_1_inv = np.linalg.inv(cov_1)
    dim = len(mean_0)
    return 0.5 * (np.log(det(cov_1)) - np.log(det(cov_0)) - dim +
                  np.trace(cov_1_inv @ cov_0) +
                  (mean_1 - mean_0).T @ cov_1_inv @ (mean_1 - mean_0))


# Originally from extrapops (SOBBH population synthesis)
def invCDFinterp(xs, pdf_func, pdf_args=None, splrep_kwargs=None):
    """
    Prepares an interpolator for 1-dimensional invCDF sampling.

    ``xs`` samples are assumed sorted.

    Integrates the pdf ``pdf_func`` using quad. Use ``pdf_args`` to pass arguments to it
    at integration.

    Uses ``scipy.interpolate.splrep`` to create an interpolator. Use ``splrep_kwargs`` to
    pass keyword arguments to it, e.g. ``{"k": 1}``.

    Returns a tuple of the the CDF samples and the interpolator.
    """
    quad_kwargs = {"args": pdf_args} if pdf_args else {}
    CDF = np.array(
        [integrate.quad(pdf_func, xs[0], x_i, **quad_kwargs)[0] for x_i in xs]
    )
    # Sometimes (very rarely) not sorted due to numerical noise. Simply delete bad entry
    for _ in range(len(CDF) - 1):
        i_unsorted_left = np.argwhere(np.diff(CDF) < 0)
        if i_unsorted_left.shape[0]:  # unsorted
            CDF = np.delete(CDF, i_unsorted_left, axis=0)
            xs = np.delete(xs, i_unsorted_left, axis=0)
        else:
            break
    else:
        raise ValueError(
            "Could not produce a sorted sample of the CDF. "
            "Maybe the pdf passed is a stochastic function?"
        )
    # Normalise to [0, 1]
    CDF = (CDF - min(CDF)) / (max(CDF) - min(CDF))
    # Remove all points until the last time CDF reaches 0 (no chance to generate them)
    # Better not use np.isclose in test, in order not to miss the tails
    last_zero = next(i for i, CDFi in enumerate(CDF) if CDFi != 0) - 1
    CDF = CDF[last_zero:]
    xs = xs[last_zero:]
    # Remove all points after CDF reaches 1 (no chance to generate them)
    # Now np.isclose should be enough
    first_one = next(i for i, CDFi in enumerate(CDF) if np.isclose(CDFi, 1))
    CDF = CDF[:first_one + 1]
    xs = xs[:first_one + 1]
    return (CDF, interpolate.splrep(CDF, xs, **(splrep_kwargs or {})))
