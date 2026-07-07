from warnings import warn

import numpy as np
from numpy.linalg import det
from scipy import integrate, interpolate  # type: ignore
from scipy.special import erfc  # type: ignore
from scipy.stats import chi2  # type: ignore
from sklearn.neighbors import KernelDensity  # type: ignore


class ColNames:
    weight = "weight"
    logpost = "logpost"
    loglike = "loglike"
    logprior = "logprior"


def generic_param_names(dim, prefix="x_", based_0=False):
    """
    Returns a list of indexed parameter names, by default as ``[x_1, ..., x_<dim>]``.
    """
    return [prefix + f"{i + (0 if based_0 else 1)}" for i in range(dim)]


def pd_to_gd_samples(samples, bounds):
    """
    Coverts a pandas samples table with columns ``weights, [params], logpost, [etc]`` to
    getdist.
    """
    from getdist.mcsamples import MCSamples  # type: ignore

    # NB: Cannot use addDerived after creation because getdist removes low weight samples
    paramnames = generic_param_names(len(samples.columns) - 4, based_0=False)
    paramnames += [getattr(ColNames, p) + "*" for p in ["logpost", "loglike", "logprior"]]
    gdsample = MCSamples(
        weights=samples[ColNames.weight].to_numpy(),
        samples=samples[[p.rstrip("*") for p in paramnames]].to_numpy(),
        names=paramnames,
        ranges=dict(zip(paramnames, bounds)),
        ignore_rows=0,
    )
    return gdsample


# From GPry
def nstd_of_1d_nstd(n1, d, warn_inf=True):
    """
    Radius of (hyper)volume in units of std's of a multivariate Gaussian of dimension
    ``d`` for a credible (hyper)volume defined by the equivalent 1-dimensional
    ``n1``-sigma interval.
    """
    nstd = np.sqrt(chi2.isf(erfc(n1 / np.sqrt(2)), d))
    if warn_inf and not np.isfinite(nstd):
        warn(f"Got -inf for n1={n1} and d={d}. This may cause errors.")
    return nstd


def kde_logp_if_needed(sample, logps_sample, weights=None, logp_func=None):
    if logp_func is None:
        # Need to "whiten" scales, in order to get bandwidth approximately right
        # (and even this way it's not very good, but order-of-mag should be OK for KL)
        avgs = np.average(sample, weights=weights, axis=0)
        scales = np.sqrt(np.diag(np.cov(sample.T, aweights=weights)))
        transf = lambda x: (x - avgs) / scales
        kde = KernelDensity(bandwidth="silverman")
        kde.fit(transf(sample), sample_weight=weights)
        # Now we need to normalize to the posterior -- use avg difference of top 2 sigma
        # to adjust it to the values provided via `logps_sample`.
        i_top = np.argwhere(
            logps_sample > max(logps_sample) - nstd_of_1d_nstd(2, len(scales))
        ).T[0]
        diffs = logps_sample[i_top] - kde.score_samples(transf(sample[i_top]))
        diffs_avg = np.average(
            diffs, weights=weights[i_top] if weights is not None else None
        )
        return lambda x: diffs_avg + kde.score_samples(transf(x))
    return logp_func


def kl_norm(mean_0, cov_0, mean_1, cov_1):
    """
    Computes the KL divergence between two normal distributions defined by their means
    and covariance matrices.

    May raise ``numpy.linalg.LinAlgError``.
    """
    cov_1_inv = np.linalg.inv(cov_1)
    dim = len(mean_0)
    return 0.5 * (
        np.log(det(cov_1))
        - np.log(det(cov_0))
        - dim
        + np.trace(cov_1_inv @ cov_0)
        + (mean_1 - mean_0).T @ cov_1_inv @ (mean_1 - mean_0)
    )


def kl_norm_sym(mean_0, cov_0, mean_1, cov_1):
    return kl_norm(mean_0, cov_0, mean_1, cov_1) + kl_norm(mean_1, cov_1, mean_0, cov_0)


def kl(sample_P, logp_P_sample_P, logp_Q_sample_P, weights_P=None, nstd=None):
    """
    Computes a Monte Carlo estimate of the KL divergence ``KL(P|Q)`` given a sample from
    ``P``, and its log-posterior values under ``P`` and ``Q``.
    """
    n = len(sample_P)
    if len(logp_P_sample_P) != n or len(logp_Q_sample_P) != n:
        raise TypeError(
            "The lenght of the samples and logp vectors must be equal. "
            f"Got respectively {n}, {len(logp_P_sample_P)}, {len(logp_Q_sample_P)}."
        )
    if weights_P is not None and len(weights_P) != n:
        raise TypeError(
            "The lenght of the weight vectors must be the same as that of the sample. "
            f"Got respectively {len(weights_P)}, {n}."
        )
    # Numerical stability: restrict to highest sigma, computed with Gaussian dynamic range
    # This avoids divergences due to very small logp_Q values *at the tails of P*, which
    # should not matter for inference.
    # If 'auto', increase nsigma until it stabilises, starting relatively high (5)
    if nstd == "auto":
        kl_old = 1e-30
        for i, nstd_i in enumerate(range(5, 100)):
            kl_new = kl(
                sample_P,
                logp_P_sample_P,
                logp_Q_sample_P,
                weights_P=weights_P,
                nstd=nstd_i,
            )
            # Stop at equality or divergence (except 1st iteration)
            if np.isclose(kl_new, kl_old) or (
                i != 0 and abs(np.log(abs(kl_new) / abs(kl_old))) > 1
            ):
                return kl_new
        return kl_new  # will amost for sure not get here.
    if nstd is not None:  # 'auto' already discarded
        d = sample_P.shape[1]
        max_diff = nstd_of_1d_nstd(nstd, d)
        i_high_P = np.argwhere(max(logp_P_sample_P) - logp_P_sample_P < max_diff)
        sample_P = sample_P[i_high_P]
        weights_P = weights_P[i_high_P] if weights_P is not None else None
        logp_P_sample_P = logp_P_sample_P[i_high_P]
        logp_Q_sample_P = logp_Q_sample_P[i_high_P]
    if weights_P is None:
        return np.sum(logp_P_sample_P - logp_Q_sample_P) / n
    # Numerical stability: make the highest weight 1
    weights_P = weights_P / max(weights_P)
    return np.sum(weights_P * (logp_P_sample_P - logp_Q_sample_P)) / np.sum(weights_P)


def kl_sym(
    sample_1,
    sample_2,
    weights_1=None,
    weights_2=None,
    logp_func_1=None,
    logp_func_2=None,
    logp_1_sample_1=None,
    logp_2_sample_2=None,
    nstd="auto",
):
    """
    Computes a Monte Carlo estimate of the symmetric (Jeffreys) KL divergence between two
    samples: ``KL(P|Q) + KL(Q|P)``.

    Only the samples are required, with optional weights.

    For each component, if the ``logp`` function is not provided, it will estimate it from
    a KDE reconstructed from the corresponding sample.

    If the log-posterior values of the samples under their own distributions are known
    (``logp_X_sampleX``), they can be passed to save some computational costs.
    """
    if logp_1_sample_1 is None:
        logp_1_sample_1 = logp_func_1(sample_1)
    if logp_2_sample_2 is None:
        logp_2_sample_2 = logp_func_2(sample_2)
    logp_func_1 = kde_logp_if_needed(
        sample_1, logp_1_sample_1, weights=weights_1, logp_func=logp_func_1
    )
    logp_func_2 = kde_logp_if_needed(
        sample_2, logp_2_sample_2, weights=weights_2, logp_func=logp_func_2
    )
    one_to_two = kl(
        sample_1, logp_1_sample_1, logp_func_2(sample_1), weights_P=weights_1, nstd=nstd
    )
    two_to_one = kl(
        sample_2, logp_2_sample_2, logp_func_1(sample_2), weights_P=weights_2, nstd=nstd
    )
    return one_to_two + two_to_one


def js(
    sample_1,
    sample_2,
    weights_1=None,
    weights_2=None,
    logp_func_1=None,
    logp_func_2=None,
    logp_1_sample_1=None,
    logp_2_sample_2=None,
    nstd="auto",
):
    """
    Computes a Monte Carlo estimate of the Jensen-Shannon divergence (distance squared)
    between two samples.

    Only the samples are required, with optional weights.

    For each component, if the ``logp`` function is not provided, it will estimate it from
    a KDE reconstructed from the corresponding sample.

    If the log-posterior values of the samples under their own distributions are known
    (``logp_X_sampleX``), they can be passed to save some computational costs.
    """
    if logp_1_sample_1 is None:
        logp_1_sample_1 = logp_func_1(sample_1)
    if logp_2_sample_2 is None:
        logp_2_sample_2 = logp_func_2(sample_2)
    logp_func_1 = kde_logp_if_needed(
        sample_1, logp_1_sample_1, weights=weights_1, logp_func=logp_func_1
    )
    logp_func_2 = kde_logp_if_needed(
        sample_2, logp_2_sample_2, weights=weights_2, logp_func=logp_func_2
    )
    # Prepare joint distribution -- give both samples the sampe weight
    logp_func_mix = lambda x: 0.5 * (logp_func_1(x) + logp_func_2(x))
    one_to_mix = kl(
        sample_1,
        logp_1_sample_1,
        logp_func_mix(sample_1),
        weights_P=weights_1,
        nstd=nstd,
    )
    two_to_mix = kl(
        sample_2,
        logp_2_sample_2,
        logp_func_mix(sample_2),
        weights_P=weights_2,
        nstd=nstd,
    )
    return 0.5 * (one_to_mix + two_to_mix)


# Originally from extrapops (SOBBH population synthesis)
def invCDFinterp(xs, pdf_func, pdf_args=None, splrep_kwargs=None):
    """
    Prepares an interpolator for 1-dimensional invCDF sampling.

    ``xs`` samples are assumed sorted.

    Integrates the pdf ``pdf_func`` using quad. Use ``pdf_args`` to pass arguments to it
    at integration.

    Uses ``scipy.interpolate.splrep`` to create an interpolator. Use ``splrep_kwargs`` to
    pass keyword arguments to it, e.g. ``{"k": 1}``.

    Returns a tuple of the ``(CDF, x)`` samples and the interpolator.
    """
    quad_kwargs = {"args": pdf_args} if pdf_args else {}
    CDF = np.array([integrate.quad(pdf_func, xs[0], x_i, **quad_kwargs)[0] for x_i in xs])
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
    CDF = CDF[: first_one + 1]
    xs = xs[: first_one + 1]
    return (CDF, xs, interpolate.splrep(CDF, xs, **(splrep_kwargs or {})))
