import numpy as np
from scipy import integrate
from scipy import interpolate


def invCDFinterp(xs, pdf_func, pdf_args=None, splrep_kwargs=None):
    """
    Prepares an interpolator for invCDF sampling.

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
