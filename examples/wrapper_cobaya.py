import os
import sys
import numpy as np

from cobaya.run import run as cobaya_run
from getdist.mcsamples import loadMCSamples

from synth_inference_tests.get_pdf import get_pdf
from synth_inference_tests.run import run as test_run


def cobaya_model_input(loglikelihood, bounds, paramnames=None):
    """
    Returns a Cobaya model input dict with the given likelihood and uniform prior bounds.

    Parameters
    ----------

    loglikelihood: callable

    bounds : list of boundaries as (lower,upper)

    paramnames : list of parameter names, optional

    Returns
    -------
    info : dict
        A Cobaya input dict containing the ``prior`` and ``likelihood`` blocks.
    """
    bounds = np.atleast_2d(bounds)
    dim = len(bounds)
    if paramnames is None:
        paramnames = [f"x_{i + 1}" for i in range(dim)]
    else:
        assert len(paramnames) == dim, (
            "If paramnames given, it must have as many parameters as the dimensionality "
            f"specified with 'bounds': {dim}.")
    info = {"params": {p: {"prior": list(b)} for p, b in zip(paramnames, bounds)}}

    # NB: `logp` methods in this package have unnamed args, which Cobaya does not support.
    # We need to create a wrapper with named args.
    def lkl(**kwargs):
        return loglikelihood(list(kwargs.values()))

    info.update({"likelihood": {"test": {
        "external": lkl, "requires": {},
        "input_params": paramnames
    }}})

    # info["timing"] = True
    # info["debug"] = True

    return info


def run_func(logpdf, bounds, output_folder=None,
                    budget=None, budget_count_inf=False, budget_count_parallel=False):
    input_dict = cobaya_model_input(logpdf, bounds, paramnames=None)
#    input_dict["sampler"] = {"mcmc": {"measure_speeds": False}}
    input_dict["sampler"] = {"polychord": {"measure_speeds": False}}

    results = {"sampler": f"cobaya:{list(input_dict['sampler'].keys())[0]}"}
    if not output_folder.endswith(r"\\"):
        output_folder = output_folder + "/"
    input_dict["output"] = output_folder
    input_dict["force"] = True
    try:
        upd_input, sampler = cobaya_run(input_dict)
    except Exception:
        results["end_state"] = "e"
        return results, None

    # TODO: for now budget not managed
    results["end_state"] = "c"

    return results, (upd_input, sampler)


def process_output_func(output_folder=None, return_values=None):
    if return_values is not None:
        upd_input, sampler = return_values
        products = sampler.products(
            to_getdist=True, combined=True, skip_samples=0.33)
        sample = products["sample"]
        logZ = products.get("logZ")
        logZstd = products.get("logZstd")
        if logZ is not None:  # yaml cannot read numpy floats
            logZ = float(logZ)
            logZstd = float(logZstd)
    elif output_folder is not None:
        sample_folder = os.path.abspath(output_folder)
        sample_folder += "/"  # to force GetDist to treat is as folder, not prefix
        sample = loadMCSamples(sample_folder)
        logZ = None
    # Create a "logpost" derived parameter with the **logposterior**
    sample.addDerived(-sample.loglikes, "logpost")
    results ={"samples": sample}
    if logZ is not None:
        results.update({"logZ": logZ, "logZstd": logZstd})
    return results
