import os
import sys

import yaml  # type: ignore
import numpy as np

from cobaya.run import run as cobaya_run  # type: ignore
from getdist.mcsamples import loadMCSamples  # type: ignore

from synth_inference_tests.get_pdf import get_pdf
from synth_inference_tests.run import run as test_run
from synth_inference_tests.mpi import is_main_process


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
            f"specified with 'bounds': {dim}."
        )
    info = {
        "params": {
            p: {
                "prior": list(b),
            }
            for p, b in zip(paramnames, bounds)
        }
    }

    # NB: `logp` methods in this package have unnamed args, which Cobaya does not support.
    # We need to create a wrapper with named args.
    def lkl(**kwargs):
        return loglikelihood(list(kwargs.values()))

    info.update(
        {
            "likelihood": {
                "test": {"external": lkl, "requires": {}, "input_params": paramnames}
            }
        }
    )
    return info


def run_func(
    logpdf,
    bounds,
    output_folder=None,
    budget=None,
    budget_count_inf=False,
    budget_count_parallel=False,
    sampler_kwargs=None,
):
    input_dict = cobaya_model_input(logpdf, bounds, paramnames=None)
    input_dict["sampler"] = sampler_kwargs or {"mcmc": None}
    sampler = list(input_dict["sampler"].keys())[0]
    input_dict["sampler"][sampler] = input_dict["sampler"][sampler] or {}
    # For now (no pdfs with param hierarchy), disable speed measurement
    input_dict["sampler"][sampler]["measure_speeds"] = False
    results = {"sampler": f"cobaya:{sampler}"}
    if not output_folder.endswith(r"\\"):
        output_folder = output_folder + "/"
    input_dict["output"] = output_folder
    input_dict["force"] = True
    try:
        upd_input, sampler = cobaya_run(input_dict)
    except Exception:
        results["end_state"] = "e"
        return results, None
    results["end_state"] = "c"
    return results, (upd_input, sampler)


def process_output_func(return_values, output_folder=None):
    if not is_main_process:
        return None
    if return_values is not None:
        upd_input, sampler = return_values
        products = sampler.products(to_getdist=True, combined=True, skip_samples=0.33)
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
    results = {"samples": sample}
    if logZ is not None:
        results.update({"logZ": logZ, "logZstd": logZstd})
    return results


# Runnable as a script, just for tests
if __name__ == "__main__":
    # Build PDF
    if 2 < len(sys.argv[1:]) < 1:
        raise ValueError(
            "Pass likelihood name as first arg, e.g. 'gaussian5', and (optionally) "
            "a .yaml file for sampler configuration as 2nd argument"
        )
    pdf_name = sys.argv[1]
    pdf = get_pdf(pdf_name)
    sampler_kwargs = sys.argv[2] if len(sys.argv) >= 3 else None
    if sampler_kwargs is not None:
        with open(sampler_kwargs, "r") as f:
            sampler_kwargs = yaml.safe_load(f)
    output_folder = os.path.join("output_cobaya", pdf_name)
    test_run(
        pdf,
        run_func,
        process_output_func,
        output_folder=output_folder,
        budget=None,
        budget_count_inf=False,
        budget_count_parallel=False,
        sampler_kwargs=sampler_kwargs,
    )
