import os
import sys
from pprint import pprint

import numpy as np
import yaml  # type: ignore
from cobaya.output import load_samples  # type: ignore
from cobaya.run import run as cobaya_run  # type: ignore

from synth_inference_tests.get_pdf import get_pdf
from synth_inference_tests.mpi import is_main_process
from synth_inference_tests.run import run as test_run
from synth_inference_tests.utils import ColNames, generic_param_names

skip = 0.33  # fraction of initial samples to be removed


def cobaya_model_input(loglikelihood, bounds, paramnames=None, ref_bounds=None):
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
        paramnames = generic_param_names(dim, based_0=False)
    else:
        assert len(paramnames) == dim, (
            "If paramnames given, it must have as many parameters as the dimensionality "
            f"specified with 'bounds': {dim}."
        )
    info = {
        "params": {p: {"prior": list(b), "ref": None} for p, b in zip(paramnames, bounds)}
    }
    if ref_bounds is not None:
        for v, rb in zip(info["params"].values(), np.atleast_2d(ref_bounds)):
            v["ref"] = {"dist": "uniform", "min": rb[0], "max": rb[1]}

    # NB: `logp` methods in this package have unnamed args, which Cobaya does not support.
    # We need to create a wrapper with named args.
    def lkl(**kwargs):
        return loglikelihood(list(kwargs.values()))

    info["likelihood"] = {
        "test": {"external": lkl, "requires": {}, "input_params": paramnames}
    }
    return info


def run_func(
    logpdf,
    bounds,
    ref_bounds=None,
    output_folder=None,
    budget=None,
    budget_count_inf=False,
    budget_count_parallel=False,
    sampler_kwargs=None,
    fiducial_samples=None,
):
    input_dict = cobaya_model_input(logpdf, bounds, paramnames=None, ref_bounds=ref_bounds)
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
    if return_values is not None:
        upd_input, sampler = return_values
        # Next line NEEDS to run MPI-parallelised, in order to combine MCMC chains.
        samples = sampler.samples(combined=True, skip_samples=skip).data
        products = sampler.products()
        logZ = products.get("logZ")
        logZstd = products.get("logZstd")
        if logZ is not None:  # yaml cannot read numpy floats
            logZ = float(logZ)
            logZstd = float(logZstd)
    elif output_folder is not None:
        if is_main_process:
            samples = load_samples(
                os.path.abspath(output_folder) + "/", combined=True, skip=skip
            ).data
        else:
            samples = None
        # TODO: read from hard drive for e.g. PolyChord!
        logZ, logZstd = None, None
    else:
        raise ValueError("Neither resturn_values nor output_folder provided.")
    if not is_main_process:
        return None
    # Process samples to get the expected columns
    samples.drop(columns="minuslogpost", inplace=True)
    cols_drop = [c for c in samples.columns if "chi2" in c or "prior" in c]
    samples.drop(columns=cols_drop, inplace=True)
    colnames = [ColNames.weight]
    colnames += generic_param_names(len(samples.columns) - 1, based_0=False)
    samples.rename(columns=dict(zip(samples.columns, colnames)), inplace=True)
    results = {"samples": samples}
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
    results = test_run(
        pdf,
        run_func,
        process_output_func,
        output_folder=output_folder,
        budget=None,
        budget_count_inf=False,
        budget_count_parallel=False,
        sampler_kwargs=sampler_kwargs,
    )
    if is_main_process:
        print("\n----RESULTS----\n")
        pprint(results)
        print()
