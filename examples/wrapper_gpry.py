import os
import sys
import warnings

import yaml
import numpy as np

from synth_inference_tests.get_pdf import get_pdf
from synth_inference_tests.run import run as test_run
from synth_inference_tests.mpi import is_main_process

from gpry.run import Runner  # type: ignore
from getdist.mcsamples import loadMCSamples  # type: ignore


sample_subdir = "sample"


def run_func(
    logpdf,
    bounds,
    output_folder=None,
    budget=None,
    budget_count_inf=False,
    budget_count_parallel=False,
    sampler_kwargs=None,
):
    results = {"sampler": "gpry"}
    if sampler_kwargs is None:
        sampler_kwargs = {}
    # GPry cannot take 0/None/False budget
    if not bool(budget):
        budget = 10000000
    if sampler_kwargs.get("options") is None:
        sampler_kwargs["options"] = {}
    sampler_kwargs["options"]["max_total"] = budget
    try:
        runner = Runner(
            logpdf,
            bounds,
            checkpoint=output_folder,
            load_checkpoint="overwrite",
            plots=False,
            **sampler_kwargs,
        )
        runner.run()
    except Exception as excpt:
        warnings.warn(f"GPry finished with an error: {excpt}")
        results["end_state"] = "e"
        return results, None
    if runner.has_converged:
        results["end_state"] = "c"
    elif runner.n_total_left <= 0:  # can be negative due to MPI rounding
        results["end_state"] = "b"
    else:
        results["end_state"] = "?"  # will raise an exception later
    return results, (runner,)


def process_output_func(return_values, output_folder=None):
    if not is_main_process:
        return
    if return_values is not None:
        runner = return_values[0]
        runner.plot_progress(timing=True, trace=True, corner=True)
        sample = runner.last_mc_samples(as_getdist=True)
    elif output_folder is not None:
        runner = Runner(checkpoint=output_folder, load_checkpoint="resume")
        sample_folder = os.path.abspath(os.path.join(output_folder, sample_subdir))
        sample_folder += "/"  # to force GetDist to treat is as folder, not prefix
        sample = loadMCSamples(sample_folder)
        runner.plot_progress(timing=True, trace=True, corner=True)
    logZ, logZstd = None, None
    # Create a "logpost" derived parameter with the **logposterior**
    if "logpost" not in sample.getParamNames().list():
        sample.addDerived(-sample.loglikes, "logpost")
    results = {"sampler": "gpry", "samples": sample}
    if logZ is not None:
        results.update({"logZ": logZ, "logZstd": logZstd})
    results["logp_func"] = lambda x: runner.surrogate.logp(np.atleast_2d(x))
    return results


# Runnable as a script, just for tests
if __name__ == "__main__":
    # Build PDF
    if 2 < len(sys.argv[1:]) < 1:
        raise ValueError(
            "Pass likelihood name as first arg, e.g. 'gaussian5', and (optionally) "
            ".yaml file for sampler configuration as 2nd argument"
        )
    pdf_name = sys.argv[1]
    pdf = get_pdf(pdf_name)
    sampler_kwargs = sys.argv[2] if len(sys.argv) >= 3 else None
    if sampler_kwargs is not None:
        with open(sampler_kwargs, "r") as f:
            sampler_kwargs = yaml.safe_load(f)
    output_folder = os.path.join("output_gpry", pdf_name)
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
