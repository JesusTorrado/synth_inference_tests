import os
import sys
import warnings

import numpy as np
import pandas as pd
import yaml  # type: ignore

# GPry import(s)
from gpry.run import Runner  # type: ignore

from synth_inference_tests.get_pdf import get_pdf
from synth_inference_tests.mpi import is_main_process
from synth_inference_tests.run import run as test_run
from synth_inference_tests.utils import ColNames, generic_param_names


def run_func(
    logpdf,
    bounds,
    ref_bounds=None,
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
    # By default, suppress plots unless explicitly asked
    if "plots" not in sampler_kwargs:
        sampler_kwargs["plots"] = False
    try:
        runner = Runner(
            logpdf,
            bounds,
            ref_bounds=ref_bounds,
            checkpoint=output_folder,
            load_checkpoint="overwrite",
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
        samples = runner.last_mc_samples(as_pandas=True)
    elif output_folder is not None:
        runner = Runner(checkpoint=output_folder, load_checkpoint="resume")
        samples = pd.read_csv(
            os.path.join(output_folder, "mc_samples.txt"),
            sep=r"\s+",
            header=0,
            dtype=np.float64,
        )
        # This will have read the columns wrong, interpreting the leading `#` as a column
        # (rename last to drop it)
        dummycol = "-"
        samples.rename(
            columns=dict(
                zip(list(samples.columns), list(samples.columns)[1:] + [dummycol])
            ),
            inplace=True,
        )
        samples.drop(columns=dummycol, inplace=True)
    # Generic samples post-processing (either loaded or runtime)
    cols_drop = [c for c in samples.columns if "log" in c]
    samples.drop(columns=cols_drop, inplace=True)
    colnames = [ColNames.weight]
    colnames += generic_param_names(len(samples.columns) - 1, based_0=False)
    samples.rename(columns=dict(zip(samples.columns, colnames)), inplace=True)
    logZ, logZstd = [float(x) if x is not None else None for x in runner.last_mc_logZ()]
    # Do some plots
    runner.plot_progress(timing=True, trace=True, corner_final=True)
    results = {"sampler": "gpry", "samples": samples}
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
            "a .yaml file for sampler configuration as 2nd argument"
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
