import os
import sys
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd  # type: ignore
import yaml  # type: ignore

# Importing PolyChord. May raise importerror if not installed
from pypolychord import run_polychord  # type: ignore
from pypolychord.priors import UniformPrior  # type: ignore
from pypolychord.settings import PolyChordSettings  # type: ignore

from synth_inference_tests.get_pdf import get_pdf
from synth_inference_tests.mpi import is_main_process
from synth_inference_tests.run import run as test_run
from synth_inference_tests.utils import ColNames, generic_param_names

defaults = {
    "nlive": "25d",
    "num_repeats": "5d",
    "precision_criterion": 0.01,
    "read_resume": False,
    "write_resume": False,
    "write_live": False,
    "write_dead": True,
    "write_prior": False,
}


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
    results = {"sampler": "polychord"}
    dim = len(bounds)
    polychord_prior = UniformPrior(*(bounds.T))
    polychord_settings = PolyChordSettings(nDims=dim, nDerived=0)
    settings = deepcopy(defaults)
    settings.update(sampler_kwargs or {})
    for p, v in settings.items():
        if isinstance(v, str):
            caster = int if p.lower().startswith("n") else float
            if v.endswith("d"):
                v = v.removesuffix("d")
                v = caster(v or 1) * dim  # 'or 1' handles v="d"
            else:
                v = caster(v)
        setattr(polychord_settings, p, v)
    polychord_settings.base_dir = output_folder
    # Run PolyChord!
    # Flush stdout, since PolyChord can step over it if async (py not called with -u)
    sys.stdout.flush()
    try:
        polychord_results = run_polychord(
            lambda x: ((lambda _: _[0] if hasattr(_, "__len__") else _)(logpdf(x)), []),
            prior=polychord_prior,
            settings=polychord_settings,
            nDims=dim,
            nDerived=0,
        )
        # Create paramnames file to be able to load with getdist
        if is_main_process:
            paramnames = [f"x_{i + 1}" for i in range(dim)]
            polychord_results.make_paramnames_files([(p, p) for p in paramnames])
    except Exception as excpt:
        warnings.warn(f"PolyChord finished with an error: {excpt}")
        results["end_state"] = "e"
        return results, None
    results["end_state"] = "c"
    return results, (polychord_results,)


def load_logZ(stats_file):
    """Loads logZ and logZ std from hard drive."""
    with open(stats_file, "r") as f:
        for line in f:
            if line.startswith("log(Z)"):
                return [float(n) for n in line.split("=")[1].split("+/-")]


def process_output_func(return_values, output_folder=None):
    if not is_main_process:
        return
    # PolyChord always writes to hard drive!
    if return_values[0] is not None:
        chain_file_name = return_values[0].root + ".txt"
        logZ, logZstd = return_values[0].logZ, return_values[0].logZerr
    elif output_folder is not None:
        chain_file_name = os.path.join(output_folder, "test.txt")
        logZ, logZstd = load_logZ(os.path.join(output_folder, "products", "test.stats"))
    else:
        raise ValueError("Neither resturn_values nor output_folder provided.")
    samples = pd.read_csv(chain_file_name, sep=r"\s+", header=None, dtype=np.float64)
    samples.drop(columns=1, inplace=True)
    colnames = [ColNames.weight]
    colnames += generic_param_names(len(samples.columns) - 1, based_0=False)
    samples.rename(columns=dict(zip(samples.columns, colnames)), inplace=True)
    return {"sampler": "polychord", "samples": samples, "logZ": logZ, "logZstd": logZstd}


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
    output_folder = os.path.join("output_polychord", pdf_name)
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
