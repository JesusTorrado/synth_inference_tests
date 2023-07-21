import os
import sys
from warnings import warn
import numpy as np

from gpry.run import Runner
from getdist.mcsamples import loadMCSamples

from synth_inference_tests.get_pdf import get_pdf
from synth_inference_tests.run import run as test_run
from synth_inference_tests.mpi import is_main_process

sample_subdir = "sample"


# TODO: remove after combining with gpry.tools.create_cobaya_model
def cobaya_model_input(loglikelihood, bounds, paramnames=None):
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
    return info


def run_func(logpdf, bounds, output_folder=None,
             budget=None, budget_count_inf=False, budget_count_parallel=False):
    results = {"sampler": "gpry"}
    # TODO: [logpdf, bounds] cannot be passed to GPRY at the moment if logpdf has unnamed
    #       args. Use the same model generator as in the cobaya test.
    model_input = cobaya_model_input(logpdf, bounds)
    from cobaya.model import get_model

    kwargs = {}
    acquisition = None

    from gpry.preprocessing import Normalize_bounds
    from gpry.gp_acquisition import NORA, GPAcquisition
    acquisition = NORA(
        bounds, acq_func="LogExp",
        preprocessing_X=Normalize_bounds(bounds),
        mc_every=2 * len(bounds),
        zeta_scaling=0.85, verbose=3)
    kwargs = {"gp_acquisition": acquisition}

    # GPry cannot take 0/None/False budget
    if not bool(budget):
        budget = 10000000
    kwargs["options"] = {"max_total": budget}

    kwargs["options"].update({"max_initial": 50000,
                              })
    try:
        runner = Runner(get_model(model_input), checkpoint=output_folder,
                        load_checkpoint="overwrite", plots=False, **kwargs)
        runner.run()
    except Exception as excpt:
        warn(f"GPry finished with an error: {excpt}")
        results["end_state"] = "e"
        return results, None
    # Generating MC sample considered part of the process:
    sample_folder = os.path.abspath(os.path.join(output_folder, sample_subdir))
    sample_folder += "/"  # to force Cobaya to use as folder, not prefix
    upd_input, sampler = runner.generate_mc_sample(
        sampler="polychord", output=sample_folder)
    if runner.has_converged:
        results["end_state"] = "c"
    elif runner.n_total_left == 0:
        results["end_state"] = "b"
    else:
        results["end_state"] = "?"  # will fail later
    return results, (runner, upd_input, sampler)


def process_output_func(output_folder=None, return_values=None):
    if not is_main_process:
        return None
    if return_values is not None:
        runner, upd_input, sampler = return_values
        sampler_products = sampler.products(
            to_getdist=True, combined=True, skip_samples=0.33)
        sample = sampler_products["sample"]
        float_if_not_None = lambda x: float(x) if x is not None else None
        logZ = float_if_not_None(sampler_products.get("logZ"))
        logZstd = float_if_not_None(sampler_products.get("logZstd"))
        runner.plot_mc(upd_input, sampler, add_training=True)
        runner.plot_distance_distribution(upd_input, sampler, show_added=True)
    elif output_folder is not None:
        runner = Runner(checkpoint=output_folder, load_checkpoint="resume")
        sample_folder = os.path.abspath(os.path.join(output_folder, sample_subdir))
        sample_folder += "/"  # to force GetDist to treat is as folder, not prefix
        sample = loadMCSamples(sample_folder)
        runner.plot_mc(sample_folder, add_training=True)
        logZ = None
    # Do some GPry plots too
    runner.plot_progress()
    # Create a "logpost" derived parameter with the **logposterior**
    sample.addDerived(-sample.loglikes, "logpost")
    results = {"sampler": "gpry", "samples": sample}
    if logZ is not None:
        results.update({"logZ": logZ, "logZstd": logZstd})
    results["logp_func"] = lambda x: runner.gpr.predict(np.atleast_2d(x))
    return results
