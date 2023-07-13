import os
import sys
import numpy as np

from gpry.run import Runner

from synth_inference_tests.get_pdf import get_pdf
from synth_inference_tests.run import run as test_run


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
        return loglikelihood(*list(kwargs.values()))

    info.update({"likelihood": {"test": {
        "external": lkl, "requires": {},
        "input_params": paramnames
    }}})
    return info


def gpry_run_func(logpdf, bounds, output_folder=None):
    # TODO: [logpdf, bounds] cannot be passed to GPRY at the moment if logpdf has unnamed
    #       args. Use the same model generator as in the cobaya test.
    model_input = cobaya_model_input(logpdf, bounds)
    from cobaya.model import get_model
    runner = Runner(get_model(model_input), checkpoint=output_folder,
                    load_checkpoint="overwrite", plots=False)
    runner.run()
    # Generating MC sample considered part of the process:
    upd_input, sampler = runner.generate_mc_sample(sampler="polychord")
    return runner, upd_input, sampler

def process_gpry_output_func(gpry_return_values, output_folder=None):
    runner, upd_input, sampler = gpry_return_values
    # Do some GPry plots too
    runner.plot_progress()
    runner.plot_mc(upd_input, sampler, add_training=True)
    return {"samples": sampler.products(
        to_getdist=True, combined=True, skip_samples=0.33)["sample"]}


if __name__ == "__main__":
    # Build PDF
    if len(sys.argv[1:]) != 1:
        raise ValueError("Pass likelihood name as first arg, e.g. 'gaussian5'")
    pdf_name = sys.argv[1]
    pdf = get_pdf(pdf_name)
    output_folder = os.path.join("output_gpry", pdf_name)
    test_run(pdf, gpry_run_func, process_gpry_output_func,
             output_folder=output_folder)
