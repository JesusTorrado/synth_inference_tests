import os
import sys
import numpy as np

from cobaya.run import run as cobaya_run

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
        return loglikelihood(*list(kwargs.values()))

    info.update({"likelihood": {"test": {
        "external": lkl, "requires": {},
        "input_params": paramnames
    }}})

    # info["timing"] = True
    # info["debug"] = True
    
    return info


def cobaya_run_func(logpdf, bounds, output_folder=None):
    input_dict = cobaya_model_input(logpdf, bounds, paramnames=None)
#    input_dict["sampler"] = {"mcmc": {"measure_speeds": False}}
    input_dict["sampler"] = {"polychord": {"measure_speeds": False}}
    if not output_folder.endswith(r"\\"):
        output_folder = output_folder + "/"
    input_dict["output"] = output_folder
    input_dict["force"] = True
    return cobaya_run(input_dict)


def process_cobaya_output_func(cobaya_return_values, output_folder=None):
    upd_input, sampler = cobaya_return_values
    return {"samples": sampler.samples(to_getdist=True, combined=True, skip_samples=0.33)}



if __name__ == "__main__":
    # Build PDF
    if len(sys.argv[1:]) != 1:
        raise ValueError("Pass likelihood name as first arg, e.g. 'gaussian5'")
    pdf_name = sys.argv[1]
    pdf = get_pdf(pdf_name)
    output_folder = os.path.join("output", pdf_name)
    test_run(pdf, cobaya_run_func, process_cobaya_output_func,
             output_folder=output_folder)
