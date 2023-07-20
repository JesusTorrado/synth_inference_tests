import os
import sys
import pickle
from warnings import warn
import numpy as np

from pyvbmc import VBMC

from synth_inference_tests.get_pdf import get_pdf
from synth_inference_tests.run import run as test_run

finite_minus_inf = -1e8  # cannot be too small, or pyVBMC crashes
pyvbmc_obj_filename = "vbmc.pkl"
pyvbmc_results_filename = "result.pkl"
sample_filename = "sample.txt"
bounds_filename = "sample.txt"


def save_sample(Xs, logposts, bounds, output_folder):
    np.savetxt(os.path.join(output_folder, sample_filename),
               np.concatenate((Xs.T, logposts.T)).T)
    np.savetxt(os.path.join(output_folder, bounds_filename), bounds)


def load_sample(output_folder):
    sample_w_logpost = np.loadtxt(os.path.join(output_folder, sample_filename))
    bounds = np.loadtxt(os.path.join(output_folder, bounds_filename))
    return sample_w_logpost[:, :-1], sample_w_logpost[:, -1], bounds


def run_func(logpdf, bounds, output_folder=None,
             budget=None, budget_count_inf=False, budget_count_parallel=False):
    LB, UB = bounds.T
    x0 = LB + (UB - LB) / 2
    options = {}
    if not bool(budget):
        budget = 10000000
    options["max_fun_evals"] = budget
    # pyVBMC calls logpdf funcs with a single arg, can cannot manage -inf
    logpdf_single_arg = lambda X: max(logpdf(X), finite_minus_inf)
    try:
        vbmc = VBMC(logpdf_single_arg, x0, LB, UB, None, None, options=options)
        vp, results = vbmc.optimize()
    except Exception as excpt:
        warn(f"pyVBMC finished with an error: {excpt}")
        return "e", None, None, None, None, None
    print("VBMC done!")
    if results["func_count"] >= budget:
        end_state = "b"
        # TODO: fix this!
        if results["func_count"] > budget:
            warn(f"More lopposterior evaluations ({results['func_count']}) "
                 f"than budgeted ({budget}).")
    elif results["convergence_status"].lower() == "probable":
        end_state = "c"
    else:
        end_state = "?"
    Xs, _ = vp.sample(10000)
    print("VBMC sampled!")
    # Save everything that would be returned
    logposts = vp.log_pdf(Xs)
    try:
        os.makedirs(output_folder)
    except FileExistsError:
        pass
    # Save everything that would be returned
    # Hack: if "iteration" is not set, it complains at load time
    vp.iteration = 99
    vp.save(os.path.join(output_folder, pyvbmc_obj_filename), overwrite=True)
    with open(os.path.join(output_folder, pyvbmc_results_filename), "wb") as f:
        pickle.dump(results, f)
    save_sample(Xs, logposts, bounds, output_folder)
    # Returning (Xs, logposts, bounds) instead of a GetDist sample in order not to add
    # unnecessary overhead.
    return end_state, vp, results, Xs, logposts, bounds


def process_output_func(output_folder=None, return_values=None):
    if return_values is not None:
        _, vp, results, Xs, logposts, bounds = return_values
    elif output_folder is not None:
        vp = VBMC.load(os.path.join(output_folder, pyvbmc_obj_filename))
        with open(os.path.join(output_folder, pyvbmc_results_filename), "rb") as f:
            results = pickle.load(f)
        Xs, logposts, bounds = load_sample(output_folder)
    from getdist.mcsamples import MCSamples
    kwargs = {"names": [f"x_{i+1}" for i in range(vp.D)]}
    kwargs["ranges"] = {p: bounds[i] for i, p in enumerate(kwargs["names"])}
    gdsample = MCSamples(samples=Xs, **kwargs)
    gdsample.addDerived(logposts, "logpost")
    # Do some pyVBMC plots too
    plots_folder = os.path.join(output_folder, "plots")
    try:
        os.makedirs(plots_folder)
    except FileExistsError:
        pass
    vp.plot(plot_data=True, plot_vp_centres=True)
    import matplotlib.pyplot as plt
    plt.savefig(os.path.join(plots_folder, "vp.png"))
    products = {"sampler": "pyvbmc", "samples": gdsample}
    # NB: logZ here are ELBO's!
    products["logZ"] = results["elbo"]
    products["logZstd"] = results["elbo_sd"]
    products["logp_func"] = lambda X: vp.log_pdf(X).T[0]
    products["notes"] = "logZ is an ELBO"
    return products
