import os
import sys
from warnings import warn
import numpy as np

from gpry.run import Runner

from synth_inference_tests.get_pdf import get_pdf
from synth_inference_tests.run import run as test_run

finite_minus_inf = -1e8  # cannot be too small, or pyVBMC crashes


def run_func(logpdf, bounds, output_folder=None,
                    budget=None, budget_count_inf=False, budget_count_parallel=False):
    from pyvbmc import VBMC
    LB, UB = bounds.T
    x0 = LB + (UB - LB) / 2
    options = {}
    if not bool(budget):
        budget = 10000000
    options["max_fun_evals"] = budget
    # pyVBMC calls logpdf funcs with a single arg, can cannot manage -inf
    logpdf_single_arg = lambda X: max(logpdf(*X), finite_minus_inf)
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
    return end_state, vp, results, Xs, logpdf


def process_output_func(pyvbmc_return_values, output_folder=None):
    _, vp, results, Xs, logpdf = pyvbmc_return_values
    from getdist.mcsamples import MCSamples
    gdsample = MCSamples(samples=Xs, names=[f"x_{i+1}" for i in range(vp.D)])
    chi2s = -2 * np.array([logpdf(*x) for x in Xs])
    gdsample.addDerived(chi2s, "chi2")
    # Do some pyVBMC plots too
    vp.plot(plot_data=True, plot_vp_centres=True)
    import matplotlib.pyplot as plt
    plots_folder = os.path.join(output_folder, "plots")
    try:
        os.makedirs(plots_folder)
    except FileExistsError:
        pass
    plt.savefig(os.path.join(plots_folder, "vp.png"))
    return {"samples": gdsample}
