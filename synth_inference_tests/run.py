import os
from time import time
from typing import Mapping, Sequence

import numpy as np
import yaml

from .pdf import PDF
from .mpi import is_main_process, multiple_processes, mpi_comm, mpi_size, get_num_threads
from .utils import kl_sym, kl_norm_sym

result_file = "result.yaml"


def run(pdf, run_func, process_output_func, output_folder,
        budget=None, budget_count_inf=False, budget_count_parallel=False):
    """
    run_func: callable
        Must take as arguments [logpdf function, bounds], and return ......

    process_output_func: callable
        Takes as args the return values of ``run_func``. Returns a dict with a GetDist
        sample under `samples`, and the evidence as `logZ` and `logZstd` (None if not
        computed).
    """
    if not isinstance(pdf, PDF):
        raise NotImplementedError("Only implemented for PDF instances.")
    products_folder = os.path.join(output_folder, "products")
    plots_folder = os.path.join(output_folder, "plots")
    # Basic inputs
    if is_main_process:
        results = {
            "pdf": pdf.__class__.__name__,
            "dim": pdf.dim,
            "n_processes": mpi_size,
            "n_threads_per_process": get_num_threads(),
        }
    # Run external sampler
    start_total = time()
    results_upd, return_values = run_func(
        pdf.logpdf, pdf.bounds, output_folder=products_folder,
        budget=None, budget_count_inf=False, budget_count_parallel=False)
    delta_total = time() - start_total
    # Timings and # evals
    time_pdf = mpi_comm.gather(pdf.t)
    n_evals_pdf = mpi_comm.gather(pdf.n)
    time_overhead = mpi_comm.gather(delta_total - pdf.t)
    if is_main_process:
        results.update(results_upd)
        results.update({
            "time_truth": max(time_pdf),
            "n_truth": max(n_evals_pdf),
            "time_overhead": max(time_overhead),
        })
    end_state = results.get("end_state")
    if not isinstance(end_state, str) or \
       not end_state.lower() in ["c", "b", "e", "?"]:
        raise ValueError("'end_state' must be one of 'c' (converged), 'b' "
                         f"(budget exhausted), 'e' (errored). Got {end_state}")
    # NB: 'c' means *spontaneous* stop. If budget exhausted and convergence judged likely
    #     by internal diagnostics, that's still 'b'.
    if end_state == "?":
        print("The sampler has finished with unknown end state.")
    elif end_state == 'e':
        dump_result(results, output_folder)
        return results

    # Compute/process necessary quantities for the report, do sampler-internal plots, etc.
    sampler_results = process_output_func(
        output_folder=products_folder, return_values=return_values)
    results.update(sampler_results)

    # Symmetric (Jeffrey's) KL against the true pdf. Only if we have a sampler from the
    # true posterior AND a surrogate model (otherwise it is tiny and meaningless).
    sample_orig = results.pop("samples")
    sample_ref = pdf.samples()
    logp_func = results.pop("logp_func", None)
    if sample_ref is not None and logp_func is not None:
        sampled_params = [
            p.name for p in sample_orig.getParamNames().names
            if not p.isDerived]
        sample = np.array([sample_orig[p] for p in sampled_params]).T
        weights = sample_orig.weights
        # This is a log-posterior sample, not a log-likelihood,
        sample_logp = sample_orig["logpost"]
        # so the reference pdf must be the logposterior too!
        sample_logp_ref = pdf.logpost(sample)
        sample_ref_logp_ref = pdf.logpost(sample_ref)
        sample_ref_logp = logp_func(sample_ref)
        results["kl"] = float(kl_sym(sample, sample_logp, sample_logp_ref,
                                    sample_ref, sample_ref_logp_ref, sample_ref_logp,
                                    weights_1=weights))
        results["kl_norm"] = float(kl_norm_sym(
            np.average(sample, weights=weights, axis=0),
            np.cov(sample.T, aweights=weights),
            np.average(sample_ref, axis=0), np.cov(sample_ref.T)))

    # Evidence
    results["logZ_truth"] = float(pdf.logZ) if pdf.logZ is not None else None

    # Save results object
    dump_result(results, output_folder)

    # Plots
    if is_main_process:
        plot_triangle(sample_orig, pdf=pdf, output_folder=plots_folder)
    return results

def dump_result(result, output_folder):
    with open(os.path.join(output_folder, result_file), "w") as f:
        yaml.dump(result, f)


def plot_triangle(sample, output_folder, filename="triangle.png", pdf=None,
                  filled=True):
    tab10_colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")
    from getdist import plots as gdplt
    g = gdplt.get_subplot_plotter()
    if not isinstance(sample, Mapping) and not isinstance(sample, Sequence):
        sample = {"This run": sample}
    elif isinstance(sample, Sequence):
        sample = {f"Run #{i + 1}": s for i, s in enumerate(sample)}
    to_plot = list(sample.values())
    filled = [filled] * len(to_plot)
    labels = list(sample.keys())
    colors = list(tab10_colors[:len(to_plot)])
    paraminfos = to_plot[0].getParamNames().names
    sampled_paramnames = [p.name for p in paraminfos if not p.isDerived]
    paramnames = sampled_paramnames + ["logpost"]
    if pdf is not None:
        truth_sample = pdf.samples()
        if truth_sample is not None:
            from getdist.mcsamples import MCSamples
            kwargs = {"names": sampled_paramnames}
            kwargs["ranges"] = {
                p: pdf.bounds[i] for i, p in enumerate(sampled_paramnames)}
            truth_sample = MCSamples(samples=truth_sample, **kwargs)
            to_plot += [truth_sample]
            filled += [False]
            labels += ["Truth"]
            colors += ["k"]
    g.triangle_plot(to_plot, params=paramnames, filled=filled,
                    legend_labels=labels, contour_colors=colors)
    g.export(os.path.join(output_folder, filename))
