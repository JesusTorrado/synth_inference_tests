import os
from time import time

import yaml

from .pdf import PDF
from .mpi import is_main_process, multiple_processes, mpi_comm, mpi_size, get_num_threads


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
        result = {
            "pdf": pdf.__class__.__name__,
            "dim": pdf.dim,
            "n_processes": mpi_size,
            "n_threads_per_process": get_num_threads(),
        }
    # Run external sampler
    start_total = time()
    return_values = run_func(pdf.logpdf, pdf.bounds, output_folder=products_folder,
                             budget=None, budget_count_inf=False,
                             budget_count_parallel=False)
    delta_total = time() - start_total
    # Timings and # evals
    time_pdf = mpi_comm.gather(pdf.t)
    n_evals_pdf = mpi_comm.gather(pdf.n)
    time_overhead = mpi_comm.gather(delta_total - pdf.t)
    if is_main_process:
        result.update({
            "time_truth": max(time_pdf),
            "n_truth": max(n_evals_pdf),
            "time_overhead": max(time_overhead),
        })
    try:
        end_state = return_values[0]
        assert isinstance(end_state, str) and end_state.lower() in ["c", "b", "e"]
    except (IndexError, AssertionError) as excpt:
        if hasattr(return_values, "__len__"):
            what_return_value_msg = f"Got {return_values[0]}."
        else:
            what_return_value_msg = ("Cannot get first element of return values "
                                     "(not a tuple?).")
        raise ValueError("The first return value for the 'run' function must be the end "
                         "state, in particular one of 'c' (converged), 'b' "
                         f"(budget exhausted), 'e' (errored). {what_return_value_msg}")
    result["end_state"] = end_state.lower()
    # NB: 'c' means *spontaneous* stop. If budget exhausted and convergence judged likeky
    #     by internal diagnostics, that's still 'b'.
    if end_state == 'e':
        dump_result(result, output_folder)
        return

    # Compute/process necessary quantities for the report
    sample_results = process_output_func(
        return_values, output_folder=products_folder)


    # Save results object
    dump_result(result, output_folder)

    # Plots
    if is_main_process:
        plot_triangle(sample_results["samples"], pdf=pdf, output_folder=plots_folder)


def dump_result(result, output_folder):
    with open(os.path.join(output_folder, result_file), "w") as f:
        yaml.dump(result, f)


def plot_triangle(sample, output_folder, pdf=None):
    from getdist import plots as gdplt
    g = gdplt.get_subplot_plotter()
    paraminfos = sample.getParamNames().names
    sampled_paramnames = [p.name for p in paraminfos if not p.isDerived]
    paramnames = sampled_paramnames + ["chi2"]
    to_plot = [sample]
    filled = [True]
    labels = ["This run"]
    colors = ["blue"]
    if pdf is not None:
        truth_sample = pdf.samples()
        if truth_sample is not None:
            from getdist.mcsamples import MCSamples
            truth_sample = MCSamples(samples=truth_sample, names=sampled_paramnames)
            to_plot += [truth_sample]
            filled += [False]
            labels += ["Truth"]
            colors += ["k"]
    g.triangle_plot(to_plot, params=paramnames, filled=filled,
                    legend_labels=labels, contour_colors=colors)
    g.export(os.path.join(output_folder, "triangle.png"))
