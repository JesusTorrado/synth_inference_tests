import os
from time import time

import yaml

from .pdf import PDF
from .mpi import is_main_process, multiple_processes, mpi_comm, mpi_size, get_num_threads


result_file = "result.yaml"


def run(pdf, run_func, process_output_func, output_folder):
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
    return_values = run_func(pdf.logpdf, pdf.bounds, output_folder=products_folder)
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
    # Compute/process necessary quantities for the report
    sample_results = process_output_func(return_values, output_folder=products_folder)

    # Save results object
    with open(os.path.join(output_folder, result_file), "w") as f:
        yaml.dump(result, f)
    
    # Plots
    if is_main_process:
        plot_triangle(sample_results["samples"], output_folder=plots_folder)


def plot_triangle(sample, output_folder):
    from getdist import plots as gdplt
    g = gdplt.get_subplot_plotter()
    g.triangle_plot([sample], filled=True)
    g.export(os.path.join(output_folder, "triangle.png"))

