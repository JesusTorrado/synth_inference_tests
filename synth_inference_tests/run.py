import os
from time import time
from warnings import warn

import numpy as np

from .io import create_path, dump_result
from .mpi import get_num_threads, is_main_process, mpi_comm, mpi_size
from .pdf import PDF
from .plots import plot_triangle
from .utils import (
    ColNames,
    generic_param_names,
    js,
    kde_logp_if_needed,
    kl_norm_sym,
    kl_sym,
)


def run(
    pdf,
    run_func,
    process_output_func,
    output_folder,
    i=None,
    budget=None,
    budget_count_inf=False,
    budget_count_parallel=False,
    sampler_kwargs=None,
):
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
    for d in [output_folder, products_folder, plots_folder]:
        if is_main_process:
            create_path(d)
    if budget is True:
        budget = pdf.budget_appinf
    # Basic inputs
    if is_main_process:
        results = {
            "pdf": pdf.__class__.__name__,
            "dim": pdf.dim,
            "n_processes": mpi_size,
            "n_threads_per_process": get_num_threads(),
            "budget": budget,
            "logZ_truth": float(pdf.logZ) if pdf.logZ is not None else None
        }
        if i is not None:
            results["i"] = i
    # Get the fiducial samples early, to pass them to the sampler for plots/checks/...
    if is_main_process:
        sample_ref = pdf.samples()
        # Copy in case it is modified inside run_func
        sample_ref_copy = sample_ref.copy() if sample_ref is not None else None
    # Run external sampler
    mpi_comm.barrier()
    start_total = time()
    try:
        sampler_results, return_values = run_func(
            pdf.logpdf,
            pdf.bounds,
            ref_bounds=getattr(pdf, "ref_bounds", None),
            output_folder=products_folder,
            budget=budget,
            budget_count_inf=False,
            budget_count_parallel=False,
            sampler_kwargs=sampler_kwargs,
            fiducial_samples=sample_ref_copy if is_main_process else None,
        )
    except Exception as excpt:
        warn(f"Sampling ended with unmanaged/unexpected error '{excpt}'.")
    if is_main_process:
        results.update(sampler_results)
    delta_total = time() - start_total
    mpi_comm.barrier()
    # Timings and # evals
    time_pdf = mpi_comm.gather(pdf.t)
    n_evals_pdf = mpi_comm.gather(pdf.n)
    time_overhead = mpi_comm.gather(delta_total - pdf.t)
    if is_main_process:
        results.update(
            {
                "time_truth": max(time_pdf),
                "n_truth": sum(n_evals_pdf),
                "n_truth_max_process": max(n_evals_pdf),
                "time_overhead": max(time_overhead),
            }
        )
    end_state = mpi_comm.bcast(results.get("end_state") if is_main_process else None)
    if not isinstance(end_state, str) or end_state.lower() not in ["c", "b", "e", "?"]:
        raise ValueError(
            "'end_state' must be one of 'c' (converged), 'b' "
            f"(budget exhausted), 'e' (errored). Got {end_state}"
        )
    # NB: 'c' means *spontaneous* stop. If budget exhausted and convergence judged likely
    #     by internal diagnostics, that's still 'b'.
    if end_state == "c":
        if is_main_process:
            print("The sampler has finished after convergence (budget not exhausted).")
    elif end_state == "b":
        if is_main_process:
            print("The sampler has finished because the budget was exhausted.")
    elif end_state == "?":
        if is_main_process:
            print("The sampler has finished with unknown end state.")
            dump_result(results, output_folder)
        return results if is_main_process else None
    elif end_state == "e":
        if is_main_process:
            print("The sampler has finished in an error end state.")
            dump_result(results, output_folder)
        return results if is_main_process else None
    # Compute/process necessary quantities for the report, do sampler-internal plots, etc.
    mpi_comm.barrier()
    if is_main_process:
        print("Processing sampler output...")
    try:
        # NB: expects to be called by all processes (e.g. MPI-parallel processes with
        #     a separate chain per process).
        #     If not needed, put a `if not mpi.is_main_process: return None` on top of the
        #     definition.
        sampler_results = process_output_func(
            return_values,
            output_folder=products_folder,
        )
        processing_success = True
    except Exception as excpt:
        print(f"Error processing output: {excpt}")
        processing_success = False  # only MPI rank 0 ever gets here
    processing_success = mpi_comm.bcast(processing_success)
    if not processing_success:
        return results if is_main_process else None
    mpi_comm.barrier()
    paramnames = generic_param_names(pdf.dim, based_0=False)
    if is_main_process:
        # (Re)create derived parameters for log-posterior, log-likelihood, log-prior
        sample = sampler_results["samples"]
        sample_X = sample[paramnames].to_numpy()
        loglikes = pdf.logp(sample_X)
        sample[ColNames.logpost] = loglikes + pdf.logprior_density
        sample[ColNames.loglike] = loglikes
        sample[ColNames.logprior] = pdf.logprior_density
    # Do our side of the tests and plots
    if is_main_process:
        results.update(sampler_results)
        sample_orig = results["samples"]
        print("Plotting...")
        plot_triangle(sample_orig, pdf=pdf, output_folder=plots_folder)
        print("Computing metrics...")
        weights = sample[ColNames.weight].to_numpy()
        if sample_ref is not None and sample_ref.shape[1] == pdf.dim + 1:
            weights_ref = sample_ref[:, 0]
            sample_ref = sample_ref[:, 1:]
        else:
            weights_ref = None
        if sample_ref is not None:
            # Symmetric (Jeffrey's) KL against the true pdf; Gaussian approx.
            results["kl_norm_sym"] = float(
                kl_norm_sym(
                    np.average(sample_X, weights=weights, axis=0),
                    np.cov(sample_X.T, aweights=weights),
                    np.average(sample_ref, axis=0, weights=weights_ref),
                    np.cov(sample_ref.T, aweights=weights_ref),
                )
            )
            # Precompute common functions and quantities
            logp_func_ref = lambda x: pdf.logp(x) + pdf.logprior_density
            logp_func_sample_kde = kde_logp_if_needed(
                sample_X, logp_func_ref(sample_X), weights=weights
            )
            logp_ref_ref = logp_func_ref(sample_ref)
            logp_mc_mc = logp_func_sample_kde(sample_X)
            # Symmetric (Jeffrey's) KL against the true pdf, MC-sum with KDE approx
            results["kl_sym"] = float(
                kl_sym(
                    sample_ref,
                    sample_X,
                    weights_1=weights_ref,
                    weights_2=weights,
                    logp_func_1=logp_func_ref,
                    logp_func_2=logp_func_sample_kde,
                    logp_1_sample_1=logp_ref_ref,
                    logp_2_sample_2=logp_mc_mc,
                )
            )
            # Jensen-Shannon divergence, MC-sum with KDE approx
            results["js"] = float(
                js(
                    sample_ref,
                    sample_X,
                    weights_1=weights_ref,
                    weights_2=weights,
                    logp_func_1=logp_func_ref,
                    logp_func_2=logp_func_sample_kde,
                    logp_1_sample_1=logp_ref_ref,
                    logp_2_sample_2=logp_mc_mc,
                )
            )
            # If the sampler has a surrogate log-posterior, try using it instead of KDE.
            logp_func_surr = results.pop("logp_func", None)
            if logp_func_surr is not None:
                logp_func_surr_clipped = lambda x: np.clip(logp_func_surr(x), -1e30, None)
                results["kl_sym_surr"] = float(
                    kl_sym(
                        sample_ref,
                        sample_X,
                        weights_1=weights_ref,
                        weights_2=weights,
                        logp_func_1=logp_func_ref,
                        logp_func_2=logp_func_surr_clipped,
                        logp_1_sample_1=logp_ref_ref,
                    )
                )
                results["js_surr"] = float(
                    js(
                        sample_ref,
                        sample_X,
                        weights_1=weights_ref,
                        weights_2=weights,
                        logp_func_1=logp_func_ref,
                        logp_func_2=logp_func_surr_clipped,
                        logp_1_sample_1=logp_ref_ref,
                    )
                )
        # Save results object and samples
        dump_result(results, output_folder)
    mpi_comm.barrier()
    if is_main_process:
        print("Done!")
    return results if is_main_process else None
