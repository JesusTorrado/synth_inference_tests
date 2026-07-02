import os
from time import time
from warnings import warn

import numpy as np

from .io import create_path, dump_result
from .mpi import get_num_threads, is_main_process, mpi_comm, mpi_size
from .pdf import PDF
from .plots import plot_triangle
from .utils import ColNames, generic_param_names, kl, kl_norm_sym, kl_sym


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
    paramnames = generic_param_names(pdf.dim, based_0=False)
    if is_main_process:
        # (Re)create derived parameters for log-posterior, log-likelihood, log-prior
        sample = sampler_results["samples"]
        sample_X = sample[paramnames].to_numpy()
        loglikes = pdf.logp(sample_X)
        sample[ColNames.logpost] = loglikes + pdf.logprior_density
        sample[ColNames.loglike] = loglikes
        sample[ColNames.logprior] = pdf.logprior_density
    if not processing_success:
        return results if is_main_process else None
    mpi_comm.barrier()
    # Do our side of the tests and plots
    if is_main_process:
        print("Computing results...")
        results.update(sampler_results)
        # Symmetric (Jeffrey's) KL against the true pdf. Only if we have a sampler from the
        # true posterior AND a surrogate model (otherwise it is tiny and meaningless).
        sample_orig = results["samples"]
        if sample_ref is not None and sample_ref.shape[1] == pdf.dim + 1:
            weights_ref = sample_ref[:, 0]
            sample_ref = sample_ref[:, 1:]
        else:
            weights_ref = None
        logp_func = results.pop("logp_func", None)
        if sample_ref is not None and logp_func is not None:
            sampled_params = [
                p.name for p in sample_orig.getParamNames().names if not p.isDerived
            ]
            sample = np.array([sample_orig[p] for p in sampled_params]).T
            weights = sample_orig.weights
            # This is a log-posterior sample, not a log-likelihood,
            sample_logp = sample_orig["logpost"]
            # so the reference pdf must be the logposterior too!
            sample_logp_ref = pdf.logpost(sample)
            sample_ref_logp_ref = pdf.logpost(sample_ref)
            sample_ref_logp = logp_func(sample_ref)
            # Let's put a bottom to the logp function to avoid infinite terms in KL sum:
            # use the smallest still-finite value
            sample_ref_logp[~np.isfinite(sample_ref_logp)] = np.min(
                sample_ref_logp[np.isfinite(sample_ref_logp)]
            )
            results["kl_left"] = float(
                kl(
                    sample_ref,
                    sample_ref_logp_ref,
                    sample_ref_logp,
                    weights_P=weights_ref,
                )
            )
            results["kl_right"] = float(
                kl(sample, sample_logp, sample_logp_ref, weights_P=weights)
            )
            results["kl"] = float(
                kl_sym(
                    sample,
                    sample_logp,
                    sample_logp_ref,
                    sample_ref,
                    sample_ref_logp_ref,
                    sample_ref_logp,
                    weights_1=weights,
                    weights_2=weights_ref,
                )
            )
            results["kl_norm"] = float(
                kl_norm_sym(
                    np.average(sample, weights=weights, axis=0),
                    np.cov(sample.T, aweights=weights),
                    np.average(sample_ref, axis=0, weights=weights_ref),
                    np.cov(sample_ref.T, aweights=weights_ref),
                )
            )
        # Evidence
        results["logZ_truth"] = float(pdf.logZ) if pdf.logZ is not None else None
        # Save results object and samples
        dump_result(results, output_folder)
    # Plots
    if is_main_process:
        print("Plotting...")
        plot_triangle(sample_orig, pdf=pdf, output_folder=plots_folder)
    mpi_comm.barrier()
    if is_main_process:
        print("Done!")
    return results if is_main_process else None
