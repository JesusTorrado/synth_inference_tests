import os
import warnings
from itertools import chain
from typing import Mapping

import numpy as np
import pandas as pd  # type: ignore

from .io import yaml_load
from .plots import metric_boxplot

results_filename = "result.yaml"
col_parallel = "(MPI, thr)"

# Operations for columns
# As list of tuples/lists; 2 elements if a metric has (avg,std) for weighted mean
_average_cols = [
    ["kl_norm_sym"],
    ["kl_sym"],
    ["kl_sym_surr"],
    ["js"],
    ["js_surr"],
    ["logZ", "logZstd"],
    ["time_overhead"],
    ["n_truth"],
]
_max_cols = ["n_truth_max_process"]


def create_table(output_folder, aggregate=False):
    """
    Parameters
    ----------

    aggregate: bool (default: False)
        Aggregate realisations by ... [what's done with each variable] -- drop "i" column!
    ...
    """
    # Load all results data
    rows = []
    for dir in os.listdir(output_folder):
        absdir = os.path.abspath(os.path.join(output_folder, dir))
        if not os.path.isdir(absdir):
            continue
        try:
            rows.append(yaml_load(os.path.join(absdir, results_filename)))
        except FileNotFoundError:
            warnings.warn(f"Unfinished run?: {dir}")
    if not rows:
        raise ValueError(
            f"The given folder {output_folder} does not contain any finished runs."
        )
    # Create results table
    columns = list(rows[0])
    table_dict = {col: [row.get(col, None) for row in rows] for col in columns}
    table = pd.DataFrame(table_dict)
    table[col_parallel] = [
        (m, t) for _, (m, t) in table[["n_processes", "n_threads_per_process"]].iterrows()
    ]
    first_columns = [
        # PDF
        "pdf",
        "dim",
        "i",
        # Sampler and environment
        "sampler",
        col_parallel,
        # Convergence and results
        "end_state",
        "kl_norm_sym",
        "kl_sym",
        "js",
        "kl_sym_surr",
        "js_surr",
        "logZ",
        "logZstd",
        "logZ_truth",
        # Efficiency
        "time_overhead",
        "n_truth",
        "n_truth_max_process",
    ]
    last_columns = ["notes"]
    ignore_columns = ["time_truth", "n_processes", "n_threads_per_process"]
    new_columns_order = [col for col in first_columns if col in table.columns]
    new_columns_order += [col for col in table.columns if col not in new_columns_order]
    new_columns_order = [
        col
        for col in new_columns_order
        if col not in set(ignore_columns).union(last_columns)
    ]
    new_columns_order += [col for col in last_columns if col in table.columns]
    table = table[new_columns_order]
    table.sort_values(["pdf", "dim", "i"], inplace=True)
    table.reset_index(drop=True, inplace=True)
    if aggregate:
        table = aggregate_table(table)
    return table


def aggregate_table(table, return_non_converged=False):
    """
    Aggregates rows corresponding to *converged* runs of the same pdf with the same
    sampler. Returns list of values for metric-related columns (and by default for any
    other unknown column).

    If ``return_non_converged``, it returns a tuple ``(aggregated_table, errors_table)``,
    where the errors table is formatted as the input one.
    """
    # Work with a copy of the table, in case leftover runs returned
    table = table.copy(deep=True)
    common_cols = ["pdf", "dim", "sampler", "logZ_truth"]
    ignore_cols = [col_parallel, "end_state", "budget", "i"]
    # Prepare aggregated table, copying
    agg_table_cols = [col for col in table.columns if col not in ignore_cols]
    agg_table = pd.DataFrame(
        {col: pd.Series([], dtype=table.dtypes[col]) for col in agg_table_cols}
    )
    # Find unique combinations to aggregate (np.unique does not work well with str)
    combinations = []
    for comb in table[common_cols].to_numpy():
        comb = tuple(comb)
        if comb not in combinations:
            combinations.append(comb)
    # Start filling up the new table, removing the used rows
    N_comb = []
    for comb in combinations:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            i_comb = table[table[common_cols[0]] == comb[0]][
                table[common_cols[1]] == comb[1]
            ][table[common_cols[2]] == comb[2]][table["end_state"] == "c"].index.to_list()
        rows = table.loc[i_comb].copy()
        table.drop(i_comb, inplace=True)
        N_comb.append(len(rows))
        # Common data to be added: uniquify and merge, or concatenate if >1 different
        row_data = {col: set(rows[col]) for col in agg_table_cols}
        row_data = {
            col: (data.pop() if len(data) == 1 else list(data))
            for col, data in row_data.items()
        }
        # Columns not dealt-with so far: add list of values
        for col in rows.columns:
            if any(col in colset for colset in [common_cols, ignore_cols]):
                continue
            row_data[col] = list(rows[col])
        agg_table.loc[len(agg_table)] = row_data
    # Add aggregation counters
    agg_table.insert(len(common_cols), "N", N_comb)
    if return_non_converged:
        return agg_table, table
    if len(table):
        warnings.warn(
            f"There were {len(table)} non-converged runs left. "
            "Use argument `return_non_converged=True` to return a tuple "
            "(aggregated_table, non_converged_table)."
        )
    return agg_table


def summarize_aggregated_table(agg_table):
    """
    Summarizes values for metrics in an aggregated table.
    """
    # Prepare summary table, copying
    to_be_dropped_cols = list(chain.from_iterable(_average_cols)) + _max_cols
    summ_table_cols = [col for col in agg_table.columns if col not in to_be_dropped_cols]
    summ_table = pd.DataFrame(
        {col: pd.Series([], dtype=agg_table.dtypes[col]) for col in summ_table_cols}
    )
    for _, row in agg_table.iterrows():
        # Preprare the row data for the summary table
        row_data = {col: row[col] for col in summ_table_cols}
        # Computed data
        for col in _average_cols:
            if len(col) == 1:
                row_data[col[0] + "_agg_avg"] = np.average(row[col[0]])
                row_data[col[0] + "_agg_std"] = np.std(row[col[0]])
            else:  # 2 cols: inv var weighting
                row_data[col[0] + "_agg_avg"] = np.sum(
                    row[col[0]] / np.power(row[col[1]], 2)
                ) / np.sum(1 / np.power(row[col[1]], 2))
                row_data[col[1] + "_agg_std"] = np.sqrt(
                    1 / np.sum(1 / np.power(row[col[1]], 2))
                )
        for col in _max_cols:
            row_data[col + "_agg_max"] = max(row[col])
        for new_col in row_data:
            if new_col not in summ_table.columns:
                summ_table[new_col] = []
        # Add the computed summaries (and the common cols) to the summary table
        summ_table.loc[len(summ_table)] = row_data
    return summ_table


def plot_metrics(agg_table, output_folder, filename="metric", ext=".png"):
    """
    From an aggredated (nor summarized!) table (or dict of tables), plots bloxplots
    summarizing of the metrics.
    """
    if not isinstance(agg_table, Mapping):
        agg_table = {None: agg_table}
    # Gather all dists, merge and sort (Gaussian first)
    dists = [
        [dist + str(dim) for dist, dim in zip(table["pdf"], table["dim"])]
        for table in agg_table.values()
    ]
    dists = list(set(chain(*dists)))
    dists_bare = list(set(d.rstrip("0123456789") for d in dists))
    dists_bare = (["Gaussian"] if "Gaussian" in dists_bare else []) + [
        d for d in sorted(dists_bare) if d != "Gaussian"
    ]
    sorted_dists = []
    for db in dists_bare:
        # Make sure str("2") comes before str("10")
        sorted_dists += sorted(
            (d for d in dists if d.startswith(db)), key=lambda x: int(x[len(db) :])
        )
    for metric_cols in _average_cols:
        col = metric_cols[0]
        data = {}
        for k, table in agg_table.items():
            data_k = list(table[["pdf", "dim", col]].to_dict(orient="list").values())
            data_k = {pdf + str(dim): data for pdf, dim, data in zip(*data_k)}
            if k is not None:
                k = k.replace(os.path.sep, "")
            data[k] = {dist: data_k.get(dist, []) for dist in sorted_dists}
        # Ref values: look up in first table
        ref_values = None
        lookup_table = list(agg_table.values())[0]
        for suff in ["truth", "_truth"]:
            if col + suff in lookup_table.columns:
                dists_ref = list(
                    lookup_table[["pdf", "dim", col + suff]]
                    .to_dict(orient="list")
                    .values()
                )
                ref_values = {pdf + str(dim): ref for pdf, dim, ref in zip(*dists_ref)}
        filename = "metric"
        if list(data) != [None]:
            filename += "_" + "_".join(data)
        if len(data) == 1:
            data = list(data.values())[0]
        metric_boxplot(
            data, output_folder, name=col, ref_values=ref_values, filename=filename
        )
