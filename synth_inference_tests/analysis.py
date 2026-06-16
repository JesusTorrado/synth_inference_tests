import os
import warnings
from itertools import chain

import numpy as np
import pandas as pd  # type: ignore
import yaml  # type: ignore

results_filename = "result.yaml"


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
            with open(os.path.join(absdir, results_filename), "r") as f:
                rows.append(yaml.safe_load(f))
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
    col_parallel = "(MPI, thr)"
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
        "kl",
        "kl_norm",
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
    table.sort_values(["pdf", "dim"], inplace=True)
    if aggregate:
        table = aggregate_table(table)
    return table


def aggregate_table(table, return_non_converged=False):
    """
    Aggregates rows corresponding to *converged* runs of the same pdf with the same
    sampler.
    
    If ``return_non_converged``, it returns a tuple ``(aggregated_table, errors_table)``,
    where the errors table is formatted as the input one.
    """
    # Work with a copy of the table, in case leftover runs returned
    table = table.copy(deep=True)
    common_cols = ["pdf", "dim", "sampler"]
    # Operations for columns (default operation: concatenate)
    average_cols = [["logZ", "logZstd"], ["time_overhead"], ["n_truth"]]
    max_cols = ["n_truth_max_process"]
    # ...
    ignore_cols = list(chain.from_iterable(average_cols)) + max_cols
    # Prepare aggregated table, copying
    agg_table_cols = [
        col for col in table.columns if col != "i" and col not in ignore_cols
    ]
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
    for comb in combinations:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            i_comb = table[table[common_cols[0]] == comb[0]][
                table[common_cols[1]] == comb[1]
            ][table[common_cols[2]] == comb[2]][table["end_state"] == "c"].index.to_list()
        rows = table.loc[i_comb].copy()
        table.drop(i_comb, inplace=True)
        # Common data to be added: uniquify and merge, or concatenate if >1 different
        row_data = {col: set(rows[col]) for col in agg_table_cols}
        row_data = {
            col: (data.pop() if len(data) == 1 else list(data))
            for col, data in row_data.items()
        }
        # Computed data
        for cols in average_cols:
            if len(cols) == 1:
                row_data[cols[0] + "_agg_avg"] = np.average(rows[cols[0]])
                row_data[cols[0] + "_agg_std"] = np.std(rows[cols[0]])
            else:  # 2 cols: inv var weighting
                row_data[cols[0] + "_agg_avg"] = np.sum(
                    rows[cols[0]] / rows[cols[1]] ** 2
                ) / np.sum(1 / rows[cols[1]] ** 2)
                row_data[cols[1] + "_agg_std"] = np.sqrt(
                    1 / np.sum(1 / rows[cols[1]] ** 2)
                )
        for col in max_cols:
            row_data[col + "_agg_max"] = max(rows[col])
        for new_col in row_data:
            if new_col not in agg_table.columns:
                agg_table[new_col] = []
        agg_table.loc[len(agg_table)] = row_data
    if return_non_converged:
        return agg_table, table
    if len(table):
        warnings.warn(
            f"There were {len(table)} non-converged runs left. "
            "Use argument `return_non_converged=True` to return a tuple "
            "(aggregated_table, non_converged_table)."
        )
    return agg_table
