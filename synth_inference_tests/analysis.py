import os
import yaml
import pandas as pd

results_filename = "result.yaml"


def create_table(output_folder):
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
            print(f"Unfinished run?: {dir}")
    if not rows:
        raise ValueError(
            f"The given folder {output_folder} does not contain any finished runs.")
    # Create results table
    columns = list(rows[0])
    table_dict = {col: [row.get(col, None) for row in rows] for col in columns}
    table = pd.DataFrame(table_dict)
    col_parallel = "(MPI, threads)"
    table["(MPI, threads)"] = [
        (m, t) for _, (m, t)
        in table[["n_processes", "n_threads_per_process"]].iterrows()]
    first_columns = [
        # PDF
        "pdf",
        "dim",
        # Environment
        "(MPI, threads)",
        # Result
        "end_state",
        "n_truth",
        "time_overhead"
        ]
    last_columns = ["notes"]
    ignore_columns = ["time_truth", "n_processes", "n_threads_per_process"]
    new_columns_order = [col for col in first_columns if col in table.columns]
    new_columns_order += [col for col in table.columns if col not in new_columns_order]
    new_columns_order = [col for col in new_columns_order
                         if col not in set(ignore_columns).union(last_columns)]
    new_columns_order += [col for col in last_columns if col in table.columns]
    table = table[new_columns_order]
    table.sort_values(["pdf", "dim"], inplace=True)
    return table
