import os
import sys
import yaml
import pandas as pd

results_filename = "result.yaml"

if __name__ == "__main__":
    if len(sys.argv[1:]) != 1:
        raise ValueError("Pass a folder as first argument")
    output_folder = sys.argv[1]
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
    preferred_order = [
        # PDF
        "pdf",
        "dim",
        # Environment
        "n_processes",
        "n_threads_per_process",
        # Result
        "end_state",
        "n_truth",
        "time_overhead"
        ]
    ignore_columns = ["time_truth"]
    new_columns_order = [col for col in preferred_order if col in table.columns]
    new_columns_order += [col for col in table.columns if col not in new_columns_order]
    new_columns_order = [col for col in new_columns_order if col not in ignore_columns]
    table = table[new_columns_order]
    table.sort_values(["pdf", "dim"], inplace=True)
    print(table)
