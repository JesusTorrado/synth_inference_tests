import os
import sys

import pandas as pd  # type: ignore

from synth_inference_tests.analysis import (
    aggregate_table,
    create_table,
    plot_metrics,
    summarize_aggregated_table,
)

if __name__ == "__main__":
    if len(sys.argv[1:]) < 1:
        raise ValueError("Pass a folder as first argument (or several)")
    output_folders = sys.argv[1:]
    agg_tables = {}
    for output_folder in output_folders:
        table = create_table(output_folder)
        pd.set_option("display.max_rows", 999)
        pd.set_option("display.max_columns", 100)
        pd.set_option("display.width", 500)
        print("+ Results table +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(table)
        print("+ Aggregated table ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        agg_tables[output_folder] = aggregate_table(table)
        print(summarize_aggregated_table(agg_tables[output_folder]))
        plot_metrics({output_folder.replace(os.path.sep, ""): agg_tables[output_folder]}, ".")
    # Create comparison plots
    plot_metrics(agg_tables, ".")
