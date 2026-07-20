import sys

import pandas as pd

from synth_inference_tests.analysis import (
    create_table,
    aggregate_table,
    summarize_aggregated_table,
    plot_metrics,
)

if __name__ == "__main__":
    if len(sys.argv[1:]) != 1:
        raise ValueError("Pass a folder as first argument")
    output_folder = sys.argv[1]
    table = create_table(output_folder)
    pd.set_option("display.max_rows", 999)
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.width", 500)
    print("+ Results table +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(table)
    #    # Plot for outliers in evidence
    #    import matplotlib.pyplot as plt
    #    plt.figure()
    # pdf  dim   i    sampler (MPI, thr) end_state        logZ   logZstd

    #    plt.errorbar(table.index, table["logZ"] , yerr=table["logZstd"])
    #    plt.show()

    print("+ Aggregated table ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    agg_table = aggregate_table(table)
    print(summarize_aggregated_table(agg_table))
    plot_metrics(agg_table, ".")
