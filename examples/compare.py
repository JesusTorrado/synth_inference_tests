import os
import sys
import pandas as pd

from synth_inference_tests.analysis import create_table
from synth_inference_tests.run import plot_triangle
from synth_inference_tests.get_pdf import get_pdf

from test_sampler import get_wrapper

table_file = "comparison.txt"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise ValueError("Pass some folders as arguments.")
    output_folders = sys.argv[1:]
    tables = {}
    folders = {}
    process_funcs = {}
    for folder in output_folders:
        sampler = folder.rstrip("/")
        if sampler.startswith("output_"):
            sampler = sampler[len("output_"):]
        folders[sampler] = folder
        process_funcs[sampler] = get_wrapper(sampler)[1]
        tables[sampler] = create_table(folder)
        tables[sampler]["folder"] = folder
        sampler = tables[sampler]["sampler"]
    # Simple for now: merge tables and re-sort with folder as last priority.
    # TODO: delete non-common cases
    comparison_table = pd.concat(list(tables.values()))
    comparison_table.sort_values(["pdf", "dim", "sampler"], inplace=True)
    print(comparison_table)
    compare_folder = "compare_" + "_".join(output_folders)
    try:
        os.makedirs(compare_folder)
    except FileExistsError:
        pass
    with open(os.path.join(compare_folder, table_file), "w") as f:
        f.write(comparison_table.to_string() + "\n")
    # Plots!
    pdf_names = [pdf + str(dim)
                 for pdf, dim in zip(comparison_table["pdf"], comparison_table["dim"])]
    # Uniquify preserving order
    pdf_names = [name for i, name in enumerate(pdf_names) if name not in pdf_names[:i]]
    for pdf_name in pdf_names:
        print(f"Plotting {pdf_name}...")
        gdsamples = {}
        for sampler, proc_func in process_funcs.items():
            products_folder = os.path.abspath(
                os.path.join(folders[sampler], pdf_name, "products"))
            try:
                gdsamples[sampler] = proc_func(output_folder=products_folder)["samples"]
            except FileNotFoundError:
                print(f"PDF {pdf_name} not computed with {sampler}")
        plot_triangle(
            gdsamples, output_folder=compare_folder,
            filename=pdf_name + ".png", pdf=get_pdf(pdf_name), filled=False)
    print("Done plotting!")
