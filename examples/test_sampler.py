import os
import sys
from importlib import import_module
from pprint import pprint
import numpy as np

from synth_inference_tests.get_pdf import get_pdfs
from synth_inference_tests.run import run as test_run


def get_wrapper(sampler_name):
    wrapper_name = "wrapper_" + sampler_name
    try:
        wrapper = import_module(wrapper_name)
    except ModuleNotFoundError as excpt:
        raise ValueError(f"No wrapper {wrapper_name}.py found for {sampler_name}.") \
            from excpt
    try:
        run_func = wrapper.run_func
        process_output_func = wrapper.process_output_func
    except AttributeError as excpt:
        raise ValueError("The wrapper should contain two functions: 'run_func' and "
                         "'process_output_func' (see documentation).") from excpt
    return run_func, process_output_func


if __name__ == "__main__":
    if len(sys.argv[1:]) != 2:
        raise ValueError("Pass a valid sampler and a likelihood name as first and second "
                         "arg, e.g. 'gpry gaussian5'")
    sampler_name = sys.argv[1].lower()
    run_func, process_output_func = get_wrapper(sampler_name)
    # Build PDF (or list of them)
    pdfs = get_pdfs(sys.argv[2])
    for pdf in pdfs:
        msg = f"*** Sampling from pdf {pdf.NameDim} " + "*" * 50
        print("\n" + "*" * len(msg) + "\n" + msg + "\n" + "*" * len(msg) + "\n")
        output_folder = os.path.join("output_" + sampler_name, pdf.NameDim)
        try:
            os.makedirs(output_folder)
        except FileExistsError:
            pass
        results = test_run(pdf, run_func, process_output_func,
                           output_folder=output_folder)
        print("\n----RESULTS----\n")
        pprint(results)
        print()
