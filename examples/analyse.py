import sys

from synth_inference_tests.analysis import create_table

if __name__ == "__main__":
    if len(sys.argv[1:]) != 1:
        raise ValueError("Pass a folder as first argument")
    output_folder = sys.argv[1]
    table = create_table(output_folder)
    import pandas as pd
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 500)
    print(table)
