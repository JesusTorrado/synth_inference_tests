import sys

from synth_inference_tests.analysis import create_table

if __name__ == "__main__":
    if len(sys.argv[1:]) != 1:
        raise ValueError("Pass a folder as first argument")
    output_folder = sys.argv[1]
    table = create_table(output_folder)
    print(table)
