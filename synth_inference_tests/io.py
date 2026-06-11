import os

import numpy as np
import pandas as pd  # type: ignore
import yaml  # type: ignore

from .utils import ColNames, generic_param_names

result_file = "result.yaml"
samples_file = "samples.npy"  # array: weights [parameters] logpost loglike logprior
path_pdfs_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pdfs", "data")


def create_path(path):
    """Creates the folder path ``path`` if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def dump_samples(sample, output_folder):
    # The following assumes the sample contains "[parameters] logpost, loglike, logprior"
    np.save(os.path.join(output_folder, samples_file), sample.to_numpy().T)


def load_samples(output_folder):
    # The following assumes the sample contains "[parameters] logpost, loglike, logprior"
    samples_arr = np.load(os.path.join(output_folder, samples_file))
    colnames = [ColNames.weight]
    colnames += generic_param_names(samples_arr.shape[0] - 4, based_0=False)
    colnames += [ColNames.logpost, ColNames.loglike, ColNames.logprior]
    return pd.DataFrame(dict(zip(colnames, samples_arr)), dtype=np.float64)


def dump_result(result, output_folder):
    samples = result.pop("samples")
    dump_samples(samples, output_folder)
    with open(os.path.join(output_folder, result_file), "w") as f:
        yaml.dump(result, f)
