import os

import numpy as np
import pandas as pd  # type: ignore
import yaml  # type: ignore

from .utils import ColNames, generic_param_names

# Ignore unconstruct-able left-over tags (e.g. surrogate log-posterior)
yaml.add_multi_constructor("tag:", lambda *x, **kw: None, Loader=yaml.SafeLoader)

result_file = "result.yaml"
_default_samples_file = "samples.npy"  # weights [parameters] logpost loglike logprior
path_pdfs_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pdfs", "data")


def create_path(path):
    """Creates the folder path ``path`` if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def dump_samples(sample, output_folder, samples_file=_default_samples_file):
    # The following assumes the sample contains
    # "weights [parameters] logpost, loglike, logprior"
    np.save(os.path.join(output_folder, samples_file), sample.to_numpy().T)


def load_samples(output_folder, samples_file=_default_samples_file):
    # The following assumes the sample contains
    # "weights [parameters] logpost, loglike, logprior"
    if not samples_file.lower().endswith(".npy"):
        samples_file += ".npy"
    samples_arr = np.load(os.path.join(output_folder, samples_file))
    colnames = [ColNames.weight]
    colnames += generic_param_names(samples_arr.shape[0] - 4, based_0=False)
    colnames += [ColNames.logpost, ColNames.loglike, ColNames.logprior]
    return pd.DataFrame(dict(zip(colnames, samples_arr)), dtype=np.float64)


def dump_result(result, output_folder):
    samples = result.pop("samples", None)
    if samples is not None:
        dump_samples(samples, output_folder)
    with open(os.path.join(output_folder, result_file), "w") as f:
        yaml.dump(result, f)


def yaml_load(filename):
    with open(filename, "r") as f:
        return yaml.safe_load(f)


def yaml_dump(data, filename):
    with open(filename, "w") as f:
        return yaml.dump(data, f)
