"""
Builds the pdf database and retrieves one or more pdf's based on some dimensionality and tag requirements.
"""

import os
from inspect import isclass
from itertools import chain

import yaml
from numpy.random import default_rng

import synth_inference_tests.pdfs as pdfs

# TODO: cached database

# Import all classes


def import_pdf(name):
    """
    Imports pdf by name.

    NB: pdf's must be importable as ``from synth_inference_tests.pdfs import [pdf]`.
    """
    for k, class_or_module in pdfs.__dict__.items():
        if isclass(class_or_module):
            if k.lower() == name.lower():
                return class_or_module


def get_pdf(name, dim=None, rng=None, tags=None):
    """
    Retrieves pdf according to some criteria.

    Use cases:
    - gaussian10: gaussian in d=10
    """
    # If there is a number at the end of the name, split into (name, dim)
    pdfname = name.rstrip("0123456789")
    dim_from_name = name.removeprefix(pdfname)
    if dim_from_name:
        dim = int(dim_from_name)
    pdf_class = import_pdf(pdfname)
    if pdf_class is None:
        raise ValueError(f"pdf '{name}' not recognized.")
    # Check if it takes dimensionality
    if dim and not pdf_class.can_be_dim(dim):
        raise ValueError(f"pdf '{pdfname}' cannot have dimensionality {dim}.")
    pdf_args = {}
    if dim is not None:
        pdf_args["dim"] = dim
    if getattr(pdf_class, "random", False):
        pdf_args["rng"] = rng or default_rng()
    pdf = pdf_class(**pdf_args)
    return pdf


def get_pdfs(pdf_or_file_name):
    """
    Retrieves pdfs from an input yaml file (as keys). Returns instances.
    """
    if os.path.splitext(pdf_or_file_name)[1]:
        try:
            with open(pdf_or_file_name, "r") as f:
                pdfs_dict = yaml.safe_load(f)
        except FileNotFoundError as excpt:
            raise FileNotFoundError(
                f"PDFs file {pdf_or_file_name} not found."
            ) from excpt
        rng = default_rng(seed=pdfs_dict.pop("seed", None))
        pdfs = [
            [get_pdf(pdf_name, rng=rng) for _ in range(n or 1)]
            for pdf_name, n in pdfs_dict.items()
        ]
        pdfs = list(chain(*pdfs))  # flatten list
        return pdfs
    else:  # is pdf name
        return [get_pdf(pdf_or_file_name)]
