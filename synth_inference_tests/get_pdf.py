"""
Builds the pdf database and retrieves one or more pdf's based on some dimensionality and tag requirements.
"""

import importlib
from inspect import isclass

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


def get_pdf(name, dim=None, tags=None):
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
    pdf = pdf_class(**pdf_args)
    return pdf
