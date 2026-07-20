import os
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import LinAlgError

from .utils import ColNames, pd_to_gd_samples


def plot_triangle(sample, output_folder, filename="triangle.png", pdf=None, filled=True):
    gdsample = pd_to_gd_samples(sample, pdf.bounds)
    plot_triangle_getdist(
        gdsample, output_folder, filename=filename, pdf=pdf, filled=filled
    )


def plot_triangle_getdist(
    sample, output_folder, filename="triangle.png", pdf=None, filled=True
):
    tab10_colors = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    )
    from getdist import plots as gdplt  # type: ignore
    from getdist.mcsamples import MCSamples  # type: ignore

    g = gdplt.get_subplot_plotter()
    if not isinstance(sample, Mapping) and not isinstance(sample, Sequence):
        sample = {"This run": sample}
    elif isinstance(sample, Sequence):
        sample = {f"Run #{i + 1}": s for i, s in enumerate(sample)}
    to_plot = list(sample.values())
    filled = [filled] * len(to_plot)
    labels = list(sample.keys())
    colors = list(tab10_colors[: len(to_plot)])
    paraminfos = to_plot[0].getParamNames().names
    sampled_paramnames = [p.name for p in paraminfos if not p.isDerived]
    paramnames = sampled_paramnames + [ColNames.logpost]
    if pdf is not None:
        truth_sample = pdf.samples()
        if truth_sample is not None:
            truth_weights = None
            if truth_sample.shape[1] == pdf.dim + 1:
                truth_weights, truth_sample = truth_sample[:, 0], truth_sample[:, 1:]
            kwargs = {"names": sampled_paramnames}
            kwargs["ranges"] = {
                p: pdf.bounds[i] for i, p in enumerate(sampled_paramnames)
            }
            truth_sample = MCSamples(
                weights=truth_weights, samples=truth_sample, **kwargs
            )
            to_plot += [truth_sample]
            filled += [False]
            labels += ["Truth"]
            colors += ["k"]
    try:
        g.triangle_plot(
            to_plot,
            params=paramnames,
            filled=filled,
            legend_labels=labels,
            contour_colors=colors,
        )
        g.export(os.path.join(output_folder, filename))
    except LinAlgError:
        # write mock file with traceback!
        pass


def metric_boxplot(
    values, output_folder, name=None, ref_values=None, filename="metric", ext=".png"
):
    """
    Creates a boxplot to show the distributions of a metric.

    Parameters
    ----------
    values: dict, list of dicts
        Dict ``{"d_1": [m_1, m_2, ...], "d_2": [m_1, m_2,...]}`` where ``d_i`` are the
        different distributions or experiments, and ``m_i`` are the metric values.

    ref_values: dict
        Dict ``{"d_1": "r_1", "d_2": r_2}`` where ``d_i`` are the different distributions
        or experiments, and ``r_i`` are the reference metric values.

    nam: str
        Name of the metric, to be used as the y axis label. Assumes latex, without ``$``.
    """
    fig = plt.figure()
    ax = fig.gca()
    ax.boxplot(list(values.values()))
    ax.set_xticklabels(list(values))
    ax.tick_params("x", rotation=45, rotation_mode="xtick")
    for i, dist in enumerate(values):
        ref = (ref_values or {}).get(dist)
        if ref is None or np.isnan(ref):
            continue
        ax.scatter(i + 1, ref, c="r", marker="*")
    if name is None or not name.startswith("log"):
        ax.set_yscale("log")
    if name is not None:
        ax.set_ylabel("$" + name + "$")
        filename += "_" + name
    filename += "." + ext.lstrip(".")
    # bbox_inches='tight' guarantees that the tick labels are not cropped.
    fig.savefig(os.path.join(output_folder, filename), bbox_inches="tight")
