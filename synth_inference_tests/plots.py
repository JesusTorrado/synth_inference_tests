import os
from typing import Mapping, Sequence

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
    g.triangle_plot(
        to_plot,
        params=paramnames,
        filled=filled,
        legend_labels=labels,
        contour_colors=colors,
    )
    g.export(os.path.join(output_folder, filename))
