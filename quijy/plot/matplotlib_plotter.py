"""
Functions for plotting datasets nicely.
"""
# TODO: unify options
# TODO: matplotlib style hlines, vlines
# TODO: logcolor
from itertools import cycle
from collections import OrderedDict
from itertools import repeat
import numpy as np


# -------------------------------------------------------------------------- #
# Plots with matplotlib only                                                 #
# -------------------------------------------------------------------------- #

def mpl_markers():
    marker_dict = OrderedDict([
        ("o", "circle"),
        ("x", "x"),
        ("D", "diamond"),
        ("+", "plus"),
        ("s", "square"),
        (".", "point"),
        ("^", "triangle_up"),
        ("3", "tri_left"),
        (">", "triangle_right"),
        ("d", "thin_diamond"),
        ("*", "star"),
        ("v", "triangle_down"),
        ("|", "vline"),
        ("1", "tri_down"),
        ("p", "pentagon"),
        (",", "pixel"),
        ("2", "tri_up"),
        ("<", "triangle_left"),
        ("h", "hexagon1"),
        ("4", "tri_right"),
        (0, "tickleft"),
        (2, "tickup"),
        (3, "tickdown"),
        (4, "caretleft"),
        ("_", "hline"),
        (5, "caretright"),
        ("H", "hexagon2"),
        (1, "tickright"),
        (6, "caretup"),
        ("8", "octagon"),
        (7, "caretdown"),
    ])
    marker_keys = [*marker_dict.keys()]
    return marker_keys


def mplot(x, y_i, fignum=1, logx=False, logy=False,
          xlims=None, ylims=None, markers=True,
          color=False, colormap="viridis", **kwargs):
    """ Function for automatically plotting multiple sets of data
    using matplot lib. """
    import matplotlib.pyplot as plt
    y_i = np.array(np.squeeze(y_i), ndmin=2)
    if np.size(x) == y_i.shape[0]:
        y_i = np.transpose(y_i)
    n_y = y_i.shape[0]
    fig = plt.figure(fignum, figsize=(8, 6), dpi=100)
    axes = fig.add_axes([0.1, 0.1, 0.85, 0.8])

    if markers:
        mrkrs = cycle(mpl_markers())
    else:
        repeat(None)

    if color:
        from matplotlib import cm
        cmap = getattr(cm, colormap)
        cns = np.linspace(0, 1, n_y)
        cols = [cmap(cn, 1) for cn in cns]
    else:
        cols = repeat(None)

    for y, col, mrkr in zip(y_i, cols, mrkrs):
        axes.plot(x, y, ".-", c=col, marker=mrkr, **kwargs)
    xlims = (np.min(x), np.max(x)) if xlims is None else xlims
    ylims = (np.min(y_i), np.max(y_i)) if ylims is None else ylims
    axes.set_xlim(xlims)
    axes.set_ylim(ylims)
    axes.set_xscale("log" if logx else "linear")
    axes.set_yscale("log" if logy else "linear")
    return fig


# -------------------------------------------------------------------------- #
# Plots with matplotlib and xarray                                           #
# -------------------------------------------------------------------------- #

def xmlineplot(ds, y_coo, x_coo, z_coo,
               color=False, colormap="viridis", legend=None, markers=None,
               xlabel=None, xlims=None, xticks=None, logx=False,
               ylabel=None, ylims=None, yticks=None, logy=False,
               zlabel=None, padding=0.0, vlines=None, hlines=None,
               title=None, fignum=1, font='Roboto', **kwargs):
    """ Function for automatically plotting multiple sets of data
    using matplotlib and xarray. """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rc('font', family=font)
    fig = plt.figure(fignum, figsize=(8, 6), dpi=100)
    axes = fig.add_axes([0.15, 0.15, 0.8, 0.75],
                        title=("" if title is None else title))
    axes.tick_params(labelsize=16)
    n_z = len(ds[z_coo])
    n_y = len(ds[y_coo])
    if color:
        from matplotlib import cm
        cmap = getattr(cm, colormap)
        zmin, zmax = ds[z_coo].values.min(), ds[z_coo].values.max()
        cols = [cmap(1 - (z-zmin)/(zmax-zmin)) for z in ds[z_coo].values]
    else:
        cols = repeat(None)
    markers = (n_y <= 50) if markers is None else markers
    mrkrs = cycle(mpl_markers()) if markers else repeat(None)
    for z, col, mrkr in zip(ds[z_coo].data, cols, mrkrs):
        x = ds.loc[{z_coo: z}][x_coo].data.flatten()
        y = ds.loc[{z_coo: z}][y_coo].data.flatten()
        axes.plot(x, y, ".-", c=col, lw=1.3, marker=mrkr,
                  label=str(z), zorder=3, **kwargs)
    if xlims is None:
        xmax, xmin = ds[x_coo].max(), ds[x_coo].min()
        xrange = xmax - xmin
        xlims = (xmin - padding * xrange, xmax + padding * xrange)
    if ylims is None:
        ymax, ymin = ds[y_coo].max(), ds[y_coo].min()
        yrange = ymax - ymin
        ylims = (ymin - padding * yrange, ymax + padding * yrange)
    axes.set_xlim(xlims)
    axes.set_ylim(ylims)
    axes.grid(True, color="0.666")
    if vlines is not None:
        for x in vlines:
            axes.axvline(x)
    if hlines is not None:
        for y in hlines:
            axes.axhline(y)
    axes.set_xscale("log" if logx else "linear")
    axes.set_yscale("log" if logy else "linear")
    axes.set_xlabel(x_coo if xlabel is None else xlabel, fontsize=20)
    axes.set_ylabel(y_coo if ylabel is None else ylabel, fontsize=20)
    if xticks is not None:
        axes.set_xticks(xticks)
        axes.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    if yticks is not None:
        axes.set_yticks(yticks)
        axes.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    if legend or not (legend is False or n_z > 10):
        legend = axes.legend(title=(z_coo if zlabel is None else zlabel),
                             loc="best", fontsize=16, frameon=False)
        legend.get_title().set_fontsize(20)
    return fig
