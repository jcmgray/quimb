"""
Misc. functions not quantum related.
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import (atleast_2d, squeeze, array, shape, argwhere, linspace)
from tqdm import tqdm


def progbar(x, ascii=True, leave=True, **kwargs):
    """
    tqdm with changed defaults. Wraps any iterable and outputs progress bar
    with statistics.
    """
    return tqdm(x, ascii=ascii, leave=leave, **kwargs)


def ezplot(x, y_i, fignum=1, xlog=False, ylog=False, **kwargs):
    """
    Function for automatically plotting multiple sets of data
    """
    # TODO colormap data and legend
    y_i = atleast_2d(squeeze(y_i))
    dimsy = array(shape(y_i))
    xaxis = argwhere(len(x) == dimsy)[0]  # 0 or 1
    fig = plt.figure(fignum, figsize=(8, 6), dpi=100)
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    colors = linspace(0, 1, dimsy[1 - xaxis])

    for i in range(dimsy[xaxis - 1]):
        if xaxis:
            y = y_i[i, :]
        else:
            y = y_i[:, i]
        if xlog:
            axes.set_xscale("log")
        if ylog:
            axes.set_yscale("log")
        axes.plot(x, y, '.-', c=cm.viridis(colors[i], 1), **kwargs)
    return axes
