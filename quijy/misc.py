"""
Misc. functions not quantum related.
"""

import numpy as np
from tqdm import tqdm


def progbar(x, ascii=True, leave=True, **kwargs):
    """
    tqdm with changed defaults. Wraps any iterable and outputs progress bar
    with statistics.
    """
    return tqdm(x, ascii=ascii, leave=leave, **kwargs)


def mplot(x, y_i, fignum=1, xlog=False, ylog=False, **kwargs):
    from matplotlib import cm
    """
    Function for automatically plotting multiple sets of data
    """
    import matplotlib.pyplot as plt
    # TODO colormap data and legend
    y_i = np.array(np.squeeze(y_i), ndmin=2)
    dimsy = np.array(y_i.shape)
    xaxis = np.argwhere(np.size(x) == dimsy)[0]  # 0 or 1
    fig = plt.figure(fignum, figsize=(8, 6), dpi=100)
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    colors = np.linspace(0, 1, dimsy[1 - xaxis])

    for i in range(dimsy[xaxis - 1]):
        if xaxis:
            y = y_i[i, :]
        else:
            y = y_i[:, i]
        if xlog:
            axes.set_xscale("log")
        if ylog:
            axes.set_yscale("log")
        axes.plot(x, y, '.-', c=cm.plasma(colors[i], 1), **kwargs)
    return axes


def iplot(x, y_i, xlog=False, ylog=False):
    import plotly.graph_objs as go
    from plotly.offline import init_notebook_mode, iplot
    init_notebook_mode()

    # Parse data
    y_i = np.array(np.squeeze(y_i), ndmin=2)
    ydims = np.array(y_i.shape)
    xaxis = np.argwhere(np.size(x) == ydims)[0]  # 0 or 1
    if xaxis == 1:
        y_i = y_i.transpose()

    traces = [go.Scatter(x=x,
                         y=y_i[:, i],
                         mode='lines+markers')
              for i in range(ydims[xaxis - 1])]
    layout = go.Layout(width=900, height=600)
    fig = {'data': traces, 'layout': layout}
    iplot(fig)
