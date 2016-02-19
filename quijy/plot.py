"""
Functions for plotting datasets nicely.
"""
from itertools import cycle
import numpy as np


# -------------------------------------------------------------------------- #
# Plots with matplotlib                                                      #
# -------------------------------------------------------------------------- #

def mplot(x, y_i, fignum=1, xlog=False, ylog=False, **kwargs):
    from matplotlib import cm
    """
    Function for automatically plotting multiple sets of data
    using matplot lib
    """
    import matplotlib.pyplot as plt
    # TODO colormap data and legend
    y_i = np.array(np.squeeze(y_i), ndmin=2)
    dimsy = np.array(y_i.shape)
    xaxis = np.argwhere(np.size(x) == dimsy)[0, 0]  # 0 or 1
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


# -------------------------------------------------------------------------- #
# Plots with plotly                                                          #
# -------------------------------------------------------------------------- #

def iplot(x, y_i, name=None, color='viridis', nb=True,
          go_dict={}, ly_dict={}, **kwargs):
    """
    Multi line plot with plotly.
    """
    # TODO: name data, log scale
    from plotly.graph_objs import Scatter
    import matplotlib.cm as cm
    # Parse data
    y_i = np.array(np.squeeze(y_i), ndmin=2)
    ydims = np.array(y_i.shape)
    xaxis = np.argwhere(np.size(x) == ydims).flat[0]  # 0 or 1
    if xaxis == 1:
        y_i = y_i.transpose()
    x = cycle(x)
    n_y = ydims[xaxis - 1]
    cmap = getattr(cm, color)
    cols = ["rgba" + str(cmap(i / (n_y - 1))) for i in range(n_y)]
    traces = [Scatter({'x': next(x),
                       'y': y_i[:, i],
                       'name': (name[i] if name is not None else None),
                       'line': {"color": col},
                       'marker': {"color": col},
                       **go_dict})
              for i, col in enumerate(cols)]
    layout = {"width": 750,
              "height": 600,
              "xaxis": {"showline": True,
                        "mirror": "ticks",
                        "ticks": "inside"},
              "yaxis": {"showline": True,
                        "mirror": "ticks",
                        "ticks": "inside"},
              **ly_dict}
    fig = {"data": traces, "layout": layout}
    if nb:
        from plotly.offline import init_notebook_mode
        from plotly.offline import iplot as plot
        init_notebook_mode()
    else:
        from plotly.plotly import plot
    plot(fig, **kwargs)


def ihist(xs, nb=True, go_dict={}, ly_dict={}, **kwargs):
    """
    Multi histogram plot with plotly.
    """
    # TODO: name data, log scale
    from plotly.graph_objs import Histogram

    traces = [Histogram({'x': x,
                        **go_dict}) for x in xs]
    layout = {"width": 750,
              "height": 600,
              "xaxis": {"showline": True,
                        "mirror": "ticks",
                        "ticks": "inside"},
              "yaxis": {"showline": True,
                        "mirror": "ticks",
                        "ticks": "inside"},
              **ly_dict}
    fig = {"data": traces, "layout": layout}
    if nb:
        from plotly.offline import init_notebook_mode
        from plotly.offline import iplot as plot
        init_notebook_mode()
    else:
        from plotly.plotly import plot
    plot(fig, **kwargs)


# -------------------------------------------------------------------------- #
# Plots with plotly and xarray                                               #
# -------------------------------------------------------------------------- #

def iheatmap(ds, data_name, x_coo, y_coo, colormap='Portland',
             go_dict={}, ly_dict={}, nb=True, **kwargs):
    """
    Automatic 2D-Heatmap plot using plotly.
    """
    from plotly.graph_objs import Heatmap
    hm = Heatmap({'z': (ds[data_name]
                        .dropna(x_coo, how='all')
                        .dropna(y_coo, how='all')
                        .squeeze()
                        .transpose(y_coo, x_coo)
                        .data),
                  'x': ds.coords[x_coo].values,
                  'y': ds.coords[y_coo].values,
                  'colorscale': colormap,
                  'colorbar': {'title': data_name},
                  **go_dict})
    ly = {'height': 600,
          'width': 650,
          "xaxis": {"showline": True,
                    "mirror": "ticks",
                    "ticks": "outside",
                    "title": x_coo},
          "yaxis": {"showline": True,
                    "mirror": "ticks",
                    "ticks": "outside",
                    "title": y_coo},
          **ly_dict}
    fig = {'data': [hm],
           'layout': ly}
    if nb:
        from plotly.offline import init_notebook_mode, iplot
        init_notebook_mode()
        iplot(fig, **kwargs)
    else:
        from plotly.plotly import plot
        plot(fig, **kwargs)


def ilineplot(ds, data_name, x_coo, y_coo=None, logx=False, logy=False,
              erry=None, errx=None, go_dict={}, ly_dict={}, nb=True,
              color=False, colormap='Spectral', **kwargs):
    # TODO: generate traces from multiple data_names
    from plotly.graph_objs import Scatter

    if y_coo is None:
        traces = [Scatter({
                    'x': ds[x_coo].values,
                    'y': ds[data_name].values.flatten(),
                    **go_dict})]
    else:
        if color:
            import matplotlib.cm as cm
            cmap = getattr(cm, colormap)
            ymin = ds[y_coo].values.min()
            ymax = ds[y_coo].values.max()
            cols = ["rgba" + str(cmap(1 - (y-ymin)/(ymax-ymin)))
                    for y in ds[y_coo].values]
        else:
            cols = [None for y in ds[y_coo].values]

        traces = [Scatter({
                    'x': ds[x_coo].values,
                    'y': ds[data_name].loc[{y_coo: y}].values.flatten(),
                    'name': str(y),
                    'line': {"color": col},
                    'marker': {"color": col},
                    **go_dict})
                  for y, col in zip(ds[y_coo].values, cols)]
    layout = {"width": 750,
              "height": 600,
              "xaxis": {"showline": True,
                        "mirror": "ticks",
                        "ticks": "inside",
                        "title": x_coo,
                        "type": "log" if logx else "linear"},
              "yaxis": {"showline": True,
                        "mirror": "ticks",
                        "ticks": "inside",
                        "title": data_name,
                        "type": "log" if logy else "linear"},
              **ly_dict}
    fig = {"data": traces, "layout": layout}
    if nb:
        from plotly.offline import init_notebook_mode
        from plotly.offline import iplot as plot
        init_notebook_mode()
    else:
        from plotly.plotly import plot
    plot(fig, **kwargs)
