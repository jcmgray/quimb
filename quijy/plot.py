"""
Functions for plotting datasets nicely.
"""
from itertools import cycle
import numpy as np


# -------------------------------------------------------------------------- #
# Plots with matplotlib                                                      #
# -------------------------------------------------------------------------- #

def mplot(x, y_i, fignum=1, logx=False, logy=False, **kwargs):
    """
    Function for automatically plotting multiple sets of data
    using matplot lib
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
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
        if logx:
            axes.set_xscale("log")
        if logy:
            axes.set_yscale("log")
        axes.plot(x, y, '.-', c=cm.viridis(colors[i], 1), **kwargs)
    return fig


# -------------------------------------------------------------------------- #
# Plots with matplotlib and xarray                                           #
# -------------------------------------------------------------------------- #

def xmlineplot(ds, y_coo, x_coo, z_coo, title=None, legend=None,
               xlabel=None, ylabel=None, zlabel=None,
               vlines=None, hlines=None, colormap='viridis',
               fignum=1, logx=False, logy=False, **kwargs):
    """
    Function for automatically plotting multiple sets of data
    using matplotlib and xarray.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    cmap = getattr(cm, colormap)
    fig = plt.figure(fignum, figsize=(8, 6), dpi=100)
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8],
                        title=('' if title is None else title))
    cols = np.linspace(0, 1, len(ds[z_coo]))
    for z, col in zip(ds[z_coo].data, cols):
        x = ds.loc[{z_coo: z}][x_coo].data.flatten()
        y = ds.loc[{z_coo: z}][y_coo].data.flatten()
        if logx:
            axes.set_xscale("log")
        if logy:
            axes.set_yscale("log")
        axes.plot(x, y, '.-', c=cmap(col, 1), lw=1.3,
                  label=str(z), zorder=3, **kwargs)
    if vlines is not None:
        for x in vlines:
            axes.axvline(x)
    if hlines is not None:
        for y in hlines:
            axes.axhline(y)
    axes.grid(True, color='0.666')
    axes.set_xlim((ds[x_coo].min(), ds[x_coo].max()))
    axes.set_ylim((ds[y_coo].min(), ds[y_coo].max()))
    axes.set_xlabel(x_coo if xlabel is None else xlabel)
    axes.set_ylabel(y_coo if ylabel is None else ylabel)
    if legend or not (legend is False or len(ds[z_coo]) > 20):
        axes.legend(title=(z_coo if zlabel is None else zlabel), loc='best')
    return fig


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
    xaxis = np.argwhere(np.size(x[0]) == ydims).flat[0]  # 0 or 1
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


def ilineplot(ds, data_name, x_coo, z_coo=None, logx=False, logy=False,
              erry=None, errx=None, nb=True, color=False, colormap='viridis',
              legend=None, traces=[], go_dict={}, ly_dict={}, **kwargs):
    from plotly.graph_objs import Scatter

    if z_coo is None:
        traces = [Scatter({
                    'x': ds[x_coo].values,
                    'y': ds[data_name].values.flatten(),
                    **go_dict})]
    else:
        if color:
            import matplotlib.cm as cm
            cmap = getattr(cm, colormap)
            zmin = ds[z_coo].values.min()
            zmax = ds[z_coo].values.max()
            cols = ["rgba" + str(cmap(1 - (z-zmin)/(zmax-zmin)))
                    for z in ds[z_coo].values]
        else:
            cols = [None for z in ds[z_coo].values]

        traces = [Scatter({
                    'x': ds.loc[{z_coo: z}][x_coo].values.flatten(),
                    'y': ds.loc[{z_coo: z}][data_name].values.flatten(),
                    'name': str(z),
                    'line': {"color": col},
                    'marker': {"color": col},
                    **go_dict})
                  for z, col in zip(ds[z_coo].values, cols)]
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
              'showlegend': legend or not (legend is False or
                                           len(ds[z_coo]) > 20),
              **ly_dict}
    fig = {"data": traces, "layout": layout}
    if nb:
        from plotly.offline import init_notebook_mode
        from plotly.offline import iplot as plot
        init_notebook_mode()
    else:
        from plotly.plotly import plot
    plot(fig, **kwargs)
