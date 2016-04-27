"""
Functions for plotting datasets nicely.
"""
# TODO: unify options
# TODO: plotly hlines, vlines
from math import log
from itertools import repeat
import numpy as np


def ishow(figs, nb=True, **kwargs):
    """ Show multiple plotly figures in notebook or on web. """
    if isinstance(figs, (list, tuple)):
        fig_main = figs[0]
        for fig in figs[1:]:
            fig_main['data'] += fig['data']
    else:
        fig_main = figs
    if nb:
        from plotly.offline import init_notebook_mode
        from plotly.offline import iplot as plot
        init_notebook_mode()
    else:
        from plotly.plotly import plot
    plot(fig_main, **kwargs)


# -------------------------------------------------------------------------- #
# Plots with plotly only                                                     #
# -------------------------------------------------------------------------- #

def iplot(x, y_i, logx=False, logy=False, color=False, colormap="viridis",
          return_fig=False, nb=True, go_dict={}, ly_dict={},
          **kwargs):
    """ Multi line plot with plotly. """
    from plotly.graph_objs import Scatter
    y_i = np.array(np.squeeze(y_i), ndmin=2)
    if np.size(x) == y_i.shape[0]:
        y_i = np.transpose(y_i)
    n_y = y_i.shape[0]
    if color:
        import matplotlib.cm as cm
        cmap = getattr(cm, colormap)
        cns = np.linspace(0, 1, n_y)
        cols = ["rgba" + str(cmap(cn)) for cn in cns]
    else:
        cols = repeat(None)
    traces = [Scatter({"x": x, "y": y, "line": {"color": col},
                       "marker": {"color": col},  **go_dict})
              for y, col in zip(y_i, cols)]
    layout = {"width": 750, "height": 600, "showlegend": False,
              "xaxis": {"showline": True, "mirror": "ticks",
                        "ticks": "inside",
                        "type": "log" if logx else "linear"},
              "yaxis": {"showline": True, "mirror": "ticks",
                        "ticks": "inside",
                        "type": "log" if logy else "linear"}, **ly_dict}
    fig = {"data": traces, "layout": layout}
    if return_fig:
        return fig
    ishow(fig, nb=nb, **kwargs)


def iscatter(x, y, cols=None, logx=False, logy=False, nb=True,
             return_fig=False, ly_dict={}, **kwargs):
    from plotly.graph_objs import Scatter, Marker
    mkr = Marker({"color": cols, "opacity": 0.9,
                  "colorscale": "Portland", "showscale": cols is not None})
    traces = [Scatter({"x": x, "y": y, "mode": "markers", "marker": mkr})]
    layout = {"width": 700, "height": 700, "showlegend": False,
              "xaxis": {"showline": True, "mirror": "ticks",
                        "ticks": "inside",
                        "type": "log" if logx else "linear"},
              "yaxis": {"showline": True, "mirror": "ticks",
                        "ticks": "inside",
                        "type": "log" if logy else "linear"}, **ly_dict}
    fig = {"data": traces, "layout": layout}
    if return_fig:
        return fig
    ishow(fig, nb=nb, **kwargs)


def ihist(xs, nb=True, go_dict={}, ly_dict={}, return_fig=False,
          **kwargs):
    """ Histogram plot with plotly. """
    from plotly.graph_objs import Histogram
    traces = [Histogram({"x": x, **go_dict}) for x in xs]
    layout = {"width": 750, "height": 600,
              "xaxis": {"showline": True, "mirror": "ticks",
                        "ticks": "inside"},
              "yaxis": {"showline": True, "mirror": "ticks",
                        "ticks": "inside"}, **ly_dict}
    fig = {"data": traces, "layout": layout}
    if return_fig:
        return fig
    ishow(fig, nb=nb, **kwargs)


# -------------------------------------------------------------------------- #
# Plots with plotly and xarray                                               #
# -------------------------------------------------------------------------- #

def iheatmap(ds, data_name, x_coo, y_coo, colormap="Portland",
             go_dict={}, ly_dict={}, nb=True, return_fig=False,
             **kwargs):
    """
    Automatic 2D-Heatmap plot using plotly.
    """
    from plotly.graph_objs import Heatmap
    traces = Heatmap({"z": (ds[data_name]
                            .dropna(x_coo, how="all")
                            .dropna(y_coo, how="all")
                            .squeeze()
                            .transpose(y_coo, x_coo)
                            .data),
                      "x": ds.coords[x_coo].values,
                      "y": ds.coords[y_coo].values,
                      "colorscale": colormap,
                      "colorbar": {"title": data_name}, **go_dict})
    layout = {"height": 600, "width": 650,
              "xaxis": {"showline": True, "mirror": "ticks",
                        "ticks": "outside", "title": x_coo},
              "yaxis": {"showline": True, "mirror": "ticks",
                        "ticks": "outside", "title": y_coo}, **ly_dict}
    fig = {"data": traces, "layout": layout}
    if return_fig:
        return fig
    ishow(fig, nb=nb, **kwargs)


def ilineplot(ds, data_name, x_coo, z_coo=None, logx=False, logy=False,
              erry=None, errx=None, nb=True, color=False, colormap="viridis",
              log_color=False, legend=None, traces=[], go_dict={},
              ly_dict={}, return_fig=False, **kwargs):
    # TODO: add hlines, vlines, xlims, ylims, title
    from plotly.graph_objs import Scatter
    if z_coo is None:
        traces = [Scatter({
                    "x": ds[x_coo].values,
                    "y": ds[data_name].values.flatten(), **go_dict})]
    else:
        cols = (calc_colors(ds, z_coo, colormap=colormap, log_scale=log_color)
                if color else repeat(None))
        traces = [Scatter({
                    "x": ds.loc[{z_coo: z}][x_coo].values.flatten(),
                    "y": ds.loc[{z_coo: z}][data_name].values.flatten(),
                    "name": str(z), "line": {"color": col},
                    "marker": {"color": col}, **go_dict})
                  for z, col in zip(ds[z_coo].values, cols)]
    layout = {"width": 750, "height": 600,
              "xaxis": {"showline": True, "title": x_coo,
                        "mirror": "ticks", "ticks": "inside",
                        "type": "log" if logx else "linear"},
              "yaxis": {"showline": True, "title": data_name,
                        "mirror": "ticks", "ticks": "inside",
                        "type": "log" if logy else "linear"},
              "showlegend": legend or not (legend is False or z_coo is None or
                                           len(ds[z_coo]) > 20), **ly_dict}
    fig = {"data": traces, "layout": layout}
    if return_fig:
        return fig
    ishow(fig, nb=nb, **kwargs)


def calc_colors(ds, z_coo, colormap="viridis", log_scale=False):
    import matplotlib.cm as cm
    cmap = getattr(cm, colormap)
    zmin = ds[z_coo].values.min()
    zmax = ds[z_coo].values.max()
    if log_scale:
        cols = ["rgba" + str(cmap(1 -
                                  (log(z)-log(zmin))/(log(zmax)-log(zmin))))
                for z in ds[z_coo].values]
    else:
        cols = ["rgba" + str(cmap(1 - (z - zmin) / (zmax - zmin)))
                for z in ds[z_coo].values]
    return cols
