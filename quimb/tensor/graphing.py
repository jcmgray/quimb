"""Functionailty for drawing tensor networks.
"""

from ..utils import valmap

import numpy as np


def graph(
    tn,
    color=None,
    highlight_inds=(),
    show_inds=None,
    show_tags=None,
    custom_colors=None,
    legend=True,
    fix=None,
    k=None,
    iterations=200,
    initial_layout='spectral',
    node_size=None,
    edge_scale=1.0,
    edge_alpha=1 / 3,
    figsize=(6, 6),
    return_fig=False,
    ax=None,
    **plot_opts
):
    """Plot this tensor network as a networkx graph using matplotlib,
    with edge width corresponding to bond dimension.

    Parameters
    ----------
    color : sequence of tags, optional
        If given, uniquely color any tensors which have each of the tags.
        If some tensors have more than of the tags, only one color will
    highlight_inds : iterable:
        Highlight these edges in red.
    show_inds : {None, False, True, 'all'}, optional
        Explicitly turn on labels for each tensors indices.
    show_tags : {None, False, True}, optional
        Explicitly turn on labels for each tensors tags.
    custom_colors : sequence of colors, optional
        Supply a custom sequence of colors to match the tags given
        in ``color``.
    legend : bool, optional
        Whether to draw a legend for the colored tags.
    fix : dict[tags, (float, float)], optional
        Used to specify actual relative positions for each tensor node.
        Each key should be a sequence of tags that uniquely identifies a
        tensor, and each value should be a x, y coordinate tuple.
    k : float, optional
        The optimal distance between nodes.
    iterations : int, optional
        How many iterations to perform when when finding the best layout
        using node repulsion. Ramp this up if the graph is drawing messily.
    initial_layout : {'spectral', 'kamada_kawai', 'circular', 'planar',
                      'random', 'shell', 'bipartite', ...}, optional
        The name of a networkx layout to use before iterating with the
        spring layout. Set ``iterations=0`` if you just want to use this
        layout only.
    node_size : None
        How big to draw the tensors.
    edge_scale : float, optional
        How much to scale the width of the edges.
    edge_alpha : float, optional
        Set the alpha (opacity) of the drawn edges.
    figsize : tuple of int
        The size of the drawing.
    return_fig : bool, optional
        If True and ``ax is None`` then return the figure created rather than
        executing ``pyplot.show()``.
    ax : matplotlib.Axis, optional
        Draw the graph on this axis rather than creating a new figure.
    plot_opts
        Supplied to ``networkx.draw``.
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import math

    # automatically decide whether to show tags and inds
    if show_inds is None:
        show_inds = (tn.num_tensors <= 20)
    if show_tags is None:
        show_tags = (tn.num_tensors <= 20)

    hyperedges = []
    node_labels = dict()
    edge_labels = dict()

    # build the graph
    G = nx.Graph()

    for ix, tids in tn.ind_map.items():
        edge_attrs = {
            'color': ((1.0, 0.2, 0.2, 1.0) if ix in highlight_inds else
                      (0.0, 0.0, 0.0, 1.0)),
            'ind': ix,
            'edge_size': edge_scale * math.log2(tn.ind_size(ix))
        }
        if len(tids) == 2:
            # standard edge
            G.add_edge(*tids, **edge_attrs)
            if show_inds == 'all':
                edge_labels[tids] = ix
        else:
            # hyper or outer edge - needs dummy 'node' shown with zero size
            hyperedges.append(ix)
            for tid in tids:
                G.add_edge(tid, ix, **edge_attrs)

    # color the nodes
    colors = get_colors(color, custom_colors)

    # set the size of the nodes
    if node_size is None:
        node_size = 1000 / tn.num_tensors**0.7
    node_outline_size = min(3, node_size**0.5 / 5)

    # set parameters for all the nodes
    for tid, t in tn.tensor_map.items():
        if t.ndim == 0:
            continue

        G.nodes[tid]['size'] = node_size
        G.nodes[tid]['outline_size'] = node_outline_size
        color = (0.4, 0.4, 0.4, 1.0)
        for tag in colors:
            if tag in t.tags:
                color = colors[tag]
        G.nodes[tid]['color'] = color
        G.nodes[tid]['outline_color'] = tuple(0.8 * c for c in color)
        if show_tags:
            node_labels[tid] = '{' + str(list(t.tags))[1:-1] + '}'

    for hix in hyperedges:
        G.nodes[hix]['ind'] = hix
        G.nodes[hix]['color'] = (1.0, 1.0, 1.0, 1.0)
        G.nodes[hix]['size'] = 0.0
        G.nodes[hix]['outline_size'] = 0.0
        G.nodes[hix]['outline_color'] = (1.0, 1.0, 1.0, 1.0)
        if show_inds == 'all':
            node_labels[hix] = hix

    if show_inds:
        for oix in tn.outer_inds():
            node_labels[oix] = oix

    pos = _get_positions(tn, G, fix, initial_layout, k, iterations)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.axis('off')
        ax.set_aspect('equal')

    nx.draw_networkx_edges(
        G, pos,
        width=tuple(x[2]['edge_size'] for x in G.edges(data=True)),
        edge_color=tuple(x[2]['color'] for x in G.edges(data=True)),
        alpha=edge_alpha,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        G, pos,
        node_color=tuple(x[1]['color'] for x in G.nodes(data=True)),
        edgecolors=tuple(x[1]['outline_color'] for x in G.nodes(data=True)),
        node_size=tuple(x[1]['size'] for x in G.nodes(data=True)),
        linewidths=tuple(x[1]['outline_size'] for x in G.nodes(data=True)),
        ax=ax,
    )
    if show_inds == 'all':
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=10,
            ax=ax,
        )
    if show_tags or show_inds:
        nx.draw_networkx_labels(
            G, pos,
            labels=node_labels,
            font_size=10,
            ax=ax,
        )

    # create legend
    if colors and legend:
        handles = []
        for color in colors.values():
            handles += [plt.Line2D([0], [0], marker='o', color=color,
                                   linestyle='', markersize=10)]

        # needed in case '_' is the first character
        lbls = [f" {lbl}" for lbl in colors]

        plt.legend(handles, lbls, ncol=max(round(len(handles) / 20), 1),
                   loc='center left', bbox_to_anchor=(1, 0.5))

    if ax is not None:
        return
    elif return_fig:
        return fig
    else:
        plt.show()


# colorblind palettes by Bang Wong (https://www.nature.com/articles/nmeth.1618)

_COLORS_DEFAULT = (
    '#56B4E9',  # light blue
    '#E69F00',  # orange
    '#009E73',  # green
    '#D55E00',  # red
    '#F0E442',  # yellow
    '#CC79A7',  # purple
    '#0072B2',  # dark blue
)

_COLORS_SORTED = (
    '#0072B2',  # dark blue
    '#56B4E9',  # light blue
    '#009E73',  # green
    '#F0E442',  # yellow
    '#E69F00',  # orange
    '#D55E00',  # red
    '#CC79A7',  # purple
)


def mod_sat(c, mod):
    """Modify the luminosity of rgb color ``c``.
    """
    from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

    h, s, v = rgb_to_hsv(c[:3])
    return (*hsv_to_rgb((h, mod * s, v)), 1.0)


def auto_colors(nc):
    import math
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list('wong', _COLORS_SORTED)

    xs = list(map(cmap, np.linspace(0, 1.0, nc)))

    # modulate color saturation with sine to generate local distinguishability
    # ... but only turn on gradually for increasing number of nodes
    sat_mod_period = min(4, nc / 7)
    sat_mod_factor = max(0.0, 2 / 3 * math.tanh((nc - 7) / 4))

    return [
        mod_sat(
            c, 1 - sat_mod_factor * math.sin(math.pi * i / sat_mod_period)**2
        )
        for i, c in enumerate(xs)
    ]


def get_colors(color, custom_colors=None):
    """Generate a sequence of rgbs for tag(s) ``color``.
    """
    from matplotlib.colors import to_rgba

    if color is None:
        return dict()

    if isinstance(color, str):
        color = (color,)

    if custom_colors is not None:
        rgbs = list(map(to_rgba, custom_colors))
        return dict(zip(color, rgbs))

    nc = len(color)
    if nc <= 7:
        return dict(zip(color, list(map(to_rgba, _COLORS_DEFAULT))))

    rgbs = auto_colors(nc)
    return dict(zip(color, rgbs))


def _get_positions(tn, G, fix, initial_layout, k, iterations):
    import networkx as nx

    if fix is None:
        fix = dict()
    else:
        # find range with which to scale spectral points with
        xmin, xmax, ymin, ymax = (
            f(fix.values(), key=lambda xy: xy[i])[i]
            for f, i in [(min, 0), (max, 0), (min, 1), (max, 1)])
        if xmin == xmax:
            xmin, xmax = xmin - 1, xmax + 1
        if ymin == ymax:
            ymin, ymax = ymin - 1, ymax + 1
        xymin, xymax = min(xmin, ymin), max(xmax, ymax)

    # identify tensors by tid
    fixed_positions = dict()
    for tags_or_ind, pos in tuple(fix.items()):
        try:
            tid, = tn._get_tids_from_tags(tags_or_ind)
            fixed_positions[tid] = pos
        except KeyError:
            # assume index
            fixed_positions[tags_or_ind] = pos

    # use spectral layout as starting point
    pos0 = getattr(nx, initial_layout + '_layout')(G)

    # scale points to fit with specified positions
    if fix:
        # but update with fixed positions
        pos0.update(valmap(lambda xy: np.array(
            (2 * (xy[0] - xymin) / (xymax - xymin) - 1,
             2 * (xy[1] - xymin) / (xymax - xymin) - 1)), fixed_positions))
        fixed = fixed_positions.keys()
    else:
        fixed = None

    # and then relax remaining using spring layout
    pos = nx.spring_layout(
        G, pos=pos0, fixed=fixed, k=k, iterations=iterations)

    return pos
