"""Functionailty for drawing tensor networks.
"""
import textwrap

import numpy as np

from ..utils import valmap


def _add_or_merge_edge(G, u, v, attrs):
    if not G.has_edge(u, v):
        G.add_edge(u, v, **attrs)
    else:
        # multibond - update attrs
        attrs0 = G.edges[u, v]
        # average colors
        attrs0['color'] = tuple(
            (x + y) / 2 for x, y in zip(attrs0['color'], attrs['color']))
        attrs0['ind'] += ' ' + attrs['ind']
        # adding log size == multiplying bond dim
        attrs0['edge_size'] += attrs['edge_size']


def draw_tn(
    tn,
    color=None,
    *,
    highlight_inds=(),
    highlight_tids=(),
    highlight_inds_color=(1.0, 0.2, 0.2, 1.0),
    highlight_tids_color=(1.0, 0.2, 0.2, 1.0),
    show_inds=None,
    show_tags=None,
    show_scalars=True,
    custom_colors=None,
    title=None,
    legend=True,
    fix=None,
    k=None,
    iterations=200,
    initial_layout='spectral',
    node_color=None,
    outline_darkness=0.8,
    node_size=None,
    edge_color=None,
    edge_scale=1.0,
    edge_alpha=1 / 2,
    label_color=None,
    font_size=10,
    font_size_inner=7,
    figsize=(6, 6),
    margin=None,
    xlims=None,
    ylims=None,
    get=None,
    return_fig=False,
    ax=None,
):
    """Plot this tensor network as a networkx graph using matplotlib,
    with edge width corresponding to bond dimension.

    Parameters
    ----------
    color : sequence of tags, optional
        If given, uniquely color any tensors which have each of the tags.
        If some tensors have more than of the tags, only one color will show.
    highlight_inds : iterable, optional
        Highlight these edges.
    highlight_tids : iterable, optional
        Highlight these nodes.
    highlight_inds_color
        What color to use for ``highlight_inds`` nodes.
    highlight_tids_color : tuple[float], optional
        What color to use for ``highlight_tids`` nodes.
    show_inds : {None, False, True, 'all'}, optional
        Explicitly turn on labels for each tensors indices.
    show_tags : {None, False, True}, optional
        Explicitly turn on labels for each tensors tags.
    show_scalars : bool, optional
        Whether to show scalar tensors (floating nodes with no edges).
    custom_colors : sequence of colors, optional
        Supply a custom sequence of colors to match the tags given
        in ``color``.
    title : str, optional
        Set a title for the axis.
    legend : bool, optional
        Whether to draw a legend for the colored tags.
    fix : dict[tags, (float, float)], optional
        Used to specify actual relative positions for each tensor node.
        Each key should be a sequence of tags that uniquely identifies a
        tensor, a ``tid``, or a ``ind``, and each value should be a ``(x, y)``
        coordinate tuple.
    k : float, optional
        The optimal distance between nodes.
    iterations : int, optional
        How many iterations to perform when when finding the best layout
        using node repulsion. Ramp this up if the graph is drawing messily.
    initial_layout : {'spectral', 'kamada_kawai', 'circular', 'planar', \\
                      'random', 'shell', 'bipartite', ...}, optional
        The name of a networkx layout to use before iterating with the
        spring layout. Set ``iterations=0`` if you just want to use this
        layout only.
    node_color : tuple[float], optional
        Default color of nodes.
    outline_darkness : float, optional
        Darkening of nodes outlines.
    node_size : None
        How big to draw the tensors.
    edge_color : tuple[float], optional
        Default color of edges.
    edge_scale : float, optional
        How much to scale the width of the edges.
    edge_alpha : float, optional
        Set the alpha (opacity) of the drawn edges.
    label_color : tuple[float], optional
        Color to draw labels with.
    font_size : int, optional
        Font size for drawing tags and outer indices.
    font_size_inner : int, optional
        Font size for drawing inner indices.
    figsize : tuple of int
        The size of the drawing.
    margin : None or float, optional
        Specify an argument for ``ax.margin``, else the plot limits will try
        and be computed based on the node positions and node sizes.
    xlims : None or tuple, optional
        Explicitly set the x plot range.
    xlims : None or tuple, optional
        Explicitly set the y plot range.
    get : {Nonr, 'pos'}, optional
        If ``'pos'``, return the plotting positions of each ``tid`` and ``ind``
        drawn as a node, this can supplied to subsequent calls as ``fix=pos``
        to maintain positions, even as the graph structure changes.
    return_fig : bool, optional
        If True and ``ax is None`` then return the figure created rather than
        executing ``pyplot.show()``.
    ax : matplotlib.Axis, optional
        Draw the graph on this axis rather than creating a new figure.
    """
    import networkx as nx
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    import math

    # automatically decide whether to show tags and inds
    if show_inds is None:
        show_inds = (tn.num_tensors <= 20)
    if show_tags is None:
        show_tags = (tn.num_tensors <= 20)

    isdark = sum(to_rgb(mpl.rcParams['figure.facecolor'])) / 3 < 0.5
    if isdark:
        draw_color = (0.75, 0.77, 0.80, 1.0)
    else:
        draw_color = (0.45, 0.47, 0.50, 1.0)

    if edge_color is None:
        edge_color = draw_color

    if node_color is None:
        node_color = draw_color

    # set the size of the nodes
    if node_size is None:
        node_size = 1000 / tn.num_tensors**0.7
    node_outline_size = min(3, node_size**0.5 / 5)

    if label_color is None:
        label_color = mpl.rcParams['axes.labelcolor']

    # build the graph
    G = nx.Graph()

    hyperedges = []
    node_labels = dict()
    edge_labels = dict()

    for ix, tids in tn.ind_map.items():
        edge_attrs = {
            'color': (highlight_inds_color if ix in highlight_inds else
                      edge_color),
            'ind': ix,
            'edge_size': edge_scale * math.log2(tn.ind_size(ix))
        }
        if len(tids) == 2:
            # standard edge
            _add_or_merge_edge(G, *tids, edge_attrs)
            if show_inds == 'all':
                edge_labels[tuple(tids)] = ix
        else:
            # hyper or outer edge - needs dummy 'node' shown with zero size
            hyperedges.append(ix)
            for tid in tids:
                _add_or_merge_edge(G, tid, ix, edge_attrs)

    # color the nodes
    colors = get_colors(color, custom_colors)

    # set parameters for all the nodes
    for tid, t in tn.tensor_map.items():

        if tid not in G.nodes:
            # e.g. tensor is a scalar
            if show_scalars:
                G.add_node(tid)
            else:
                continue

        G.nodes[tid]['size'] = node_size
        G.nodes[tid]['outline_size'] = node_outline_size
        color = node_color
        for tag in colors:
            if tag in t.tags:
                color = colors[tag]
        if tid in highlight_tids:
            color = highlight_tids_color
        G.nodes[tid]['color'] = color
        G.nodes[tid]['outline_color'] = tuple(
            (1.0 if i == 3 else outline_darkness) * c
            for i, c in enumerate(color)
        )
        if show_tags:
            # make the tags appear with auto vertical extend
            node_label = '{' + str(list(t.tags))[1:-1] + '}'
            node_labels[tid] = "\n".join(textwrap.wrap(
                node_label, max(2 * len(node_label) ** 0.5, 16)
            ))

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

    if get == 'pos':
        return pos

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.axis('off')
        ax.set_aspect('equal')
        if title is not None:
            ax.set_title(str(title))

        xmin = ymin = +float('inf')
        xmax = ymax = -float('inf')
        for xy in pos.values():
            xmin = min(xmin, xy[0])
            xmax = max(xmax, xy[0])
            ymin = min(ymin, xy[1])
            ymax = max(ymax, xy[1])

        if margin is None:
            # XXX: pad the plot range so that node circles are not clipped,
            #     using the networkx node_size parameter, *which is in absolute
            #     units* and so must be inverse transformed using matplotlib!
            inv = ax.transData.inverted()
            real_node_size = (abs(
                inv.transform((0, node_size))[1] -
                inv.transform((0, 0))[1]
            ) ** 0.5) / 4
            ax.set_xlim(xmin - real_node_size, xmax + real_node_size)
            ax.set_ylim(ymin - real_node_size, ymax + real_node_size)
        else:
            ax.margins(margin)

        created_fig = True
    else:
        created_fig = False

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
            font_size=font_size_inner,
            font_color=label_color,
            ax=ax,
        )
    if show_tags or show_inds:
        nx.draw_networkx_labels(
            G, pos,
            labels=node_labels,
            font_size=font_size,
            font_color=label_color,
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

    if not created_fig:
        # we added to axisting axes
        return

    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    if return_fig:
        return fig
    else:
        plt.show()
        plt.close(fig)


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


def _rotate(xy, theta):
    """Return a rotated set of points.
    """
    s = np.sin(theta)
    c = np.cos(theta)

    xyr = np.empty_like(xy)
    xyr[:, 0] = c * xy[:, 0] - s * xy[:, 1]
    xyr[:, 1] = s * xy[:, 0] + c * xy[:, 1]

    return xyr


def _span(xy):
    """Return the vertical span of the points.
    """
    return xy[:, 1].max() - xy[:, 1].min()


def _massage_pos(pos, nangles=360, flatten=False):
    """Rotate a position dict's points to cover a small vertical span
    """
    xy = np.empty((len(pos), 2))
    for i, (x, y) in enumerate(pos.values()):
        xy[i, 0] = x
        xy[i, 1] = y

    thetas = np.linspace(0, 2 * np.pi, nangles, endpoint=False)
    rxys = (_rotate(xy, theta) for theta in thetas)
    rxy0 = min(rxys, key=lambda rxy: _span(rxy))

    if flatten:
        rxy0[:, 1] /= 2

    return dict(zip(pos, rxy0))


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
            if (tags_or_ind in tn.tensor_map) or (tags_or_ind in tn.ind_map):
                fixed_positions[tags_or_ind] = pos

    if all(node in fix for node in G.nodes):
        # everything is already fixed
        return fix

    # use spectral or other layout as starting point
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

    if not fix:
        # finally rotate them to cover a small vertical span
        pos = _massage_pos(pos)

    return pos
