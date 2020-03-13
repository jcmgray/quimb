"""Functionailty for drawing tensor networks.
"""

import numpy as np
from cytoolz import valmap


def graph(
    tn,
    color=None,
    show_inds=None,
    show_tags=None,
    node_size=None,
    iterations=200,
    k=None,
    fix=None,
    figsize=(6, 6),
    legend=True,
    return_fig=False,
    highlight_inds=(),
    initial_layout='spectral',
    edge_alpha=1 / 3,
    custom_colors=None,
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
    show_inds : bool, optional
        Explicitly turn on labels for each tensors indices.
    show_tags : bool, optional
        Explicitly turn on labels for each tensors tags.
    iterations : int, optional
        How many iterations to perform when when finding the best layout
        using node repulsion. Ramp this up if the graph is drawing messily.
    k : float, optional
        The optimal distance between nodes.
    fix : dict[tags, (float, float)], optional
        Used to specify actual relative positions for each tensor node.
        Each key should be a sequence of tags that uniquely identifies a
        tensor, and each value should be a x, y coordinate tuple.
    figsize : tuple of int
        The size of the drawing.
    legend : bool, optional
        Whether to draw a legend for the colored tags.
    node_size : None
        How big to draw the tensors.
    initial_layout : {'spectral', 'kamada_kawai', 'circular', 'planar',
                      'random', 'shell', 'bipartite', ...}, optional
        The name of a networkx layout to use before iterating with the
        spring layout. Set ``iterations=0`` if you just want to use this
        layout only.
    edge_alpha : float, optional
        Set the alpha (opacity) of the drawn edges.
    plot_opts
        Supplied to ``networkx.draw``.
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import math

    # build the graph
    G = nx.Graph()
    ts = list(tn.tensors)
    n = len(ts)

    if show_inds is None:
        show_inds = (n <= 20)
    if show_tags is None:
        show_tags = (n <= 20)

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
    fix_tids = dict()
    for tags_or_ind, pos in tuple(fix.items()):
        try:
            tid, = tn._get_tids_from_tags(tags_or_ind)
            fix_tids[tid] = pos
        except KeyError:
            # assume index
            fix_tids[f"ext{tags_or_ind}"] = pos

    labels = dict()
    fixed_positions = dict()

    for i, (tid, t1) in enumerate(tn.tensor_map.items()):

        if tid in fix_tids:
            fixed_positions[i] = fix_tids[tid]

        if not t1.inds:
            # is a scalar
            G.add_node(i)
            continue

        for ix in t1.inds:
            found_ind = False
            edge_color = ((1.0, 0.2, 0.2) if ix in highlight_inds else
                          (0.0, 0.0, 0.0))

            # check to see if index is linked to another tensor
            for j in range(0, n):
                if j == i:
                    continue

                t2 = ts[j]
                if ix in t2.inds:
                    found_ind = True
                    G.add_edge(i, j, weight=t1.shared_bond_size(t2),
                               color=edge_color)

            # else it must be an 'external' index
            if not found_ind:
                ext_lbl = f"ext{ix}"
                G.add_edge(i, ext_lbl, weight=t1.ind_size(ix),
                           color=edge_color)

                # optionally label the external index
                if show_inds:
                    labels[ext_lbl] = ix

                if ext_lbl in fix_tids:
                    fixed_positions[ext_lbl] = fix_tids[ext_lbl]

    edge_weights = [x[2]['weight'] for x in G.edges(data=True)]
    edge_colors = [x[2]['color'] for x in G.edges(data=True)]

    # color the nodes
    colors = get_colors(color, custom_colors)

    for i, t1 in enumerate(ts):
        G.nodes[i]['color'] = None
        for col_tag in colors:
            if col_tag in t1.tags:
                G.nodes[i]['color'] = colors[col_tag]
        # optionally label the tensor's tags
        if show_tags:
            labels[i] = str(t1.tags)

    # Set the size of the nodes, so that dangling inds appear so.
    # Also set the colors of any tagged tensors.
    if node_size is None:
        node_size = 1000 / n**0.7
    node_outline_size = min(3, node_size**0.5 / 5)

    szs = []
    node_colors = []
    node_outline_colors = []
    for nd in G.nodes():

        # 'node' is actually a open index
        if isinstance(nd, str):
            szs += [0]
            node_colors += [(1.0, 1.0, 1.0)]
        else:
            szs += [node_size]
            if G.nodes[nd]['color'] is not None:
                node_colors += [G.nodes[nd]['color']]
            else:
                node_colors += [(0.4, 0.4, 0.4)]

        node_outline_colors.append(
            tuple(0.8 * x for x in node_colors[-1])
        )

    edge_weights = [math.log2(d) for d in edge_weights]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.axis('off')
    ax.set_aspect('equal')

    # use spectral layout as starting point
    pos0 = getattr(nx, initial_layout + '_layout')(G, weight=None)
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
    pos = nx.spring_layout(G, pos=pos0, fixed=fixed,
                           k=k, iterations=iterations, weight=None)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=szs,
                           ax=ax, linewidths=node_outline_size,
                           edgecolors=node_outline_colors)
    nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                           alpha=edge_alpha, width=edge_weights, ax=ax)

    # create legend
    if colors and legend:
        handles = []
        for color in colors.values():
            handles += [plt.Line2D([0], [0], marker='o', color=color,
                                   linestyle='', markersize=10)]

        # needed in case '_' is the first character
        lbls = [f" {l}" for l in colors]

        plt.legend(handles, lbls, ncol=max(int(len(handles) / 20), 1),
                   loc='center left', bbox_to_anchor=(1, 0.5))

    if return_fig:
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
    return tuple(hsv_to_rgb((h, mod * s, v)))


def auto_colors(nc):
    import math
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list('wong', _COLORS_SORTED)

    xs = list(map(cmap, np.linspace(0, 1.0, nc)))

    # modulate color saturation with sine to generate local distinguishability
    # ... but only turn on gradually for increasing number of nodes
    sat_mod_period = min(4, nc / 7)
    sat_mod_factor = max(0.0, 2 / 3 * math.tanh((nc - 7) / 7))

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
