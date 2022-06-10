"""Functionailty for drawing tensor networks.
"""
import textwrap
import importlib
import collections

import numpy as np

from ..utils import valmap


HAS_FA2 = importlib.util.find_spec('fa2') is not None


def parse_dict_to_tids_or_inds(spec, tn, default='__NONE__'):
    """Parse a dictionary possibly containing a mix of tags, tids and inds, to
    a dictionary with only sinlge tids and inds as keys. If a tag or set of
    tags are given as a key, all matching tensor tids will receive the value.
    """
    #
    if (spec is not None) and (not isinstance(spec, dict)):
        # assume new default value for everything
        return collections.defaultdict(lambda: spec)

    # allow not specifying a default value
    if default != '__NONE__':
        new = collections.defaultdict(lambda: default)
    else:
        new = {}

    if spec is None:
        return new

    # parse the special values
    for k, v in spec.items():
        if (
            # given as tid
            (isinstance(k, int) and k in tn.tensor_map) or
            # given as ind
            (isinstance(k, str) and k in tn.ind_map)
        ):
            # already a tid
            new[k] = v
            continue

        try:
            for tid in tn._get_tids_from_tags(k):
                new[tid] = v
        except KeyError:
            # just ignore keys that don't match any tensor
            pass

    return new


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
        # hide original edge and instead track multiple bond sizes
        attrs0['multiedge_inds'].append(attrs['ind'])
        attrs0['multiedge_sizes'].append(attrs['edge_size'])
        attrs0['spring_weight'] /= (attrs['edge_size'] + 1)
        attrs0['edge_size'] = 0


def draw_tn(
    tn,
    color=None,
    *,
    output_inds=None,
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
    use_forceatlas2=1000,
    use_spring_weight=False,
    node_color=None,
    node_scale=1.0,
    node_size=None,
    node_shape='o',
    node_outline_size=None,
    node_outline_darkness=0.8,
    node_hatch='',
    edge_color=None,
    edge_scale=1.0,
    edge_alpha=1 / 2,
    multiedge_spread=0.1,
    show_left_inds=True,
    arrow_closeness=1.1,
    arrow_length=1.0,
    arrow_overhang=1.0,
    arrow_linewidth=1.0,
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
    output_inds : sequence of str, optional
        For hyper tensor networks explicitly specify which indices should be
        drawn as outer indices. If not set, the outer indices are assumed to be
        those that only appear on a single tensor.
    highlight_inds : iterable, optional
        Highlight these edges.
    highlight_tids : iterable, optional
        Highlight these nodes.
    highlight_inds_color
        What color to use for ``highlight_inds`` nodes.
    highlight_tids_color : tuple[float], optional
        What color to use for ``highlight_tids`` nodes.
    show_inds : {None, False, True, 'all', 'bond-size'}, optional
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
    fix : dict[tags_ind_or_tid], (float, float)], optional
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
    use_forceatlas2 : bool or int, optional
        Whether to try and use ``forceatlas2`` (``fa2``) for the spring layout
        relaxation instead of ``networkx``. If an integer, only try and use
        beyond that many nodes (it can give messier results on smaller graphs).
    use_spring_weight : bool, optional
        Whether to use inverse bond sizes as spring weights to the force
        repulsion layout algorithms.
    node_color : tuple[float], optional
        Default color of nodes.
    node_size : None, float or dict, optional
        How big to draw the tensors. Can be a global single value, or a dict
        containing values for specific tags or tids. This is in absolute
        figure units. See ``node_scale`` simply scale the node sizes up or
        down.
    node_scale : float, optional
        Scale the node sizes by this factor, in addition to the automatica
        scaling based on the number of tensors.
    node_shape : None, str or dict, optional
        What shape to draw the tensors. Should correspond to a matplotlib
        scatter marker. Can be a global single value, or a dict containing
        values for specific tags or tids.
    node_outline_size : None, float or dict, optional
        The width of the border of each node. Can be a global single value, or
        a dict containing values for specific tags or tids.
    node_outline_darkness : float, optional
        Darkening of nodes outlines.
    edge_color : tuple[float], optional
        Default color of edges.
    edge_scale : float, optional
        How much to scale the width of the edges.
    edge_alpha : float, optional
        Set the alpha (opacity) of the drawn edges.
    multiedge_spread : float, optional
        How much to spread the lines of multi-edges.
    show_left_inds : bool, optional
        Whether to show ``tensor.left_inds`` as incoming arrows.
    arrow_closeness : float, optional
        How close to draw the arrow to its target.
    arrow_length : float, optional
        The size of the arrow with respect to the edge.
    arrow_overhang : float, optional
        Varies the arrowhead between a triangle (0.0) and 'V' (1.0).
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
    get : {None, 'pos', 'graph'}, optional
        If ``None`` then plot as normal, else if:

            - ``'pos'``, return the plotting positions of each ``tid`` and
              ``ind`` drawn as a node, this can supplied to subsequent calls as
              ``fix=pos`` to maintain positions, even as the graph structure
              changes.
            - ``'graph'``, return the ``networkx.Graph`` object. Note that this
              will potentially have extra nodes representing output and hyper
              indices.

    return_fig : bool, optional
        If True and ``ax is None`` then return the figure created rather than
        executing ``pyplot.show()``.
    ax : matplotlib.Axis, optional
        Draw the graph on this axis rather than creating a new figure.
    """
    import networkx as nx
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import to_rgb, to_rgba
    import math

    if output_inds is None:
        output_inds = set(tn.outer_inds())
    elif isinstance(output_inds, str):
        output_inds = {output_inds}
    else:
        output_inds = set(output_inds)

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
    else:
        edge_color = mpl.colors.to_rgb(edge_color)

    if node_color is None:
        node_color = draw_color
    else:
        node_color = mpl.colors.to_rgb(node_color)

    highlight_tids_color = to_rgba(highlight_tids_color)
    highlight_inds_color = to_rgba(highlight_inds_color)

    # set the size of the nodes and their border
    node_size = parse_dict_to_tids_or_inds(
        node_size, tn,
        default=node_scale * 1000 / tn.num_tensors**0.7)
    node_outline_size = parse_dict_to_tids_or_inds(
        node_outline_size, tn,
        default=min(3, node_size.default_factory()**0.5 / 5))
    node_shape = parse_dict_to_tids_or_inds(
        node_shape, tn, default='o')
    node_hatch = parse_dict_to_tids_or_inds(
        node_hatch, tn, default='')

    if label_color is None:
        label_color = mpl.rcParams['axes.labelcolor']

    # build the graph
    G = nx.Graph()
    hyperedges = []
    node_labels = dict()
    edge_labels = dict()

    for ix, tids in tn.ind_map.items():
        # general information for this index
        edge_attrs = {
            'color': (highlight_inds_color if ix in highlight_inds else
                      edge_color),
            'ind': ix,
            'edge_size': edge_scale * math.log2(tn.ind_size(ix)),
        }
        edge_attrs['multiedge_inds'] = [edge_attrs['ind']]
        edge_attrs['multiedge_sizes'] = [edge_attrs['edge_size']]
        edge_attrs['spring_weight'] = 1 / sum(t.ndim for t in tn._inds_get(ix))

        if (ix in output_inds) or (len(tids) != 2):
            # hyper or outer edge - needs dummy 'node' shown with zero size
            hyperedges.append(ix)
            for tid in tids:
                _add_or_merge_edge(G, tid, ix, edge_attrs)
        else:
            # standard edge
            _add_or_merge_edge(G, *tids, edge_attrs)
            if show_inds == 'all':
                edge_labels[tuple(tids)] = ix
            elif show_inds == 'bond-size':
                edge_labels[tuple(tids)] = tn.ind_size(ix)

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

        G.nodes[tid]['size'] = node_size[tid]
        G.nodes[tid]['outline_size'] = node_outline_size[tid]
        color = node_color
        for tag in colors:
            if tag in t.tags:
                color = colors[tag]
        if tid in highlight_tids:
            color = highlight_tids_color
        G.nodes[tid]['color'] = color
        G.nodes[tid]['outline_color'] = tuple(
            (1.0 if i == 3 else node_outline_darkness) * c
            for i, c in enumerate(color)
        )
        G.nodes[tid]['marker'] = node_shape[tid]
        G.nodes[tid]['hatch'] = node_hatch[tid]

        if show_tags == 'tids':
            node_labels[tid] = str(tid)
        elif show_tags:
            # make the tags appear with auto vertical extent
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
        G.nodes[hix]['marker'] = '.'  # set this to avoid warning - size is 0
        G.nodes[hix]['hatch'] = ''
        if show_inds == 'all':
            node_labels[hix] = hix
        elif show_inds == 'bond-size':
            node_labels[hix] = tn.ind_size(hix)

    if get == 'graph':
        return G

    if show_inds == 'bond-size':
        font_size = font_size_inner
        for oix in output_inds:
            node_labels[oix] = tn.ind_size(oix)
    elif show_inds:
        for oix in output_inds:
            node_labels[oix] = oix

    pos = get_positions(tn, G, fix, initial_layout, k, iterations,
                        use_forceatlas2, use_spring_weight)

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
                inv.transform((0, node_size.default_factory()))[1] -
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

    # draw multiedges
    multiedge_centers = {}
    for i, j, attrs in G.edges(data=True):
        sizes = attrs['multiedge_sizes']
        multiplicity = len(sizes)
        if multiplicity > 1:
            rads = np.linspace(
                multiplicity * -multiedge_spread,
                multiplicity * +multiedge_spread,
                multiplicity
            )

            xa, ya = pos[i]
            xb, yb = pos[j]
            xab, yab = (xa + xb) / 2., (ya + yb) / 2.
            dx, dy = xb - xa, yb - ya

            inds = attrs['multiedge_inds']
            for sz, rad, ix in zip(sizes, rads, inds):

                # store the central point of the arc in case its needed by
                # the arrow drawing functionality
                cx, cy = xab + rad * dy * 0.5, yab - rad * dx * 0.5
                multiedge_centers[ix] = (cx, cy)

                ax.add_patch(patches.FancyArrowPatch(
                    (xa, ya), (xb, yb),
                    connectionstyle=patches.ConnectionStyle.Arc3(rad=rad),
                    alpha=edge_alpha,
                    linewidth=sz,
                    color=attrs['color'],
                    zorder=1,
                ))

    scatters = collections.defaultdict(lambda: collections.defaultdict(list))

    for node, attrs in G.nodes(data=True):
        # need to group by marker and hatch as matplotlib doesn't map these
        key = (attrs['marker'], attrs['hatch'])
        scatters[key]['x'].append(pos[node][0])
        scatters[key]['y'].append(pos[node][1])
        scatters[key]['s'].append(attrs['size'])
        scatters[key]['c'].append(attrs['color'])
        scatters[key]['linewidths'].append(attrs['outline_size'])
        scatters[key]['edgecolors'].append(attrs['outline_color'])

    # plot the nodes
    for (marker, hatch), data in scatters.items():
        ax.scatter(
            data['x'],
            data['y'],
            s=data['s'],
            c=data['c'],
            marker=marker,
            linewidths=data['linewidths'],
            edgecolors=data['edgecolors'],
            hatch=hatch,
            zorder=2,
        )

    # draw incomcing arrows for tensor left_inds
    if show_left_inds:
        for tid, t in tn.tensor_map.items():
            if t.left_inds is not None:
                for ind in t.left_inds:
                    if ind in hyperedges:
                        tida = ind
                    else:
                        tida, = (x for x in tn.ind_map[ind] if x != tid)
                    tidb = tid
                    (xa, ya), (xb, yb) = pos[tida], pos[tidb]

                    edge_width = G.get_edge_data(tida, tidb)['edge_size']
                    edge_length = ((xb - xa)**2 + (yb - ya)**2)**0.5
                    arrow_scale = (
                        0.02 * arrow_length * edge_width / edge_length**0.5
                    )

                    # arrow start and change
                    if ind in multiedge_centers:
                        x, y = multiedge_centers[ind]
                    else:
                        x = (xa + arrow_closeness * xb) / (1 + arrow_closeness)
                        y = (ya + arrow_closeness * yb) / (1 + arrow_closeness)

                    dx = (xb - xa) * arrow_scale
                    dy = (yb - ya) * arrow_scale

                    ax.add_patch(patches.FancyArrow(
                        x, y, dx, dy,
                        width=0,  # don't draw tail
                        length_includes_head=True,
                        head_width=(dx**2 + dy**2)**0.5,
                        head_length=(dx**2 + dy**2)**0.5,
                        linewidth=arrow_linewidth,
                        color=(
                            highlight_inds_color if ind in highlight_inds else
                            edge_color
                        ),
                        alpha=edge_alpha,
                        fill=True,
                        shape='full',
                        overhang=arrow_overhang,
                    ))

    if show_inds in {'all', 'bond-size'}:
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


def get_positions(
    tn,
    G,
    fix=None,
    initial_layout='spectral',
    k=None,
    iterations=200,
    use_forceatlas2=False,
    use_spring_weight=False,
):
    import networkx as nx

    if fix is None:
        fix = dict()
    else:
        fix = parse_dict_to_tids_or_inds(fix, tn)
        # find range with which to scale spectral points with
        xmin, xmax, ymin, ymax = (
            f(fix.values(), key=lambda xy: xy[i])[i]
            for f, i in [(min, 0), (max, 0), (min, 1), (max, 1)])
        if xmin == xmax:
            xmin, xmax = xmin - 1, xmax + 1
        if ymin == ymax:
            ymin, ymax = ymin - 1, ymax + 1
        xymin, xymax = min(xmin, ymin), max(xmax, ymax)

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
             2 * (xy[1] - xymin) / (xymax - xymin) - 1)), fix))
        fixed = fix.keys()
    else:
        fixed = None

    # and then relax remaining using spring layout
    if iterations:

        if use_forceatlas2 is True:
            use_forceatlas2 = 1
        elif use_forceatlas2 in (0, False):
            use_forceatlas2 = float('inf')

        should_use_fa2 = (
            (fixed is None) and HAS_FA2 and (len(G) > use_forceatlas2)
        )

        weight = 'spring_weight' if use_spring_weight else None

        if should_use_fa2:
            from fa2 import ForceAtlas2
            pos = ForceAtlas2(verbose=False).forceatlas2_networkx_layout(
                G, pos=pos0, iterations=iterations, weight_attr=weight)
        else:
            pos = nx.spring_layout(
                G, pos=pos0, fixed=fixed, k=k, iterations=iterations,
                weight=weight)
    else:
        pos = pos0

    if not fix:
        # finally rotate them to cover a small vertical span
        pos = _massage_pos(pos)

    return pos
