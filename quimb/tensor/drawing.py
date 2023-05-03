"""Functionailty for drawing tensor networks.
"""
import textwrap
import importlib
import collections

import numpy as np

from ..utils import valmap


HAS_FA2 = importlib.util.find_spec("fa2") is not None


def draw_tn(
    tn,
    color=None,
    *,
    output_inds=None,
    highlight_inds=(),
    highlight_tids=(),
    highlight_inds_color=(1.0, 0.2, 0.2),
    highlight_tids_color=(1.0, 0.2, 0.2),
    show_inds=None,
    show_tags=None,
    show_scalars=True,
    custom_colors=None,
    title=None,
    legend=True,
    dim=2,
    fix=None,
    layout="auto",
    initial_layout="auto",
    iterations="auto",
    k=None,
    use_forceatlas2=1000,
    use_spring_weight=False,
    pos=None,
    node_color=None,
    node_scale=1.0,
    node_size=None,
    node_alpha=1.0,
    node_shape="o",
    node_outline_size=None,
    node_outline_darkness=0.8,
    node_hatch="",
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
    font_family="monospace",
    isdark=None,
    backend="matplotlib",
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
    dim : {2, 2.5, 3}, optional
        What dimension to position the graph nodes in. 2.5 positions the nodes
        in 3D but then projects then down to 2D.
    fix : dict[tags_ind_or_tid], (float, float)], optional
        Used to specify actual relative positions for each tensor node.
        Each key should be a sequence of tags that uniquely identifies a
        tensor, a ``tid``, or a ``ind``, and each value should be a ``(x, y)``
        coordinate tuple.
    layout : str, optional
        How to layout the graph. Can be any of the following:

            - ``'auto'``: layout the graph using a networkx method then relax
              the layout using a force-directed algorithm.
            - a networkx layout method name, e.g. ``'kamada_kawai'``: just
              layout the graph using a networkx method, with no relaxation.
            - a graphviz method such as ``'dot'``, ``'neato'`` or ``'sfdp'``:
              layout the graph using ``pygraphviz``.

    initial_layout : {'auto', 'spectral', 'kamada_kawai', 'circular', \\
                      'planar', 'random', 'shell', 'bipartite', ...}, optional
        If ``layout == 'auto'`` The name of a networkx layout to use before
        iterating with the spring layout. Set `layout` directly or
        ``iterations=0`` if you don't want any spring relaxation.
    iterations : int, optional
        How many iterations to perform when when finding the best layout
        using node repulsion. Ramp this up if the graph is drawing messily.
    k : float, optional
        The optimal distance between nodes.
    use_forceatlas2 : bool or int, optional
        Whether to try and use ``forceatlas2`` (``fa2``) for the spring layout
        relaxation instead of ``networkx``. If an integer, only try and use
        beyond that many nodes (it can give messier results on smaller graphs).
    use_spring_weight : bool, optional
        Whether to use inverse bond sizes as spring weights to the force
        repulsion layout algorithms.
    pos : dict, optional
        Pre-computed positions for the nodes. If given, this will override
        ``layout``. The nodes shouuld be exactly the same as the nodes in the
        graph returned by ``draw(get='graph')``.
    node_color : tuple[float], optional
        Default color of nodes.
    node_scale : float, optional
        Scale the node sizes by this factor, in addition to the automatic
        scaling based on the number of tensors.
    node_size : None, float or dict, optional
        How big to draw the tensors. Can be a global single value, or a dict
        containing values for specific tags or tids. This is in absolute
        figure units. See ``node_scale`` simply scale the node sizes up or
        down.
    node_alpha : float, optional
        Transparency of the nodes.
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
    arrow_linewidth : float, optional
        The width of the arrow line itself.
    label_color : tuple[float], optional
        Color to draw labels with.
    font_size : int, optional
        Font size for drawing tags and outer indices.
    font_size_inner : int, optional
        Font size for drawing inner indices.
    font_family : str, optional
        Font family to use for all labels.
    isdark : bool, optional
        Explicitly specify that the background is dark, and use slightly
        different default drawing colors. If not specified detects
        automatically from `matplotlib.rcParams`.
    figsize : tuple of int, optional
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
        show_inds = len(tn.outer_inds()) <= 20
    if show_tags is None:
        show_tags = len(tn.tag_map) <= 20

    if isdark is None:
        isdark = sum(to_rgb(mpl.rcParams["figure.facecolor"])) / 3 < 0.5

    if isdark:
        default_draw_color = (0.55, 0.57, 0.60, 1.0)
        default_label_color = (0.85, 0.85, 0.85, 1.0)
    else:
        default_draw_color = (0.45, 0.47, 0.50, 1.0)
        default_label_color = (0.25, 0.25, 0.25, 1.0)

    if edge_color is None:
        edge_color = mpl.colors.to_rgba(default_draw_color, edge_alpha)
    else:
        edge_color = mpl.colors.to_rgba(edge_color, edge_alpha)

    if node_color is None:
        node_color = mpl.colors.to_rgba(default_draw_color, node_alpha)
    else:
        node_color = mpl.colors.to_rgba(node_color, node_alpha)

    if label_color is None:
        label_color = default_label_color
    elif label_color == "inherit":
        label_color = mpl.rcParams["axes.labelcolor"]

    highlight_tids_color = to_rgba(highlight_tids_color, node_alpha)
    highlight_inds_color = to_rgba(highlight_inds_color, edge_alpha)

    # set the size of the nodes and their border
    default_node_size = node_scale * 1000 / tn.num_tensors**0.7
    node_size = parse_dict_to_tids_or_inds(
        node_size,
        tn,
        default=default_node_size,
    )
    node_outline_size = parse_dict_to_tids_or_inds(
        node_outline_size, tn, default=min(3, default_node_size**0.5 / 5)
    )
    node_shape = parse_dict_to_tids_or_inds(node_shape, tn, default="o")
    node_hatch = parse_dict_to_tids_or_inds(node_hatch, tn, default="")

    # build the graph
    G = nx.Graph()
    hyperedges = []

    for ix, tids in tn.ind_map.items():
        # general information for this index
        edge_attrs = {
            "color": (
                highlight_inds_color if ix in highlight_inds else edge_color
            ),
            "ind": ix,
            "ind_size": str(tn.ind_size(ix)),
            "edge_size": edge_scale * math.log2(tn.ind_size(ix)),
        }
        edge_attrs["multiedge_inds"] = [edge_attrs["ind"]]
        edge_attrs["multiedge_sizes"] = [edge_attrs["edge_size"]]
        edge_attrs["spring_weight"] = 1 / sum(t.ndim for t in tn._inds_get(ix))

        if (ix in output_inds) or (len(tids) != 2):
            # hyper or outer edge - needs dummy 'node' shown with zero size
            hyperedges.append(ix)
            for tid in tids:
                _add_or_merge_edge(G, tid, ix, edge_attrs)
        else:
            # standard edge
            _add_or_merge_edge(G, *tids, edge_attrs)
            if show_inds == "all":
                G.edges[tuple(tids)]["label"] = ix
            elif show_inds == "bond-size":
                G.edges[tuple(tids)]["label"] = tn.ind_size(ix)

    # color the nodes
    colors = get_colors(color, custom_colors, node_alpha)

    # set parameters for all the nodes
    for tid, t in tn.tensor_map.items():
        if tid not in G.nodes:
            # e.g. tensor is a scalar -> has not been added via an edge
            if not show_scalars:
                continue
            G.add_node(tid)

        G.nodes[tid]["tid"] = tid
        G.nodes[tid]["tags"] = str(list(t.tags))
        G.nodes[tid]["shape"] = str(t.shape)
        G.nodes[tid]["size"] = node_size[tid]
        G.nodes[tid]["outline_size"] = node_outline_size[tid]
        color = node_color
        for tag in colors:
            if tag in t.tags:
                color = colors[tag]
        if tid in highlight_tids:
            color = highlight_tids_color
        G.nodes[tid]["color"] = color
        G.nodes[tid]["outline_color"] = tuple(
            (1.0 if i == 3 else node_outline_darkness) * c
            for i, c in enumerate(color)
        )
        G.nodes[tid]["marker"] = node_shape[tid]
        G.nodes[tid]["hatch"] = node_hatch[tid]

        if show_tags == "tids":
            G.nodes[tid]["label"] = str(tid)
        elif show_tags:
            # make the tags appear with auto vertical extent
            # node_label = '{' + str(list(t.tags))[1:-1] + '}'
            node_label = ", ".join(map(str, t.tags))
            G.nodes[tid]["label"] = "\n".join(
                textwrap.wrap(node_label, max(2 * len(node_label) ** 0.5, 16))
            )

    for hix in hyperedges:
        G.nodes[hix]["ind"] = hix
        G.nodes[hix]["color"] = (1.0, 1.0, 1.0, 1.0)
        G.nodes[hix]["size"] = 0.0
        G.nodes[hix]["outline_size"] = 0.0
        G.nodes[hix]["outline_color"] = (1.0, 1.0, 1.0, 1.0)
        G.nodes[hix]["marker"] = "."  # set this to avoid warning - size is 0
        G.nodes[hix]["hatch"] = ""
        if show_inds == "all":
            G.nodes[hix]["label"] = hix
        elif show_inds == "bond-size":
            G.nodes[hix]["label"] = tn.ind_size(hix)

    if show_inds == "bond-size":
        for oix in output_inds:
            G.nodes[oix]["label"] = tn.ind_size(oix)
    elif show_inds:
        for oix in output_inds:
            G.nodes[oix]["label"] = oix

    if get == "graph":
        return G

    if pos is None:
        pos = get_positions(
            tn=tn,
            G=G,
            fix=fix,
            layout=layout,
            initial_layout=initial_layout,
            k=k,
            dim=dim,
            iterations=iterations,
            use_forceatlas2=use_forceatlas2,
            use_spring_weight=use_spring_weight,
        )

    if get == "pos":
        return pos
    if get == "graph,pos":
        return G, pos

    if backend == "matplotlib":
        return _draw_matplotlib(
            G=G,
            pos=pos,
            tn=tn,
            hyperedges=hyperedges,
            highlight_inds=highlight_inds,
            highlight_inds_color=highlight_inds_color,
            edge_color=edge_color,
            default_node_size=default_node_size,
            show_inds=show_inds,
            label_color=label_color,
            show_tags=show_tags,
            colors=colors,
            node_outline_darkness=node_outline_darkness,
            title=title,
            legend=legend,
            multiedge_spread=multiedge_spread,
            show_left_inds=show_left_inds,
            arrow_closeness=arrow_closeness,
            arrow_length=arrow_length,
            arrow_overhang=arrow_overhang,
            arrow_linewidth=arrow_linewidth,
            font_size=font_size,
            font_size_inner=font_size_inner,
            font_family=font_family,
            figsize=figsize,
            margin=margin,
            xlims=xlims,
            ylims=ylims,
            return_fig=return_fig,
            ax=ax,
        )

    if backend == "matplotlib3d":
        # TODO: support more style options
        return _draw_matplotlib3d(
            G,
            pos,
            figsize=figsize,
            ax=ax,
            return_fig=return_fig,
        )

    if backend == "plotly":
        # TODO: support more style options
        return _draw_plotly(
            G,
            pos,
            figsize=figsize,
        )


def parse_dict_to_tids_or_inds(spec, tn, default="__NONE__"):
    """Parse a dictionary possibly containing a mix of tags, tids and inds, to
    a dictionary with only sinlge tids and inds as keys. If a tag or set of
    tags are given as a key, all matching tensor tids will receive the value.
    """
    #
    if (spec is not None) and (not isinstance(spec, dict)):
        # assume new default value for everything
        return collections.defaultdict(lambda: spec)

    # allow not specifying a default value
    if default != "__NONE__":
        new = collections.defaultdict(lambda: default)
    else:
        new = {}

    if spec is None:
        return new

    # parse the special values
    for k, v in spec.items():
        if (
            # given as tid
            (isinstance(k, int) and k in tn.tensor_map)
            or
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
        attrs0["color"] = tuple(
            (x + y) / 2 for x, y in zip(attrs0["color"], attrs["color"])
        )
        attrs0["ind"] += " " + attrs["ind"]
        attrs0["ind_size"] = f"{attrs0['ind_size']} {attrs['ind_size']}"
        # hide original edge and instead track multiple bond sizes
        attrs0["multiedge_inds"].append(attrs["ind"])
        attrs0["multiedge_sizes"].append(attrs["edge_size"])
        attrs0["spring_weight"] /= attrs["edge_size"] + 1
        attrs0["edge_size"] = 0


def _draw_matplotlib(
    G,
    pos,
    *,
    tn,
    hyperedges,
    highlight_inds,
    highlight_inds_color,
    edge_color,
    default_node_size,
    show_inds,
    label_color,
    show_tags,
    colors,
    node_outline_darkness,
    title=None,
    legend=True,
    multiedge_spread=0.1,
    show_left_inds=True,
    arrow_closeness=1.1,
    arrow_length=1.0,
    arrow_overhang=1.0,
    arrow_linewidth=1.0,
    font_size=10,
    font_size_inner=7,
    font_family="monospace",
    figsize=(6, 6),
    margin=None,
    xlims=None,
    ylims=None,
    return_fig=False,
    ax=None,
):
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        fig.patch.set_alpha(0.0)
        ax.axis("off")
        ax.set_aspect("equal")
        if title is not None:
            ax.set_title(str(title))

        xmin = ymin = +float("inf")
        xmax = ymax = -float("inf")
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
            real_node_size = (
                abs(
                    inv.transform((0, default_node_size))[1]
                    - inv.transform((0, 0))[1]
                )
                ** 0.5
            ) / 4
            ax.set_xlim(xmin - real_node_size, xmax + real_node_size)
            ax.set_ylim(ymin - real_node_size, ymax + real_node_size)
        else:
            ax.margins(margin)
    else:
        fig = None

    nx.draw_networkx_edges(
        G,
        pos,
        width=tuple(x[2]["edge_size"] for x in G.edges(data=True)),
        edge_color=tuple(x[2]["color"] for x in G.edges(data=True)),
        ax=ax,
    )

    # draw multiedges
    multiedge_centers = {}
    for i, j, attrs in G.edges(data=True):
        sizes = attrs["multiedge_sizes"]
        multiplicity = len(sizes)
        if multiplicity > 1:
            rads = np.linspace(
                multiplicity * -multiedge_spread,
                multiplicity * +multiedge_spread,
                multiplicity,
            )

            xa, ya = pos[i]
            xb, yb = pos[j]
            xab, yab = (xa + xb) / 2.0, (ya + yb) / 2.0
            dx, dy = xb - xa, yb - ya

            inds = attrs["multiedge_inds"]
            for sz, rad, ix in zip(sizes, rads, inds):
                # store the central point of the arc in case its needed by
                # the arrow drawing functionality
                cx, cy = xab + rad * dy * 0.5, yab - rad * dx * 0.5
                multiedge_centers[ix] = (cx, cy)

                ax.add_patch(
                    patches.FancyArrowPatch(
                        (xa, ya),
                        (xb, yb),
                        connectionstyle=patches.ConnectionStyle.Arc3(rad=rad),
                        linewidth=sz,
                        color=attrs["color"],
                        zorder=1,
                    )
                )

    scatters = collections.defaultdict(lambda: collections.defaultdict(list))
    for node, attrs in G.nodes(data=True):
        # need to group by marker and hatch as matplotlib doesn't map these
        key = (attrs["marker"], attrs["hatch"])
        scatters[key]["x"].append(pos[node][0])
        scatters[key]["y"].append(pos[node][1])
        scatters[key]["s"].append(attrs["size"])
        scatters[key]["c"].append(attrs["color"])
        scatters[key]["linewidths"].append(attrs["outline_size"])
        scatters[key]["edgecolors"].append(attrs["outline_color"])

    # plot the nodes
    for (marker, hatch), data in scatters.items():
        ax.scatter(
            data["x"],
            data["y"],
            s=data["s"],
            c=data["c"],
            marker=marker,
            linewidths=data["linewidths"],
            edgecolors=data["edgecolors"],
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
                        (tida,) = (x for x in tn.ind_map[ind] if x != tid)
                    tidb = tid
                    (xa, ya), (xb, yb) = pos[tida], pos[tidb]

                    edge_width = G.get_edge_data(tida, tidb)["edge_size"]
                    edge_length = ((xb - xa) ** 2 + (yb - ya) ** 2) ** 0.5
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

                    ax.add_patch(
                        patches.FancyArrow(
                            x,
                            y,
                            dx,
                            dy,
                            width=0,  # don't draw tail
                            length_includes_head=True,
                            head_width=(dx**2 + dy**2) ** 0.5,
                            head_length=(dx**2 + dy**2) ** 0.5,
                            linewidth=arrow_linewidth,
                            color=(
                                highlight_inds_color
                                if ind in highlight_inds
                                else edge_color
                            ),
                            fill=True,
                            shape="full",
                            overhang=arrow_overhang,
                        )
                    )

    if show_inds in {"all", "bond-size"}:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={
                (nodea, nodeb): data.get("label", "")
                for nodea, nodeb, data in G.edges(data=True)
            },
            font_size=font_size_inner,
            font_color=label_color,
            font_family=font_family,
            bbox={"ec": (0, 0, 0, 0), "fc": (0, 0, 0, 0)},
            ax=ax,
        )
    if show_tags or show_inds:
        nx.draw_networkx_labels(
            G,
            pos,
            labels={
                node: data.get("label", "")
                for node, data in G.nodes(data=True)
            },
            font_size=(
                font_size_inner if show_inds == "bond-size" else font_size
            ),
            font_color=label_color,
            font_family=font_family,
            ax=ax,
        )

    # create legend
    if colors and legend:
        handles = []
        for color in colors.values():
            ecolor = tuple(
                (1.0 if i == 3 else node_outline_darkness) * c
                for i, c in enumerate(color)
            )
            linewidth = min(3, default_node_size**0.5 / 5)

            handles += [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=color,
                    markeredgecolor=ecolor,
                    markeredgewidth=linewidth,
                    linestyle="",
                    markersize=10,
                )
            ]

        # needed in case '_' is the first character
        lbls = [f" {lbl}" for lbl in colors]

        legend = plt.legend(
            handles,
            lbls,
            ncol=max(round(len(handles) / 20), 1),
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            labelcolor=label_color,
            prop={"family": font_family},
        )
        # do this manually as otherwise can't make only face transparent
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor((0.0, 0.0, 0.0, 0.0))
        legend.get_frame().set_edgecolor((0.6, 0.6, 0.6, 0.2))

    if fig is None:
        # ax was supplied, don't modify and simply return
        return
    else:
        # axes and figure were created
        if xlims is not None:
            ax.set_xlim(xlims)
        if ylims is not None:
            ax.set_ylim(ylims)

    if return_fig:
        return fig
    else:
        plt.show()
        plt.close(fig)


def _linearize_graph_data(G, pos):
    edge_source = collections.defaultdict(list)
    for nodea, nodeb, data in G.edges(data=True):
        x0, y0, *maybe_z0 = pos[nodea]
        x1, y1, *maybe_z1 = pos[nodeb]
        edge_source["x0"].append(x0)
        edge_source["y0"].append(y0)
        edge_source["x1"].append(x1)
        edge_source["y1"].append(y1)
        if maybe_z0:
            edge_source["z0"].extend(maybe_z0)
            edge_source["z1"].extend(maybe_z1)

        for k in ("color", "edge_size", "ind", "ind_size", "label"):
            edge_source[k].append(data.get(k, None))

    node_source = collections.defaultdict(list)
    for node, data in G.nodes(data=True):
        if "ind" in data:
            continue
        x, y, *maybe_z = pos[node]
        node_source["x"].append(x)
        node_source["y"].append(y)
        if maybe_z:
            node_source["z"].extend(maybe_z)

        for k in (
            "size",
            "color",
            "outline_color",
            "outline_size",
            "hatch",
            "tags",
            "shape",
            "tid",
            "label",
        ):
            node_source[k].append(data.get(k, None))

    return dict(edge_source), dict(node_source)


def _draw_matplotlib3d(
    G,
    pos,
    figsize=(6, 6),
    return_fig=False,
    ax=None,
):
    import matplotlib.pyplot as plt

    edge_source, node_source = _linearize_graph_data(G, pos)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        fig.patch.set_alpha(0.0)
        ax = plt.axes([0, 0, 1, 1], projection="3d")
        ax.patch.set_alpha(0.0)

        xmin = min(node_source["x"])
        xmax = max(node_source["x"])
        ymin = min(node_source["y"])
        ymax = max(node_source["y"])
        zmin = min(node_source["z"])
        zmax = max(node_source["z"])
        xyzmin = min((xmin, ymin, zmin))
        xyzmax = max((xmax, ymax, zmax))

        ax.set_xlim(xyzmin, xyzmax)
        ax.set_ylim(xyzmin, xyzmax)
        ax.set_zlim(xyzmin, xyzmax)
        ax.set_aspect("equal")
        ax.axis("off")

    # draw the edges
    # TODO: multiedges and left_inds
    for i in range(len(edge_source["x0"])):
        x0, x1 = edge_source["x0"][i], edge_source["x1"][i]
        xm = (x0 + x1) / 2
        y0, y1 = edge_source["y0"][i], edge_source["y1"][i]
        ym = (y0 + y1) / 2
        z0, z1 = edge_source["z0"][i], edge_source["z1"][i]
        zm = (z0 + z1) / 2
        ax.plot3D(
            [x0, x1],
            [y0, y1],
            [z0, z1],
            c=edge_source["color"][i],
            linewidth=edge_source["edge_size"][i],
        )
        label = edge_source["label"][i]
        if label:
            ax.text(
                xm,
                ym,
                zm,
                s=label,
                ha="center",
                va="center",
                color=edge_source["color"][i],
                fontsize=6,
            )

    # draw the nodes
    ax.scatter3D(
        xs="x",
        ys="y",
        zs="z",
        c="color",
        s="size",
        data=node_source,
        depthshade=False,
        edgecolors=node_source["outline_color"],
        linewidth=node_source["outline_size"],
    )

    for x, y, label in zip(
        node_source["x"], node_source["y"], node_source["label"]
    ):
        if label:
            ax.text(
                x,
                y,
                0,
                s=label,
                ha="center",
                va="center",
                color="black",
                fontsize=6,
            )

    if return_fig:
        return fig
    else:
        plt.show()
        plt.close(fig)


def _draw_plotly(
    G,
    pos,
    figsize=(6, 6),
):
    import plotly.graph_objects as go

    edge_source, node_source = _linearize_graph_data(G, pos)

    fig = go.Figure()
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        width=100 * figsize[0],
        height=100 * figsize[1],
        margin=dict(l=10, r=10, b=10, t=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
    )

    for i in range(len(edge_source["x0"])):
        x0, x1 = edge_source["x0"][i], edge_source["x1"][i]
        y0, y1 = edge_source["y0"][i], edge_source["y1"][i]
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        *rgb, alpha = edge_source["color"][i]
        edge_kwargs = dict(
            x=[x0, xm, x1],
            y=[y0, ym, y1],
            opacity=alpha,
            line=dict(
                color=to_rgba_str(rgb, 1.0),
                width=2.5 * edge_source["edge_size"][i],
            ),
            customdata=[[edge_source["ind"][i], edge_source["ind_size"][i]]]
            * 2,
            # show ind and ind_size on hover:
            hovertemplate="%{customdata[0]}<br>size: %{customdata[1]}",
            mode="lines",
            name="",
        )
        if "z0" in edge_source:
            z0, z1 = edge_source["z0"][i], edge_source["z1"][i]
            zm = (z0 + z1) / 2
            edge_kwargs["z"] = [z0, zm, z1]
            fig.add_trace(go.Scatter3d(**edge_kwargs))
        else:
            fig.add_trace(go.Scatter(**edge_kwargs))

    node_kwargs = dict(
        x=node_source["x"],
        y=node_source["y"],
        marker=dict(
            opacity=1.0,
            color=list(map(to_rgba_str, node_source["color"])),
            size=[s / 3 for s in node_source["size"]],
            line=dict(
                color=list(map(to_rgba_str, node_source["outline_color"])),
                width=2,
            ),
        ),
        customdata=list(
            zip(node_source["tid"], node_source["shape"], node_source["tags"])
        ),
        hovertemplate=(
            "tid: %{customdata[0]}<br>"
            "shape: %{customdata[1]}<br>"
            "tags: %{customdata[2]}"
        ),
        mode="markers",
        name="",
    )
    if "z" in node_source:
        node_kwargs["z"] = node_source["z"]
        fig.add_trace(go.Scatter3d(**node_kwargs))
    else:
        fig.add_trace(go.Scatter(**node_kwargs))
    fig.show()


# colorblind palettes by Bang Wong (https://www.nature.com/articles/nmeth.1618)

_COLORS_DEFAULT = (
    "#56B4E9",  # light blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # red
    "#F0E442",  # yellow
    "#CC79A7",  # purple
    "#0072B2",  # dark blue
)

_COLORS_SORTED = (
    "#0072B2",  # dark blue
    "#56B4E9",  # light blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#E69F00",  # orange
    "#D55E00",  # red
    "#CC79A7",  # purple
)


def mod_sat(c, mod, alpha):
    """Modify the luminosity of rgb color ``c``."""
    from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

    h, s, v = rgb_to_hsv(c[:3])
    return (*hsv_to_rgb((h, mod * s, v)), alpha)


def auto_colors(nc, alpha=None):
    import math
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("wong", _COLORS_SORTED)

    xs = list(map(cmap, np.linspace(0, 1.0, nc)))

    # modulate color saturation with sine to generate local distinguishability
    # ... but only turn on gradually for increasing number of nodes
    sat_mod_period = min(4, nc / 7)
    sat_mod_factor = max(0.0, 2 / 3 * math.tanh((nc - 7) / 4))

    if alpha is None:
        alpha = 1.0

    return [
        mod_sat(
            c,
            1 - sat_mod_factor * math.sin(math.pi * i / sat_mod_period) ** 2,
            alpha,
        )
        for i, c in enumerate(xs)
    ]


def get_colors(color, custom_colors=None, alpha=None):
    """Generate a sequence of rgbs for tag(s) ``color``."""
    from matplotlib.colors import to_rgba

    if color is None:
        return dict()

    if isinstance(color, str):
        color = (color,)

    if custom_colors is not None:
        rgbs = [to_rgba(c, alpha=alpha) for c in custom_colors]
        return dict(zip(color, rgbs))

    nc = len(color)
    if nc <= 7:
        rgbs = [to_rgba(c, alpha=alpha) for c in _COLORS_DEFAULT]
        return dict(zip(color, rgbs))

    rgbs = auto_colors(nc, alpha)
    return dict(zip(color, rgbs))


def to_rgba_str(color, alpha=None):
    from matplotlib.colors import to_rgba

    rgba = to_rgba(color, alpha)
    r = int(rgba[0] * 255) if isinstance(rgba[0], float) else rgba[0]
    g = int(rgba[1] * 255) if isinstance(rgba[1], float) else rgba[1]
    b = int(rgba[2] * 255) if isinstance(rgba[2], float) else rgba[2]
    return f"rgba({r}, {g}, {b}, {rgba[3]})"


def _rotate(xy, theta):
    """Return a rotated set of points."""
    s = np.sin(theta)
    c = np.cos(theta)

    xyr = np.empty_like(xy)
    xyr[:, 0] = c * xy[:, 0] - s * xy[:, 1]
    xyr[:, 1] = s * xy[:, 0] + c * xy[:, 1]

    return xyr


def _span(xy):
    """Return the vertical span of the points."""
    return xy[:, 1].max() - xy[:, 1].min()


def _massage_pos(pos, nangles=360, flatten=False):
    """Rotate a position dict's points to cover a small vertical span"""
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


def layout_pygraphviz(
    G,
    prog="neato",
    dim=2,
    **kwargs,
):
    # TODO: fix nodes with pin attribute
    # TODO: initial positions
    # TODO: max iters
    # TODO: spring parameter
    import pygraphviz as pgv

    aG = pgv.AGraph()
    mapping = {}
    for nodea, nodeb in G.edges():
        s_nodea = str(nodea)
        s_nodeb = str(nodeb)
        mapping[s_nodea] = nodea
        mapping[s_nodeb] = nodeb
        aG.add_edge(s_nodea, s_nodeb)

    kwargs = {}

    if dim == 2.5:
        kwargs["dim"] = 3
        kwargs["dimen"] = 2
    else:
        kwargs["dim"] = kwargs["dimen"] = dim
    args = " ".join(f"-G{k}={v}" for k, v in kwargs.items())

    # run layout algorithm
    aG.layout(prog=prog, args=args)

    # extract layout
    pos = {}
    for snode, node in mapping.items():
        spos = aG.get_node(snode).attr["pos"]
        pos[node] = tuple(map(float, spos.split(",")))

    # normalize to unit square
    xmin = ymin = zmin = float("inf")
    xmax = ymax = zmaz = float("-inf")
    for x, y, *maybe_z in pos.values():
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)
        for z in maybe_z:
            zmin = min(zmin, z)
            zmaz = max(zmaz, z)

    for node, (x, y, *maybe_z) in pos.items():
        pos[node] = (
            2 * (x - xmin) / (xmax - xmin) - 1,
            2 * (y - ymin) / (ymax - ymin) - 1,
            *(2 * (z - zmin) / (zmaz - zmin) - 1 for z in maybe_z),
        )

    return pos


def get_positions(
    tn,
    G,
    *,
    dim=2,
    fix=None,
    layout="auto",
    initial_layout="auto",
    iterations="auto",
    k=None,
    use_forceatlas2=False,
    use_spring_weight=False,
):
    if layout in ("dot", "neato", "fdp", "sfdp"):
        if fix or k or iterations != "auto":
            import warnings

            warnings.warn(
                "Layout is being done by pygraphviz, so `fix`, "
                "`k`, and `iterations` are currently ignored."
            )
        return layout_pygraphviz(G, prog=layout, dim=dim)

    if layout != "auto":
        initial_layout = layout
        iterations = 0

    import networkx as nx

    if fix is None:
        fix = dict()
    else:
        fix = parse_dict_to_tids_or_inds(fix, tn)
        # find range with which to scale spectral points with
        xmin, xmax, ymin, ymax = (
            f(fix.values(), key=lambda xy: xy[i])[i]
            for f, i in [(min, 0), (max, 0), (min, 1), (max, 1)]
        )
        if xmin == xmax:
            xmin, xmax = xmin - 1, xmax + 1
        if ymin == ymax:
            ymin, ymax = ymin - 1, ymax + 1
        xymin, xymax = min(xmin, ymin), max(xmax, ymax)

    if all(node in fix for node in G.nodes):
        # everything is already fixed
        return fix

    if initial_layout == "auto":
        # automatically select
        if len(G) <= 100:
            # usually nicest
            initial_layout = "kamada_kawai"
        else:
            # faster, but not as nice
            initial_layout = "spectral"

    if iterations == "auto":
        # the smaller the graph, the more iterations we can afford
        iterations = max(200, 1000 - len(G))

    if dim == 2.5:
        dim = 3
        project_back_to_2d = True
    else:
        project_back_to_2d = False

    # use spectral or other layout as starting point
    ly_opts = {"dim": dim} if dim != 2 else {}
    pos0 = getattr(nx, initial_layout + "_layout")(G, **ly_opts)

    # scale points to fit with specified positions
    if fix:
        # but update with fixed positions
        pos0.update(
            valmap(
                lambda xy: np.array(
                    (
                        2 * (xy[0] - xymin) / (xymax - xymin) - 1,
                        2 * (xy[1] - xymin) / (xymax - xymin) - 1,
                    )
                ),
                fix,
            )
        )
        fixed = fix.keys()
    else:
        fixed = None

    # and then relax remaining using spring layout
    if iterations:
        if use_forceatlas2 is True:
            # turn on for more than 1 node
            use_forceatlas2 = 1
        elif use_forceatlas2 in (0, False):
            # never turn on
            use_forceatlas2 = float("inf")

        should_use_fa2 = (
            (fixed is None)
            and HAS_FA2
            and (len(G) > use_forceatlas2)
            and (dim == 2)
        )

        weight = "spring_weight" if use_spring_weight else None

        if should_use_fa2:
            from fa2 import ForceAtlas2

            # NB: some versions of fa2 don't support the `weight_attr` option
            pos = ForceAtlas2(verbose=False).forceatlas2_networkx_layout(
                G, pos=pos0, iterations=iterations
            )
        else:
            pos = nx.spring_layout(
                G,
                pos=pos0,
                fixed=fixed,
                k=k,
                dim=dim,
                iterations=iterations,
                weight=weight,
            )
    else:
        pos = pos0

    if project_back_to_2d:
        # project back to 2d
        pos = {k: v[:2] for k, v in pos.items()}
        dim = 2

    if (not fix) and (dim == 2):
        # finally rotate them to cover a small vertical span
        pos = _massage_pos(pos)

    return pos


def visualize_tensor(tensor, **kwargs):
    """Visualize all entries of a tensor, with indices mapped into the plane
    and values mapped into a color wheel.

    Parameters
    ----------
    tensor : Tensor
        The tensor to visualize.
    skew_factor : float, optional
        When there are more than two dimensions, a factor to scale the
        rotations by to avoid overlapping data points.
    size_map : bool, optional
        Whether to map the tensor value magnitudes to marker size.
    size_scale : float, optional
        An overall factor to scale the marker size by.
    alpha_map : bool, optional
        Whether to map the tensor value magnitudes to marker alpha.
    alpha_pow : float, optional
        The power to raise the magnitude to when mapping to alpha.
    alpha : float, optional
        The overall alpha to use for all markers if ``not alpha_map``.
    show_lattice : bool, optional
        Show a small grey dot for every 'lattice' point regardless of value.
    lattice_opts : dict, optional
        Options to pass to ``maplotlib.Axis.scatter`` for the lattice points.
    linewidths : float, optional
        The linewidth to use for the markers.
    marker : str, optional
        The marker to use for the markers.
    figsize : tuple, optional
        The size of the figure to create, if ``ax`` is not provided.
    ax : matplotlib.Axis, optional
        The axis to draw to. If not provided, a new figure will be created.

    Returns
    -------
    fig : matplotlib.Figure
        The figure containing the plot, or ``None`` if ``ax`` was provided.
    ax : matplotlib.Axis
        The axis containing the plot.
    """
    import xyzpy as xyz

    kwargs.setdefault("legend", True)
    kwargs.setdefault("compass", True)
    kwargs.setdefault("compass_labels", tensor.inds)
    return xyz.visualize_tensor(tensor.data, **kwargs)


COLORING_SEED = 8  # 8, 10


def set_coloring_seed(seed):
    """Set the seed for the random color generator.

    Parameters
    ----------
    seed : int
        The seed to use.
    """
    global COLORING_SEED
    COLORING_SEED = seed


def hash_to_nvalues(s, nval, seed=None):
    """Hash the string ``s`` to ``nval`` different floats in the range [0, 1].
    """
    import hashlib

    if seed is None:
        seed = COLORING_SEED

    m = hashlib.sha256()
    m.update(f"{seed}".encode())
    m.update(s.encode())
    hsh = m.hexdigest()

    b = len(hsh) // nval
    if b == 0:
        raise ValueError(
            f"Can't extract {nval} values from hash of length {len(hsh)}"
        )
    return tuple(
        int(hsh[i * b : (i + 1) * b], 16) / 16**b for i in range(nval)
    )


def hash_to_color(
    s,
    hmin=0.0,
    hmax=1.0,
    smin=0.3,
    smax=0.8,
    vmin=0.8,
    vmax=0.9,
):
    """Generate a random color for a string  ``s``.

    Parameters
    ----------
    s : str
        The string to generate a color for.
    hmin : float, optional
        The minimum hue value.
    hmax : float, optional
        The maximum hue value.
    smin : float, optional
        The minimum saturation value.
    smax : float, optional
        The maximum saturation value.
    vmin : float, optional
        The minimum value value.
    vmax : float, optional
        The maximum value value.

    Returns
    -------
    color : tuple
        A tuple of floats in the range [0, 1] representing the RGB color.
    """
    from matplotlib.colors import to_hex, hsv_to_rgb

    h, s, v = hash_to_nvalues(s, 3)
    h = hmin + h * (hmax - hmin)
    s = smin + s * (smax - smin)
    v = vmin + v * (vmax - vmin)

    rgb = hsv_to_rgb((h, s, v))
    return to_hex(rgb)


def auto_color_html(s):
    """Automatically hash and color a string for HTML display.
    """
    if not isinstance(s, str):
        s = str(s)
    return f'<b style="color: {hash_to_color(s)};">{s}</b>'
