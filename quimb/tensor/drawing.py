"""Functionailty for drawing tensor networks.
"""
import collections
import importlib
import textwrap
import warnings

import numpy as np

from ..utils import valmap, check_opt, autocorrect_kwargs


HAS_FA2 = importlib.util.find_spec("fa2") is not None


@autocorrect_kwargs
def draw_tn(
    tn,
    color=None,
    *,
    show_inds=None,
    show_tags=None,
    output_inds=None,
    highlight_inds=(),
    highlight_tids=(),
    highlight_inds_color=(1.0, 0.2, 0.2),
    highlight_tids_color=(1.0, 0.2, 0.2),
    custom_colors=None,
    legend="auto",
    dim=2,
    fix=None,
    layout="auto",
    initial_layout="auto",
    refine_layout="auto",
    iterations="auto",
    k=None,
    pos=None,
    node_color=None,
    node_scale=1.0,
    node_size=None,
    node_alpha=1.0,
    node_shape="o",
    node_outline_size=None,
    node_outline_darkness=0.9,
    node_hatch="",
    edge_color=None,
    edge_scale=1.0,
    edge_alpha=1 / 2,
    multi_edge_spread=0.1,
    multi_tag_style="auto",
    show_left_inds=True,
    arrow_opts=None,
    label_color=None,
    font_size=10,
    font_size_inner=7,
    font_family="monospace",
    isdark=None,
    title=None,
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
    custom_colors : sequence of colors, optional
        Supply a custom sequence of colors to match the tags given
        in ``color``.
    title : str, optional
        Set a title for the axis.
    legend : "auto" or bool, optional
        Whether to draw a legend for the colored tags. If ``"auto"`` then
        only draw a legend if there are less than 20 tags.
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
    multi_edge_spread : float, optional
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

    check_opt(
        "multi_tag_style",
        multi_tag_style,
        ("auto", "pie", "nest", "average", "last"),
    )

    if output_inds is None:
        output_inds = set(tn.outer_inds())
    elif isinstance(output_inds, str):
        output_inds = {output_inds}
    else:
        output_inds = set(output_inds)

    # automatically decide whether to show tags and inds
    if show_inds is None:
        show_inds = len(tn.outer_inds()) <= 20
    show_inds = {False: "", True: "outer"}.get(show_inds, show_inds)

    if show_tags is None:
        show_tags = len(tn.tag_map) <= 20
    show_tags = {False: "", True: "tags"}.get(show_tags, show_tags)

    if isdark is None:
        isdark = sum(to_rgb(mpl.rcParams["figure.facecolor"])) / 3 < 0.5

    if isdark:
        default_draw_color = (0.55, 0.57, 0.60, 1.0)
        default_label_color = (0.85, 0.86, 0.87, 1.0)
    else:
        default_draw_color = (0.45, 0.47, 0.50, 1.0)
        default_label_color = (0.33, 0.34, 0.35, 1.0)

    if edge_color is None:
        edge_color = mpl.colors.to_rgba(default_draw_color, edge_alpha)
    elif edge_color is True:
        # hash edge to get color
        pass
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

    # get colors for tagged nodes
    colors = get_colors(color, custom_colors, node_alpha)

    if legend == "auto":
        legend = len(colors) <= 20

    highlight_tids_color = to_rgba(highlight_tids_color, node_alpha)
    highlight_inds_color = to_rgba(highlight_inds_color, edge_alpha)

    # set the size of the nodes and their border
    node_size = parse_dict_to_tids_or_inds(
        node_size,
        tn,
        default=1,
    )
    node_outline_size = parse_dict_to_tids_or_inds(
        node_outline_size,
        tn,
        default=1,
    )
    node_shape = parse_dict_to_tids_or_inds(node_shape, tn, default="o")
    node_hatch = parse_dict_to_tids_or_inds(node_hatch, tn, default="")

    # build the graph
    edges = collections.defaultdict(lambda: collections.defaultdict(list))
    nodes = collections.defaultdict(dict)

    # parse all indices / edges
    for ix, tids in tn.ind_map.items():
        tids = sorted(tids)

        isouter = ix in output_inds
        ishyper = isouter or (len(tids) != 2)
        ind_size = tn.ind_size(ix)
        edge_size = edge_scale * math.log2(ind_size)

        # compute a color for this index
        color = (
            highlight_inds_color
            if ix in highlight_inds
            else to_rgba(hash_to_color(ix))
            if edge_color is True
            else edge_color
        )

        # compute a label for this index
        if ishyper:
            # each tensor connects to the dummy node represeting the hyper edge
            pairs = [(tid, ix) for tid in tids]
            if isouter and len(tids) > 1:
                # 'hyper outer' index
                pairs.append((("outer", ix), ix))
            # hyper labels get put on dummy node
            label = ""

            nodes[ix]["ind"] = ix
            nodes[ix]["ind_size"] = ind_size
            # make actual node invisible
            nodes[ix]["color"] = (1.0, 1.0, 1.0, 1.0)
            nodes[ix]["size"] = 0.0
            nodes[ix]["outline_size"] = 0.0
            nodes[ix]["outline_color"] = (1.0, 1.0, 1.0, 1.0)
            nodes[ix]["marker"] = "."  # set this to avoid warning - size is 0
            nodes[ix]["hatch"] = ""

            # set these for plotly hover info
            nodes[ix]["tid"] = nodes[ix]["shape"] = nodes[ix]["tags"] = ""

            if ((show_inds == "outer") and isouter) or (show_inds == "all"):
                # show as outer index or inner index name
                nodes[ix]["label"] = ix
            elif show_inds == "bond-size":
                # show all bond sizes
                nodes[ix]["label"] = f"{tn.ind_size(ix)}"
            else:
                # labels hidden or inner edge
                nodes[ix]["label"] = ""

            nodes[ix]["label_fontsize"] = font_size_inner
            nodes[ix]["label_color"] = label_color
            nodes[ix]["label_fontfamily"] = font_family

        else:
            # standard edge
            pairs = [tuple(tids)]

            if show_inds == "all":
                # show inner index name
                label = ix
            elif show_inds == "bond-size":
                # show all bond sizes
                label = f"{ind_size}"
            else:
                # labels hidden or inner edge
                label = ""

        for pair in pairs:
            edges[pair]["color"].append(color)
            edges[pair]["ind"].append(ix)
            edges[pair]["ind_size"].append(ind_size)
            edges[pair]["edge_size"].append(edge_size)
            edges[pair]["label"].append(label)
            edges[pair]["label_fontsize"] = font_size_inner
            edges[pair]["label_color"] = label_color
            edges[pair]["label_fontfamily"] = font_family

            if isinstance(pair[0], tuple):
                # dummy hyper outer edge - no arrows
                edges[pair]["arrow_left"].append(False)
                edges[pair]["arrow_right"].append(False)
            else:
                # tensor side can always have an incoming arrow
                tl_left_inds = tn.tensor_map[pair[0]].left_inds
                edges[pair]["arrow_left"].append(
                    show_left_inds
                    and (tl_left_inds is not None)
                    and (ix in tl_left_inds)
                )
                if ishyper:
                    # hyper edge can't have an incoming arrow
                    edges[pair]["arrow_right"].append(False)
                else:
                    # standard edge can
                    tr_left_inds = tn.tensor_map[pair[1]].left_inds
                    edges[pair]["arrow_right"].append(
                        show_left_inds
                        and (tr_left_inds is not None)
                        and (ix in tr_left_inds)
                    )

    # parse all tensors / nodes
    for tid, t in tn.tensor_map.items():
        nodes[tid]["tid"] = tid
        nodes[tid]["tags"] = str(list(t.tags))
        nodes[tid]["shape"] = str(t.shape)
        nodes[tid]["size"] = node_size[tid]
        nodes[tid]["outline_size"] = node_outline_size[tid]
        nodes[tid]["marker"] = node_shape[tid]
        nodes[tid]["hatch"] = node_hatch[tid]

        if show_tags == "tags":
            node_label = ", ".join(map(str, t.tags))
            # make the tags appear with auto vertical extent
            nodes[tid]["label"] = "\n".join(
                textwrap.wrap(node_label, max(2 * len(node_label) ** 0.5, 16))
            )
        elif show_tags == "tids":
            nodes[tid]["label"] = str(tid)
        elif show_tags == "shape":
            nodes[tid]["label"] = nodes[tid]["shape"]
        else:
            nodes[tid]["label"] = ""

        nodes[tid]["label_fontsize"] = font_size
        nodes[tid]["label_color"] = label_color
        nodes[tid]["label_fontfamily"] = font_family

        if tid in highlight_tids:
            nodes[tid]["color"] = highlight_tids_color
            nodes[tid]["outline_color"] = darken_color(
                highlight_tids_color, node_outline_darkness
            )
        else:
            # collect all relevant tag colors
            multi_colors = []
            multi_outline_colors = []
            for tag in colors:
                if tag in t.tags:
                    multi_colors.append(colors[tag])
                    multi_outline_colors.append(
                        darken_color(colors[tag], node_outline_darkness)
                    )

            if len(multi_colors) >= 1:
                # set the basic color to the last tag
                nodes[tid]["color"] = multi_colors[-1]
                nodes[tid]["outline_color"] = multi_outline_colors[-1]
                if len(multi_colors) >= 2:
                    # have multiple relevant tags - store them, but some
                    # backends might support, so store alongside basic color
                    nodes[tid]["multi_colors"] = multi_colors
                    nodes[tid]["multi_outline_colors"] = multi_outline_colors
            else:
                # untagged node
                nodes[tid]["color"] = node_color
                nodes[tid]["outline_color"] = darken_color(
                    node_color, node_outline_darkness**2
                )

    G = nx.Graph()
    for edge, edge_data in edges.items():
        G.add_edge(*edge, **edge_data)
    for node, node_data in nodes.items():
        G.add_node(node, **node_data)

    if pos is None:
        pos = get_positions(
            tn=tn,
            G=G,
            fix=fix,
            layout=layout,
            initial_layout=initial_layout,
            refine_layout=refine_layout,
            k=k,
            dim=dim,
            iterations=iterations,
        )
    else:
        pos = _normalize_positions(pos)

    # compute a base size using the position and number of tensors
    # first get plot volume:
    node_packing_factor = tn.num_tensors**-0.45
    xs, ys, *zs = zip(*pos.values())
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    # if there only a few tensors we don't want to limit the node size
    # because of flatness, also don't allow the plot volume to go to zero
    xrange = max(((xmax - xmin) / 2, node_packing_factor, 0.1))
    yrange = max(((ymax - ymin) / 2, node_packing_factor, 0.1))
    plot_volume = xrange * yrange
    if zs:
        zmin, zmax = min(zs[0]), max(zs[0])
        zrange = max(((zmax - zmin) / 2, node_packing_factor, 0.1))
        plot_volume *= zrange
    # in total we account for:
    #     - user specified scaling
    #     - number of tensors
    #     - how flat the plot area is (flatter requires smaller nodes)
    full_node_scale = 0.2 * node_scale * node_packing_factor * plot_volume**0.5

    default_outline_size = 6 * full_node_scale**0.5

    # update node size and position attributes
    for node, node_data in nodes.items():
        nodes[node]["size"] = G.nodes[node]["size"] = (
            full_node_scale * node_data["size"]
        )
        nodes[node]["outline_size"] = G.nodes[node]["outline_size"] = (
            default_outline_size * node_data["outline_size"]
        )
        nodes[node]["coo"] = G.nodes[node]["coo"] = pos[node]

    for (i, j), edge_data in edges.items():
        edges[i, j]["coos"] = G.edges[i, j]["coos"] = pos[i], pos[j]

    if get == "pos":
        return pos
    if get == "graph,pos":
        return G, pos

    opts = {
        "colors": colors,
        "node_outline_darkness": node_outline_darkness,
        "title": title,
        "legend": legend,
        "multi_edge_spread": multi_edge_spread,
        "multi_tag_style": multi_tag_style,
        "arrow_opts": arrow_opts,
        "label_color": label_color,
        "font_family": font_family,
        "figsize": figsize,
        "margin": margin,
        "xlims": xlims,
        "ylims": ylims,
        "return_fig": return_fig,
        "ax": ax,
    }

    if get == "data":
        return edges, nodes, opts

    if backend == "matplotlib":
        return _draw_matplotlib(edges=edges, nodes=nodes, **opts)

    if backend == "matplotlib3d":
        return _draw_matplotlib3d(G, **opts)

    if backend == "plotly":
        return _draw_plotly(G, **opts)


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


def _add_legend_matplotlib(
    ax, colors, legend, node_outline_darkness, label_color, font_family
):
    import matplotlib.pyplot as plt

    # create legend
    if colors and legend:
        handles = []
        for color in colors.values():
            ecolor = tuple(
                (1.0 if i == 3 else node_outline_darkness) * c
                for i, c in enumerate(color)
            )
            handles += [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=color,
                    markeredgecolor=ecolor,
                    markeredgewidth=1,
                    linestyle="",
                    markersize=10,
                )
            ]

        # needed in case '_' is the first character
        lbls = [f" {lbl}" for lbl in colors]

        legend = ax.legend(
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


def _draw_matplotlib(
    edges,
    nodes,
    *,
    colors=None,
    node_outline_darkness=0.9,
    title=None,
    legend=True,
    multi_edge_spread=0.1,
    multi_tag_style="auto",
    arrow_opts=None,
    label_color=None,
    font_family="monospace",
    figsize=(6, 6),
    margin=None,
    xlims=None,
    ylims=None,
    return_fig=False,
    ax=None,
):
    import matplotlib.pyplot as plt
    from quimb.schematic import Drawing

    d = Drawing(figsize=figsize, ax=ax)
    if ax is None:
        fig = d.fig
        ax = d.ax
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
    else:
        fig = None

    arrow_opts = arrow_opts or {}
    arrow_opts.setdefault("center", 3 / 4)
    arrow_opts.setdefault("linewidth", 1)
    arrow_opts.setdefault("width", 0.08)
    arrow_opts.setdefault("length", 0.12)

    if title is not None:
        ax.set_title(str(title))

    for _, edge_data in edges.items():
        cooa, coob = edge_data["coos"]
        edge_colors = edge_data["color"]
        edge_sizes = edge_data["edge_size"]
        labels = edge_data["label"]
        arrow_lefts = edge_data["arrow_left"]
        arrow_rights = edge_data["arrow_right"]
        multiplicity = len(edge_colors)

        if multiplicity > 1:
            offsets = np.linspace(
                +multiplicity * multi_edge_spread / 2,
                -multiplicity * multi_edge_spread / 2,
                multiplicity,
            )
        else:
            offsets = None

        for m in range(multiplicity):
            line_opts = dict(
                cooa=cooa,
                coob=coob,
                linewidth=edge_sizes[m],
                color=edge_colors[m],
            )

            arrowhead, reverse = {
                (False, False): (None, False),  # no arrow
                (False, True): (True, False),  # arrowhead to right
                (True, False): (True, True),  # arrowhead to left
                (True, True): (True, "both"),  # arrowheads both sides
            }[arrow_lefts[m], arrow_rights[m]]

            if arrowhead:
                line_opts["arrowhead"] = dict(
                    reverse=reverse,
                    **arrow_opts,
                )

            if labels[m]:
                line_opts["text"] = dict(
                    text=labels[m],
                    fontsize=edge_data["label_fontsize"],
                    color=edge_data["label_color"],
                    fontfamily=edge_data["label_fontfamily"],
                )

            if multiplicity > 1:
                d.line_offset(offset=offsets[m], **line_opts)
            else:
                d.line(**line_opts)

    # draw the tensors
    for _, node_data in nodes.items():
        patch_opts = dict(
            coo=node_data["coo"],
            radius=node_data["size"],
            facecolor=node_data["color"],
            edgecolor=node_data["outline_color"],
            linewidth=node_data["outline_size"],
            hatch=node_data["hatch"],
        )
        marker = node_data["marker"]

        if "multi_colors" in node_data:
            # tensor has multiple tags which are colored

            if multi_tag_style in ("pie", "auto"):
                # draw a mini pie chart
                if marker not in ("o", "."):
                    warnings.warn(
                        "Can only draw multi-colored nodes as circles."
                    )

                angles = np.linspace(
                    0, 360, len(node_data["multi_colors"]) + 1
                )
                for i, (color, outline_color) in enumerate(
                    zip(
                        node_data["multi_colors"],
                        node_data["multi_outline_colors"],
                    )
                ):
                    patch_opts["facecolor"] = color
                    patch_opts["edgecolor"] = outline_color
                    d.wedge(
                        theta1=angles[i] - 67.5,
                        theta2=angles[i + 1] - 67.5,
                        **patch_opts,
                    )
            elif multi_tag_style == "nest":
                # draw nested markers of decreasing size
                radii = np.linspace(
                    node_data["size"], 0, len(node_data["multi_colors"]) + 1
                )
                for i, (color, outline_color) in enumerate(
                    zip(
                        node_data["multi_colors"],
                        node_data["multi_outline_colors"],
                    )
                ):
                    patch_opts["facecolor"] = color
                    patch_opts["edgecolor"] = outline_color
                    d.marker(
                        marker=marker,
                        **{**patch_opts, "radius": radii[i], "linewidth": 0},
                    )
            elif multi_tag_style == "last":
                # draw a single marker with last tag
                patch_opts["facecolor"] = node_data["multi_colors"][-1]
                patch_opts["edgecolor"] = node_data["multi_outline_colors"][-1]
                d.marker(marker=marker, **patch_opts)
            else:  # multi_tag_style == "average":
                # draw a single marker with average color
                patch_opts["facecolor"] = average_color(
                    node_data["multi_colors"]
                )
                patch_opts["edgecolor"] = average_color(
                    node_data["multi_outline_colors"]
                )
                d.marker(marker=marker, **patch_opts)

        else:
            d.marker(marker=marker, **patch_opts)

        if node_data["label"]:
            d.text(
                node_data["coo"],
                node_data["label"],
                fontsize=node_data["label_fontsize"],
                color=node_data["label_color"],
                fontfamily=node_data["label_fontfamily"],
            )

    _add_legend_matplotlib(
        ax, colors, legend, node_outline_darkness, label_color, font_family
    )

    if fig is None:
        # ax was supplied, don't modify and simply return
        return
    else:
        # axes and figure were created
        if xlims is not None:
            ax.set_xlim(xlims)
        if ylims is not None:
            ax.set_ylim(ylims)
        if margin is not None:
            ax.margins(margin)

    if return_fig:
        return fig
    else:
        plt.show()
        plt.close(fig)


def _linearize_graph_data(G, multi_tag_style="auto"):
    edge_source = collections.defaultdict(list)
    for _, _, edge_data in G.edges(data=True):
        cooa, coob = edge_data["coos"]
        x0, y0, *maybe_z0 = cooa
        x1, y1, *maybe_z1 = coob
        edge_source["x0"].append(x0)
        edge_source["y0"].append(y0)
        edge_source["x1"].append(x1)
        edge_source["y1"].append(y1)
        if maybe_z0:
            edge_source["z0"].extend(maybe_z0)
            edge_source["z1"].extend(maybe_z1)

        # we just aggregate all multi-edges into one
        edge_source["color"].append(average_color(edge_data["color"]))
        edge_source["edge_size"].append(sum(edge_data["edge_size"]))
        edge_source["ind"].append(" ".join(edge_data["ind"]))
        edge_source["ind_size"].append(np.prod(edge_data["ind_size"]))
        edge_source["label"].append(" ".join(edge_data["label"]))

    node_source = collections.defaultdict(list)
    for _, node_data in G.nodes(data=True):
        if "ind" in node_data:
            continue

        x, y, *maybe_z = node_data["coo"]

        if "multi_colors" not in node_data:
            # single marker
            mcs = [node_data["color"]]
            mocs = [node_data["outline_color"]]
            szs = [node_data["size"]]
            os = node_data["outline_size"]
        elif multi_tag_style == "average":
            # plot a single marker with average color
            mcs = [average_color(node_data["multi_colors"])]
            mocs = [average_color(node_data["multi_outline_colors"])]
            szs = [node_data["size"]]
            os = node_data["outline_size"]
        elif multi_tag_style == "last":
            # plot a single marker with last tag
            mcs = [node_data["multi_colors"][-1]]
            mocs = [node_data["multi_outline_colors"][-1]]
            szs = [node_data["size"]]
            os = node_data["outline_size"]
        else:  # multi_tag_style in ("auto", "nest"):
            # plot multiple nested markers
            mcs = node_data["multi_colors"]
            mocs = node_data["multi_outline_colors"]
            szs = np.linspace(node_data["size"], 0, len(mcs) + 1)
            os = 0.0

        for mc, moc, sz in zip(mcs, mocs, szs):
            node_source["x"].append(x)
            node_source["y"].append(y)
            if maybe_z:
                node_source["z"].extend(maybe_z)

            node_source["color"].append(mc)
            node_source["outline_color"].append(moc)
            node_source["size"].append(sz)
            node_source["outline_size"].append(os)

            for k in ("hatch", "tags", "shape", "tid", "label"):
                node_source[k].append(node_data.get(k, None))

    return dict(edge_source), dict(node_source)


def _draw_matplotlib3d(G, **kwargs):
    import matplotlib.pyplot as plt

    edge_source, node_source = _linearize_graph_data(
        G, multi_tag_style=kwargs["multi_tag_style"]
    )

    ax = kwargs.pop("ax")
    if ax is None:
        fig = plt.figure(figsize=kwargs["figsize"])
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

    node_source["color"] = [rgba[:3] for rgba in node_source["color"]]
    node_source["size"] = [100000 * s**2 for s in node_source["size"]]
    node_source["linewdith"] = [lw / 50 for lw in node_source["outline_size"]]

    # draw the nodes
    ax.scatter3D(
        xs="x",
        ys="y",
        zs="z",
        c="color",
        s="size",
        alpha=1.0,
        marker="o",
        data=node_source,
        depthshade=False,
        edgecolor=node_source["outline_color"],
        linewidth=node_source["outline_size"],
    )

    for _, node_data in G.nodes(data=True):
        label = node_data["label"]
        if label:
            ax.text(
                *node_data["coo"],
                s=label,
                ha="center",
                va="center",
                color=node_data["label_color"],
                fontsize=node_data["label_fontsize"],
                fontfamily=node_data["label_fontfamily"],
            )

    _add_legend_matplotlib(
        ax,
        kwargs["colors"],
        kwargs["legend"],
        kwargs["node_outline_darkness"],
        kwargs["label_color"],
        kwargs["font_family"],
    )

    if kwargs["return_fig"]:
        return fig
    else:
        plt.show()
        plt.close(fig)


def _draw_plotly(G, **kwargs):
    import plotly.graph_objects as go

    edge_source, node_source = _linearize_graph_data(
        G, multi_tag_style=kwargs["multi_tag_style"]
    )

    fig = go.Figure()
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        width=100 * kwargs["figsize"][0],
        height=100 * kwargs["figsize"][1],
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
                width=edge_source["edge_size"][i],
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
            # edges appear much thinner in 3D
            edge_kwargs["line"]["width"] *= 2
            fig.add_trace(go.Scatter3d(**edge_kwargs))
        else:
            fig.add_trace(go.Scatter(**edge_kwargs))

    node_kwargs = dict(
        x=node_source["x"],
        y=node_source["y"],
        marker=dict(
            opacity=1.0,
            color=list(map(to_rgba_str, node_source["color"])),
            size=[300 * s for s in node_source["size"]],
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


# ---------------------------- layout functions ----------------------------- #


def _normalize_positions(pos):
    # normalize to unit square
    xmin = ymin = zmin = float("inf")
    xmax = ymax = zmax = float("-inf")
    for x, y, *maybe_z in pos.values():
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)
        for z in maybe_z:
            zmin = min(zmin, z)
            zmax = max(zmax, z)

    # maintain aspect ratio:
    # center each dimension separately
    xmid, ymid, zmid = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    # but scale all dimensions by the largest range
    xdiameter, ydiameter, zdiameter = xmax - xmin, ymax - ymin, zmax - zmin
    radius = max((xdiameter, ydiameter, zdiameter)) / 2

    for node, (x, y, *maybe_z) in pos.items():
        pos[node] = (
            (x - xmid) / radius,
            (y - ymid) / radius,
            *((z - zmid) / radius for z in maybe_z),
        )

    return pos


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


def phyllotaxis_points(n):
    """J. Kogan, "A New Computationally Efficient Method for Spacing Points on
    a Sphere," Rose-Hulman Undergraduate Mathematics Journal, 18(2), 2017
    Article 5. scholar.rose-hulman.edu/rhumj/vol18/iss2/5.
    """

    def spherical_coordinate(x, y):
        return [np.cos(x) * np.cos(y), np.sin(x) * np.cos(y), np.sin(y)]

    x = 0.1 + 1.2 * n
    pts = []
    start = -1.0 + 1.0 / (n - 1.0)
    increment = (2.0 - 2.0 / (n - 1.0)) / (n - 1.0)
    for j in range(n):
        s = start + j * increment
        pts.append(
            spherical_coordinate(
                s * x,
                np.pi
                / 2.0
                * np.copysign(1, s)
                * (1.0 - np.sqrt(1.0 - abs(s))),
            )
        )
    return pts


def layout_single_tensor(tn, dim=2):
    """Manually layout indices around a tensor either in a circle or sphere."""
    ((tid, t),) = tn.tensor_map.items()

    if dim == 2.5:
        dim = 3
        project_back_to_2d = True
    else:
        project_back_to_2d = False

    pos = {tid: (0.0,) * dim}
    if dim == 2:
        # fix around a circle
        angles = np.linspace(0, 2 * np.pi, t.ndim, endpoint=False)
        for ind, angle in zip(t.inds, angles):
            pos[ind] = (-np.cos(angle), np.sin(angle))
    else:
        # fix around a sphere
        for ind, coo in zip(t.inds, phyllotaxis_points(t.ndim)):
            pos[ind] = coo

    if project_back_to_2d:
        pos = {k: v[:2] for k, v in pos.items()}

    return pos


def layout_networkx(
    G,
    layout="kamada_kawai",
    pos0=None,
    fixed=None,
    dim=2,
    **kwargs,
):
    import networkx as nx

    layout_fn = getattr(nx, layout + "_layout")

    if pos0 is not None:
        if layout not in ("spring", "kamada_kawai"):
            warnings.warn(
                "Initial positions supplied but layout is not spring-based, "
                "so `pos0` is being ignored."
            )
        else:
            kwargs["pos"] = pos0

    if fixed is not None:
        if layout != "spring":
            warnings.warn(
                "Fixed positions supplied but layout is not spring-based, "
                "so `fixed` is being ignored."
            )
        else:
            kwargs["fixed"] = fixed

    return layout_fn(G, dim=dim, **kwargs)


def layout_pygraphviz(
    G,
    layout="neato",
    pos0=None,
    fixed=None,
    dim=2,
    iterations=None,
    k=None,
    **kwargs,
):
    # TODO: max iters
    # TODO: spring parameter
    # TODO: work out why pos0 and fix don't work
    import pygraphviz as pgv

    if k is not None:
        warnings.warn(
            "`k` is being ignored as layout is being done by pygraphviz."
        )

    aG = pgv.AGraph()

    # create nodes
    if pos0 is not None:
        fixed = fixed or set()
        for node, coo in pos0.items():
            pos = ",".join((f"{w:f}" for w in coo))
            pin = "true" if node in fixed else "false"
            aG.add_node(str(node), pos=pos, pin=pin)

        warnings.warn(
            "Initial and fixed positions don't seem "
            "to work currently with pygraphviz."
        )

    # create edges
    mapping = {}
    for nodea, nodeb in G.edges():
        s_nodea = str(nodea)
        s_nodeb = str(nodeb)
        mapping[s_nodea] = nodea
        mapping[s_nodeb] = nodeb
        aG.add_edge(s_nodea, s_nodeb)

    # layout options
    if iterations is not None:
        kwargs["maxiter"] = iterations
    if dim == 2.5:
        kwargs["dim"] = 3
        kwargs["dimen"] = 2
    else:
        kwargs["dim"] = kwargs["dimen"] = dim
    args = " ".join(f"-G{k}={v}" for k, v in kwargs.items())

    # run layout algorithm
    aG.layout(prog=layout, args=args)

    # extract layout
    pos = {}
    for snode, node in mapping.items():
        spos = aG.get_node(snode).attr["pos"]
        pos[node] = tuple(map(float, spos.split(",")))

    pos = _normalize_positions(pos)
    if dim < 3:
        pos = _massage_pos(pos)

    return pos


def get_positions(
    tn,
    G,
    *,
    dim=2,
    fix=None,
    layout="auto",
    initial_layout="auto",
    refine_layout="auto",
    iterations="auto",
    k=None,
):
    if (tn.num_tensors == 1) and (fix is None):
        # single tensor, layout manually
        return layout_single_tensor(tn, dim=dim)

    if layout != "auto":
        # don't use two step layout with relaxation
        initial_layout = layout
        iterations = 0

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
        # everything is already fixed -> simply normalize
        return _normalize_positions(fix)

    if initial_layout == "auto":
        # automatically select
        if len(G) <= 100:
            # usually nicest
            initial_layout = "kamada_kawai"
        else:
            # faster, but not as nice
            initial_layout = "spectral"

    if refine_layout == "auto":
        # automatically select
        refine_layout = "spring"
        # if len(G) <= 100:
        #     # usually nicest
        #     refine_layout = "fdp"
        # else:
        #     # faster, but not as nice
        #     refine_layout = "sfdp"

    if iterations == "auto":
        # the smaller the graph, the more iterations we can afford
        iterations = max(200, 1000 - len(G))

    if dim == 2.5:
        dim = 3
        project_back_to_2d = True
    else:
        project_back_to_2d = False

    # use spectral or other layout as starting point
    if initial_layout in ("neato", "fdp", "sfdp", "dot"):
        pos0 = layout_pygraphviz(G, initial_layout, dim=dim)
    else:
        pos0 = layout_networkx(G, initial_layout, dim=dim)

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
        if refine_layout == "spring":
            pos = layout_networkx(
                G,
                "spring",
                pos0=pos0,
                fixed=fixed,
                k=k,
                dim=dim,
                iterations=iterations,
            )
        elif refine_layout in ("fdp", "sfdp", "neato"):
            # XXX: currently doesn't seem to work with pos0 and fixed
            pos = layout_pygraphviz(
                G,
                refine_layout,
                pos0=pos0,
                fixed=fixed,
                k=k,
                dim=dim,
                iterations=iterations,
            )
        else:
            raise ValueError(f"Unknown refining layout {refine_layout}.")
    else:
        # no relaxation
        pos = pos0

    if project_back_to_2d:
        # ignore z-coordinate
        pos = {k: v[:2] for k, v in pos.items()}
        dim = 2

    # map all to range [-1, +1], but preserving aspect ratio
    pos = _normalize_positions(pos)

    if (not fix) and (dim == 2):
        # finally rotate them to cover a small vertical span
        pos = _massage_pos(pos)

    return pos


# ----------------------------- color functions ----------------------------- #

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


def average_color(colors):
    from matplotlib.colors import to_rgba

    # first map to rgba
    colors = [to_rgba(c) for c in colors]

    r, g, b, a = zip(*colors)

    # then RMS average each channel
    rm = (sum(ri**2 for ri in r) / len(r)) ** 0.5
    gm = (sum(gi**2 for gi in g) / len(g)) ** 0.5
    bm = (sum(bi**2 for bi in b) / len(b)) ** 0.5
    am = sum(a) / len(a)

    return (rm, gm, bm, am)


def darken_color(rgba, darkness=0.8):
    """Return a darker color."""
    return tuple(x if i == 3 else (darkness) * x for i, x in enumerate(rgba))


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
    """Hash the string ``s`` to ``nval`` different floats in the range [0, 1]."""
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
    """Automatically hash and color a string for HTML display."""
    if not isinstance(s, str):
        s = str(s)
    return f'<b style="color: {hash_to_color(s)};">{s}</b>'


# ---------------------------- tensor functions ----------------------------- #


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


def choose_squarest_grid(x):
    p = x**0.5
    if p.is_integer():
        m = n = int(p)
    else:
        m = int(round(p))
        p = int(p)
        n = p if m * p >= x else p + 1
    return m, n


def visualize_tensors(
    tn,
    mode="network",
    r=None,
    r_scale=1.0,
    figsize=None,
    **visualize_opts,
):
    """Visualize all the entries of every tensor in this network.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to visualize.
    mode : {'network', 'grid', 'row', 'col'}, optional
        How to arrange each tensor's visualization.

        - ``'network'``: arrange each tensor's visualization according to the
            automatic layout given by ``draw``.
        - ``'grid'``: arrange each tensor's visualization in a grid.
        - ``'row'``: arrange each tensor's visualization horizontally.
        - ``'col'``: arrange each tensor's visualization vertically.

    r : float, optional
        The absolute radius of each tensor's visualization, when
        ``mode='network'``.
    r_scale : float, optional
        A relative scaling factor for the radius of each tensor's
        visualization, when ``mode='network'``.
    figsize : tuple, optional
        The size of the figure to create, if ``ax`` is not provided.
    visualize_opts
        Supplied to ``visualize_tensor``.
    """
    from matplotlib import pyplot as plt

    if figsize is None:
        figsize = (2 * tn.num_tensors**0.4, 2 * tn.num_tensors**0.4)
    if r is None:
        r = 1.0 / tn.num_tensors**0.5
    r *= r_scale

    max_mag = None
    visualize_opts.setdefault("max_mag", max_mag)
    visualize_opts.setdefault("size_scale", r)

    if mode == "network":
        fig = plt.figure(figsize=figsize)
        pos = tn.draw(get="pos")
        for tid, (x, y) in pos.items():
            if tid not in tn.tensor_map:
                # hyper indez
                continue
            x = (x + 1) / 2 - r / 2
            y = (y + 1) / 2 - r / 2
            ax = fig.add_axes((x, y, r / 2, r / 2))
            tn.tensor_map[tid].visualize(ax=ax, **visualize_opts)
    else:
        if mode == "grid":
            px, py = choose_squarest_grid(tn.num_tensors)
        elif mode == "row":
            px, py = tn.num_tensors, 1
            figsize = (2 * figsize[0], figsize[1] / 2)
        elif mode == "col":
            px, py = 1, tn.num_tensors
            figsize = (figsize[0] / 2, 2 * figsize[1])

        fig, axs = plt.subplots(py, px, figsize=figsize)
        for i, t in enumerate(tn):
            t.visualize(ax=axs.flat[i], **visualize_opts)
        for ax in axs.flat[i:]:
            ax.set_axis_off()

    # transparent background
    fig.patch.set_alpha(0.0)

    plt.show()
    plt.close()
