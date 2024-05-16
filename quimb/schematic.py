"""Draw psuedo-3D diagrams using matplotlib.
"""

import functools
import warnings
from math import atan2, cos, pi, sin

import matplotlib as mpl
import matplotlib.pyplot as plt


class Drawing:
    """Draw 2D or pseudo-3D diagrams using matplotlib. This handles the
    axonometric projection and the z-ordering of the elements, as well as named
    preset styles for repeated elements, and the automatic adjustment of the
    figure limits. It also has basic support for drawing smooth curves and
    shaded areas around certain elements automatically.

    Parameters
    ----------
    background : color, optional
        The background color of the figure, defaults to transparent.
    drawcolor : color, optional
        The default color to draw lines and text in.
    shapecolor : color, optional
        The default color to fill shapes with.
    a : float
        The axonometric angle of the x-axis in degrees.
    b : float
        The axonometric angle of the y-axis in degrees.
    xscale : float
        A factor to scale the x-axis by.
    yscale : float
        A factor to scale the y-axis by.
    zscale : float
        A factor to scale the z-axis by.
    presets : dict
        A dictionary of named style presets. When you add an element to the
        drawing, you can specify a preset name to use as default styling.
    ax : matplotlib.axes.Axes
        The axes to draw on. If None, a new figure is created.
    kwargs
        Passed to ``plt.figure`` if ``ax`` is None.
    """

    def __init__(
        self,
        background=(0, 0, 0, 0),
        drawcolor=(0.14, 0.15, 0.16, 1.0),
        shapecolor=(0.45, 0.50, 0.55, 1.0),
        a=50,
        b=12,
        xscale=1,
        yscale=1,
        zscale=1,
        presets=None,
        ax=None,
        **kwargs,
    ):
        if ax is None:
            self.fig = plt.figure(**kwargs)
            self.fig.set_facecolor(background)
            self.ax = self.fig.add_subplot(111)
        else:
            self.ax = ax
            self.fig = self.ax.figure

        self.ax.set_axis_off()
        self.ax.set_aspect("equal")
        self.ax.set_clip_on(False)

        self.drawcolor = drawcolor
        self.shapecolor = shapecolor

        self._xmin = None
        self._xmax = None
        self._ymin = None
        self._ymax = None
        self.presets = {} if presets is None else dict(presets)
        self.presets.setdefault(None, {})

        self._3d_project = functools.partial(
            axonometric_project,
            a=a,
            b=b,
            xscale=xscale,
            yscale=yscale,
            zscale=zscale,
        )
        self._coo_to_zorder = functools.partial(
            coo_to_zorder,
            xscale=xscale,
            yscale=yscale,
            zscale=zscale,
        )

    def _adjust_lims(self, x, y):
        xchange = ychange = False
        if self._xmin is None or x < self._xmin:
            xchange = True
            self._xmin = x

        if self._xmax is None or x > self._xmax:
            xchange = True
            self._xmax = x

        if self._ymin is None or y < self._ymin:
            ychange = True
            self._ymin = y

        if self._ymax is None or y > self._ymax:
            ychange = True
            self._ymax = y

        if xchange and self._xmin != self._xmax:
            dx = self._xmax - self._xmin
            plot_xmin = self._xmin - dx * 0.01
            plot_xmax = self._xmax + dx * 0.01
            self.ax.set_xlim(plot_xmin, plot_xmax)
        if ychange and self._ymin != self._ymax:
            dy = self._ymax - self._ymin
            plot_ymin = self._ymin - dy * 0.01
            plot_ymax = self._ymax + dy * 0.01
            self.ax.set_ylim(plot_ymin, plot_ymax)

    def text(self, coo, text, preset=None, **kwargs):
        """Place text at the specified coordinate.

        Parameters
        ----------
        coo : tuple[int, int] or tuple[int, int, int]
            The 2D or 3D coordinate of the text. If 3D, the coordinate will be
            projected onto the 2D plane, and a z-order will be assigned.
        text : str
            The text to place.
        preset : str, optional
            A preset style to use for the text.
        kwargs
            Specific style options passed to ``matplotlib.axes.Axes.text``.
        """
        style = parse_style_preset(self.presets, preset, **kwargs)
        style.setdefault("color", self.drawcolor)
        style.setdefault("horizontalalignment", "center")
        style.setdefault("verticalalignment", "center")
        style.setdefault("clip_on", False)

        if len(coo) == 2:
            x, y = coo
            style.setdefault("zorder", +0.02)
        else:
            x, y = self._3d_project(*coo)
            style.setdefault("zorder", self._coo_to_zorder(*coo) + 0.02)

        self.ax.text(x, y, text, **style)
        self._adjust_lims(x, y)

    def text_between(self, cooa, coob, text, preset=None, **kwargs):
        """Place text between two coordinates.

        Parameters
        ----------
        cooa, coob : tuple[int, int] or tuple[int, int, int]
            The 2D or 3D coordinates of the text endpoints. If 3D, the
            coordinates will be projected onto the 2D plane, and a z-order
            will be assigned based on average z-order of the endpoints.
        text : str
            The text to place.
        center : float, optional
            The position of the text along the line, where 0.0 is the start and
            1.0 is the end. Default is 0.5.
        preset : str, optional
            A preset style to use for the text.
        kwargs
            Specific style options passed to ``matplotlib.axes.Axes.text``.
        """
        style = parse_style_preset(self.presets, preset, **kwargs)
        style.setdefault("color", self.drawcolor)
        style.setdefault("horizontalalignment", "center")
        style.setdefault("verticalalignment", "center")
        style.setdefault("clip_on", False)
        center = style.pop("center", 0.5)

        if len(cooa) == 2:
            xa, ya = cooa
            xb, yb = coob
            style.setdefault("zorder", +0.02)
        else:
            style.setdefault(
                "zorder",
                mean(self._coo_to_zorder(*coo) for coo in [cooa, coob]) + 0.02,
            )
            xa, ya = self._3d_project(*cooa)
            xb, yb = self._3d_project(*coob)

        # compute midpoint
        x = xa * (1 - center) + xb * center
        y = ya * (1 - center) + yb * center

        # compute angle
        if xa <= xb:
            angle = atan2(yb - ya, xb - xa) * 180 / pi
        else:
            angle = atan2(ya - yb, xa - xb) * 180 / pi
        style.setdefault("rotation", angle)

        self.ax.text(x, y, text, **style)
        self._adjust_lims(x, y)

    def label_ax(self, x, y, text, preset=None, **kwargs):
        """Place text at the specified location, using the axis coordinates
        rather than 2D or 3D data coordinates.

        Parameters
        ----------
        x, y : float
            The x and y positions of the text, relative to the axis.
        text : str
            The text to place.
        preset : str, optional
            A preset style to use for the text.
        kwargs
            Specific style options passed to ``matplotlib.axes.Axes.text``.
        """
        style = parse_style_preset(self.presets, preset, **kwargs)
        style.setdefault("color", self.drawcolor)
        style.setdefault("horizontalalignment", "center")
        style.setdefault("verticalalignment", "center")
        style.setdefault("transform", self.ax.transAxes)
        self.ax.text(x, y, text, **style)
        self._adjust_lims(x, y)

    def label_fig(self, x, y, text, preset=None, **kwargs):
        """Place text at the specified location, using the figure coordinates
        rather than 2D or 3D data coordinates.

        Parameters
        ----------
        x, y : float
            The x and y positions of the text, relative to the figure.
        text : str
            The text to place.
        preset : str, optional
            A preset style to use for the text.
        kwargs
            Specific style options passed to ``matplotlib.axes.Axes.text``.
        """
        style = parse_style_preset(self.presets, preset, **kwargs)
        style.setdefault("color", self.drawcolor)
        style.setdefault("horizontalalignment", "center")
        style.setdefault("verticalalignment", "center")
        style.setdefault("transform", self.fig.transFigure)
        self.ax.text(x, y, text, **style)
        self._adjust_lims(x, y)

    def _parse_style_for_marker(self, coo, preset=None, **kwargs):
        style = parse_style_preset(self.presets, preset, **kwargs)
        if "color" in style:
            # assume coloring whole shape
            style.setdefault("facecolor", style.pop("color"))
        style.setdefault("facecolor", self.shapecolor)
        style.setdefault("edgecolor", darken_color(style["facecolor"]))
        style.setdefault("linewidth", 1)
        style.setdefault("radius", 0.25)

        if len(coo) == 2:
            x, y = coo
            style.setdefault("zorder", +0.01)
        else:
            x, y = self._3d_project(*coo)
            style.setdefault("zorder", self._coo_to_zorder(*coo) + 0.01)

        return x, y, style

    def _adjust_lims_for_marker(self, x, y, r):
        for x, y in [
            (x - 1.1 * r, y),
            (x + 1.1 * r, y),
            (x, y - 1.1 * r),
            (x, y + 1.1 * r),
        ]:
            self._adjust_lims(x, y)

    def circle(self, coo, preset=None, **kwargs):
        """Draw a circle at the specified coordinate.

        Parameters
        ----------
        coo : tuple[int, int] or tuple[int, int, int]
            The 2D or 3D coordinate of the circle. If 3D, the coordinate will
            be projected onto the 2D plane, and a z-order will be assigned.
        preset : str, optional
            A preset style to use for the circle.
        kwargs
            Specific style options passed to ``matplotlib.patches.Circle``.
        """
        x, y, style = self._parse_style_for_marker(
            coo, preset=preset, **kwargs
        )
        circle = mpl.patches.Circle((x, y), **style)
        self.ax.add_artist(circle)
        self._adjust_lims_for_marker(x, y, style["radius"])

    def wedge(self, coo, theta1, theta2, preset=None, **kwargs):
        """Draw a wedge at the specified coordinate.

        Parameters
        ----------
        coo : tuple[int, int] or tuple[int, int, int]
            The 2D or 3D coordinate of the wedge. If 3D, the coordinate will
            be projected onto the 2D plane, and a z-order will be assigned.
        theta1 : float
            The angle in degrees of the first edge of the wedge.
        theta2 : float
            The angle in degrees of the second edge of the wedge.
        preset : str, optional
            A preset style to use for the wedge.
        kwargs
            Specific style options passed to ``matplotlib.patches.Wedge``.
        """
        x, y, style = self._parse_style_for_marker(
            coo, preset=preset, **kwargs
        )

        # wedge uses r, not radius
        style["r"] = style.pop("radius")
        # and is not filled by default
        style.setdefault("fill", True)

        wedge = mpl.patches.Wedge(
            (x, y), theta1=theta1, theta2=theta2, **style
        )

        self.ax.add_artist(wedge)
        self._adjust_lims_for_marker(x, y, style["r"])

    def dot(self, coo, preset=None, **kwargs):
        """Draw a small circle with no border. Alias for circle with defaults
        `radius=0.1` and `linewidth=0.0`.

        Parameters
        ----------
        coo : tuple[int, int] or tuple[int, int, int]
            The 2D or 3D coordinate of the dot. If 3D, the coordinate will
            be projected onto the 2D plane, and a z-order will be assigned.
        preset : str, optional
            A preset style to use for the dot.
        kwargs
            Specific style options passed to ``matplotlib.patches.Circle``.
        """
        style = parse_style_preset(self.presets, preset, **kwargs)
        style.setdefault("radius", 0.1)
        style.setdefault("linewidth", 0.0)
        self.circle(coo, **style)

    def regular_polygon(self, coo, preset=None, **kwargs):
        """Draw a regular polygon at the specified coordinate.

        Parameters
        ----------
        coo : tuple[int, int] or tuple[int, int, int]
            The 2D or 3D coordinate of the polygon. If 3D, the coordinate will
            be projected onto the 2D plane, and a z-order will be assigned.
        n : int
            The number of sides of the polygon.
        orientation : float, optional
            The orientation of the polygon in radians. Default is 0.0.
        preset : str, optional
            A preset style to use for the polygon.
        kwargs
            Specific style options passed to ``matplotlib.patches.Polygon``.
        """
        x, y, style = self._parse_style_for_marker(
            coo, preset=preset, **kwargs
        )

        n = style.pop("n", 3)
        orientation = style.pop("orientation", 0.0)

        rpoly = mpl.patches.RegularPolygon(
            (x, y), numVertices=n, orientation=orientation, **style
        )
        self.ax.add_artist(rpoly)
        self._adjust_lims_for_marker(x, y, style["radius"])

    def marker(self, coo, preset=None, **kwargs):
        """Draw a 'marker' at the specified coordinate. This is a shorthand for
        creating polygons with shape specified by a single character.

        Parameters
        ----------
        coo : tuple[int, int] or tuple[int, int, int]
            The 2D or 3D coordinate of the marker. If 3D, the coordinate will
            be projected onto the 2D plane, and a z-order will be assigned.
        marker : str, optional
            The marker shape to draw. One of ``"o.v^<>sDphH8"``.
        preset : str, optional
            A preset style to use for the marker.
        kwargs
            Specific style options.
        """
        style = parse_style_preset(self.presets, preset, **kwargs)
        marker = style.pop("marker", "s")
        if marker in ("o", "."):
            return self.circle(coo, preset=preset, **style)

        if isinstance(marker, int):
            n = marker
            orientation = 0.0
        else:
            n, orientation = {
                "v": (3, pi / 3),
                "^": (3, 0),
                "<": (3, pi / 2),
                ">": (3, -pi / 2),
                "s": (4, pi / 4),
                "D": (4, 0),
                "p": (5, 0),
                "h": (6, 0),
                "H": (6, pi / 2),
                "8": (8, 0),
            }[marker]

        self.regular_polygon(
            coo, preset=preset, n=n, orientation=orientation, **style
        )

    def cube(self, coo, preset=None, **kwargs):
        """Draw a cube at the specified coordinate, which must be 3D.

        Parameters
        ----------
        coo : tuple[int, int, int]
            The 3D coordinate of the cube. The coordinate will
            be projected onto the 2D plane, and a z-order will be assigned.
        preset : str, optional
            A preset style to use for the cube.
        kwargs
            Specific style options passed to ``matplotlib.patches.Polygon``.
        """
        style = parse_style_preset(self.presets, preset, **kwargs)

        r = style.pop("radius", 0.25)
        x, y, z = coo
        xm, xp = x - r, x + r
        ym, yp = y - r, y + r
        zm, zp = z - r, z + r

        faces = [
            ((xm, ym, zm), (xm, ym, zp), (xm, yp, zp), (xm, yp, zm)),
            ((xp, ym, zm), (xp, ym, zp), (xp, yp, zp), (xp, yp, zm)),
            ((xp, ym, zm), (xp, ym, zp), (xm, ym, zp), (xm, ym, zm)),
            ((xp, yp, zm), (xp, yp, zp), (xm, yp, zp), (xm, yp, zm)),
            ((xp, ym, zm), (xp, yp, zm), (xm, yp, zm), (xm, ym, zm)),
            ((xp, ym, zp), (xp, yp, zp), (xm, yp, zp), (xm, ym, zp)),
        ]
        for face in faces:
            self.shape(face, preset=preset, **style)

    def line(self, cooa, coob, preset=None, **kwargs):
        """Draw a line between two coordinates.

        Parameters
        ----------
        cooa, coob : tuple[int, int] or tuple[int, int, int]
            The 2D or 3D coordinates of the line endpoints. If 3D, the
            coordinates will be projected onto the 2D plane, and a z-order
            will be assigned based on average z-order of the endpoints.
        stretch : float
            Stretch the line by this factor. 1.0 is no stretch, 0.5 is half
            length, 2.0 is double length. Default is 1.0.
        arrowhead : bool or dict, optional
            Draw an arrowhead at the end of the line. Default is False. If a
            dict, it is passed as keyword arguments to the arrowhead method.
        text_between : str, optional
            Add text along the line.
        preset : str, optional
            A preset style to use for the line.
        kwargs
            Specific style options passed to ``matplotlib.lines.Line2D``.

        See Also
        --------
        Drawing.arrowhead
        """
        style = parse_style_preset(self.presets, preset, **kwargs)
        style.setdefault("color", self.drawcolor)
        style.setdefault("solid_capstyle", "round")
        style.setdefault("stretch", 1.0)
        style.setdefault("arrowhead", None)
        style.setdefault("text", None)
        stretch = style.pop("stretch")
        arrowhead = style.pop("arrowhead")
        text = style.pop("text")

        if len(cooa) == 2:
            xs, ys = zip(*(cooa, coob))
            style.setdefault("zorder", +0.0)
        else:
            style.setdefault(
                "zorder",
                mean(self._coo_to_zorder(*coo) for coo in [cooa, coob]),
            )
            xs, ys = zip(*[self._3d_project(*coo) for coo in [cooa, coob]])

        if stretch != 1.0:
            # shorten around center
            center = mean(xs), mean(ys)
            xs = [center[0] + stretch * (x - center[0]) for x in xs]
            ys = [center[1] + stretch * (y - center[1]) for y in ys]

        if arrowhead is not None:
            if arrowhead is True:
                arrowhead = {}
            else:
                arrowhead = dict(arrowhead)
            self.arrowhead(cooa, coob, preset=preset, **(style | arrowhead))

        line = mpl.lines.Line2D(xs, ys, **style)
        self.ax.add_artist(line)

        if text:
            if isinstance(text, str):
                text = {"text": text}
            else:
                text = dict(text)

            # don't want to pass full style dict to text_between
            text.setdefault("zorder", style["zorder"])
            self.text_between(cooa, coob, **text)

        for x, y in zip(xs, ys):
            self._adjust_lims(x, y)

    def line_offset(
        self,
        cooa,
        coob,
        offset,
        midlength=0.5,
        relative=True,
        preset=None,
        **kwargs,
    ):
        """Draw a line between two coordinates, but curving out by a given
        offset perpendicular to the line.

        Parameters
        ----------
        cooa, coob : tuple[int, int] or tuple[int, int, int]
            The 2D or 3D coordinates of the line endpoints. If 3D, the
            coordinates will be projected onto the 2D plane, and a z-order
            will be assigned based on average z-order of the endpoints.
        offset : float
            The offset of the curve from the line, as a fraction of the total
            line length. This is always processed in the 2D projected plane.
        midlength : float
            The length of the middle straight section, as a fraction of the
            total line length. Default is 0.5.
        arrowhead : bool or dict, optional
            Draw an arrowhead at the end of the line. Default is False. If a
            dict, it is passed as keyword arguments to the arrowhead method.
        text_between : str, optional
            Add text along the line.
        relative : bool, optional
            If ``True`` (the default), then ``offset`` is taken as a fraction
            of the line length, else in absolute units.
        preset : str, optional
            A preset style to use for the line.
        kwargs
            Specific style options passed to ``curve``.
        """
        style = parse_style_preset(self.presets, preset, **kwargs)
        style.setdefault("arrowhead", None)
        style.setdefault("text", None)
        arrowhead = style.pop("arrowhead")
        text = style.pop("text")

        if len(cooa) == 2:
            xs, ys = zip(*(cooa, coob))
            style.setdefault("zorder", +0.0)
        else:
            style.setdefault(
                "zorder",
                mean(self._coo_to_zorder(*coo) for coo in [cooa, coob]),
            )
            xs, ys = zip(*[self._3d_project(*coo) for coo in [cooa, coob]])

        cooa = xs[0], ys[0]
        coob = xs[1], ys[1]
        forward, inverse = get_rotator_and_inverse(cooa, coob)
        R = forward(*coob)[0]

        if relative:
            offset *= R

        endlength = (1 - midlength) / 2
        cooml = inverse(endlength * R, offset)
        coomm = inverse(R / 2, offset)
        coomr = inverse((1 - endlength) * R, offset)
        curve_pts = [cooa, cooml, coomm, coomr, coob]

        if arrowhead is not None:
            if arrowhead is True:
                arrowhead = {}
            else:
                arrowhead = dict(arrowhead)

            # want to correct center for midlength
            center = arrowhead.pop("center", 0.5)
            arrowhead["center"] = min(
                max(0.0, 0.5 + (center - 0.5) / midlength), 1.0
            )
            self.arrowhead(cooml, coomr, preset=preset, **(style | arrowhead))

        self.curve(curve_pts, preset=preset, **style)

        if text:
            if isinstance(text, str):
                text = {"text": text}
            else:
                text = dict(text)
            # don't want to pass full style dict to text_between
            text.setdefault("zorder", style["zorder"])
            self.text_between(cooml, coomr, **text)

        for coo in curve_pts:
            self._adjust_lims(*coo)

    def arrowhead(self, cooa, coob, preset=None, **kwargs):
        """Draw just a arrowhead on the line between ``cooa`` and ``coob``.

        Parameters
        ----------
        cooa, coob : tuple[int, int] or tuple[int, int, int]
            The coordinates of the start and end of the line. If 3D, the
            coordinates will be projected onto the 2D plane, and a z-order
            will be assigned based on average z-order of the endpoints.
        reverse : bool or "both", optional
            Reverse the direction by switching ``cooa`` and ``coob``. If
            ``"both"``, draw an arrowhead in both directions. Default is
            False.
        center : float, optional
            The position of the arrowhead along the line, where 0 is the start
            and 1 is the end. Default is 0.5.
        width : float, optional
            The width of the arrowhead. Default is 0.05.
        length : float, optional
            The length of the arrowhead. Default is 0.1.
        preset : str, optional
            A preset style to use for the arrowhead, including the above
            options.
        kwargs
            Specific style options passed to ``matplotlib.lines.Line2D``.
        """
        style = parse_style_preset(self.presets, preset, **kwargs)
        style.setdefault("color", self.drawcolor)
        style.setdefault("center", 0.5)
        style.setdefault("width", 0.05)
        style.setdefault("length", 0.1)
        style.setdefault("reverse", False)

        reverse = style.pop("reverse")
        if reverse == "both":
            self.arrowhead(cooa, coob, preset=preset, **style)
        if reverse:
            cooa, coob = coob, cooa

        if len(cooa) != 2:
            style.setdefault(
                "zorder",
                mean(self._coo_to_zorder(*coo) for coo in [cooa, coob]),
            )
            cooa = self._3d_project(*cooa)
            coob = self._3d_project(*coob)

        forward, inverse = get_rotator_and_inverse(cooa, coob)
        rb = forward(*coob)

        center = style.pop("center")
        width = style.pop("width")
        length = style.pop("length")
        lab = rb[0]
        xc = center * lab
        arrow_x = xc - length * lab
        arrow_y = width * lab

        aa, ab, ac = [
            inverse(arrow_x, arrow_y),
            inverse(xc, 0),
            inverse(arrow_x, -arrow_y),
        ]

        line = mpl.lines.Line2D(*zip(*(aa, ab, ac)), **style)
        self.ax.add_artist(line)
        for x, y in [aa, ab, ac]:
            self._adjust_lims(x, y)

    def curve(self, coos, preset=None, **kwargs):
        """Draw a smooth line through the given coordinates.

        Parameters
        ----------
        coos : Sequence[tuple[int, int]] or Sequence[tuple[int, int, int]]
            The 2D or 3D coordinates of the line. If 3D, the coordinates will
            be projected onto the 2D plane, and a z-order will be assigned
            based on average z-order of the endpoints.
        smoothing : float, optional
            The amount of smoothing to apply to the curve. 0.0 is no smoothing,
            1.0 is maximum smoothing. Default is 0.5.
        preset : str, optional
            A preset style to use for the curve.
        kwargs
            Specific style options passed to ``matplotlib.lines.Line2D``.
        """
        from matplotlib.path import Path

        style = parse_style_preset(self.presets, preset, **kwargs)
        if "color" in style:
            # presume that edge color is being specified
            style.setdefault("edgecolor", style.pop("color"))
        style.setdefault("edgecolor", self.drawcolor)
        style.setdefault("fill", False)
        style.setdefault("capstyle", "round")
        style.setdefault("smoothing", 1 / 2)
        smoothing = style.pop("smoothing")

        if len(coos[0]) != 2:
            style.setdefault(
                "zorder", mean(self._coo_to_zorder(*coo) for coo in coos)
            )
            coos = [self._3d_project(*coo) for coo in coos]

        N = len(coos)

        if N <= 2 or smoothing == 0.0:
            path = coos
            moves = [Path.MOVETO] + [Path.LINETO] * (N - 1)
            control_pts = {}
        else:
            control_pts = {}
            for i in range(1, N - 1):
                control_pts[i, "l"], control_pts[i, "r"] = get_control_points(
                    coos[i - 1],
                    coos[i],
                    coos[i + 1],
                    spacing=smoothing / 2,
                )

            path = [coos[0], control_pts[1, "l"], coos[1]]
            moves = [Path.MOVETO, Path.CURVE3, Path.CURVE3]

            for i in range(1, N - 2):
                path.extend(
                    (control_pts[i, "r"], control_pts[i + 1, "l"], coos[i + 1])
                )
                moves.extend((Path.CURVE4, Path.CURVE4, Path.CURVE4))

            path.extend((control_pts[N - 2, "r"], coos[N - 1]))
            moves.extend((Path.CURVE3, Path.CURVE3))

        curve = mpl.patches.PathPatch(Path(path, moves), **style)
        self.ax.add_patch(curve)

        for coo in coos:
            self._adjust_lims(*coo)
        for coo in control_pts.values():
            self._adjust_lims(*coo)

    def shape(self, coos, preset=None, **kwargs):
        """Draw a closed shape with (sharp) corners at the given coordinates.

        Parameters
        ----------
        coos : sequence of coordinates
            The coordinates of the corners' of the shape.
        preset : str, optional
            A preset style to use for the shape.
        kwargs
            Specific style options passed to ``matplotlib.patches.PathPatch``.

        See Also
        --------
        Drawing.patch
        """
        from matplotlib.path import Path

        style = parse_style_preset(self.presets, preset, **kwargs)
        if "color" in style:
            style.setdefault("facecolor", style.pop("color"))
        style.setdefault("facecolor", self.shapecolor)
        style.setdefault("edgecolor", darken_color(style["facecolor"]))
        style.setdefault("linewidth", 1)
        style.setdefault("joinstyle", "round")

        if len(coos[0]) != 2:
            style.setdefault(
                "zorder", mean(self._coo_to_zorder(*coo) for coo in coos)
            )
            coos = [self._3d_project(*coo) for coo in coos]

        path = [coos[0]]
        moves = [Path.MOVETO]
        for coo in coos[1:]:
            path.append(coo)
            moves.append(Path.LINETO)
        path.append(coos[0])
        moves.append(Path.CLOSEPOLY)

        curve = mpl.patches.PathPatch(Path(path, moves), **style)
        self.ax.add_patch(curve)

        for coo in coos:
            self._adjust_lims(*coo)

    def patch(self, coos, preset=None, **kwargs):
        """Draw a closed smooth patch through given coordinates.

        Parameters
        ----------
        coos : sequence of coordinates
            The coordinates of the 'corners' of the patch, the outline is
            guaranteed to pass through these points.
        smoothing : float
            The smoothing factor, the higher the smoother. The default is
            0.5.
        preset : str, optional
            A preset style to use for the patch.
        kwargs
            Specific style options passed to ``matplotlib.patches.PathPatch``.

        See Also
        --------
        Drawing.shape, Drawing.curve
        """
        from matplotlib.path import Path

        style = parse_style_preset(self.presets, preset, **kwargs)
        style.setdefault("linestyle", ":")
        style.setdefault("edgecolor", (0.5, 0.5, 0.5, 0.75))
        style.setdefault("facecolor", (0.5, 0.5, 0.5, 0.25))
        style.setdefault("smoothing", 1 / 2)
        smoothing = style.pop("smoothing")

        if len(coos[0]) != 2:
            # use min so the patch appears *just* behind the elements
            # its meant to highlight
            style.setdefault(
                "zorder", min(self._coo_to_zorder(*coo) for coo in coos) - 0.01
            )
            coos = [self._3d_project(*coo) for coo in coos]
        else:
            style.setdefault("zorder", -0.01)

        N = len(coos)

        control_pts = {}
        for i in range(N):
            control_pts[i, "l"], control_pts[i, "r"] = get_control_points(
                coos[(i - 1) % N],
                coos[i],
                coos[(i + 1) % N],
                spacing=smoothing / 2,
            )

        path = [coos[0]]
        moves = [Path.MOVETO]
        for ia in range(N):
            ib = (ia + 1) % N
            path.append(control_pts[ia, "r"])
            path.append(control_pts[ib, "l"])
            path.append(coos[ib])
            moves.append(Path.CURVE4)
            moves.append(Path.CURVE4)
            moves.append(Path.CURVE4)

        curve = mpl.patches.PathPatch(Path(path, moves), **style)
        self.ax.add_patch(curve)

        for coo in coos:
            self._adjust_lims(*coo)
        for coo in control_pts.values():
            self._adjust_lims(*coo)

    def patch_around(
        self, coos, radius=0.0, resolution=12, preset=None, **kwargs
    ):
        """Draw a patch around the given coordinates, by contructing a convex
        hull around the points, optionally including an extra uniform or per
        coordinate radius.

        Parameters
        ----------
        coos : sequence[tuple[int, int]] or sequence[tuple[int, int, int]]
            The coordinates of the points to draw the patch around. If 3D, the
            coordinates will be projected onto the 2D plane, and a z-order will
            be assigned based on *min* z-order of the endpoints.
        radius : float or sequence[float], optional
            The radius of the patch around each point. If a sequence, must be
            the same length as ``coos``. Default is 0.0.
        resolution : int, optional
            The number of points to use pad around each point. Default is 12.
        preset : str, optional
            A preset style to use for the patch.
        kwargs
            Specific style options passed to ``matplotlib.patches.PathPatch``.
        """
        import numpy as np
        from scipy.spatial import ConvexHull

        style = parse_style_preset(self.presets, preset, **kwargs)

        if isinstance(radius, (int, float)):
            radius = [radius] * len(coos)

        if len(coos[0]) != 2:
            # use min so the patch appears *just* behind the elements
            # its meant to highlight
            style.setdefault(
                "zorder", min(self._coo_to_zorder(*coo) for coo in coos) - 0.01
            )
            coos = [self._3d_project(*coo) for coo in coos]
        else:
            style.setdefault("zorder", -0.01)

        expanded_pts = []
        for coo, r in zip(coos, radius):
            if r == 0:
                expanded_pts.append(coo)
            else:
                expanded_pts.extend(gen_points_around(coo, r, resolution))

        if len(expanded_pts) <= 3:
            # need at least 3 points to make convex hull
            boundary_pts = expanded_pts
        else:
            expanded_pts = np.array(expanded_pts)
            hull = ConvexHull(expanded_pts)
            boundary_pts = expanded_pts[hull.vertices]

        self.patch(boundary_pts, preset=preset, **style)

    def patch_around_circles(
        self,
        cooa,
        ra,
        coob,
        rb,
        padding=0.2,
        pinch=True,
        preset=None,
        **kwargs,
    ):
        """Draw a smooth patch around two circles.

        Parameters
        ----------
        cooa : tuple[int, int] or tuple[int, int, int]
            The coordinates of the center of the first circle. If 3D, the
            coordinates will be projected onto the 2D plane, and a z-order
            will be assigned based on average z-order of the endpoints.
        ra : int
            The radius of the first circle.
        coob : tuple[int, int] or tuple[int, int, int]
            The coordinates of the center of the second circle. If 3D, the
            coordinates will be projected onto the 2D plane, and a z-order
            will be assigned based on average z-order of the endpoints.
        rb : int
            The radius of the second circle.
        padding : float, optional
            The amount of padding to add around the circles. Default is 0.2.
        pinch : bool or float, optional
            If or how much to pinch the patch in between the circles.
            Default is to match the padding.
        preset : str, optional
            A preset style to use for the patch.
        kwargs
            Specific style options passed to ``matplotlib.patches.PathPatch``.

        See Also
        --------
        Drawing.patch
        """
        style = parse_style_preset(self.presets, preset, **kwargs)
        style.setdefault("smoothing", 1.0)

        if pinch is True:
            pinch = 1 - padding

        if len(cooa) != 2:
            style.setdefault(
                "zorder",
                min(self._coo_to_zorder(*coo) for coo in [cooa, coob]) - 0.01,
            )
            cooa = self._3d_project(*cooa)
            coob = self._3d_project(*coob)
        else:
            style.setdefault("zorder", -0.01)

        forward, inverse = get_rotator_and_inverse(cooa, coob)
        xb = forward(*coob)[0]
        rl = (1 + padding) * ra
        rr = (1 + padding) * rb
        rcoos = [
            # left loop
            (0, -rl),
            (-rl * 2**-0.5, -rl * 2**-0.5),
            (-rl, 0),
            (-rl * 2**-0.5, +rl * 2**-0.5),
            (0, +rl),
            # above pinch point
            (xb / 2, (1 - float(pinch)) * (ra + rb)),
            # right loop
            (xb, +rr),
            (xb + rr * 2**-0.5, +rr * 2**-0.5),
            (xb + rr, 0),
            (xb + rr * 2**-0.5, -rr * 2**-0.5),
            (xb, -rr),
            # below pinch point
            (xb / 2, (float(pinch) - 1) * (ra + rb)),
        ]

        pcoos = [inverse(*rcoo) for rcoo in rcoos]
        self.patch(pcoos, preset=preset, **style)


def parse_style_preset(presets, preset, **kwargs):
    """Parse a one or more style presets plus manual kwargs.

    Parameters
    ----------
    presets : dict
        The dictionary of presets.
    preset : str or sequence of str
        The name of the preset(s) to use. If multiple, later presets take
        precedence.
    kwargs
        Any additional manual keyword arguments are added to the style and
        override the presets.

    Returns
    -------
    style : dict
    """
    if (preset is None) or isinstance(preset, str):
        preset = (preset,)

    style = {}
    for p in preset:
        if p not in presets:
            warnings.warn(f"Drawing has no preset '{p}' yet.")
        else:
            style.update(presets[p])
    style.update(kwargs)
    return style


def axonometric_project(
    i,
    j,
    k,
    a=50,
    b=12,
    xscale=1,
    yscale=1,
    zscale=1,
):
    """Project the 3D location ``(i, j, k)`` onto the 2D plane, using
    the axonometric projection with the given angles ``a`` and ``b``.

    The ``xscale``, ``yscale`` and ``zscale`` parameters can be used to
    scale the axes, including flipping them.

    Parameters
    ----------
    i, j, k : float
        The 3D coordinates of the point to project.
    a, b : float
        The left and right angles to displace x and y axes, from horizontal,
        in degrees.
    xscale, yscale, zscale : float
        The scaling factor for the x, y and z axes. If negative, the axis
        is flipped.

    Returns
    -------
    x, y : float
        The 2D coordinates of the projected point.
    """
    i *= xscale * 0.8
    j *= yscale
    k *= zscale
    return (
        +i * cos(pi * a / 180) + j * cos(pi * b / 180),
        -i * sin(pi * a / 180) + j * sin(pi * b / 180) + k,
    )


def coo_to_zorder(i, j, k, xscale=1, yscale=1, zscale=1):
    """Given the coordinates of a point in 3D space, return a z-order value
    that can be used to draw it on top of other elements in the diagram.
    Take into account the scaling of the axes, so that the z-ordering
    is correct even if the axes flipped.
    """
    return (
        +i * xscale / abs(xscale)
        - j * yscale / abs(yscale)
        + k * zscale / abs(zscale)
    )


# colorblind palettes by Okabe & Ito (https://jfly.uni-koeln.de/color/)

_COLORS_DEFAULT = {
    "blue": "#56B4E9",  # light blue
    "orange": "#E69F00",  # orange
    "green": "#009E73",  # green
    "red": "#D55E00",  # red
    "yellow": "#F0E442",  # yellow
    "pink": "#CC79A7",  # pink
    "bluedark": "#0072B2",  # dark blue
}


def get_wong_color(
    which,
    alpha=None,
    hue_factor=0.0,
    sat_factor=1.0,
    val_factor=1.0,
):
    """Get a color by name, optionally modifying its alpha, hue, saturation
    or value.

    These colorblind friendly colors were ppularized in an article by Wong
    (https://www.nature.com/articles/nmeth.1618) but originally come from
    Okabe & Ito (https://jfly.uni-koeln.de/color/).

    Parameters
    ----------
    which : {'blue', 'orange', 'green', 'red', 'yellow', 'pink', 'bluedark'}
        The name of the color to get.
    alpha : float, optional
        The alpha channel value to set for the color. Default is 1.0.
    hue_factor : float, optional
        The amount to shift the hue of the color. Default is 0.0.
    sat_factor : float, optional
        The amount to scale the saturation of the color. Default is 1.0.
    val_factor : float, optional
        The amount to scale the value of the color. Default is 1.0.

    Returns
    -------
    color : tuple[float, float, float, float]
        The RGBA color as a tuple of floats.
    """
    import matplotlib as mpl

    h = _COLORS_DEFAULT[which]
    rgb = mpl.colors.to_rgb(h)
    h, s, v = mpl.colors.rgb_to_hsv(rgb)
    h = (h + hue_factor) % 1.0
    s = min(max(0.0, s * sat_factor), 1.0)
    v = min(max(0.0, v * val_factor), 1.0)
    r, g, b = mpl.colors.hsv_to_rgb((h, s, v))
    if alpha is not None:
        return (r, g, b, alpha)
    return r, g, b


_COLORS_SORTED = [
    _COLORS_DEFAULT["bluedark"],
    _COLORS_DEFAULT["blue"],
    _COLORS_DEFAULT["green"],
    _COLORS_DEFAULT["yellow"],
    _COLORS_DEFAULT["orange"],
    _COLORS_DEFAULT["red"],
    _COLORS_DEFAULT["pink"],
]


def mod_sat(c, mod=None, alpha=None):
    """Modify the luminosity of color ``c``, optionally set the ``alpha``
    channel, and return the final color as a RGBA tuple."""
    import matplotlib as mpl

    r, g, b, a = mpl.colors.to_rgba(c)
    if alpha is None:
        alpha = a

    if mod is None:
        return r, g, b, alpha

    h, s, v = mpl.colors.rgb_to_hsv((r, g, b))
    return (*mpl.colors.hsv_to_rgb((h, mod * s, v)), alpha)


def auto_colors(nc, alpha=None, default_sequence=False):
    """Generate a nice sequence of ``nc`` colors. By default this uses an
    interpolation between the colorblind friendly colors of Okabe & Ito in hue
    sorted order, with luminosity moderated by a sine function to increase
    local distinguishability.

    Parameters
    ----------
    nc : int
        The number of colors to generate.
    alpha : float, optional
        The alpha channel value to set for all colors. Default is 1.0.
    default_sequence : bool, optional
        If ``True``, take from the default sequence of 7 colors, un-sorted and
        un-modulated.

    Returns
    -------
    colors : list[tuple[float, float, float, float]]
    """
    import math

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    if default_sequence:
        if nc > 7:
            raise ValueError(
                "Can only generate 7 colors with default sequence"
            )
        return [
            mod_sat(c, alpha=alpha)
            for c in tuple(_COLORS_DEFAULT.values())[:nc]
        ]

    cmap = LinearSegmentedColormap.from_list("colorblind", _COLORS_SORTED)

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


def darken_color(color, factor=2 / 3):
    """Take ``color`` and darken it by ``factor``."""
    rgba = mpl.colors.to_rgba(color)
    return tuple(factor * c for c in rgba[:3]) + rgba[3:]


def average_color(colors):
    """Take a sequence of colors and return the RMS average in RGB space."""
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


def jitter_color(color, factor=0.05):
    """Take ``color`` and add a random offset to each of its components."""
    import random

    rgba = mpl.colors.to_rgba(color)
    hsv = mpl.colors.rgb_to_hsv(rgba[:3])
    hsv = (
        hsv[0],
        min(max(0, hsv[1] + random.uniform(-factor / 2, factor / 2)), 1),
        min(max(0, hsv[2] + random.uniform(-factor / 2, factor / 2)), 1),
    )
    rgb = mpl.colors.hsv_to_rgb(hsv)
    return tuple(rgb) + rgba[3:]



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


def mean(xs):
    """Get the mean of a list of numbers."""
    s = 0
    i = 0
    for x in xs:
        s += x
        i += 1
    return s / i


def distance(pa, pb):
    """Get the distance between two points, in arbtirary dimensions."""
    d = 0.0
    for a, b in zip(pa, pb):
        d += (a - b) ** 2
    return d**0.5


def get_angle(pa, pb):
    """Get the angle between the line from p1 to p2 and the x-axis."""
    (xa, ya), (xb, yb) = pa, pb
    return atan2(yb - ya, xb - xa)


def get_rotator_and_inverse(pa, pb):
    """Get a rotation matrix that rotates points by theta radians about
    the origin and then translates them by offset.
    """
    theta = get_angle(pa, pb)
    dx, dy = pa

    def forward(x, y):
        """Rotate and translate a point."""
        x, y = x - dx, y - dy
        x, y = (
            x * cos(-theta) - y * sin(-theta),
            x * sin(-theta) + y * cos(-theta),
        )
        return x, y

    def inverse(x, y):
        """Rotate and translate a point."""
        x, y = x * cos(theta) - y * sin(theta), x * sin(theta) + y * cos(theta)
        return x + dx, y + dy

    return forward, inverse


def get_control_points(pa, pb, pc, spacing=1 / 3):
    """Get two points that can be used to construct a bezier curve that
    passes smoothly through the angle `pa`, `pb`, `pc`.
    """
    # rotate onto x-axis (ra always (0, 0))
    forward, inverse = get_rotator_and_inverse(pa, pb)
    _, rb, rc = [forward(*p) for p in [pa, pb, pc]]

    # flip so third point is always above axis
    flip_y = rc[1] < 0
    if flip_y:
        rc = rc[0], -rc[1]

    phi = get_angle(rb, rc) / 2

    # lengths of the two lines
    lab = rb[0]
    lbc = distance(rb, rc)

    # lengths of perpendicular offsets
    oab = lab * cos(phi)
    obc = lbc * cos(phi)

    dx_ab = spacing * oab * cos(phi)
    dy_ab = spacing * oab * sin(phi)

    dx_bc = spacing * obc * cos(phi)
    dy_bc = spacing * obc * sin(phi)

    # get control points in this reference frame
    rc_ab = rb[0] - dx_ab, rb[1] - dy_ab
    rc_bc = rb[0] + dx_bc, rb[1] + dy_bc

    # unflip and un rotate
    if flip_y:
        rc_ab = rc_ab[0], -rc_ab[1]
        rc_bc = rc_bc[0], -rc_bc[1]

    c_ab, c_bc = inverse(*rc_ab), inverse(*rc_bc)

    return c_ab, c_bc


def gen_points_around(coo, radius=1, resolution=12):
    """Generate points around a circle."""
    x, y = coo
    dphi = 2 * pi / resolution
    phis = (i * dphi for i in range(resolution))
    for phi in phis:
        xa = x + radius * cos(phi)
        ya = y + radius * sin(phi)
        yield xa, ya
