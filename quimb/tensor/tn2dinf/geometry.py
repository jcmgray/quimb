"""Geometry of infinite, translation-invariant 2D lattices: the unit cell of
sites and bonds, with a ``GeometryInfinite2D.square`` builder for square
lattices. See the subpackage ``__init__`` for some shared definition.
"""


def is_inf_2d_site(site):
    """Check if `site` is a valid 2d infinite site specifier, that is, it is
    tuple `(cell, site_type)` where `cell` is a tuple `(x : int, y : int)`.
    """
    if not isinstance(site, tuple) or len(site) != 2:
        return False
    cell, _ = site
    if not isinstance(cell, tuple) or len(cell) != 2:
        return False
    x, y = cell
    if not isinstance(x, int) or not isinstance(y, int):
        return False
    return True


def ensure_inf_2d_sites(where):
    """Ensure ``where`` (a single site or a sequence of sites) is a tuple of
    sites.
    """
    if is_inf_2d_site(where):
        return (where,)
    return tuple(where)


def get_bond_sorted(sitea, siteb):
    """Given a bond between two sites, simply return the sorted order of the
    two, which is the canonical way to refer to it / use it as a key etc.
    """
    if sitea <= siteb:
        return (sitea, siteb)
    else:
        return (siteb, sitea)


def get_bond_type(sitea, siteb):
    """Given a bond between two sites, get its bond type: the canonical
    representative of the bond's translation class. The first endpoint is
    translated to cell (0, 0) and the orientation chosen so the two endpoints
    are in sorted order, i.e. ``cella < cellb``, or (within-cell)
    ``site_type_a < site_type_b``. Thus cells compare dx-first.
    """
    cella, site_type_a = sitea
    cellb, site_type_b = siteb

    dx = cellb[0] - cella[0]
    dy = cellb[1] - cella[1]
    if abs(dx) > 1 or abs(dy) > 1:
        raise ValueError(
            "Only edges between neighboring cells are allowed. "
            f"Got edge from {sitea} to {siteb}."
        )

    # anchor the first endpoint at cell (0, 0) and keep the sorted
    # orientation, else flip, anchoring the other endpoint instead
    first = ((0, 0), site_type_a)
    second = ((dx, dy), site_type_b)
    if first <= second:
        return (first, second)
    return (((0, 0), site_type_b), ((-dx, -dy), site_type_a))


def _average_position(positions):
    """Return the average position of a sequence of 2d positions."""
    xm, ym = 0.0, 0.0
    n = len(positions)
    for x, y in positions:
        xm += x
        ym += y
    return xm / n, ym / n


class GeometryInfinite2D:
    """Helper class to represent the geometry of an infinite 2D lattice.

    Parameters
    ----------
    edges : sequence[(((int, int), hashable), ((int, int), hashable)), ...]
        A sequence of edges, where each edge is a pair of sites. Each site is
        represented as a tuple of (cell, site_type), where cell is a tuple of
        integers representing the cell coordinates, and site_type is a hashable
        representing the type of site (e.g., an integer or string). Equivalent
        edges are automatically deduplicated and normalized to a canonical
        ``bond_type``: the first endpoint translated to cell (0, 0) and the two
        endpoints in sorted order, i.e. ``cella < cellb``, or (within-cell)
        ``site_type_a < site_type_b`` (cells compare dx-first).
    basis : (float, float), optional
        A pair of 2D vectors representing the lattice basis vectors. If not
        provided, the default basis is the standard square lattice basis.
    positions : dict, optional
        A dictionary mapping site types to their fractional positions within
        the unit cell. If not provided, the default position for each site type
        is (0.0, 0.0). Currently only used for drawing.
    """

    def __init__(
        self,
        edges,
        basis=None,
        positions=None,
    ):
        # first get all unique types of bond and site
        bond_types = set()
        site_types = set()
        for sitea, siteb in edges:
            _, site_type_a = sitea
            _, site_type_b = siteb
            site_types.add(site_type_a)
            site_types.add(site_type_b)

            # deduplicate such that each starts in cell (0, 0) and 'points up'
            bond_type = self.get_bond_type(sitea, siteb)
            bond_types.add(bond_type)

        # put in canonical order as well
        self.site_types = tuple(sorted(site_types))
        self.bond_types = tuple(sorted(bond_types))

        # build base neighbor map
        self.site_type_neighbors = {}
        for bond_type in self.bond_types:
            (_, site_type_a), (cellb, site_type_b) = bond_type

            if site_type_a == site_type_b:
                raise NotImplementedError(
                    f"Self-loops {site_type_a} <-> {site_type_b}"
                    "are not supported, consider expanding unit cell."
                )

            dx, dy = cellb  # cella is always (0, 0)
            self.site_type_neighbors.setdefault(site_type_a, []).append(
                ((dx, dy), site_type_b)
            )
            self.site_type_neighbors.setdefault(site_type_b, []).append(
                ((-dx, -dy), site_type_a)
            )

        # build covering bonds and sites ->
        #     all sites and bonds that connect to the unit cell (0, 0)
        covering_sites = set()
        covering_bonds = set()
        for site_typea in self.site_types:
            sitea = ((0, 0), site_typea)
            covering_sites.add(sitea)
            for cellb, site_typeb in self.site_type_neighbors[site_typea]:
                siteb = (cellb, site_typeb)
                covering_sites.add(siteb)
                bond = self.get_bond_sorted(sitea, siteb)
                covering_bonds.add(bond)
        self.covering_sites = tuple(sorted(covering_sites))
        self.covering_bonds = tuple(sorted(covering_bonds))

        # optional spatial embedding: lattice vectors (a1, a2) (default square)
        self.basis = basis if basis is not None else ((1, 0), (0, 1))
        # and fractional sublattice offsets within the cell, keyed by site_type
        self.positions = dict(positions) if positions is not None else {}

    @classmethod
    def square(
        cls,
        Lx=2,
        Ly=None,
        *,
        couplings=None,
        radius=None,
        basis=None,
        **kwargs,
    ):
        """Build a square-lattice geometry on an ``Lx`` by ``Ly`` unit cell,
        with ``site_type = (subx, suby)`` sitting at the 0-based lattice
        points. By default builds the minimal 2x2 cell with nearest-neighbor
        bonds. Longer range bonds can be included by specifying ``couplings``
        or ``radius``.

        Parameters
        ----------
        Lx : int
            The number of sites in the x-direction in the unit cell, default 2.
        Ly : int, optional
            The number of sites in the y-direction in the unit cell, default
            ``Lx``.
        couplings : int or sequence[tuple[int, int]], optional
            Which neighbor shells to bond, in square-lattice distance: an int
            ``k`` for the first ``k`` shells (``1`` nearest-neighbor, ``2``
            also diagonal next-nearest, ...), or an explicit sequence of
            ``(dx, dy)`` site-step displacements. Mutually exclusive with
            ``radius``, defaults to nearest-neighbor.
        radius : float, optional
            Alternatively, bond every neighbor within this square-lattice
            distance.
        basis : tuple[tuple[float, float], tuple[float, float]], optional
            Lattice vectors for the spatial embedding (drawing only), defaults
            to the unit square.
        kwargs
            Passed to the ``GeometryInfinite2D`` constructor.

        Returns
        -------
        GeometryInfinite2D
        """
        if Ly is None:
            Ly = Lx

        positions = {
            (cx, cy): (cx / Lx, cy / Ly)
            for cx in range(Lx)
            for cy in range(Ly)
        }

        edges = make_edges_inf_2d_square(
            Lx, Ly, couplings=couplings, radius=radius
        )

        return cls(
            edges=edges,
            basis=basis,
            positions=positions,
            **kwargs,
        )

    def get_site_neighbors(self, site):
        """Generate the neighbors of a given site."""
        (x, y), site_type_a = site
        for (dx, dy), site_type_b in self.site_type_neighbors[site_type_a]:
            yield ((x + dx, y + dy), site_type_b)

    def coordinate(self, site):
        """Map a site ``(cell, site_type)`` to its cartesian position:
        ``(cell + fractional_offset) @ basis``, defining a spatial embedding.
        """
        (x, y), site_type = site
        sx, sy = self.positions.get(site_type, (0.0, 0.0))
        fx = x + sx
        fy = y + sy
        (a1x, a1y), (a2x, a2y) = self.basis
        return (fx * a1x + fy * a2x, fx * a1y + fy * a2y)

    def get_graph_distance(self, sitea, siteb, max_hops=None):
        """Calculate the graph distance (number of hops) between two sites,
        found by BFS over the site graph. ``max_hops`` caps the search,
        guarding against an unreachable siteb (any connected pair terminates
        without it).
        """
        if max_hops is None:
            # TODO: some better heuristic
            max_hops = 8 * len(self.site_types) + 8
        if sitea == siteb:
            return 0
        seen = {sitea}
        boundary = [sitea]
        for d in range(1, max_hops + 1):
            next_boundary = []
            for sb in boundary:
                for sn in self.get_site_neighbors(sb):
                    if sn == siteb:
                        return d
                    if sn not in seen:
                        seen.add(sn)
                        next_boundary.append(sn)
            boundary = next_boundary
        raise ValueError(
            f"No path from {sitea} to {siteb} within {max_hops} hops."
        )

    def get_sites_within_radius(self, site, radius):
        """All sites within ``radius`` graph distance of ``site``, inclusive."""
        seen = {site}
        boundary = [site]
        for _ in range(radius):
            next_boundary = []
            for sb in boundary:
                for sn in self.get_site_neighbors(sb):
                    if sn not in seen:
                        seen.add(sn)
                        next_boundary.append(sn)
            boundary = next_boundary
        return seen

    def get_cell_size(self):
        """Return (dx, dy), the 'width' and 'height' of the unit cell in terms
        of the minimum number of bond hops for any site_type to reach the same
        site_type in the neighboring cell in the x and y directions,
        respectively. Useful for computing necessary tiling sizes.
        """
        dx = min(
            self.get_graph_distance(((0, 0), st), ((1, 0), st))
            for st in self.site_types
        )
        dy = min(
            self.get_graph_distance(((0, 0), st), ((0, 1), st))
            for st in self.site_types
        )
        return dx, dy

    def get_tiling_for_radius(self, radius):
        """Number of cells to tile out from the origin in each direction so
        that every site within ``radius`` bond-hops of the origin cell is
        contained. Returns ``(nx, ny)``, i.e. tile cells ``-nx..nx`` by
        ``-ny..ny``. Explores the graph directly, so the tiling never
        undershoots (note an oversized fragment is usually harmless, the region
        of interest is generally subselected from it).
        """
        nx = ny = 0
        for site_type in self.site_types:
            for (cx, cy), _ in self.get_sites_within_radius(
                ((0, 0), site_type), radius
            ):
                nx = max(nx, abs(cx))
                ny = max(ny, abs(cy))
        return nx, ny

    # exposed as static methods, defined as top-level functions above
    get_bond_sorted = staticmethod(get_bond_sorted)
    get_bond_type = staticmethod(get_bond_type)

    def is_canonical_bond(self, sitea, siteb):
        """Whether the bond between ``sitea`` and ``siteb`` is already in
        canonical ``bond_type`` form, i.e. sorted with its first endpoint in
        the origin cell (0, 0).
        """
        return get_bond_sorted(sitea, siteb) == get_bond_type(sitea, siteb)

    def get_auto_ordering(self, order="sort", group=False):
        """An ordering of the ``bond_types`` such that consecutive entries act
        on disjoint ``site_types`` where possible, i.e. grouped into commuting
        layers. Used to sequence gates in e.g. a simple-update sweep.

        Parameters
        ----------
        order : str, optional
            Strategy for coloring the bond-types to generate the ordering.
            Note currently only "sort" is supported.
        group : bool, optional
            If ``True``, return a list of layers (tuples of ``bond_types``),
            otherwise return a flat list of ``bond_types``.

        Returns
        -------
        list[bond_type] or list[tuple[bond_type]]
        """
        # TODO: implement networkx strategies

        bonds = list(self.bond_types)
        i = 0
        ordering = []
        current_site_types = set()
        layer = []
        while bonds:
            (_, sta), (_, stb) = bonds[i]
            sts = {sta, stb}

            if sts & current_site_types:
                # does not commute with current -> skip
                i += 1
            else:
                # commutes with current -> accept
                layer.append(bonds.pop(i))
                current_site_types.update(sts)

            if i >= len(bonds):
                # no commuting bond available -> flush
                if group:
                    ordering.append(tuple(layer))
                else:
                    ordering.extend(tuple(layer))
                i = 0
                layer.clear()
                current_site_types.clear()

        return ordering

    def draw(self, pos=None):
        """Draw the covering sites and bonds of the unit cell, with the cell
        boundary marked.

        Parameters
        ----------
        pos : dict, optional
            A mapping of ``site_type`` to fractional position within the unit
            cell. If not given, the geometry's own ``coordinate`` is used.
        """
        from ...schematic import (
            Drawing,
            # auto_colors,
            hash_to_color,
        )

        if pos is None:
            get_pos = self.coordinate
        else:

            def get_pos(site):
                cell, site_type = site
                sx, sy = pos[site_type]
                fx, fy = cell[0] + sx, cell[1] + sy
                (a1x, a1y), (a2x, a2y) = self.basis
                return (fx * a1x + fy * a2x, fx * a1y + fy * a2y)

        def _draw_site(site):
            x, y = get_pos(site)
            color = hash_to_color(str(site[1]))
            # color = site_type_to_color[site[1]]
            d.circle((x, y), color=color, radius=0.1)
            d.text(
                (x, y),
                site[-1],
                fontsize=12,
                ha="center",
                va="center",
            )

        def _draw_bond(sitea, siteb):
            bond_type = self.get_bond_type(sitea, siteb)
            # color = edge_type_to_color[bond_type]
            color = hash_to_color(str(bond_type))
            d.line(
                get_pos(sitea),
                get_pos(siteb),
                color=color,
                linewidth=3,
            )

        # site_type_colors = auto_colors(len(self.site_types))
        # site_type_to_color = dict(zip(self.site_types, site_type_colors))

        # edge_type_colors = auto_colors(len(self.bond_types))
        # edge_type_to_color = dict(zip(self.bond_types, edge_type_colors))

        d = Drawing()

        for site in self.covering_sites:
            _draw_site(site)
        for sitea, siteb in self.covering_bonds:
            _draw_bond(sitea, siteb)

        # draw the unit cell boundary
        corners = []
        for cells in [
            ((0, 0), (0, +1), (+1, 0), (+1, +1)),
            ((0, 0), (0, -1), (+1, 0), (+1, -1)),
            ((0, 0), (0, -1), (-1, 0), (-1, -1)),
            ((0, 0), (0, +1), (-1, 0), (-1, +1)),
        ]:
            corners.append(
                _average_position(
                    [
                        self.coordinate((cell, st))
                        for st in self.site_types
                        for cell in cells
                    ]
                )
            )
        d.shape(
            corners,
            edgecolor=(0.5, 0.5, 0.5),
            linestyle=":",
            color="none",
            zorder=-100,
        )

        return d.fig, d.ax

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"#site_types={len(self.site_types)}, "
            f"#bond_types={len(self.bond_types)})"
        )


def _half_grid(rmax):
    """Get all positive displacements (``(dx, dy) > (0, 0)``) with
    ``dx <= rmax`` and ``dy <= rmax``.
    """
    return [
        (dx, dy)
        for dx in range(0, rmax + 1)
        for dy in range(-rmax, rmax + 1)
        if (dx, dy) > (0, 0)
    ]


def _square_displacements(couplings=None, radius=None):
    """Calculate possible long range bonds on a square lattice, producing a
    sorted list of ``(dx, dy)`` site-step displacements, one per +/- pair,
    measured in square-lattice distance (independent of the drawing basis). At
    most one of ``couplings`` / ``radius`` may be given, defaulting to
    nearest-neighbor.

    Parameters
    ----------
    couplings : int or sequence[tuple[int, int]], optional
        An int ``k`` for the first ``k`` neighbor shells (``1``
        nearest-neighbor, ``2`` also diagonal next-nearest, ...), or an
        explicit iterable of ``(dx, dy)``.
    radius : float, optional
        All displacements within this square-lattice distance.

    Returns
    -------
    list[tuple[int, int]]
    """
    if couplings is not None and radius is not None:
        raise ValueError("give either `couplings` or `radius`, not both.")

    if radius is not None:
        return [
            d
            for d in _half_grid(int(radius))
            if d[0] ** 2 + d[1] ** 2 <= radius**2
        ]

    if couplings is None:
        couplings = 1  # nearest-neighbor default

    if isinstance(couplings, bool):
        raise TypeError("`couplings` is an int (shells) or a (dx, dy) list.")
    if isinstance(couplings, float):
        raise TypeError(
            "use `radius=` for a distance cutoff, `couplings` is an int "
            "(number of shells) or a list of (dx, dy)."
        )
    if isinstance(couplings, int):
        # get all possible displacements
        grid = _half_grid(couplings)
        # group, sort by their distance from origin, and take first k types
        keep = set(sorted({dx**2 + dy**2 for dx, dy in grid})[:couplings])
        return [d for d in grid if d[0] ** 2 + d[1] ** 2 in keep]

    # explicit iterable -> canonicalize to the positive half and dedup
    disps = set()
    for dx, dy in couplings:
        if (dx, dy) == (0, 0):
            raise ValueError("(0, 0) is not a valid coupling displacement.")
        disps.add((dx, dy) if dx > 0 or (dx == 0 and dy > 0) else (-dx, -dy))
    return sorted(disps)


def make_edges_inf_2d_square(Lx=2, Ly=None, couplings=None, radius=None):
    """Edges of an ``Lx`` by ``Ly`` square-lattice unit cell, with
    ``site_type = (subx, suby)``. By default nearest-neighbor bonds only, with
    longer range bonds included by specifying ``couplings`` or ``radius``.

    Each bond must reach a *different* sublattice in a neighboring cell, else a
    ``ValueError`` asks to expand the cell (e.g. 3rd-NN ``(2, 0)`` invalidly
    links sites to themselves in only a 2x2 cell).

    Parameters
    ----------
    Lx : int
        The number of sites in the x-direction in the unit cell, default 2.
    Ly : int, optional
        The number of sites in the y-direction in the unit cell, default
        ``Lx``.
    couplings : int or sequence[tuple[int, int]], optional
        The neighbor shells to bond (see :func:`_square_displacements`).
    radius : float, optional
        A square-lattice distance cutoff, as an alternative to ``couplings``.

    Returns
    -------
    list[tuple[site, site]]
    """
    if Ly is None:
        Ly = Lx

    disps = _square_displacements(couplings=couplings, radius=radius)
    edges = []
    for cx in range(Lx):
        for cy in range(Ly):
            a = ((0, 0), (cx, cy))
            for dx, dy in disps:
                bcx, lx = divmod(cx + dx, Lx)
                bcy, ly = divmod(cy + dy, Ly)
                if abs(bcx) > 1 or abs(bcy) > 1:
                    raise ValueError(
                        f"coupling {(dx, dy)} reaches cell {(bcx, bcy)} from "
                        f"site_type {(cx, cy)} in a {Lx}x{Ly} cell, expand it."
                    )
                if (lx, ly) == (cx, cy):
                    raise ValueError(
                        f"coupling {(dx, dy)} maps site_type {(cx, cy)} onto "
                        f"the same sublattice in a {Lx}x{Ly} cell, expand it."
                    )
                edges.append((a, ((bcx, bcy), (lx, ly))))
    return edges
