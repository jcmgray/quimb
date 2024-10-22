class RegionGraph:
    def __init__(self, regions=(), autocomplete=True):
        self.lookup = {}
        self.parents = {}
        self.children = {}
        self.counts = {}
        for region in regions:
            self.add_region(region)
        if autocomplete:
            self.autocomplete()

    @property
    def regions(self):
        return tuple(self.children)

    def neighbor_regions(self, region):
        """Get all regions that intersect with the given region."""
        region = frozenset(region)

        other_regions = set.union(*(self.lookup[node] for node in region))
        other_regions.discard(region)
        return other_regions

    def add_region(self, region):
        """Add a new region and update parent-child relationships.

        Parameters
        ----------
        region : Sequence[Hashable]
            The new region to add.
        """
        region = frozenset(region)

        if region in self.parents:
            # already added
            return

        # populate data structures
        self.parents[region] = set()
        self.children[region] = set()
        for node in region:
            # collect regions that contain nodes for fast neighbor lookup
            self.lookup.setdefault(node, set()).add(region)

        # add parent-child relationships
        for other in self.neighbor_regions(region):
            if region.issubset(other):
                self.parents[region].add(other)
                self.children[other].add(region)
            elif other.issubset(region):
                self.children[region].add(other)
                self.parents[other].add(region)

        # prune redundant parents and children
        children = sorted(self.children[region], key=len)
        for i, c in enumerate(children):
            if any(c.issubset(cc) for cc in children[i + 1 :]):
                # child is a subset of larger child -> remove
                self.children[region].remove(c)
                self.parents[c].remove(region)

        parents = sorted(self.parents[region], key=len, reverse=True)
        for i, p in enumerate(parents):
            if any(p.issuperset(pp) for pp in parents[i + 1 :]):
                # parent is a superset of smaller parent -> remove
                self.parents[region].remove(p)
                self.children[p].remove(region)

        self.counts.clear()

    def autocomplete(self):
        """Add all missing intersecting sub-regions."""
        for r in self.regions:
            for other in self.neighbor_regions(r):
                self.add_region(r & other)

    def autoextend(self, regions=None):
        """Extend this region graph upwards by adding in all pairwise unions of
        regions. If regions is specified, take this as one set of pairs.
        """
        if regions is None:
            regions = self.regions

        neighbors = {}
        for r in regions:
            for other in self.neighbor_regions(r):
                neighbors.setdefault(r, []).append(other)

        for r, others in neighbors.items():
            for other in others:
                self.add_region(r | other)

    def get_parents(self, region):
        """Get all ancestors that contain the given region, but do not contain
        any other regions that themselves contain the given region.
        """
        return self.parents[region]

    def get_children(self, region):
        """Get all regions that are contained by the given region, but are not
        contained by any other descendents of the given region.
        """
        return self.children[region]

    def get_ancestors(self, region):
        """Get all regions that contain the given region, not just direct
        parents.
        """
        seen = set()
        queue = [region]
        while queue:
            r = queue.pop()
            for rp in self.parents[r]:
                if rp not in seen:
                    seen.add(rp)
                    queue.append(rp)
        return seen

    def get_descendents(self, region):
        """Get all regions that are contained by the given region, not just
        direct children.
        """
        seen = set()
        queue = [region]
        while queue:
            r = queue.pop()
            for rc in self.children[r]:
                if rc not in seen:
                    seen.add(rc)
                    queue.append(rc)
        return seen

    def get_count(self, region):
        """Get the count of the given region, i.e. the correct weighting to
        apply when summing over all regions to avoid overcounting.
        """
        try:
            C = self.counts[region]
        except KeyError:
            # n.b. cache is cleared when any new region is added
            C = self.counts[region] = 1 - sum(
                self.get_count(a) for a in self.get_ancestors(region)
            )
        return C

    def get_total_count(self):
        return sum(map(self.get_count, self.regions))

    def get_level(self, region):
        """Get the level of the given region, i.e. the distance to an ancestor
        with no parents.
        """
        if not self.parents[region]:
            return 0
        else:
            return min(self.get_level(p) for p in self.parents[region]) - 1

    def draw(self, pos=None, a=20, scale=1.0, radius=0.1, **drawing_opts):
        from quimb.schematic import Drawing, hash_to_color

        if pos is None:
            pos = {node: node for node in self.lookup}

        def get_draw_pos(coo):
            return tuple(scale * s for s in pos[coo])

        sizes = {len(r) for r in self.regions}
        levelmap = {s: i for i, s in enumerate(sorted(sizes))}

        d = Drawing(a=a, **drawing_opts)
        for region in sorted(self.regions, key=len, reverse=True):
            # level = self.get_level(region)
            # level = len(region)
            level = levelmap[len(region)]

            coos = [(*get_draw_pos(coo), 2.0 * level) for coo in region]

            d.patch_around(
                coos,
                radius=radius,
                # edgecolor=hash_to_color(str(region)),
                facecolor=hash_to_color(str(region)),
                alpha=1 / 3,
                linestyle="",
                linewidth=3,
            )

        return d.fig, d.ax

    def __repr__(self):
        return (
            f"<RegionGraph(regions={len(self.regions)}, "
            f"total_count={self.get_total_count()})>"
        )
