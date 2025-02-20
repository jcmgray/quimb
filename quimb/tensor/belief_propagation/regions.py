"""Region graph functionality - for GBP and cluster expansions.
"""

import functools
import itertools


def cached_region_property(name):
    """Decorator for caching information about regions."""

    def wrapper(meth):
        @functools.wraps(meth)
        def getter(self, region):
            try:
                return self.info[region][name]
            except KeyError:
                region_info = self.info.setdefault(region, {})
                region_info[name] = value = meth(self, region)
                return value

        return getter

    return wrapper


class RegionGraph:
    """A graph of regions, where each region is a set of nodes. For generalized
    belief propagation or cluster expansion methods.

    Parameters
    ----------
    regions : Iterable[Sequence[Hashable]]
        Generating regions.
    autocomplete : bool, optional
        Whether to automatically add all intersecting sub-regions, to guarantee
        a complete region graph.
    autoprune : bool, optional
        Whether to automatically remove all regions with a count of zero.
    """

    def __init__(self, regions=(), autocomplete=True, autoprune=True):
        self.lookup = {}
        self.parents = {}
        self.children = {}
        self.info = {}

        for region in regions:
            self.add_region(region)
        if autocomplete:
            self.autocomplete()
        if autoprune:
            self.autoprune()

    def reset_info(self):
        """Remove all cached region properties.
        """
        self.info.clear()

    @property
    def regions(self):
        return tuple(self.children)

    def get_overlapping(self, region):
        """Get all regions that intersect with the given region."""
        region = frozenset(region)
        return {
            other_region
            for node in region
            for other_region in self.lookup[node]
            if other_region != region
        }

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
        for other in self.get_overlapping(region):
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

            for c in self.children[region]:
                if p.issuperset(c):
                    # parent is a superset of child -> ensure no link
                    self.parents[c].discard(p)
                    self.children[p].discard(c)

        self.reset_info()

    def remove_region(self, region):
        """Remove a region and update parent-child relationships.
        """
        # remove from lookup
        for node in region:
            self.lookup[node].remove(region)

        # remove from parents and children, joining those up
        parents = self.parents.pop(region)
        children = self.children.pop(region)
        for p in parents:
            self.children[p].remove(region)
            self.children[p].update(children)
        for c in children:
            self.parents[c].remove(region)
            self.parents[c].update(parents)

        self.reset_info()

    def autocomplete(self):
        """Add all missing intersecting sub-regions."""
        for r in self.regions:
            for other in self.get_overlapping(r):
                self.add_region(r & other)

    def autoprune(self):
        """Remove all regions with a count of zero."""
        for r in self.regions:
            if self.get_count(r) == 0:
                self.remove_region(r)

    def autoextend(self, regions=None):
        """Extend this region graph upwards by adding in all pairwise unions of
        regions. If regions is specified, take this as one set of pairs.
        """
        if regions is None:
            regions = self.regions

        neighbors = {}
        for r in regions:
            for other in self.get_overlapping(r):
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

    @cached_region_property("ancestors")
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

    @cached_region_property("descendents")
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

    @cached_region_property("coparent_pairs")
    def get_coparent_pairs(self, region):
        """Get all regions which are direct parents of any descendant of the
        given region, but not themselves descendants of the given region.
        """
        # start with direct parents
        coparent_pairs = [(p, region) for p in self.get_parents(region)]

        # get all descendents
        rds = self.get_descendents(region)

        # exclude the region and its descendents
        seen = {region, *rds}

        # for each descendant
        for rd in rds:
            # add only its parents...
            for rdp in self.get_parents(rd):
                # ... which are not themselves descendents
                if rdp not in seen:
                    coparent_pairs.append((rdp, rd))
                    seen.add(rdp)

        return coparent_pairs

    @cached_region_property("count")
    def get_count(self, region):
        """Get the count of the given region, i.e. the correct weighting to
        apply when summing over all regions to avoid overcounting.
        """
        return 1 - sum(self.get_count(a) for a in self.get_ancestors(region))

    def get_total_count(self):
        """Get the total count of all regions."""
        return sum(map(self.get_count, self.regions))

    @cached_region_property("level")
    def get_level(self, region):
        """Get the level of the given region, i.e. the distance to an ancestor
        with no parents.
        """
        if not self.parents[region]:
            return 0
        else:
            return min(self.get_level(p) for p in self.get_parents(region)) - 1

    @cached_region_property("message_parts")
    def get_message_parts(self, pair):
        """Get the three contribution groups for a GBP message from region
        `source` to region `target`. 1. The part of region `source` that is
        not part of target, i.e. the factors to include. 2. The messages that
        appear in the numerator of the update equation. 3. The messages that
        appear in the denominator of the update equation.

        Parameters
        ----------
        source : Region
            The source region, should be a parent of `target`.
        target : Region
            The target region, should be a child of `source`.

        Returns
        -------
        factors : Region
            The difference of `source` and `target`, which will include the
            factors to appear in the numerator of the update equation.
        pairs_mul : set[(Region, Region)]
            The messages that appear in the numerator of the update equation,
            after cancelling out those that appear in the denominator.
        pairs_div : set[(Region, Region)]
            The messages that appear in the denominator of the update equation,
            after cancelling out those that appear in the numerator.
        """
        source, target = pair
        factors = source - target

        # we want to cancel out messages that appear in both of:
        # the messages that go into the belief of region `source`
        source_pairs = set(self.get_coparent_pairs(source))
        # the messages that go into the belief of region `target`
        target_pairs = set(self.get_coparent_pairs(target))
        # the current message to be updated by defn appears directly in the
        # update numerator, but also target belief region, so can be cancelled
        target_pairs.remove((source, target))

        pairs_mul = source_pairs - target_pairs
        pairs_div = target_pairs - source_pairs

        return factors, pairs_mul, pairs_div

    def check(self):
        """Run some basic consistency checks on the region graph."""
        for r, rps in self.parents.items():
            for rp in rps:
                assert r.issubset(rp)
                assert r in self.get_children(rp)

        for r in self.regions:
            for rd in self.get_descendents(r):
                assert r.issuperset(rd)
                assert r in self.get_ancestors(rd)

            for ra in self.get_ancestors(r):
                assert r.issubset(ra)
                assert r in self.get_descendents(ra)

            rps = self.get_parents(r)
            for rpa, rpb in itertools.combinations(rps, 2):
                assert not rpa.issubset(rpb)
                assert not rpb.issubset(rpa)

            rcs = self.get_children(r)
            for rca, rcb in itertools.combinations(rcs, 2):
                assert not rca.issubset(rcb)
                assert not rcb.issubset(rca)

    def draw(self, pos=None, a=20, scale=1.0, radius=0.1, **drawing_opts):
        from quimb.schematic import Drawing, hash_to_color

        if pos is None:
            pos = {node: node for node in self.lookup}

        def get_draw_pos(coo):
            return tuple(scale * s for s in pos[coo])

        sizes = {len(r) for r in self.regions}
        levelmap = {s: i for i, s in enumerate(sorted(sizes))}
        centers = {}

        d = Drawing(a=a, **drawing_opts)
        for region in sorted(self.regions, key=len, reverse=True):
            # level = self.get_level(region)
            # level = len(region)
            level = levelmap[len(region)]

            coos = [(*get_draw_pos(coo), 2.0 * level) for coo in region]

            average_coo = tuple(map(sum, zip(*coos)))
            centers[region] = tuple(c / len(coos) for c in average_coo)

            d.patch_around(
                coos,
                radius=radius,
                # edgecolor=hash_to_color(str(region)),
                facecolor=hash_to_color(str(region)),
                alpha=1 / 3,
                linestyle="",
                linewidth=3,
            )

        for region in self.regions:
            for child in self.get_children(region):
                d.line(
                    centers[region],
                    centers[child],
                    linewidth=.5,
                    linestyle="-",
                    color=(.5, .5, .5),
                    alpha=0.5,
                    arrowhead={},
                )

        return d.fig, d.ax

    def __repr__(self):
        return (
            f"<RegionGraph(regions={len(self.regions)}, "
            f"total_count={self.get_total_count()})>"
        )
