"""Functionality for analyzing the structure of tensor networks, including
finding paths, loops, connected components, hierarchical groupings and more.
"""

import collections
import functools
import itertools
import math

from ..utils import oset, unique


class NetworkPatch:
    """A simple class to represent a patch of tensors and indices, storing
    both the tensor identifies (`tids`) and indices (`inds`) it contains.
    """

    __slots__ = ("_tids", "_inds", "_key")

    def __init__(self, tids, inds):
        self._tids = oset(tids)
        self._inds = oset(inds)
        self._key = None

    @classmethod
    def from_sequence(cls, it):
        tids = []
        inds = []
        for x in it:
            (tids if isinstance(x, int) else inds).append(x)
        return cls(tids, inds)

    @property
    def tids(self):
        return self._tids

    @property
    def inds(self):
        return self._inds

    def __iter__(self):
        return itertools.chain(self._tids, self._inds)

    @property
    def key(self):
        # build lazily as don't always need
        if self._key is None:
            self._key = frozenset(self)
        return self._key

    def merge(self, other):
        return NetworkPatch(
            tids=itertools.chain(self._tids, other._tids),
            inds=itertools.chain(self._inds, other._inds),
        )

    def __contains__(self, x):
        return x in self.key

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        if not isinstance(other, NetworkPatch):
            return NotImplemented
        return self.key == other.key

    def __repr__(self):
        return f"{self.__class__.__name__}({self._tids}, {self._inds})"


class NetworkPath(NetworkPatch):
    """A simple class to represent a path through a tensor network, storing
    both the tensor identifies (`tids`) and indices (`inds`) it passes through.
    """

    __slots__ = NetworkPatch.__slots__

    def __init__(self, tids, inds=()):
        self._tids = tuple(tids)
        self._inds = tuple(inds)
        if len(self._tids) != len(self._inds) + 1:
            raise ValueError("tids should be one longer than inds")
        self._key = None

    def __len__(self):
        return len(self._inds)

    def __iter__(self):
        # interleave tids and inds
        for tid, ind in zip(self._tids, self._inds):
            yield tid
            yield ind
        # always one more tid
        yield self._tids[-1]

    def extend(self, ind, tid):
        """Get a new path by extending this one with a new index and tensor id."""
        new = NetworkPath.__new__(NetworkPath)
        new._tids = self._tids + (tid,)
        new._inds = self._inds + (ind,)
        new._key = None if self._key is None else self._key | {tid, ind}
        return new


def istree(tn):
    """Check if this tensor network has a tree structure, (treating
    multibonds as a single edge).

    Examples
    --------

        >>> MPS_rand_state(10, 7).istree()
        True

        >>> MPS_rand_state(10, 7, cyclic=True).istree()
        False

    """
    tid0 = next(iter(tn.tensor_map))
    region = [(tid0, None)]
    seen = {tid0}
    while region:
        tid, ptid = region.pop()
        for ntid in tn._get_neighbor_tids(tid):
            if ntid == ptid:
                # ignore the previous tid we just came from
                continue
            if ntid in seen:
                # found a loop
                return False
            # expand the queue
            region.append((ntid, tid))
            seen.add(ntid)
    return True


def isconnected(tn):
    """Check whether this tensor network is connected, i.e. whether
    there is a path between any two tensors, (including size 1 indices).
    """
    tid0 = next(iter(tn.tensor_map))
    region = tn._get_subgraph_tids([tid0])
    return len(region) == len(tn.tensor_map)


def subgraphs(tn, virtual=False):
    """Split this tensor network into disconneceted subgraphs.

    Parameters
    ----------
    virtual : bool, optional
        Whether the tensor networks should view the original tensors or
        not - by default take copies.

    Returns
    -------
    list[TensorNetwork]
    """
    groups = []
    tids = oset(tn.tensor_map)

    # check all nodes
    while tids:
        # get a remaining node
        tid0 = tids.popright()
        queue = [tid0]
        group = oset(queue)

        while queue:
            # expand it until no neighbors
            tid = queue.pop()
            for tid_n in tn._get_neighbor_tids(tid):
                if tid_n in group:
                    continue
                else:
                    group.add(tid_n)
                    queue.append(tid_n)

        # remove current subgraph and continue
        tids -= group
        groups.append(group)

    return [tn._select_tids(group, virtual=virtual) for group in groups]


def get_tree_span(
    tn,
    tids,
    min_distance=0,
    max_distance=None,
    include=None,
    exclude=None,
    ndim_sort="max",
    distance_sort="min",
    sorter=None,
    weight_bonds=True,
    inwards=True,
):
    """Generate a tree on the tensor network graph, fanning out from the
    tensors identified by ``tids``, up to a maximum of ``max_distance``
    away. The tree can be visualized with
    :meth:`~quimb.tensor.tensor_core.TensorNetwork.draw_tree_span`.

    Parameters
    ----------
    tids : sequence of str
        The nodes that define the region to span out of.
    min_distance : int, optional
        Don't add edges to the tree until this far from the region. For
        example, ``1`` will not include the last merges from neighboring
        tensors in the region defined by ``tids``.
    max_distance : None or int, optional
        Terminate branches once they reach this far away. If ``None`` there
        is no limit,
    include : sequence of str, optional
        If specified, only ``tids`` specified here can be part of the tree.
    exclude : sequence of str, optional
        If specified, ``tids`` specified here cannot be part of the tree.
    ndim_sort : {'min', 'max', 'none'}, optional
        When expanding the tree, how to choose what nodes to expand to
        next, once connectivity to the current surface has been taken into
        account.
    distance_sort : {'min', 'max', 'none'}, optional
        When expanding the tree, how to choose what nodes to expand to
        next, once connectivity to the current surface has been taken into
        account.
    weight_bonds : bool, optional
        Whether to weight the 'connection' of a candidate tensor to expand
        out to using bond size as well as number of bonds.

    Returns
    -------
    list[(str, str, int)]
        The ordered list of merges, each given as tuple ``(tid1, tid2, d)``
        indicating merge ``tid1 -> tid2`` at distance ``d``.

    See Also
    --------
    draw_tree_span
    """
    # current tensors in the tree -> we will grow this
    region = oset(tids)

    # check if we should only allow a certain set of nodes
    if include is None:
        include = oset(tn.tensor_map)
    elif not isinstance(include, oset):
        include = oset(include)

    allowed = include - region

    # check if we should explicitly ignore some nodes
    if exclude is not None:
        if not isinstance(exclude, oset):
            exclude = oset(exclude)
        allowed -= exclude

    # possible merges of neighbors into the region
    candidates = []

    # actual merges we have performed, defining the tree
    merges = {}

    # distance to the original region
    distances = {tid: 0 for tid in region}

    # how many times (or weight) that neighbors are connected to the region
    connectivity = collections.defaultdict(lambda: 0)

    # given equal connectivity compare neighbors based on
    #      min/max distance and min/max ndim
    distance_coeff = {"min": -1, "max": 1, "none": 0}[distance_sort]
    ndim_coeff = {"min": -1, "max": 1, "none": 0}[ndim_sort]

    def _check_candidate(tid_surface, tid_neighb):
        """Check the expansion of ``tid_surface`` to ``tid_neighb``."""
        if (tid_neighb in region) or (tid_neighb not in allowed):
            # we've already absorbed it, or we're not allowed to
            return

        if tid_neighb not in distances:
            # defines a new spanning tree edge
            merges[tid_neighb] = tid_surface
            # graph distance to original region
            new_d = distances[tid_surface] + 1
            distances[tid_neighb] = new_d
            if (max_distance is None) or (new_d <= max_distance):
                candidates.append(tid_neighb)

        # keep track of how connected to the current surface potential new
        # nodes are
        if weight_bonds:
            connectivity[tid_neighb] += math.log2(
                tn.tensor_map[tid_surface].bonds_size(
                    tn.tensor_map[tid_neighb]
                )
            )
        else:
            connectivity[tid_neighb] += 1

    if sorter is None:

        def _sorter(t):
            # how to pick which tensor to absorb into the expanding surface
            # here, choose the candidate that is most connected to current
            # surface, breaking ties with how close it is to the original
            # tree, and how many dimensions it has
            return (
                connectivity[t],
                ndim_coeff * tn.tensor_map[t].ndim,
                distance_coeff * distances[t],
            )
    else:
        _sorter = functools.partial(
            sorter, tn=tn, distances=distances, connectivity=connectivity
        )

    # setup the initial region and candidate nodes to expand to
    for tid_surface in region:
        for tid_next in tn._get_neighbor_tids(tid_surface):
            _check_candidate(tid_surface, tid_next)

    # generate the sequence of tensor merges
    seq = []
    while candidates:
        # choose the *highest* scoring candidate
        candidates.sort(key=_sorter)
        tid_surface = candidates.pop()
        region.add(tid_surface)

        if distances[tid_surface] > min_distance:
            # checking distance allows the innermost merges to be ignored,
            # for example, to contract an environment around a region
            seq.append(
                (tid_surface, merges[tid_surface], distances[tid_surface])
            )

        # check all the neighbors of the tensor we've just expanded to
        for tid_next in tn._get_neighbor_tids(tid_surface):
            _check_candidate(tid_surface, tid_next)

    if inwards:
        # make the sequence of merges flow inwards
        seq.reverse()

    return seq


def get_local_patch(
    tn,
    tids,
    max_distance,
    include=None,
    exclude=None,
):
    """Get the local patch of tids that is within ``max_distance`` of
    the given ``tids``. This is like an unordered version of ``get_tree_span``.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to get the local patch from.
    tids : sequence of int
        The tensor ids to start from.
    max_distance : int
        The maximum distance from ``tids`` to include, in terms of graph
        distance. 0 corresponds to the original ``tids``, 1 to nearest
        neighbors and so on.
    include : sequence of int, optional
        If specified, only tids from this set can be included in the patch.
    exclude : sequence of int, optional
        If specified, tids from this set cannot be included in the patch.

    Returns
    -------
    tuple[int]
    """
    if include is None and exclude is None:

        def isvalid(tid):
            return True

    else:
        if include is None:
            include = set(tn.tensor_map)
        else:
            include = set(include)

        if exclude is None:
            exclude = set()
        else:
            exclude = set(exclude)

        def isvalid(tid):
            return (tid in include) and (tid not in exclude)

    d = 0
    if isinstance(tids, int):
        # allow single tid to be passed
        patch = {tids}
    else:
        patch = set(tids)

    boundary = tuple(patch)
    while d < max_distance:
        # expand outwards
        next_boundary = set()
        for tid in boundary:
            for tid_n in tn._get_neighbor_tids(tid):
                if isvalid(tid_n):
                    patch.add(tid_n)
                    next_boundary.add(tid_n)
        boundary = tuple(next_boundary)
        d += 1

    return tuple(sorted(patch))


def get_path_between_tids(tn, tida, tidb):
    """Find a shortest path between ``tida`` and ``tidb`` in this tensor
    network. Returns a ``NetworkPath`` if a path is found, otherwise ``None``.

    Currently ignores dangling and hyper indices.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to find a path in.
    tida : int
        The tensor id to start from.
    tidb : int
        The tensor id to end at.

    Returns
    -------
    NetworkPath or None
    """
    # expand from both points
    path_a0 = NetworkPath((tida,))
    path_b0 = NetworkPath((tidb,))
    queue_a = collections.deque((path_a0,))
    queue_b = collections.deque((path_b0,))
    # track ends of path so we identify when they meet
    # also acts a store for shortest path to that point
    ends_a = {tida: path_a0}
    ends_b = {tidb: path_b0}

    while queue_a or queue_b:
        for queue, ends_this, ends_other in [
            (queue_a, ends_a, ends_b),
            (queue_b, ends_b, ends_a),
        ]:
            if not queue:
                # no possible path
                return None

            path = queue.popleft()

            # get the tensor at the current end of the path
            last_tid = path.tids[-1]
            t = tn.tensor_map[last_tid]

            # check ways we could extend it
            for next_ind in t.inds:
                if next_ind in path:
                    # don't go back on ourselves
                    continue

                tids = tn.ind_map[next_ind]
                if len(tids) != 2:
                    # ignore dangling and hyper indices
                    continue

                next_tid = next(tid for tid in tids if tid != last_tid)

                if next_tid in ends_this:
                    # already been here in shorter or equal path
                    continue

                if next_tid in ends_other:
                    # found valid complete path!
                    other_path = ends_other[next_tid]

                    # want path to go from tida -> tidb
                    if queue is queue_a:
                        return NetworkPath(
                            tids=path.tids + other_path.tids[::-1],
                            inds=path.inds
                            + (next_ind,)
                            + other_path.inds[::-1],
                        )
                    else:
                        return NetworkPath(
                            tids=other_path.tids + path.tids[::-1],
                            inds=other_path.inds
                            + (next_ind,)
                            + path.inds[::-1],
                        )

                # valid partial path
                next_path = path.extend(next_ind, next_tid)
                ends_this[next_tid] = next_path
                queue.append(next_path)


def gen_all_paths_between_tids(tn, tida, tidb):
    """Generate all shortest paths between ``tida`` and ``tidb`` in this
    tensor network. Returns a generator of ``NetworkPath`` objects, ignores
    dangling and hyper indices currently.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to find paths in.
    tida : int
        The tensor id to start from.
    tidb : int
        The tensor id to end at.

    Yields
    ------
    NetworkPath
    """
    # map of only those neighbors which contribute to shortest paths
    predecessors = {}
    distances = {tidb: 0}
    queue = collections.deque([(tidb, 0)])
    found_start = False

    while queue:
        # get possible path extension, BFS
        last_tid, length = queue.popleft()

        # check ways we could extend it
        t = tn.tensor_map[last_tid]
        for next_ind in t.inds:
            tids = tn.ind_map[next_ind]
            if len(tids) != 2:
                # ignore dangling and hyper indices
                continue
            next_tid = next(tid for tid in tids if tid != last_tid)

            if next_tid == tida:
                found_start = True

            d = distances.get(next_tid, None)
            if d is None:
                # first time reaching this node
                distances[next_tid] = length + 1
                predecessors[next_tid] = [(last_tid, next_ind)]
                if not found_start:
                    # BFS search, so once we have found target, all
                    # possible paths will be in the queue already
                    queue.append((next_tid, length + 1))
            elif length < d:
                # another shortest path, just update predecessors
                # since extentions handled by case above
                predecessors[next_tid].append((last_tid, next_ind))

    # back track to find all paths
    queue = [NetworkPath([tida])]
    while queue:
        # this part can be DFS
        path = queue.pop()
        last_tid = path.tids[-1]
        for next_tid, next_ind in predecessors[last_tid]:
            new_path = path.extend(next_ind, next_tid)
            if next_tid == tidb:
                # reached the start
                yield new_path
            else:
                queue.append(new_path)


def gen_paths_loops(
    tn,
    max_loop_length=None,
    intersect=False,
    tids=None,
    inds=None,
    paths=None,
):
    """Generate all paths, up to a specified length, that represent loops in
    this tensor network. Unlike ``gen_loops`` this function will yield a
    `NetworkPath` objects, allowing one to differentiate between e.g. a double
    loop and a 'figure of eight' loop. Dangling and hyper indices are ignored.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to find loops in.
    max_loop_length : None or int
        Set the maximum number of indices that can appear in a loop. If
        ``None``, wait until any loop is found and set that as the maximum
        length.
    intersect : bool, optional
        Whether to allow self-intersecting loops.
    tids : None or sequence of int, optional
        If supplied, only consider loops containing one of these tensor ids.
    inds : None or sequence of str, optional
        If supplied, only consider loops containing one of these indices.
    paths : None or sequence of NetworkPath, optional
        If supplied, only consider loops starting from these paths.

    Yields
    ------
    NetworkPath

    See Also
    --------
    gen_loops, gen_inds_connected
    """
    queue = collections.deque()

    if isinstance(tids, int):
        # allow single tid to be passed
        tids = (tids,)
    if isinstance(inds, str):
        # allow single index to be passed
        inds = (inds,)

    if (tids is None) and (inds is None) and (paths is None):
        # default to finding loops everywhere
        inds = tn.ind_map

    if tids is not None:
        # generate loops starting at any of these tids
        for tid in tids:
            queue.append(NetworkPath([tid]))

    if inds is not None:
        # generate loops passing through any of these indices
        for ind in inds:
            tids = tn.ind_map[ind]
            if len(tids) != 2:
                # ignore dangling and hyper indices
                continue

            tida, tidb = tids
            # (only need one direction)
            queue.append(NetworkPath((tida, tidb), (ind,)))

    if paths is not None:
        # generate loops starting from these paths
        for path in paths:
            if not isinstance(path, NetworkPath):
                path = NetworkPath.from_sequence(path)
            queue.append(path)

    # cache index neighbor lookups for speed
    neighbormap = {}
    seen = set()

    while queue:
        path = queue.popleft()

        if intersect:
            # might have formed a closed loop, then it matter where we are
            # continuing from, so key on both ends
            search_key = (path.key, frozenset((path.tids[0], path.tids[-1])))
        else:
            # set of tids and inds is unique for non-intersecting loops
            search_key = path.key

        if search_key in seen:
            continue
        seen.add(search_key)

        last_tid = path.tids[-1]
        try:
            last_ind = path.inds[-1]
        except IndexError:
            # path is a single tid, no indices
            last_ind = None

        try:
            expansions = neighbormap[last_ind, last_tid]
        except KeyError:
            # check which ways we can continue this path
            possible_inds = tn.tensor_map[last_tid].inds
            expansions = []
            for next_ind in possible_inds:
                # don't come back the way we came
                if next_ind != last_ind:
                    next_ind_tids = tn.ind_map[next_ind]
                    # only consider normal bonds
                    if len(next_ind_tids) == 2:
                        # get the tid which isn't the direction we came
                        next_tid, next_tid_b = next_ind_tids
                        if next_tid == last_tid:
                            next_tid = next_tid_b
                        expansions.append((next_ind, next_tid))

            # cache this lookup
            neighbormap[last_ind, last_tid] = expansions

        continue_search = (max_loop_length is None) or (
            len(path) < max_loop_length - 1
        )

        for next_ind, next_tid in expansions:
            if next_ind in path:
                # can't ever double up on indices
                continue

            if next_tid == path.tids[0]:
                # finished a loop!

                loop = path.extend(next_ind, next_tid)
                if loop.key not in seen:
                    seen.add(loop.key)
                    if max_loop_length is None:
                        max_loop_length = len(loop)

                    # normalize the loop to be consistent across searches
                    # yield _normalize_loop(loop)
                    yield loop

            if continue_search and (intersect or next_tid not in path):
                # valid candidate extension!
                # -> we can double up on nodes only if intersecting
                queue.append(path.extend(next_ind, next_tid))


def gen_sloops(
    tn,
    max_loop_length=None,
    num_joins=1,
    intersect=False,
    tids=None,
    inds=None,
    paths=None,
):
    if num_joins < 1:
        return ()

    base_loops = gen_paths_loops(
        tn,
        max_loop_length=max_loop_length,
        intersect=intersect,
        tids=tids,
        inds=inds,
        paths=paths,
    )

    if num_joins == 1:
        # just return the base loops
        return base_loops

    # will reuse
    current_patches = tuple(base_loops)

    if (tids is None) and (inds is None) and (paths is None):
        # loops are already global
        base_loops = current_patches
    else:
        # need to merge local base loops with global loops
        # XXX: do this with the tids in current_patches, every join
        base_loops = tuple(
            gen_paths_loops(
                tn, max_loop_length=max_loop_length, intersect=intersect
            )
        )

    # efficient lookup of overlapping loop tids
    lookup = {}
    for sl in base_loops:
        for tid in unique(sl.tids):
            lookup.setdefault(tid, []).append(sl)

    for _ in range(num_joins - 1):
        next_patches = set()
        for p in current_patches:
            once = set()
            twice = set()

            # for each tensor
            for tid in unique(p.tids):
                # lookup possible base loops to merge with
                for po in lookup[tid]:
                    if po in once:
                        twice.add(po)
                    else:
                        once.add(po)

            for po in twice:
                # merge and add!
                next_patches.add(p.merge(po))

        current_patches = tuple(next_patches)

    return current_patches


def gen_patches(tn, max_size, tids=None, grow_from="all"):
    """Generate sets of tids that represent 'patches' of the tensor network,
    where each patch is a connected subgraph of the tensor network. Unlike
    generalized loops, patches can contain dangling nodes.
    """
    if tids is None:
        # find regions anywhere
        tids = tn.tensor_map.keys()
        grow_from = "any"
    elif isinstance(tids, int):
        # handle single tid region
        tids = (tids,)

    if grow_from == "all":
        # take `tids` as single initial region
        queue = collections.deque((frozenset(tids),))
    elif grow_from == "any":
        # take each tid as an initial region
        queue = collections.deque(frozenset([tid]) for tid in tids)
    else:
        raise ValueError("`grow_from` must be 'all' or 'any'.")

    # cache neighbors for speed
    neighbormap = {}
    seen = set()

    while queue:
        region = queue.popleft()

        if len(region) <= max_size:
            # is a valid patch
            yield tuple(sorted(region))

            if len(region) < max_size:
                # is valid to be extended
                # first get all atch neighbors
                outer = set()
                for tid in region:
                    try:
                        neighbors = neighbormap[tid]
                    except KeyError:
                        neighbors = tn._get_neighbor_tids([tid])
                        neighbormap[tid] = neighbors

                    for ntid in neighbors:
                        if ntid not in region:
                            outer.add(ntid)

                for ntid in outer:
                    # then extend this patch with each new combination
                    nregion = region | {ntid}
                    if nregion not in seen:
                        queue.append(nregion)
                        seen.add(nregion)


def _gen_gloops_single(tn, max_size=None, tids=None, grow_from="all"):
    if tids is None:
        # find loops everywhere
        tids = tn.tensor_map.keys()
        grow_from = "any"
    elif isinstance(tids, int):
        # handle single tid region
        tids = (tids,)

    if grow_from in ("all", "alldangle"):
        # take `tids` as single initial region
        queue = collections.deque((frozenset(tids),))
    elif grow_from in ("any", "anydangle"):
        # take each tid as an initial region
        queue = collections.deque(frozenset([tid]) for tid in tids)
    else:
        raise ValueError("`grow_from` must be 'all' or 'any'.")

    if "dangle" in grow_from:
        # target tids are allowed to be dangling
        dangle_tids = set(tids)
    else:
        dangle_tids = set()

    # cache neighbors for speed
    tid2inds = {}
    seen = set()

    while queue:
        region = queue.popleft()

        inds_once = set()
        inds_more = set()
        tids_next = set()

        for tid in region:
            try:
                inds = tid2inds[tid]
            except KeyError:
                inds = tid2inds[tid] = tn.tensor_map[tid].inds

            for ind in inds:
                if ind in inds_once:
                    # already seen -> is a bond
                    inds_more.add(ind)
                else:
                    inds_once.add(ind)

        valid_gloop = True
        for tid in region:
            # count number of connections each node has within region
            num_inner_connections = 0
            for ind in tid2inds[tid]:
                if ind in inds_more:
                    # bond within region
                    num_inner_connections += 1
                else:
                    # bond to outside -> add neighbors to queue
                    for ntid in tn.ind_map[ind]:
                        if ntid != tid:
                            tids_next.add(ntid)

            valid_gloop &= (
                # site is allowed to dangle
                tid in dangle_tids
                # or it is connected by at least two bonds
                or num_inner_connections >= 2
            )

        if valid_gloop:
            # valid region: no node is connected by a single bond only
            if max_size is None:
                # automatically set maximum region size
                max_size = len(region)
            yield tuple(sorted(region))

        if (max_size is None) or len(region) < max_size:
            # continue searching
            for ntid in tids_next:
                # possible extensions
                nregion = region | {ntid}
                if nregion not in seen:
                    # many ways to construct a region -> only check one
                    queue.append(nregion)
                    seen.add(nregion)


def gen_gloops(
    tn,
    max_size=None,
    tids=None,
    grow_from="all",
    num_joins=1,
):
    """Generate sets of tids that represent 'generalized loops' where every
    node is connected to at least two bonds, i.e. 2-degree connected subgraphs.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to find loops in.
    max_size : None or int
        Set the maximum number of tensors that can appear in a region. If
        ``None``, wait until any valid region is found and set that as the
        maximum size.
    tids : None or sequence of int, optional
        If supplied, only yield loops containing these tids, see
        ``grow_from``.
    grow_from : {'all', 'any', 'alldangle', 'anydangle'}, optional
        Only if ``tids`` is specified, this determines how to filter
        loops. If 'all', only yield loops containing *all* of the tids
        in ``tids``, if 'any', yield loops containing *any* of the tids
        in ``tids``. If 'alldangle' or 'anydangle', the tids are allowed to
        be dangling, i.e. 1-degree connected. This is useful for computing
        local expectations where the operator insertion breaks the loop
        assumption locally.

    Yields
    ------
    tuple[int]
    """
    if num_joins < 1:
        return ()

    base_gloops = _gen_gloops_single(
        tn,
        max_size=max_size,
        tids=tids,
        grow_from=grow_from,
    )

    if num_joins == 1:
        # just return the base loops
        return base_gloops

    # will reuse
    current_patches = tuple(map(frozenset, base_gloops))

    if tids is None:
        # loops are already global
        base_gloops = current_patches
    else:
        # need to merge local base loops with global loops
        # XXX: do this with the tids in current_patches, every join
        base_gloops = tuple(
            map(frozenset, _gen_gloops_single(tn, max_size=max_size))
        )

    # efficient lookup of overlapping gloop tids
    lookup = {}
    for gl in base_gloops:
        for tid in gl:
            lookup.setdefault(tid, []).append(gl)

    for _ in range(num_joins - 1):
        next_patches = set()
        for gl in current_patches:
            once = set()
            twice = set()

            # for each tensor
            for tid in gl:
                # lookup possible base loops to merge with
                for glo in lookup[tid]:
                    if glo in once:
                        twice.add(glo)
                    else:
                        once.add(glo)

            for glo in twice:
                # merge and add!
                next_patches.add(gl | glo)

        current_patches = tuple(next_patches)

    return current_patches


def gen_loops(tn, max_loop_length=None):
    """Generate sequences of tids that represent loops in the TN.

    Parameters
    ----------
    max_loop_length : None or int
        Set the maximum number of tensors that can appear in a loop. If
        ``None``, wait until any loop is found and set that as the
        maximum length.

    Yields
    ------
    tuple[int]

    See Also
    --------
    gen_paths_loops
    """
    from cotengra.core import get_hypergraph

    inputs = {tid: t.inds for tid, t in tn.tensor_map.items()}
    hg = get_hypergraph(inputs, accel="auto")
    return hg.compute_loops(max_loop_length=max_loop_length)


def get_loop_union(
    tn,
    tids,
    max_size=None,
    grow_from="all",
):
    """Find the union, in terms of tids, of all generliazed loops that pass
    through either all or at least one of the given tids, depending on
    `grow_from`.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to find the loop union region in.
    tids : sequence of int
        The tensor ids to consider.
    max_size : None or int, optional
        The maximum number of tensors that can appear in the region. If
        ``None``, wait until any valid region is found and set that as the
        maximum size.
    grow_from : {'all', 'any', 'alldangle', 'anydangle'}, optional
        Only if ``tids`` is specified, this determines how to filter
        loops. If 'all', only take loops containing *all* of the tids
        in ``tids``, if 'any', yield loops containing *any* of the tids
        in ``tids``. If 'alldangle' or 'anydangle', the base tids are allowed
        to be dangling, i.e. 1-degree connected.

    Returns
    -------
    tuple[int]
    """
    gloops = _gen_gloops_single(
        tn,
        max_size=max_size,
        tids=tids,
        grow_from=grow_from,
    )
    return tuple(sorted({tid for r in gloops for tid in r}))


def gen_inds_connected(tn, max_length):
    """Generate all index 'patches' of size up to ``max_length``.

    Parameters
    ----------
    max_length : int
        The maximum number of indices in the patch.

    Yields
    ------
    tuple[str]

    See Also
    --------
    gen_paths_loops
    """
    queue = [(ix,) for ix in tn.ind_map]
    seen = {frozenset(s) for s in queue}
    while queue:
        s = queue.pop()
        if len(s) == max_length:
            continue
        expansions = tn._get_neighbor_inds(s)
        for ix in expansions:
            next_s = s + (ix,)
            key = frozenset(next_s)
            if key not in seen:
                # new string
                yield next_s
                seen.add(key)
                queue.append(next_s)


def tids_are_connected(tn, tids):
    """Check whether nodes ``tids`` are connected.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to check.
    tids : sequence of int
        Nodes to check.

    Returns
    -------
    bool
    """
    enum = range(len(tids))
    groups = dict(zip(enum, enum))
    regions = [(oset([tid]), tn._get_neighbor_tids(tid)) for tid in tids]
    for i, j in itertools.combinations(enum, 2):
        mi = groups.get(i, i)
        mj = groups.get(j, j)

        if regions[mi][0] & regions[mj][1]:
            groups[mj] = mi
            regions[mi][0].update(regions[mj][0])
            regions[mi][1].update(regions[mj][1])

    return len(set(groups.values())) == 1


def compute_shortest_distances(tn, tids=None, exclude_inds=()):
    """Compute the minimum graph distances between all or some nodes
    ``tids``.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compute distances in.
    tids : None or sequence of int, optional
        If supplied, only compute distances between these nodes.
    exclude_inds : sequence of str, optional
        Exclude these indices when computing distances.

    Returns
    -------
    dict[tuple[int, int], int]
    """
    if tids is None:
        tids = tn.tensor_map
    else:
        tids = set(tids)

    visitors = collections.defaultdict(frozenset)
    for tid in tids:
        # start with target tids having 'visited' themselves only
        visitors[tid] = frozenset([tid])

    distances = {}
    N = math.comb(len(tids), 2)

    for d in itertools.count(1):
        any_change = False
        old_visitors = visitors.copy()

        # only need to iterate over touched region
        for tid in tuple(visitors):
            # at each step, each node sends its current visitors to all
            # neighboring nodes
            current_visitors = old_visitors[tid]
            for next_tid in tn._get_neighbor_tids(tid, exclude_inds):
                visitors[next_tid] |= current_visitors

        for tid in tuple(visitors):
            # check for new visitors -> those with shortest path d
            for diff_tid in visitors[tid] - old_visitors[tid]:
                any_change = True
                if (tid in tids) and (diff_tid in tids) and (tid < diff_tid):
                    distances[tid, diff_tid] = d

        if (len(distances) == N) or (not any_change):
            # all pair combinations have been computed, or everything
            # converged, presumably due to disconnected subgraphs
            break

    return distances


def compute_hierarchical_linkage(
    tn,
    tids=None,
    method="weighted",
    optimal_ordering=True,
    exclude_inds=(),
):
    from scipy.cluster import hierarchy

    if tids is None:
        tids = tn.tensor_map

    try:
        from cotengra import get_hypergraph

        hg = get_hypergraph(
            {tid: t.inds for tid, t in tn.tensor_map.items()},
            accel="auto",
        )
        for ix in exclude_inds:
            hg.remove_edge(ix)
        y = hg.all_shortest_distances_condensed(tuple(tids))
        return hierarchy.linkage(
            y, method=method, optimal_ordering=optimal_ordering
        )
    except ImportError:
        pass

    distances = tn.compute_shortest_distances(tids, exclude_inds)

    dinf = 10 * tn.num_tensors
    y = [
        distances.get(tuple(sorted((i, j))), dinf)
        for i, j in itertools.combinations(tids, 2)
    ]

    return hierarchy.linkage(
        y, method=method, optimal_ordering=optimal_ordering
    )


def compute_hierarchical_ssa_path(
    tn,
    tids=None,
    method="weighted",
    optimal_ordering=True,
    exclude_inds=(),
    are_sorted=False,
    linkage=None,
):
    """Compute a hierarchical grouping of ``tids``, as a ``ssa_path``."""
    if linkage is None:
        linkage = tn.compute_hierarchical_linkage(
            tids,
            method=method,
            exclude_inds=exclude_inds,
            optimal_ordering=optimal_ordering,
        )

    sorted_ssa_path = ((int(x[0]), int(x[1])) for x in linkage)
    if are_sorted:
        return tuple(sorted_ssa_path)

    if tids is None:
        tids = tn.tensor_map
    given_idx = {tid: i for i, tid in enumerate(tids)}
    sorted_to_given_idx = {
        i: given_idx[tid] for i, tid in enumerate(sorted(tids))
    }
    return tuple(
        (sorted_to_given_idx.get(x, x), sorted_to_given_idx.get(y, y))
        for x, y in sorted_ssa_path
    )


def compute_hierarchical_ordering(
    tn,
    tids=None,
    method="weighted",
    optimal_ordering=True,
    exclude_inds=(),
    linkage=None,
):
    """Compute a hierarchical ordering of ``tids``."""
    from scipy.cluster import hierarchy

    if tids is None:
        tids = list(tn.tensor_map)

    if linkage is None:
        linkage = tn.compute_hierarchical_linkage(
            tids,
            method=method,
            exclude_inds=exclude_inds,
            optimal_ordering=optimal_ordering,
        )

    node2tid = {i: tid for i, tid in enumerate(sorted(tids))}
    return tuple(map(node2tid.__getitem__, hierarchy.leaves_list(linkage)))


def compute_hierarchical_grouping(
    tn,
    max_group_size,
    tids=None,
    method="weighted",
    optimal_ordering=True,
    exclude_inds=(),
    linkage=None,
):
    """Group ``tids`` (by default, all tensors) into groups of size
    ``max_group_size`` or less, using a hierarchical clustering.
    """
    if tids is None:
        tids = list(tn.tensor_map)

    tids = sorted(tids)

    if linkage is None:
        linkage = tn.compute_hierarchical_linkage(
            tids,
            method=method,
            exclude_inds=exclude_inds,
            optimal_ordering=optimal_ordering,
        )

    ssa_path = tn.compute_hierarchical_ssa_path(
        tids=tids,
        method=method,
        exclude_inds=exclude_inds,
        are_sorted=True,
        linkage=linkage,
    )

    # follow ssa_path, agglomerating groups as long they small enough
    groups = {i: (tid,) for i, tid in enumerate(tids)}
    ssa = len(tids) - 1
    for i, j in ssa_path:
        ssa += 1

        if (i not in groups) or (j not in groups):
            # children already too big
            continue

        if len(groups[i]) + len(groups[j]) > max_group_size:
            # too big, skip
            continue

        # merge groups
        groups[ssa] = groups.pop(i) + groups.pop(j)

    # now sort groups by when their nodes in leaf ordering
    ordering = tn.compute_hierarchical_ordering(
        tids=tids,
        method=method,
        exclude_inds=exclude_inds,
        optimal_ordering=optimal_ordering,
        linkage=linkage,
    )
    score = {tid: i for i, tid in enumerate(ordering)}
    groups = sorted(
        groups.items(), key=lambda kv: sum(map(score.__getitem__, kv[1]))
    )

    return tuple(kv[1] for kv in groups)


def compute_centralities(tn):
    """Compute a simple centrality measure for each tensor in the network. The
    values go from 0 to 1, with 1 being the most central tensor.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compute centralities for.

    Returns
    -------
    dict[int, float]
    """
    import cotengra as ctg

    hg = ctg.get_hypergraph({tid: t.inds for tid, t in tn.tensor_map.items()})
    return hg.simple_centrality()


def most_central_tid(tn):
    """Find the most central tensor in the network."""
    cents = tn.compute_centralities()
    return max((score, tid) for tid, score in cents.items())[1]


def least_central_tid(tn):
    """Find the least central tensor in the network."""
    cents = tn.compute_centralities()
    return min((score, tid) for tid, score in cents.items())[1]
