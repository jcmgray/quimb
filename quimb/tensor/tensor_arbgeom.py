"""Classes and algorithms related to arbitrary geometry tensor networks."""

import functools
import itertools
import warnings
from operator import add, mul

from autoray import dag, do

from ..utils import LRU, check_opt, deprecated, ensure_dict
from ..utils import progbar as Progbar
from .contraction import get_symbol
from .tensor_core import (
    TensorNetwork,
    oset,
    rand_uuid,
    tags_to_oset,
)


def get_coordinate_formatter(ndims):
    return ",".join("{}" for _ in range(ndims))


def prod(xs):
    """Product of all elements in ``xs``."""
    return functools.reduce(mul, xs)


def tensor_network_align(
    *tns: "TensorNetworkGen",
    ind_ids=None,
    trace=False,
    inplace=False,
):
    r"""Align an arbitrary number of tensor networks in a stack-like geometry::

        a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a-a
        | | | | | | | | | | | | | | | | | | <- ind_ids[0] (defaults to 1st id)
        b-b-b-b-b-b-b-b-b-b-b-b-b-b-b-b-b-b
        | | | | | | | | | | | | | | | | | | <- ind_ids[1]
                       ...
        | | | | | | | | | | | | | | | | | | <- ind_ids[-2]
        y-y-y-y-y-y-y-y-y-y-y-y-y-y-y-y-y-y
        | | | | | | | | | | | | | | | | | | <- ind_ids[-1]
        z-z-z-z-z-z-z-z-z-z-z-z-z-z-z-z-z-z

    Parameters
    ----------
    tns : sequence of TensorNetwork
        The TNs to align, should be structured and either effective 'vectors'
        (have a ``site_ind_id``) or 'operators' (have a ``up_ind_id`` and
        ``lower_ind_id``).
    ind_ids : None, or sequence of str
        String with format specifiers to id each level of sites with. Will be
        automatically generated like ``(tns[0].site_ind_id, "__ind_a{}__",
        "__ind_b{}__", ...)`` if not given.
    inplace : bool
        Whether to modify the input tensor networks inplace.

    Returns
    -------
    tns_aligned : sequence of TensorNetwork
    """
    if not inplace:
        tns = [tn.copy() for tn in tns]

    n = len(tns)
    coordinate_formatter = get_coordinate_formatter(tns[0]._NDIMS)

    if ind_ids is None:
        if hasattr(tns[0], "site_ind_id"):
            ind_ids = [tns[0].site_ind_id]
        else:
            ind_ids = [tns[0].lower_ind_id]
        ind_ids.extend(
            f"__ind_{get_symbol(i)}{coordinate_formatter}__"
            for i in range(n - 2)
        )
    else:
        ind_ids = tuple(ind_ids)

    for i, tn in enumerate(tns):
        if hasattr(tn, "site_ind_id"):
            if i == 0:
                tn.site_ind_id = ind_ids[i]
            elif i == n - 1:
                tn.site_ind_id = ind_ids[i - 1]
            else:
                raise ValueError(
                    "An TN 'vector' can only be aligned as the "
                    "first or last TN in a sequence."
                )

        elif hasattr(tn, "upper_ind_id") and hasattr(tn, "lower_ind_id"):
            if i != 0:
                tn.upper_ind_id = ind_ids[i - 1]
            if i != n - 1:
                tn.lower_ind_id = ind_ids[i]

        else:
            raise ValueError("Can only align vectors and operators currently.")

    if trace:
        tns[-1].lower_ind_id = tns[0].upper_ind_id

    return tns


def tensor_network_apply_op_vec(
    A: "TensorNetworkGenOperator",
    x: "TensorNetworkGenVector",
    which_A="lower",
    contract=False,
    fuse_multibonds=True,
    compress=False,
    inplace=False,
    inplace_A=False,
    **compress_opts,
):
    """Apply a general a general tensor network representing an operator (has
    ``upper_ind_id`` and ``lower_ind_id``) to a tensor network representing a
    vector (has ``site_ind_id``), by contracting each pair of tensors at each
    site then compressing the resulting tensor network. How the compression
    takes place is determined by the type of tensor network passed in. The
    returned tensor network has the same site indices as ``x``, and it is
    the ``lower_ind_id`` of ``A`` that is contracted.

    This is like performing ``A.to_dense() @ x.to_dense()``, or the transpose
    thereof, depending on the value of ``which_A``.

    Parameters
    ----------
    A : TensorNetworkGenOperator
        The tensor network representing the operator.
    x : TensorNetworkGenVector
        The tensor network representing the vector.
    which_A : {"lower", "upper"}, optional
        Whether to contract the lower or upper indices of ``A`` with the site
        indices of ``x``.
    contract : bool
        Whether to contract the tensors at each site after applying the
        operator, yielding a single tensor at each site.
    fuse_multibonds : bool
        If ``contract=True``, whether to fuse any multibonds after contracting
        the tensors at each site.
    compress : bool
        Whether to compress the resulting tensor network.
    inplace : bool
        Whether to modify ``x``, the input vector tensor network inplace.
    inplace_A : bool
        Whether to reindex the operator tensor network ``A`` inplace, a
        minor performance gain if you don't need to use ``A`` afterwards.
    compress_opts
        Options to pass to ``tn.compress``, where ``tn`` is the resulting
        tensor network, if ``compress=True``.

    Returns
    -------
    TensorNetworkGenVector
        The same type as ``x``.
    """
    x = x if inplace else x.copy()
    A = A if inplace_A else A.copy()

    coordinate_formatter = get_coordinate_formatter(A._NDIMS)
    inner_ind_id = rand_uuid() + f"{coordinate_formatter}"

    if which_A == "lower":
        # align the indices
        #
        #     |       <- upper_ind_id to be site_ind_id (outerid)
        #    -A- ...
        #     |       <- lower_ind_id to be innerid
        #     :
        #     |       <- site_ind_id to be innerid
        #    -x- ...
        #
        A.lower_ind_id = inner_ind_id
        A.upper_ind_id = x.site_ind_id
    elif which_A == "upper":
        # transposed application
        A.upper_ind_id = inner_ind_id
        A.lower_ind_id = x.site_ind_id
    else:
        raise ValueError(
            f"Invalid `which_A`: {which_A}, should be 'lower' or 'upper'."
        )

    # only want to reindex on sites that being acted on
    sites_present_in_A = tuple(A.gen_sites_present())
    x.reindex_sites_(inner_ind_id, where=sites_present_in_A)

    # combine the tensor networks
    x |= A

    if contract:
        # optionally contract all tensor at each site
        for site in sites_present_in_A:
            x ^= site

        if fuse_multibonds:
            x.fuse_multibonds_()

    # optionally compress
    if compress:
        x.compress(**compress_opts)

    return x


def tensor_network_apply_op_op(
    A: "TensorNetworkGenOperator",
    B: "TensorNetworkGenOperator",
    which_A="lower",
    which_B="upper",
    contract=False,
    fuse_multibonds=True,
    compress=False,
    inplace=False,
    inplace_A=False,
    **compress_opts,
):
    """Apply the operator (has upper and lower site inds) represented by tensor
    network ``A`` to the operator represented by tensor network ``B``. The
    resulting tensor network has the same upper and lower indices as ``B``.
    Optionally contract the tensors at each site, fuse any multibonds, and
    compress the resulting tensor network.

    This is like performing ``A.to_dense() @ B.to_dense()``, or various
    combinations of tranposes thereof, depending on the values of ``which_A``
    and ``which_B``.

    Parameters
    ----------
    A : TensorNetworkGenOperator
        The tensor network representing the operator to apply.
    B : TensorNetworkGenOperator
        The tensor network representing the target operator.
    which_A : {"lower", "upper"}, optional
        Whether to contract the lower or upper indices of ``A``.
    which_B : {"lower", "upper"}, optional
        Whether to contract the lower or upper indices of ``B``.
    contract : bool
        Whether to contract the tensors at each site after applying the
        operator, yielding a single tensor at each site.
    fuse_multibonds : bool
        If ``contract=True``, whether to fuse any multibonds after contracting
        the tensors at each site.
    compress : bool
        Whether to compress the resulting tensor network.
    inplace : bool
        Whether to modify ``B``, the target tensor network inplace.
    inplace_A : bool
        Whether to modify ``A``, the applied operator tensor network inplace.
    compress_opts
        Options to pass to ``tn.compress``, where ``tn`` is the resulting
        tensor network, if ``compress=True``.

    Returns
    -------
    TensorNetworkGenOperator
        The same type as ``B``.
    """
    B = B if inplace else B.copy()
    A = A if inplace_A else A.copy()

    coordinate_formatter = get_coordinate_formatter(A._NDIMS)
    inner_ind_id = rand_uuid() + f"{coordinate_formatter}"

    if (which_A, which_B) == ("lower", "upper"):
        # align the indices (by default lower of A joined with upper of B
        # which corresponds to matrix multiplication):
        #
        #     |       <- A upper_ind_id to be upper_ind_id
        #    -A- ...
        #     |       <- A lower_ind_id to be innerid
        #     :
        #     |       <- B upper_ind_id to be innerid
        #    -B- ...
        #     |       <- B lower_ind_id to be lower_ind_id
        #
        A.lower_ind_id = inner_ind_id
        A.upper_ind_id = B.upper_ind_id
        B.reindex_upper_sites_(inner_ind_id)
    elif (which_A, which_B) == ("lower", "lower"):
        # rest are just permutations of above ...
        A.lower_ind_id = inner_ind_id
        A.upper_ind_id = B.lower_ind_id
        B.reindex_lower_sites_(inner_ind_id)
    elif (which_A, which_B) == ("upper", "upper"):
        A.upper_ind_id = inner_ind_id
        A.lower_ind_id = B.upper_ind_id
        B.reindex_upper_sites_(inner_ind_id)
    elif (which_A, which_B) == ("upper", "lower"):
        A.upper_ind_id = inner_ind_id
        A.lower_ind_id = B.lower_ind_id
        B.reindex_lower_sites_(inner_ind_id)
    else:
        raise ValueError("Invalid `which_A` and `which_B` combination.")

    # combine the tensor networks
    B |= A

    if contract:
        # optionally contract all tensor at each site
        for site in B.gen_sites_present():
            B ^= site

        if fuse_multibonds:
            B.fuse_multibonds_()

    if compress:
        B.compress(**compress_opts)

    return B


def create_lazy_edge_map(tn: "TensorNetworkGen", site_tags=None):
    """Given a tensor network, where each tensor is in exactly one group or
    'site', compute which sites are connected to each other, without checking
    each pair.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to analyze.
    site_tags : None or sequence of str, optional
        Which tags to consider as 'sites', by default uses ``tn.site_tags``.

    Returns
    -------
    edges : dict[tuple[str, str], list[str]]
        Each key is a sorted pair of tags, which are connected, and the value
        is a list of the indices connecting them.
    neighbors : dict[str, list[str]]
        For each site tag, the other site tags it is connected to.
    """
    if site_tags is None:
        site_tags = set(tn.site_tags)
    else:
        site_tags = set(site_tags)

    edges = {}
    neighbors = {}

    for ix in tn.ind_map:
        ts = tn._inds_get(ix)
        tags = {tag for t in ts for tag in t.tags if tag in site_tags}
        if len(tags) >= 2:
            # index spans multiple sites
            i, j = tuple(sorted(tags))

            if (i, j) not in edges:
                # record indices per edge
                edges[(i, j)] = [ix]

                # add to neighbor map
                neighbors.setdefault(i, []).append(j)
                neighbors.setdefault(j, []).append(i)
            else:
                # already processed this edge
                edges[(i, j)].append(ix)

    return edges, neighbors


def tensor_network_ag_sum(
    tna: "TensorNetworkGen",
    tnb: "TensorNetworkGen",
    site_tags=None,
    negate=False,
    compress=False,
    inplace=False,
    **compress_opts,
):
    """Add two tensor networks with arbitrary, but matching, geometries. They
    should have the same site tags, with a single tensor per site and sites
    connected by a single index only (but the name of this index can differ in
    the two TNs).

    Parameters
    ----------
    tna : TensorNetworkGen
        The first tensor network to add.
    tnb : TensorNetworkGen
        The second tensor network to add.
    site_tags : None or sequence of str, optional
        Which tags to consider as 'sites', by default uses ``tna.site_tags``.
    negate : bool, optional
        Whether to negate the second tensor network before adding.
    compress : bool, optional
        Whether to compress the resulting tensor network, by calling the
        ``compress`` method with the given options.
    inplace : bool, optional
        Whether to modify the first tensor network inplace.

    Returns
    -------
    TensorNetworkGen
        The resulting tensor network.
    """
    tna = tna if inplace else tna.copy()

    edges_a, neighbors_a = create_lazy_edge_map(tna, site_tags)
    edges_b, _ = create_lazy_edge_map(tnb, site_tags)

    reindex_map = {}
    for (si, sj), inds in edges_a.items():
        (ixa,) = inds
        (ixb,) = edges_b.pop((si, sj))
        reindex_map[ixb] = ixa

    if edges_b:
        raise ValueError("Not all edges matched.")

    for si in neighbors_a:
        ta, tb = tna[si], tnb[si]

        # the local outer indices
        sum_inds = [ix for ix in tb.inds if ix not in reindex_map]

        tb = tb.reindex(reindex_map)
        if negate:
            tb.negate_()
            # only need to negate a single tensor
            negate = False

        ta.direct_product_(tb, sum_inds)

    if compress:
        tna.compress(**compress_opts)

    return tna


def tensor_network_ag_gate(
    self: "TensorNetworkGen",
    G,
    where,
    contract=False,
    tags=None,
    propagate_tags=False,
    which=None,
    info=None,
    inplace=False,
    **compress_opts,
):
    r"""Apply a gate to this vector tensor network at sites ``where``. This is
    essentially a wrapper around
    :meth:`~quimb.tensor.tensor_core.TensorNetwork.gate_inds` apart from
    ``where`` can be specified as a list of sites, and tags can be optionally,
    intelligently propagated to the new gate tensor.

    .. math::

        | \psi \rangle \rightarrow G_\mathrm{where} | \psi \rangle

    Parameters
    ----------
    G : array_ike
        The gate array to apply, should match or be factorable into the shape
        ``(*phys_dims, *phys_dims)``.
    where : node or sequence[node]
        The sites to apply the gate to.
    contract : {False, True, 'split', 'reduce-split', 'split-gate',
                'swap-split-gate', 'auto-split-gate'}, optional
        How to apply the gate, see
        :meth:`~quimb.tensor.tensor_core.TensorNetwork.gate_inds`.
    tags : str or sequence of str, optional
        Tags to add to the new gate tensor.
    propagate_tags : {False, True, 'register', 'sites'}, optional
        Whether to propagate tags to the new gate tensor::

        - False: no tags are propagated
        - True: all tags are propagated
        - 'register': only site tags corresponding to ``where`` are
            added.
        - 'sites': all site tags on the current sites are propgated,
            resulting in a lightcone like tagging.

    info : None or dict, optional
        Used to store extra optional information such as the singular
        values if not absorbed.
    inplace : bool, optional
        Whether to perform the gate operation inplace on the tensor network
        or not.
    compress_opts
        Supplied to :func:`~quimb.tensor.tensor_core.tensor_split` for any
        ``contract`` methods that involve splitting. Ignored otherwise.

    Returns
    -------
    TensorNetworkGenVector

    See Also
    --------
    TensorNetwork.gate_inds
    """
    check_opt("propagate_tags", propagate_tags, _VALID_GATE_PROPAGATE)

    tn = self if inplace else self.copy()

    if which is None:
        site_ind_fn = tn.site_ind
    elif which == "upper":
        site_ind_fn = tn.upper_ind
    elif which == "lower":
        site_ind_fn = tn.lower_ind
    else:
        raise ValueError("`which` should be None, 'upper' or 'lower'.")

    if not isinstance(where, (tuple, list)):
        where = (where,)
    inds = tuple(map(site_ind_fn, where))

    # potentially add tags from current tensors to the new ones,
    # only do this if we are lazily adding the gate tensor(s)
    if (contract in _LAZY_GATE_CONTRACT) and (
        propagate_tags in (True, "sites")
    ):
        old_tags = oset.union(*(t.tags for t in tn._inds_get(*inds)))
        if propagate_tags == "sites":
            old_tags = tn.filter_valid_site_tags(old_tags)

        tags = tags_to_oset(tags)
        tags.update(old_tags)

    # perform the actual gating
    tn.gate_inds_(
        G, inds, contract=contract, tags=tags, info=info, **compress_opts
    )

    # possibly add tags based on where the gate was applied
    if propagate_tags == "register":
        for ix, site in zip(inds, where):
            (t,) = tn._inds_get(ix)
            t.add_tag(tn.site_tag(site))

    return tn


class TensorNetworkGen(TensorNetwork):
    """A tensor network which notionally has a single tensor per 'site',
    though these could be labelled arbitrarily could also be linked in an
    arbitrary geometry by bonds.
    """

    _NDIMS = 1
    _EXTRA_PROPS = (
        "_sites",
        "_site_tag_id",
    )

    def _compatible_arbgeom(self, other):
        """Check whether ``self`` and ``other`` represent the same set of
        sites and are tagged equivalently.
        """
        return isinstance(other, TensorNetworkGen) and all(
            getattr(self, e, 0) == getattr(other, e, 1)
            for e in TensorNetworkGen._EXTRA_PROPS
        )

    def combine(self, other, *, virtual=False, check_collisions=True):
        """Combine this tensor network with another, returning a new tensor
        network. If the two are compatible, cast the resulting tensor network
        to a :class:`TensorNetworkGen` instance.

        Parameters
        ----------
        other : TensorNetworkGen or TensorNetwork
            The other tensor network to combine with.
        virtual : bool, optional
            Whether the new tensor network should copy all the incoming tensors
            (``False``, the default), or view them as virtual (``True``).
        check_collisions : bool, optional
            Whether to check for index collisions between the two tensor
            networks before combining them. If ``True`` (the default), any
            inner indices that clash will be mangled.

        Returns
        -------
        TensorNetworkGen or TensorNetwork
        """
        new = super().combine(
            other, virtual=virtual, check_collisions=check_collisions
        )
        if self._compatible_arbgeom(other):
            new.view_as_(TensorNetworkGen, like=self)
        return new

    @property
    def nsites(self):
        """The total number of sites."""
        return len(self._sites)

    def gen_site_coos(self):
        """Generate the coordinates of all sites, same as ``self.sites``."""
        return self._sites

    @property
    def sites(self):
        """Tuple of the possible sites in this tensor network."""
        sites = getattr(self, "_sites", None)
        if sites is None:
            sites = tuple(self.gen_site_coos())
        return sites

    def _get_site_set(self):
        """The set of all sites."""
        if getattr(self, "_site_set", None) is None:
            self._site_set = set(self.sites)
        return self._site_set

    def gen_sites_present(self):
        """Generate the sites which are currently present (e.g. if a local view
        of a larger tensor network), based on whether their tags are present.

        Examples
        --------

            >>> tn = qtn.TN3D_rand(4, 4, 4, 2)
            >>> tn_sub = tn.select_local('I1,2,3', max_distance=1)
            >>> list(tn_sub.gen_sites_present())
            [(0, 2, 3), (1, 1, 3), (1, 2, 2), (1, 2, 3), (1, 3, 3), (2, 2, 3)]

        """
        return (
            site
            for site in self.gen_site_coos()
            if self.site_tag(site) in self.tag_map
        )

    @property
    def site_tag_id(self):
        """The string specifier for tagging each site of this tensor network."""
        return self._site_tag_id

    def site_tag(self, site):
        """The name of the tag specifiying the tensor at ``site``."""
        return self.site_tag_id.format(site)

    def retag_sites(self, new_id, where=None, inplace=False):
        """Modify the site tags for all or some tensors in this tensor network
        (without changing the ``site_tag_id``).

        Parameters
        ----------
        new_id : str
            A string with a format placeholder to accept a site, e.g. "S{}".
        where : None or sequence
            Which sites to update the index labels on. If ``None`` (default)
            all sites.
        inplace : bool
            Whether to retag in place.
        """
        if where is None:
            where = self.gen_sites_present()

        return self.retag(
            {self.site_tag(x): new_id.format(x) for x in where},
            inplace=inplace,
        )

    @property
    def site_tags(self):
        """All of the site tags."""
        if getattr(self, "_site_tags", None) is None:
            self._site_tags = tuple(map(self.site_tag, self.gen_site_coos()))
        return self._site_tags

    @property
    def site_tags_present(self):
        """All of the site tags still present in this tensor network."""
        return tuple(map(self.site_tag, self.gen_sites_present()))

    @site_tag_id.setter
    def site_tag_id(self, new_id):
        if self._site_tag_id != new_id:
            self.retag_sites(new_id, inplace=True)
            self._site_tag_id = new_id
            self._site_tags = None

    def retag_all(self, new_id, inplace=False):
        """Retag all sites and change the ``site_tag_id``."""
        tn = self if inplace else self.copy()
        tn.site_tag_id = new_id
        return tn

    retag_all_ = functools.partialmethod(retag_all, inplace=True)

    def _get_site_tag_set(self):
        """The oset of all site tags."""
        if getattr(self, "_site_tag_set", None) is None:
            self._site_tag_set = set(self.site_tags)
        return self._site_tag_set

    def filter_valid_site_tags(self, tags):
        """Get the valid site tags from ``tags``."""
        return oset(sorted(self._get_site_tag_set().intersection(tags)))

    def maybe_convert_coo(self, x):
        """Check if ``x`` is a valid site and convert to the corresponding site
        tag if so, else return ``x``.
        """
        try:
            if x in self._get_site_set():
                return self.site_tag(x)
        except TypeError:
            pass
        return x

    def gen_tags_from_coos(self, coos):
        """Generate the site tags corresponding to the given coordinates."""
        return map(self.site_tag, coos)

    def _get_tids_from_tags(self, tags, which="all"):
        """This is the function that lets coordinates such as ``site`` be
        used for many 'tag' based functions.
        """
        tags = self.maybe_convert_coo(tags)
        return super()._get_tids_from_tags(tags, which=which)

    def _get_tid_to_site_map(self):
        """Get a mapping from low level tensor id to the site it represents,
        assuming there is a single tensor per site.
        """
        tid2site = {}
        for site in self.sites:
            (tid,) = self._get_tids_from_tags(site)
            tid2site[tid] = site
        return tid2site

    def bond(self, coo1, coo2):
        """Get the name of the index defining the bond between sites at
        ``coo1`` and ``coo2``. This will error if there is not exactly one bond
        between the sites.

        Parameters
        ----------
        coo1 : hashable or str
            The first site, or site tag.
        coo2 : hashable or str
            The second site, or site tag.

        Returns
        -------
        str
        """
        (b_ix,) = self[coo1].bonds(self[coo2])
        return b_ix

    def bond_size(self, coo1, coo2):
        """Return the (combined) size of the bond(s) between sites at ``coo1``
        and ``coo2``.

        Parameters
        ----------
        coo1 : hashable or str
            The first site, or site tag.
        coo2 : hashable or str
            The second site, or site tag.

        Returns
        -------
        int
        """
        b_ix = self.bond(coo1, coo2)
        return self[coo1].ind_size(b_ix)

    def gen_bond_coos(self):
        """Generate the coordinates (pairs of sites) of all bonds."""
        tid2site = self._get_tid_to_site_map()
        seen = set()
        for tida in self.tensor_map:
            for tidb in self._get_neighbor_tids(tida):
                sitea, siteb = tid2site[tida], tid2site[tidb]
                if sitea > siteb:
                    sitea, siteb = siteb, sitea
                bond = (sitea, siteb)
                if bond not in seen:
                    yield bond
                    seen.add(bond)

    def get_site_neighbor_map(self):
        """Get a mapping from each site to its neighbors."""
        tid2site = self._get_tid_to_site_map()
        return {
            tid2site[tid]: tuple(
                tid2site[ntid] for ntid in self._get_neighbor_tids(tid)
            )
            for tid in self.tensor_map
        }

    def gen_gloops_sites(
        self,
        max_size=None,
        sites=None,
        grow_from="all",
        num_joins=1,
    ):
        """Generate sets of sites that represent 'generalized loops' where
        every node is connected to at least two other loop nodes. This is a
        simple wrapper around ``TensorNewtork.gen_gloops`` that works with the
        sites rather than ``tids``.

        Parameters
        ----------
        max_size : None or int
            Set the maximum number of tensors that can appear in a loop. If
            ``None``, wait until any valid loop is found and set that as the
            maximum size.
        sites : None or sequence[hashable]
            If supplied, only consider loops containing these sites.

        Yields
        ------
        tuple[hashable]
        """
        if sites is not None:
            tags = tuple(map(self.site_tag, sites))
            tids = self._get_tids_from_tags(tags, "any")
        else:
            tids = None

        tid2site = self._get_tid_to_site_map()

        for gloop in self.gen_gloops(
            max_size=max_size,
            tids=tids,
            grow_from=grow_from,
            num_joins=num_joins,
        ):
            yield tuple(tid2site[tid] for tid in gloop)

    def reset_cached_properties(self):
        """Reset any cached properties, one should call this when changing the
        actual geometry of a TN inplace, for example.
        """
        self._site_set = None
        self._site_tag_set = None
        self._site_tags = None

    @functools.wraps(tensor_network_align)
    def align(self, *args, inplace=False, **kwargs):
        return tensor_network_align(self, *args, inplace=inplace, **kwargs)

    align_ = functools.partialmethod(align, inplace=True)

    def __add__(self, other):
        return tensor_network_ag_sum(self, other)

    def __sub__(self, other):
        return tensor_network_ag_sum(self, other, negate=True)

    def __iadd__(self, other):
        return tensor_network_ag_sum(self, other, inplace=True)

    def __isub__(self, other):
        return tensor_network_ag_sum(self, other, negate=True, inplace=True)

    def normalize_simple(self, gauges, **contract_opts):
        """Normalize this network using simple local gauges. After calling
        this, any tree-like sub network gauged with ``gauges`` will have
        2-norm 1. Inplace operation on both the tensor network and ``gauges``.

        Parameters
        ----------
        gauges : dict[str, array_like]
            The gauges to normalize with.
        """
        # normalize gauges
        for ix, g in gauges.items():
            gauges[ix] = g / do("linalg.norm", g)

        nfactor = 1.0

        # normalize sites
        for site in self.sites:
            tn_site = self.select(site)
            tn_site_gauged = tn_site.copy()
            tn_site_gauged.gauge_simple_insert(gauges)
            lnorm = (tn_site_gauged.H | tn_site_gauged).contract(
                all, **contract_opts
            ) ** 0.5
            tn_site /= lnorm
            nfactor *= lnorm

        return nfactor

    def get_local_sloops(
        self,
        *,
        where=None,
        sloops=None,
        num_joins=1,
        intersect=False,
        grow_from="all",
        strict_size=False,
        info=None,
    ):
        r"""Parse the `sloops` argument to get the relevant simple loops for
        the given `where` sites. If `sloops` is an integer, auto generate loops
        locally, otherwise filter the given loops to only include those
        relevant to `where`. Simple loops are loops where each site is
        connected to exactly two other sites.
        """
        if isinstance(sloops, int):
            # auto generate sloops locally
            max_loop_length = sloops
            sloops = None
            if strict_size is True:
                # generate sloops up to size `max_loop_length`, but filter out
                # any which are larger once the target size are included
                strict_size = max_loop_length
        else:
            # filter out from set of manually supplied sloops
            max_loop_length = None
            if strict_size is True:
                warnings.warn(
                    "If manually supplying a set of sloops, `strict_size` "
                    "should be an integer, not a boolean - ignoring.",
                )
                strict_size = False

        if info is None:
            info = {}

        tags = tuple(map(self.site_tag, where))
        tids = self._get_tids_from_tags(tags, "any")

        # always include as base region target
        # sites + any bonds internal to those sites
        r0 = list(tids)
        r0_seen = set()
        for tid in tids:
            t = self.tensor_map[tid]
            for ix in t.inds:
                if ix in r0_seen:
                    # internal bond
                    r0.append(ix)
                else:
                    r0_seen.add(ix)
        r0 = frozenset(r0)

        if sloops is None:
            # find loops automatically
            # first find loop paths that pass through any of sites
            sloops = tuple(
                self.gen_sloops(
                    max_loop_length=max_loop_length,
                    num_joins=num_joins,
                    intersect=intersect,
                    tids=tids,
                )
            )

            if grow_from == "all":
                # further refine to loops that pass through all sites only
                sloops = tuple(
                    loop
                    for loop in sloops
                    if all(tid in loop.tids for tid in tids)
                )

        else:
            # have explicit loop specification, maybe as a larger set ->
            # need to select only the loops covering all or any  of `where`
            sloops = tuple(sloops)

            if "lookup" in info and hash(sloops) == info["lookup_hash"]:
                # want to share info across different expectations and also
                # different sets of loops -> but if the loop set has changed
                # then "lookup" specifically is probably incomplete
                lookup = info["lookup"]
            else:
                # build cache of which coordinates are in which loop to avoid
                # quadratic loop checking cost doing every time
                lookup = {}
                tid2site = self._get_tid_to_site_map()
                for loop in sloops:
                    for tid in loop.tids:
                        site = tid2site[tid]
                        lookup.setdefault(site, set()).add(loop)

                info["lookup"] = lookup
                info["lookup_hash"] = hash(sloops)

            if sloops:
                if grow_from == "all":
                    # get all loops which contain *all* sites in `where`
                    sloops = set.intersection(*(lookup[coo] for coo in where))
                elif grow_from == "any":
                    # get all loops which contain *any* site in `where`
                    sloops = set.union(*(lookup[coo] for coo in where))
                else:
                    raise ValueError(
                        f"Invalid `grow_from` value: {grow_from}."
                        "Should be 'all' or 'any'."
                    )

        if grow_from == "any":
            # loops only guaranteed to include any 1 site from `where`
            #     -> always include base sites as its own region
            #     -> make sure every loop includes base region
            clusters = (r0, *(r0.union(r) for r in sloops))
        else:
            # gloops guaranteed to include all sites from `where`, but might
            # not include the base region if its not a valid cluster on its own
            clusters = (r0, *map(frozenset, sloops))

        if strict_size:
            # only allow clusters below max_loop_length *including* base region
            clusters = (
                r0,
                *(
                    r
                    for r in clusters
                    if sum(isinstance(x, int) for x in r) <= strict_size
                ),
            )

        return clusters

    def get_local_gloops(
        self,
        *,
        tids=None,
        where=None,
        gloops=None,
        grow_from="all",
        num_joins=1,
        strict_size=False,
        info=None,
    ):
        r"""Parse the `gloops` argument to get the relevant generalized loops
        for the given `where` sites. If `gloops` is an integer, auto generate
        loops locally, otherwise filter the given loops to only include those
        relevant to `where`.

        An example loop with where=("A", "B"), for `grow_from='all'`::

             |   |
            -o---o-
             |   |
            -A---B-
             |   |

        Setting `grow_from='any'` in would also generates loops like::

             |   |
            -o---o-
             |   |   |
            -o---A---B-
             |   |   |

        where only site A is part of the original, generating loop, but both
        A and B are part of the returned cluster. For `"alldangle"` the same
        cluster would be generated but its size would be considered 5.

        Parameters
        ----------
        tids : sequence[int], optional
            The tensor ids to consider. Either this or `where` must be
            supplied.
        where : sequence[hashable], optional
            The sites to consider. Either this or `tids` must be supplied.
        info : dict
            A dictionary to store information across different calls.
        gloops : None, int, or sequence of sequence of hashable, optional
            The gloops to consider. If ``None``, auto generate local gloops
            up to the smallest non-trivial cluster size. If an integer,
            generate gloops locally with up to that size. If a sequence of
            sequences, use those gloops.
        grow_from : {"all", "any"}, optional
            Whether to generate gloops that originally contain all or just
            any of the target sites in `where`. The base sites are always
            included in all returned gloops, even if they were not part of
            the supplied or auto-generated cluster.

        Returns
        -------
        tuple[frozenset[hashable]]
        """
        if tids is None and where is None:
            raise ValueError("Either `tids` or `where` must be supplied.")

        if isinstance(gloops, int):
            max_size = gloops
            gloops = None
            if strict_size is True:
                # generate gloops up to size `max_loop_length`, but filter out
                # any which are larger once the target size are included
                strict_size = max_size
        else:
            max_size = None
            if strict_size is True:
                warnings.warn(
                    "If manually supplying a set of gloops, `strict_size` "
                    "should be an integer, not a boolean - ignoring.",
                )
                strict_size = False

        if info is None:
            info = {}

        if gloops is None:
            # auto generate gloops locally
            if where is not None:
                # sites given as coordinates
                r0 = frozenset(where)
                gloops = tuple(
                    self.gen_gloops_sites(
                        max_size=max_size,
                        sites=where,
                        grow_from=grow_from,
                        num_joins=num_joins,
                    )
                )
            else:
                # sites given as tensor ids
                r0 = frozenset(tids)
                gloops = tuple(
                    self.gen_gloops(
                        max_size=max_size,
                        tids=tids,
                        grow_from=grow_from,
                        num_joins=num_joins,
                    )
                )
        else:
            if tids:
                raise NotImplementedError(
                    "Supplying `tids` and `gloops` is not yet supported."
                )

            r0 = frozenset(where)

            # gloops already given, get the ones relevant to `where`
            gloops = tuple(gloops)

            if not gloops:
                # always include base sites
                gloops = (tuple(where),)

            if "lookup" in info and hash(gloops) == info["lookup_hash"]:
                # want to share info across different expectations and also
                # different sets of gloops -> but if the cluster set has
                # changed then "lookup" specifically is probably incomplete
                lookup = info["lookup"]
            else:
                # build cache of which coordinates are in which cluster to
                # avoid quadratic cluster checking cost doing every time
                lookup = {}
                for cluster in gloops:
                    for site in cluster:
                        lookup.setdefault(site, set()).add(cluster)

                info["lookup"] = lookup
                info["lookup_hash"] = hash(gloops)

            if grow_from == "all":
                # get all gloops which contain *all* sites in `where`
                gloops = set.intersection(*(lookup[coo] for coo in where))
            elif grow_from == "any":
                # get all gloops which contain *any* site in `where`
                gloops = set.union(*(lookup[coo] for coo in where))
            else:
                raise ValueError(
                    f"Invalid `grow_from` value: {grow_from}."
                    "Should be 'all' or 'any'."
                )

        if grow_from == "any":
            # gloops only guaranteed to include any 1 site from `where`
            #     -> always include base sites as its own region
            #     -> make sure every cluster includes base region
            clusters = (r0, *(r0.union(r) for r in gloops))
        else:
            # gloops guaranteed to include all sites from `where`, but might
            # not include the base region if its not a valid cluster on its own
            clusters = (r0, *map(frozenset, gloops))

        if strict_size:
            # only allow clusters below max_size *including* base region
            clusters = (r0, *(r for r in clusters if len(r) <= strict_size))

        return clusters


def gauge_product_boundary_vector(
    tn,
    tags,
    which="all",
    max_bond=1,
    smudge=1e-6,
    canonize_distance=0,
    select_local_distance=None,
    select_local_opts=None,
    **contract_around_opts,
):
    tids = tn._get_tids_from_tags(tags, which)

    # form the double layer tensor network - this is the TN we will
    #     generate the actual gauges with
    if select_local_distance is None:
        # use the whole tensor network ...
        outer_inds = tn.outer_inds()
        dtn = tn.H & tn
    else:
        # ... or just a local patch
        select_local_opts = ensure_dict(select_local_opts)
        ltn = tn._select_local_tids(
            tids,
            max_distance=select_local_distance,
            virtual=False,
            **select_local_opts,
        )
        outer_inds = ltn.outer_inds()
        dtn = ltn.H | ltn

    # get all inds in the tagged region
    region_inds = set.union(*(set(tn.tensor_map[tid].inds) for tid in tids))

    # contract all 'physical' indices so that we have a single layer TN
    #     outside region and double layer sandwich inside region
    for ix in outer_inds:
        if (ix in region_inds) or (ix not in dtn.ind_map):
            # 1st condition - don't contract region sandwich
            # 2nd condition - if local selecting, will get multibonds so
            #     some indices already contracted
            continue
        dtn.contract_ind(ix)

    # form the single layer boundary of double layer tagged region
    dtids = dtn._get_tids_from_tags(tags, which)
    dtn._contract_around_tids(
        dtids,
        min_distance=1,
        max_bond=max_bond,
        canonize_distance=canonize_distance,
        **contract_around_opts,
    )

    # select this boundary and compress to ensure it is a product operator
    dtn = dtn._select_without_tids(dtids, virtual=True)
    dtn.compress_all_(max_bond=1)
    dtn.squeeze_()

    # each tensor in the boundary should now have exactly two inds
    #     connecting to the top and bottom of the tagged region double
    #     layer. Iterate over these, inserting the gauge into the original
    #     tensor network that would turn each of these boundary tensors
    #     into identities.
    for t in dtn:
        (ix,) = [i for i in t.inds if i in region_inds]
        _, s, VH = do("linalg.svd", t.data)
        s = s + smudge
        G = do("reshape", s**0.5, (-1, 1)) * VH
        Ginv = dag(VH) * do("reshape", s**-0.5, (1, -1))

        tid_l, tid_r = sorted(tn.ind_map[ix], key=lambda tid: tid in tids)
        tn.tensor_map[tid_l].gate_(Ginv.T, ix)
        tn.tensor_map[tid_r].gate_(G, ix)

    return tn


def gloop_remove_dangling(sites, neighbors, where=()):
    """Given cluster `sites` and edges given by the `neighbors` mapping, remove
    all sites that are not connected to at least two other sites, reducing it
    in this way to a generalised loop.

    Parameters
    ----------
    sites : sequence[hashable]
        The sites in the original cluster.
    neighbors : dict[hashable, sequence[hashable]]
        The neighbors of each site.
    where : sequence[hashable], optional
        The sites to keep, even if they are dangling.

    Returns
    -------
    frozenset[hashable]
    """
    sites = list(sites)
    i = 0
    while i < len(sites):
        # check next site
        site = sites[i]
        # can only reduce non target sites
        if site not in where:
            num_neighbors = sum(nsite in sites for nsite in neighbors[site])
            if num_neighbors < 2:
                # dangling -> remove!
                sites.pop(i)
                # back to beginning
                i = -1
        i += 1
    return frozenset(sites)


def sloop_remove_dangling(path, neighbor_inds, where_tids):
    loop = set(path)
    while True:
        for x in loop:
            if isinstance(x, int) and (x not in where_tids):
                ninds = [ix for ix in neighbor_inds[x] if ix in loop]
                if len(ninds) <= 1:
                    # dangling tid and index -> remove
                    loop.remove(x)
                    loop.difference_update(ninds)
                    # go back to beginning
                    break
        else:
            # checked all without finding anything
            break
    return frozenset(loop)


_VALID_GATE_PROPAGATE = {"sites", "register", False, True}
_LAZY_GATE_CONTRACT = {
    False,
    "split-gate",
    "swap-split-gate",
    "auto-split-gate",
}


class TensorNetworkGenVector(TensorNetworkGen):
    """A tensor network which notionally has a single tensor and outer index
    per 'site', though these could be labelled arbitrarily and could also be
    linked in an arbitrary geometry by bonds.
    """

    _EXTRA_PROPS = (
        "_sites",
        "_site_tag_id",
        "_site_ind_id",
    )

    @property
    def site_ind_id(self):
        """The string specifier for the physical indices."""
        return self._site_ind_id

    def site_ind(self, site):
        return self.site_ind_id.format(site)

    @property
    def site_inds(self):
        """Return a tuple of all site indices."""
        if getattr(self, "_site_inds", None) is None:
            self._site_inds = tuple(map(self.site_ind, self.gen_site_coos()))
        return self._site_inds

    @property
    def site_inds_present(self):
        """All of the site inds still present in this tensor network."""
        return tuple(map(self.site_ind, self.gen_sites_present()))

    def reset_cached_properties(self):
        """Reset any cached properties, one should call this when changing the
        actual geometry of a TN inplace, for example.
        """
        self._site_inds = None
        return super().reset_cached_properties()

    def reindex_sites(self, new_id, where=None, inplace=False):
        """Modify the site indices for all or some tensors in this vector
        tensor network (without changing the ``site_ind_id``).

        Parameters
        ----------
        new_id : str
            A string with a format placeholder to accept a site, e.g. "ket{}".
        where : None or sequence
            Which sites to update the index labels on. If ``None`` (default)
            all sites.
        inplace : bool
            Whether to reindex in place.
        """
        if where is None:
            where = self.gen_sites_present()

        return self.reindex(
            {self.site_ind(x): new_id.format(x) for x in where},
            inplace=inplace,
        )

    reindex_sites_ = functools.partialmethod(reindex_sites, inplace=True)

    @site_ind_id.setter
    def site_ind_id(self, new_id):
        if self._site_ind_id != new_id:
            self.reindex_sites_(new_id)
            self._site_ind_id = new_id
            self._site_inds = None

    def reindex_all(self, new_id, inplace=False):
        """Reindex all physical sites and change the ``site_ind_id``."""
        tn = self if inplace else self.copy()
        tn.site_ind_id = new_id
        return tn

    reindex_all_ = functools.partialmethod(reindex_all, inplace=True)

    def gen_inds_from_coos(self, coos):
        """Generate the site inds corresponding to the given coordinates."""
        return map(self.site_ind, coos)

    def phys_dim(self, site=None):
        """Get the physical dimension of ``site``, defaulting to the first site
        if not specified.
        """
        if site is None:
            site = next(iter(self.gen_sites_present()))
        return self.ind_size(self.site_ind(site))

    def to_dense(
        self, *inds_seq, to_qarray=False, to_ket=None, **contract_opts
    ):
        """Contract this tensor network 'vector' into a dense array. By
        default, turn into a 'ket' ``qarray``, i.e. column vector of shape
        ``(d, 1)``.

        Parameters
        ----------
        inds_seq : sequence of sequences of str
            How to group the site indices into the dense array. By default,
            use a single group ordered like ``sites``, but only containing
            those sites which are still present.
        to_qarray : bool
            Whether to turn the dense array into a ``qarray``, if the backend
            would otherwise be ``'numpy'``.
        to_ket : None or str
            Whether to reshape the dense array into a ket (shape ``(d, 1)``
            array). If ``None`` (default), do this only if the ``inds_seq`` is
            not supplied.
        contract_opts
            Options to pass to
            :meth:`~quimb.tensor.tensor_core.TensorNewtork.contract`.

        Returns
        -------
        array
        """
        if not inds_seq:
            inds_seq = (self.site_inds_present,)
            if to_ket is None:
                to_ket = True

        x = TensorNetwork.to_dense(
            self, *inds_seq, to_qarray=to_qarray, **contract_opts
        )

        if to_ket:
            x = do("reshape", x, (-1, 1))

        return x

    to_qarray = functools.partialmethod(to_dense, to_qarray=True)

    def gate_with_op_lazy(
        self,
        A,
        transpose=False,
        inplace=False,
        inplace_op=False,
        **kwargs,
    ):
        r"""Act lazily with the operator tensor network ``A``, which should
        have matching structure, on this vector/state tensor network, like
        ``A @ x``. The returned tensor network will have the same structure as
        this one, but with the operator gated in lazily, i.e. uncontracted.

        .. math::

            | x \rangle \rightarrow A | x \rangle

        or (if ``transpose=True``):

        .. math::

            | x \rangle \rightarrow A^T | x \rangle

        Parameters
        ----------
        A : TensorNetworkGenOperator
            The operator tensor network to gate with, or apply to this tensor
            network.
        transpose : bool, optional
            Whether to contract the lower or upper indices of ``A`` with the
            site indices of ``x``. If ``False`` (the default), the lower
            indices of ``A`` will be contracted with the site indices of ``x``,
            if ``True`` the upper indices of ``A`` will be contracted with
            the site indices of ``x``, which is like applying ``A.T @ x``.
        inplace : bool, optional
            Whether to perform the gate operation inplace on this tensor
            network.
        inplace_op : bool, optional
            Whether to reindex the operator tensor network ``A`` inplace, a
            minor performance gain if you don't need to use ``A`` afterwards.

        Returns
        -------
        TensorNetworkGenVector
        """
        return tensor_network_apply_op_vec(
            A=A,
            x=self,
            which_A="upper" if transpose else "lower",
            contract=False,
            inplace=inplace,
            inplace_A=inplace_op,
            **kwargs,
        )

    gate_with_op_lazy_ = functools.partialmethod(
        gate_with_op_lazy, inplace=True
    )

    gate = tensor_network_ag_gate
    gate_ = functools.partialmethod(tensor_network_ag_gate, inplace=True)

    def gate_simple_(
        self,
        G,
        where,
        gauges,
        renorm=True,
        smudge=1e-12,
        power=1.0,
        info=None,
        **gate_opts,
    ):
        """Apply a gate to this vector tensor network at sites ``where``, using
        simple update style gauging of the tensors first, as supplied in
        ``gauges``. The new singular values for the bond are reinserted into
        ``gauges``.

        Parameters
        ----------
        G : array_like
            The gate to be applied.
        where : node or sequence[node]
            The sites to apply the gate to.
        gauges : dict[str, array_like]
            The store of gauge bonds, the keys being indices and the values
            being the vectors. Only bonds present in this dictionary will be
            used.
        renorm : bool, optional
            Whether to renormalise the singular after the gate is applied,
            before reinserting them into ``gauges``.
        smudge : float, optional
            A small value to add to the gauges before multiplying them in and
            inverting them to avoid numerical issues.
        power : float, optional
            The power to raise the singular values to before multiplying them
            in and inverting them.
        gate_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.gate_inds`.
        """
        if isinstance(where, int):
            where = (where,)

        site_tags = tuple(map(self.site_tag, where))
        tids = self._get_tids_from_tags(site_tags, "any")

        if len(tids) == 1:
            # gate acts on a single tensor
            return self.gate_(G, where, contract=True)

        gate_opts.setdefault("absorb", None)
        gate_opts.setdefault("contract", "reduce-split")
        tn_where = self._select_tids(tids)

        with tn_where.gauge_simple_temp(
            gauges,
            smudge=smudge,
            power=power,
            ungauge_inner=False,
        ):
            if info is None:
                info = {}

            tn_where.gate_(G, where, info=info, **gate_opts)

            # inner ungauging is performed by tracking the new singular values
            (((_, ix), s),) = info.items()
            if renorm:
                s = s / do("linalg.norm", s)
            gauges[ix] = s

        return self

    def gate_fit_local_(
        self,
        G,
        where,
        max_distance=1,
        mode="graphdistance",
        fillin=False,
        gauges=None,
        connect_path=True,
        **fit_opts,
    ):
        """Apply gate ``G`` to sites ``where`` by directly fitting a local
        patch of tensors, optionally gauging the boudary of this local patch
        first.

        Parameters
        ----------
        G : array_like
            The gate to be applied.
        where : node or sequence[node]
            The sites to apply the gate to.
        max_distance : int, optional
            The maximum distance from ``where`` to select tensors up to.
        mode : {'graphdistance', 'loopunion'}, optional
            How to select the local tensors, either by graph distance or by
            selecting the union of all loopy regions containing ``where``, of
            size up to ``max_distance``, ensuring no dangling tensors.
        fillin : bool or int, optional
            Whether to fill in the local patch with additional tensors, or not.
            `fillin` tensors are those connected by two or more bonds to the
            original local patch, the process is repeated int(fillin) times.
        gauges : dict[str, array_like], optional
            The store of gauge bonds, the keys being indices and the values
            being the vectors. Only bonds present in this dictionary will be
            used.
        connect_path : bool, optional
            If the sites in ``where `` are a non-adjacent pair, whether to set
            the initial patch as a path that connects them.
        fit_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.fit`.
        """

        # select a local neighborhood of tensors
        tags = tuple(map(self.site_tag, where))
        tids = self._get_tids_from_tags(tags, "any")

        if connect_path and (len(tids) == 2):
            tids = self.get_path_between_tids(*tids).tids

        k = self._select_local_tids(
            tids,
            max_distance=max_distance,
            mode=mode,
            fillin=fillin,
            virtual=True,
        )

        if gauges:
            outer, inner = k.gauge_simple_insert(gauges)

        # target TN with gate applied
        Gk = k.gate(G, where)

        # perform fit!
        k.fit_(Gk, **fit_opts)

        # ungauge
        if gauges:
            k.gauge_simple_remove(outer, inner)

        if gauges is not None:
            k.gauge_all_simple_(gauges=gauges, touched_tids=tids)

    def make_reduced_density_matrix(
        self,
        where,
        allow_dangling=True,
        bra_ind_id="b{}",
        mangle_append="*",
        layer_tags=("KET", "BRA"),
    ):
        """Form the tensor network representation of the reduced density
        matrix, taking special care to handle potential hyper inner and outer
        indices.

        Parameters
        ----------
        where : node or sequence[node]
            The sites to keep.
        allow_dangling : bool, optional
            Whether to allow dangling indices in the resulting density matrix.
            These are non-physical indices, that usually result from having
            cut a region of the tensor network.
        bra_ind_id : str, optional
            The string format to use for the bra indices.
        mangle_append : str, optional
            The string to append to indices that are not traced out.
        layer_tags : tuple of str, optional
            The tags to apply to the ket and bra tensor network layers.
        """
        where = set(where)
        reindex_map = {}
        phys_inds = set()

        for coo in self.gen_site_coos():
            kix = self.site_ind(coo)
            if coo in where:
                reindex_map[kix] = bra_ind_id.format(coo)
            phys_inds.add(kix)

        for ix, tids in self.ind_map.items():
            if ix in phys_inds:
                # traced out or handled above
                continue

            if (len(tids) == 1) and allow_dangling:
                # dangling indices appear most often in cluster methods
                continue

            reindex_map[ix] = ix + mangle_append

        ket = self.copy()
        bra = self.reindex(reindex_map).conj_()

        if layer_tags:
            ket.add_tag(layer_tags[0])
            bra.add_tag(layer_tags[1])

        # index collisions already handled above
        return ket.combine(bra, virtual=True, check_collisions=False)

    def partial_trace_exact(
        self,
        where,
        optimize="auto-hq",
        normalized=True,
        rehearse=False,
        get="matrix",
        **contract_opts,
    ):
        """Compute the reduced density matrix at sites ``where`` by exactly
        contracting the full overlap tensor network.

        Parameters
        ----------
        where : sequence[node]
            The sites to keep.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer to use, when exactly contracting the
            full tensor network.
        normalized : bool or "return", optional
            Whether to normalize the result. If "return", return the norm
            separately.
        rehearse : bool, optional
            Whether to perform the computation or not, if ``True`` return a
            rehearsal info dict.
        get : {'matrix', 'array', 'tensor'}, optional
            Whether to return the result as a dense array, the data itself, or
            a tensor network.

        Returns
        -------
        array or Tensor or dict or (array, float), (Tensor, float)
        """
        k_inds = tuple(map(self.site_ind, where))
        bra_ind_id = "_bra{}"
        b_inds = tuple(map(bra_ind_id.format, where))

        tn = self.make_reduced_density_matrix(where, bra_ind_id=bra_ind_id)

        if rehearse:
            return _handle_rehearse(
                rehearse, tn, optimize, output_inds=k_inds + b_inds
            )

        rho = tn.contract(
            output_inds=(*k_inds, *b_inds),
            optimize=optimize,
            **contract_opts,
        )

        if normalized:
            rho_array_fused = rho.to_dense(k_inds, b_inds)
            nfactor = do("trace", rho_array_fused)
        else:
            rho_array_fused = nfactor = None

        if get == "matrix":
            if rho_array_fused is None:
                # might have computed already
                rho_array_fused = rho.to_dense(k_inds, b_inds)
            if normalized is True:
                # multiply norm in
                rho = rho_array_fused / nfactor
            else:
                rho = rho_array_fused
        elif get == "array":
            if normalized is True:
                # multiply norm in
                rho = rho.data / nfactor
            else:
                rho = rho.data
        elif get == "tensor":
            if normalized is True:
                # multiply norm in, inplace
                rho.multiply_(1 / nfactor)
        else:
            raise ValueError(f"Unrecognized 'get' value: {get}")

        if normalized == "return":
            return rho, nfactor
        else:
            return rho

    def local_expectation_exact(
        self,
        G,
        where,
        optimize="auto-hq",
        normalized=True,
        rehearse=False,
        **contract_opts,
    ):
        """Compute the local expectation of operator ``G`` at sites ``where``
        by exactly contracting the full overlap tensor network.

        Parameters
        ----------
        G : array_like
            The operator to compute the expectation of.
        where : sequence[node]
            The sites to compute the expectation at.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer to use, when exactly contracting the
            full tensor network.
        normalized : bool or "return", optional
            Whether to normalize the result. If "return", return the norm
            separately.
        rehearse : bool, optional
            Whether to perform the computation or not, if ``True`` return a
            rehearsal info dict.
        contract_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract`.

        Returns
        -------
        float or (float, float)
        """
        rho = self.partial_trace_exact(
            where=where,
            optimize=optimize,
            rehearse=rehearse,
            normalized=normalized,
            get="array",
            **contract_opts,
        )

        if rehearse:
            # immediately return the info dict
            return rho

        if normalized == "return":
            # separate out the norm
            rho, nfactor = rho

        ng = len(where)
        if do("ndim", G) != 2 * ng:
            # might be supplied in matrix form
            G = do("reshape", G, rho.shape)

        # contract the expectation!
        expec = do(
            "tensordot",
            rho,
            G,
            axes=(
                tuple(range(2 * ng)),
                tuple(range(ng, 2 * ng)) + tuple(range(0, ng)),
            ),
        )

        if normalized == "return":
            return expec, nfactor
        else:
            return expec

    def compute_local_expectation_exact(
        self,
        terms,
        optimize="auto-hq",
        *,
        normalized=True,
        return_all=False,
        rehearse=False,
        executor=None,
        progbar=False,
        **contract_opts,
    ):
        """Compute the local expectations of many operators,
        by exactly contracting the full overlap tensor network.

        Parameters
        ----------
        terms : dict[node or (node, node), array_like]
            The terms to compute the expectation of, with keys being the sites
            and values being the local operators.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer to use, when exactly contracting the
            full tensor network.
        normalized : bool, optional
            Whether to normalize the result.
        return_all : bool, optional
            Whether to return all results, or just the summed expectation.
        rehearse : {False, 'tn', 'tree', True}, optional
            Whether to perform the computations or not::

                - False: perform the computation.
                - 'tn': return the tensor networks of each local expectation,
                  without running the path optimizer.
                - 'tree': run the path optimizer and return the
                  ``cotengra.ContractonTree`` for each local expectation.
                - True: run the path optimizer and return the ``PathInfo`` for
                  each local expectation.

        executor : Executor, optional
            If supplied compute the terms in parallel using this executor.
        progbar : bool, optional
            Whether to show a progress bar.
        contract_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract`.

        Returns
        -------
        expecs : float or dict[node or (node, node), float]
            If ``return_all==False``, return the summed expectation value of
            the given terms. Otherwise, return a dictionary mapping each term's
            location to the expectation value.
        """
        return _compute_expecs_maybe_in_parallel(
            fn=_tn_local_expectation_exact,
            tn=self,
            terms=terms,
            return_all=return_all,
            executor=executor,
            progbar=progbar,
            optimize=optimize,
            normalized=normalized,
            rehearse=rehearse,
            **contract_opts,
        )

    def get_cluster(
        self,
        where,
        gauges=None,
        max_distance=0,
        mode="graphdistance",
        fillin=0,
        grow_from="all",
        smudge=1e-12,
        power=1.0,
    ):
        """Get the wavefunction cluster tensor network for the sites
        surrounding ``where``, potentially gauging the region with the simple
        update style bond gauges in ``gauges``.

        Parameters
        ----------
        where : sequence[node]
            The sites around which to form the cluster.
        gauges : dict[str, array_like], optional
            The store of gauge bonds, the keys being indices and the values
            being the vectors. Only bonds present in this dictionary will be
            used.
        max_distance : int, optional
            The maximum graph distance to include tensors neighboring ``where``
            when computing the expectation. The default 0 means only the
            tensors at sites ``where`` are used, 1 includes their direct
            neighbors, etc.
        mode : {'graphdistance', 'loopunion'}, optional
            How to select the local tensors, either by graph distance or by
            selecting the union of all loopy regions containing ``where``, of
            size up to ``max_distance``, ensuring no dangling tensors.
        fillin : bool or int, optional
            When selecting the local tensors, whether and how many times to
            'fill-in' corner tensors attached multiple times to the local
            region. On a lattice this fills in the corners. See
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.select_local`.
        grow_from : {"all", "any"}, optional
            If mode is 'loopunion', whether each loop should contain *all* of
            the initial tagged tensors, or just *any* of them (generating a
            larger region).
        smudge : float, optional
            A small value to add to the gauges before multiplying them in and
            inverting them to avoid numerical issues.
        power : float, optional
            The power to raise the singular values to before multiplying them
            in and inverting them.

        Returns
        -------
        TensorNetworkGenVector
        """
        # select a local neighborhood of tensor tids
        tids = self._get_tids_from_tags(
            tuple(map(self.site_tag, where)), "any"
        )

        if len(tids) == 2:
            # connect the sites up
            tids = self.get_path_between_tids(*tids).tids

        # select the local patch!
        k = self._select_local_tids(
            tids,
            max_distance=max_distance,
            mode=mode,
            fillin=fillin,
            grow_from=grow_from,
            virtual=False,
        )

        if gauges is not None:
            # gauge the region with simple update style bond gauges
            k.gauge_simple_insert(gauges, smudge=smudge, power=power)

        return k

    def partial_trace_cluster(
        self,
        where,
        gauges=None,
        optimize="auto-hq",
        normalized=True,
        max_distance=0,
        mode="graphdistance",
        fillin=0,
        grow_from="all",
        smudge=1e-12,
        power=1.0,
        get="matrix",
        rehearse=False,
        **contract_opts,
    ):
        """Compute the approximate reduced density matrix at sites ``where`` by
        contracting a local cluster of tensors, potentially gauging the region
        with the simple update style bond gauges in ``gauges``.

        Parameters
        ----------
        where : sequence[node]
            The sites to keep.
        gauges : dict[str, array_like], optional
            The store of gauge bonds, the keys being indices and the values
            being the vectors. Only bonds present in this dictionary will be
            used.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer to use, when exactly contracting the
            local tensors.
        normalized : bool or "return", optional
            Whether to normalize the result. If "return", return the norm
            separately.
        max_distance : int, optional
            The maximum graph distance to include tensors neighboring ``where``
            when computing the expectation. The default 0 means only the
            tensors at sites ``where`` are used, 1 includes there direct
            neighbors, etc.
        mode : {'graphdistance', 'loopunion'}, optional
            How to select the local tensors, either by graph distance or by
            selecting the union of all loopy regions containing ``where``, of
            size up to ``max_distance``, ensuring no dangling tensors.
        fillin : bool or int, optional
            When selecting the local tensors, whether and how many times to
            'fill-in' corner tensors attached multiple times to the local
            region. On a lattice this fills in the corners. See
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.select_local`.
        grow_from : {"all", "any"}, optional
            If mode is 'loopunion', whether each loop should contain *all* of
            the initial tagged tensors, or just *any* of them (generating a
            larger region).
        smudge : float, optional
            A small value to add to the gauges before multiplying them in and
            inverting them to avoid numerical issues.
        power : float, optional
            The power to raise the singular values to before multiplying them
            in and inverting them.
        get : {'matrix', 'array', 'tensor'}, optional
            Whether to return the result as a fused matrix (i.e. always 2D),
            unfused array, or still labeled Tensor.
        rehearse : bool, optional
            Whether to perform the computation or not, if ``True`` return a
            rehearsal info dict.
        contract_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract`.
        """
        k = self.get_cluster(
            where,
            gauges=gauges,
            max_distance=max_distance,
            mode=mode,
            fillin=fillin,
            grow_from=grow_from,
            smudge=smudge,
            power=power,
        )

        return k.partial_trace_exact(
            where=where,
            optimize=optimize,
            normalized=normalized,
            rehearse=rehearse,
            get=get,
            **contract_opts,
        )

    def local_expectation_cluster(
        self,
        G,
        where,
        normalized=True,
        max_distance=0,
        mode="graphdistance",
        fillin=False,
        grow_from="all",
        gauges=None,
        smudge=0.0,
        power=1.0,
        optimize="auto",
        max_bond=None,
        rehearse=False,
        **contract_opts,
    ):
        r"""Approximately compute a single local expectation value of the gate
        ``G`` at sites ``where``, either treating the environment beyond
        ``max_distance`` as the identity, or using simple update style bond
        gauges as supplied in ``gauges``.

        This selects a local neighbourhood of tensors up to distance
        ``max_distance`` away from ``where``, then traces over dangling bonds
        after potentially inserting the bond gauges, to form an approximate
        version of the reduced density matrix.

        .. math::

            \langle \psi | G | \psi \rangle
            \approx
            \frac{
            \mathrm{Tr} [ G \tilde{\rho}_\mathrm{where} ]
            }{
            \mathrm{Tr} [ \tilde{\rho}_\mathrm{where} ]
            }

        assuming ``normalized==True``.

        Parameters
        ----------
        G : array_like
            The gate to compute the expecation of.
        where : node or sequence[node]
            The sites to compute the expectation at.
        normalized : bool, optional
            Whether to locally normalize the result, i.e. divide by the
            expectation value of the identity.
        max_distance : int, optional
            The maximum graph distance to include tensors neighboring ``where``
            when computing the expectation. The default 0 means only the
            tensors at sites ``where`` are used, 1 includes there direct
            neighbors, etc.
        mode : {'graphdistance', 'loopunion'}, optional
            How to select the local tensors, either by graph distance or by
            selecting the union of all loopy regions containing ``where``, of
            size up to ``max_distance``, ensuring no dangling tensors.
        fillin : bool or int, optional
            When selecting the local tensors, whether and how many times to
            'fill-in' corner tensors attached multiple times to the local
            region. On a lattice this fills in the corners. See
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.select_local`.
        grow_from : {"all", "any"}, optional
            If mode is 'loopunion', whether each loop should contain *all* of
            the initial tagged tensors, or just *any* of them (generating a
            larger region).
        gauges : dict[str, array_like], optional
            The store of gauge bonds, the keys being indices and the values
            being the vectors. Only bonds present in this dictionary will be
            used.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer to use, when exactly contracting the
            local tensors.
        max_bond : None or int, optional
            If specified, use compressed contraction.
        rehearse : {False, 'tn', 'tree', True}, optional
            Whether to perform the computations or not::

            - False: perform the computation.
            - 'tn': return the tensor networks of each local expectation,
                without running the path optimizer.
            - 'tree': run the path optimizer and return the
                ``cotengra.ContractonTree`` for each local expectation.
            - True: run the path optimizer and return the ``PathInfo`` for
                each local expectation.

        Returns
        -------
        expectation : float
        """
        k = self.get_cluster(
            where,
            gauges=gauges,
            max_distance=max_distance,
            mode=mode,
            fillin=fillin,
            grow_from=grow_from,
            smudge=smudge,
            power=power,
        )

        if max_bond is not None:
            return k.local_expectation(
                G=G,
                where=where,
                max_bond=max_bond,
                optimize=optimize,
                normalized=normalized,
                rehearse=rehearse,
                **contract_opts,
            )

        return k.local_expectation_exact(
            G=G,
            where=where,
            optimize=optimize,
            normalized=normalized,
            rehearse=rehearse,
            **contract_opts,
        )

    local_expectation_simple = deprecated(
        local_expectation_cluster,
        "local_expectation_simple",
        "local_expectation_cluster",
    )

    def compute_local_expectation_cluster(
        self,
        terms,
        *,
        max_distance=0,
        mode="graphdistance",
        fillin=False,
        grow_from="all",
        normalized=True,
        gauges=None,
        optimize="auto",
        max_bond=None,
        return_all=False,
        rehearse=False,
        executor=None,
        progbar=False,
        **contract_opts,
    ):
        r"""Compute all local expectations of the given terms, either treating
        the environment beyond ``max_distance`` as the identity, or using
        simple update style bond gauges as supplied in ``gauges``.

        This selects a local neighbourhood of tensors up to distance
        ``max_distance`` away from each term's sites, then traces over
        dangling bonds after potentially inserting the bond gauges, to form
        an approximate version of the reduced density matrix.

        .. math::

            \sum_\mathrm{i}
            \langle \psi | G_\mathrm{i} | \psi \rangle
            \approx
            \sum_\mathrm{i}
            \frac{
            \mathrm{Tr} [ G_\mathrm{i} \tilde{\rho}_\mathrm{i} ]
            }{
            \mathrm{Tr} [ \tilde{\rho}_\mathrm{i} ]
            }

        assuming ``normalized==True``.

        Parameters
        ----------
        terms : dict[node or (node, node), array_like]
            The terms to compute the expectation of, with keys being the sites
            and values being the local operators.
        max_distance : int, optional
            The maximum graph distance to include tensors neighboring ``where``
            when computing the expectation. The default 0 means only the
            tensors at sites ``where`` are used, 1 includes there direct
            neighbors, etc.
        mode : {'graphdistance', 'loopunion'}, optional
            How to select the local tensors, either by graph distance or by
            selecting the union of all loopy regions containing ``where``, of
            size up to ``max_distance``, ensuring no dangling tensors.
        fillin : bool or int, optional
            When selecting the local tensors, whether and how many times to
            'fill-in' corner tensors attached multiple times to the local
            region. On a lattice this fills in the corners. See
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.select_local`.
        grow_from : {"all", "any"}, optional
            If mode is 'loopunion', whether each loop should contain *all* of
            the initial tagged tensors, or just *any* of them (generating a
            larger region).
        normalized : bool, optional
            Whether to locally normalize the result, i.e. divide by the
            expectation value of the identity. This implies that a different
            normalization factor is used for each term.
        gauges : dict[str, array_like], optional
            The store of gauge bonds, the keys being indices and the values
            being the vectors. Only bonds present in this dictionary will be
            used.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer to use, when exactly contracting the
            local tensors.
        max_bond : None or int, optional
            If specified, use compressed contraction.
        return_all : bool, optional
            Whether to return all results, or just the summed expectation.
        rehearse : {False, 'tn', 'tree', True}, optional
            Whether to perform the computations or not::

                - False: perform the computation.
                - 'tn': return the tensor networks of each local expectation,
                  without running the path optimizer.
                - 'tree': run the path optimizer and return the
                  ``cotengra.ContractonTree`` for each local expectation.
                - True: run the path optimizer and return the ``PathInfo`` for
                  each local expectation.

        executor : Executor, optional
            If supplied compute the terms in parallel using this executor.
        progbar : bool, optional
            Whether to show a progress bar.
        contract_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract`.

        Returns
        -------
        expecs : float or dict[node or (node, node), float]
            If ``return_all==False``, return the summed expectation value of
            the given terms. Otherwise, return a dictionary mapping each term's
            location to the expectation value.
        """
        return _compute_expecs_maybe_in_parallel(
            fn=_tn_local_expectation_cluster,
            tn=self,
            terms=terms,
            return_all=return_all,
            executor=executor,
            progbar=progbar,
            normalized=normalized,
            max_distance=max_distance,
            mode=mode,
            fillin=fillin,
            grow_from=grow_from,
            gauges=gauges,
            optimize=optimize,
            rehearse=rehearse,
            max_bond=max_bond,
            **contract_opts,
        )

    compute_local_expectation_simple = deprecated(
        compute_local_expectation_cluster,
        "compute_local_expectation_simple",
        "compute_local_expectation_cluster",
    )

    def local_expectation_sloop_expand(
        self,
        G,
        where,
        sloops=None,
        gauges=None,
        *,
        num_joins=1,
        grow_from="all",
        strict_size=False,
        intersect=False,
        autocomplete=True,
        autoreduce=True,
        combine="prod",
        normalized=True,
        info=None,
        optimize="auto",
        tn_cache_maxsize=512,
        progbar=False,
        **contract_opts,
    ):
        """Compute the expectation of operator ``G`` at site(s) ``where`` by
        expanding the expectation in terms of loops of tensors.

        Parameters
        ----------
        G : array_like
            The operator to compute the expectation of.
        where : node or sequence[node]
            The sites to compute the expectation at.
        sloops : None or sequence[NetworkPath], optional
            The loops to use. If an integer, all loops up to and including that
            length will be used if the loop passes through all sites in
            ``where``. If ``None`` the maximum loop length is set as the
            shortest loop found. If an explicit set of loops is given, only
            these loops are considered, but only if they pass through all sites
            in ``where``, and ``intersect`` is ignored.
        gauges : dict[str, array_like], optional
            The store of gauge bonds, the keys being indices and the values
            being the vectors. Only bonds present in this dictionary will be
            gauged.
        intersect : bool, optional
            If ``loops`` is not an explicit set of loops, whether to consider
            self intersecting loops in the search for loops passing through
            ``where``.
        normalized : bool or "local", optional
            Whether to normalize the result. If "local" each loop term is
            normalized separately. If ``True`` each term is normalized using
            a loop expansion estimate of the norm. If ``False`` no
            normalization is performed.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer to use.
        info : dict, optional
            A dictionary to store intermediate results in to avoid recomputing
            them. This is useful when computing various expectations with
            different sets of loops. This should only be reused when both the
            tensor network and gauges remain the same.
        tn_cache_maxsize : int, optional
            The maximum size of the cache for the locally gauged tensor
            networks used in computing the local expectations.
        contract_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract`.

        Returns
        -------
        expec : scalar
        """
        from quimb.tensor.belief_propagation import gen_region_counts

        info = info if info is not None else {}
        info.setdefault("tns", LRU(tn_cache_maxsize))
        info.setdefault("expecs", {})

        sloops = self.get_local_sloops(
            where=where,
            sloops=sloops,
            num_joins=num_joins,
            intersect=intersect,
            grow_from=grow_from,
            strict_size=strict_size,
            info=info,
        )

        if autoreduce:
            where_tags = tuple(map(self.site_tag, where))
            where_tids = self._get_tids_from_tags(where_tags, "any")
            try:
                neighbor_inds = info["neighbor_inds"]
            except KeyError:
                neighbor_inds = info["neighbor_inds"] = {
                    tid: t.inds for tid, t in self.tensor_map.items()
                }
        else:
            neighbor_inds = where_tids = None

        expecs = []
        norms = []
        counts = []

        loop_counts = gen_region_counts(sloops, autocomplete=autocomplete)
        if progbar:
            loop_counts = Progbar(loop_counts)

        for loop, C in loop_counts:
            if autoreduce:
                # check if we can map loop to a smaller cluster
                # this is only valid at a BP fixed point
                loop = sloop_remove_dangling(loop, neighbor_inds, where_tids)

            try:
                # have already computed this term in a different expectation
                # e.g. with different set of loops
                expec_loop, norm_loop = info["expecs"][loop, where]
            except KeyError:
                # get the gauged loop tn
                try:
                    tnl = info["tns"][loop]
                except KeyError:
                    tnl = self.select_path(loop, gauges=gauges)
                    info["tns"][loop] = tnl

                # compute the expectation with exact contraction
                expec_loop, norm_loop = tnl.local_expectation_exact(
                    G,
                    where,
                    normalized="return",
                    optimize=optimize,
                    **contract_opts,
                )
                # store for efficient calls with multiply loop sets
                info["expecs"][loop, where] = expec_loop, norm_loop

            expecs.append(expec_loop)
            norms.append(norm_loop)
            counts.append(C)

        return _combine_expansion_expectations(
            expecs, counts, norms, combine=combine, normalized=normalized
        )

    def compute_local_expectation_sloop_expand(
        self,
        terms,
        sloops=None,
        gauges=None,
        *,
        num_joins=1,
        grow_from="all",
        strict_size=False,
        intersect=False,
        autocomplete=True,
        autoreduce=True,
        combine="prod",
        normalized=True,
        info=None,
        return_all=False,
        optimize="auto",
        executor=None,
        progbar=False,
        **contract_opts,
    ):
        info = info if info is not None else {}

        if (sloops is None) or isinstance(sloops, int):
            sloops = tuple(
                self.gen_sloops(
                    sloops,
                    num_joins=num_joins,
                    intersect=intersect,
                )
            )

        return _compute_expecs_maybe_in_parallel(
            fn=_tn_local_expectation_sloop_expand,
            tn=self,
            terms=terms,
            sloops=sloops,
            gauges=gauges,
            grow_from=grow_from,
            strict_size=strict_size,
            autocomplete=autocomplete,
            autoreduce=autoreduce,
            combine=combine,
            normalized=normalized,
            info=info,
            return_all=return_all,
            optimize=optimize,
            executor=executor,
            progbar=progbar,
            **contract_opts,
        )

    def local_expectation_gloop_expand(
        self,
        G,
        where,
        gloops=None,
        gauges=None,
        combine="prod",
        normalized=True,
        autocomplete=True,
        autoreduce=True,
        grow_from="all",
        num_joins=1,
        strict_size=False,
        optimize="auto",
        info=None,
        tn_cache_maxsize=512,
        progbar=False,
        **contract_opts,
    ):
        """Compute the expectation of operator ``G`` at site(s) ``where`` by
        expanding the expectation in terms of generalized loops of tensors.

        Parameters
        ----------
        G : array_like
            The operator to compute the expectation of.
        where : node or sequence[node]
            The sites to compute the expectation at.
        gloops : None, int, or sequence[sequence[node]], optional
            The generalized loops to use. If an integer, generate loops up to
            and including this size. If ``None`` the maximum loop size is set
            as the smallest non-trivial loop found. If an explicit set of
            loops is given, only these loops are considered.
        gauges : dict[str, array_like], optional
            The store of gauge bonds, the keys being indices and the values
            being the vectors. Only bonds present in this dictionary will be
            gauged.
        combine : {"prod", "sum"}, optional
            How to combine the local expectations of each cluster.
        normalized : bool, "prod", "local" or "separate", "global", optional
            Whether and how to normalize the result.

            - True: uses "prod" if combine="prod", "local" if combine="sum".
            - "prod": shorthand for `combine="prod"` and `normalized=True`.
            - "local": each cluster within a term is normalized separately.
            - "separate": the expectation and normalization are computed
              separately and then combined for each term.

        autocomplete : bool, optional
            Whether to automatically complete the local region graph by adding
            all required intersecting regions automatically.
        autoreduce : bool, optional
            Whether to reduce any clusters with dangling bonds (which can
            still be generated as intersecting regions of loops) to generalized
            loops with dangling bonds removed. Should only be used at a BP
            fixed point.
        grow_from : {"all", "any"}, optional
            If auto generating loops, whether generate only those which contain
            *all* sites in the term, or just *any*. The loops generated by
            "any" are a superset of those generated by "all", giving a more
            accuracte estimate.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer to use.
        info : dict, optional
            A dictionary to store intermediate results in to avoid recomputing
            them. This is useful when computing various expectations with
            different sets of loops. This should only be reused when both the
            tensor network and gauges remain the same.
        tn_cache_maxsize : int, optional
            The maximum size of the cache for the locally gauged tensor
            networks used in computing the local expectations.
        contract_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract`.

        Returns
        -------
        expec : scalar
        """
        from quimb.tensor.belief_propagation import gen_region_counts

        info = info if info is not None else {}
        info.setdefault("tns", LRU(tn_cache_maxsize))
        info.setdefault("expecs", {})

        gloops = self.get_local_gloops(
            where=where,
            gloops=gloops,
            grow_from=grow_from,
            num_joins=num_joins,
            strict_size=strict_size,
            info=info,
        )

        if autoreduce:
            try:
                neighbors = info["neighbors"]
            except KeyError:
                neighbors = info["neighbors"] = self.get_site_neighbor_map()
        else:
            neighbors = None

        expecs = []
        norms = []
        counts = []

        region_counts = gen_region_counts(gloops, autocomplete=autocomplete)
        if progbar:
            region_counts = Progbar(region_counts)

        for r, C in region_counts:
            if autoreduce:
                # check if we can map cluster to a smaller generalized loop
                # this is only valid at a BP fixed point
                r = gloop_remove_dangling(r, neighbors, where)

            try:
                # have already computed this term in a different expectation
                # e.g. with different set of gloops
                expec_r, norm_r = info["expecs"][r, where]
            except KeyError:
                # get the gauged cluster tn
                try:
                    tnr = info["tns"][r]
                except KeyError:
                    tags = tuple(map(self.site_tag, r))
                    # take copy as inserting gauges
                    tnr = self.select_any(tags, virtual=False)
                    tnr.gauge_simple_insert(gauges)
                    info["tns"][r] = tnr

                # compute the expectation with exact contraction
                expec_r, norm_r = tnr.local_expectation_exact(
                    G,
                    where,
                    normalized="return",
                    optimize=optimize,
                    **contract_opts,
                )
                # store for efficient calls with multiple cluster sets
                info["expecs"][r, where] = expec_r, norm_r

            expecs.append(expec_r)
            norms.append(norm_r)
            counts.append(C)

        return _combine_expansion_expectations(
            expecs, counts, norms, combine=combine, normalized=normalized
        )

    def norm_gloop_expand(
        self,
        gloops=None,
        autocomplete=False,
        autoreduce=True,
        gauges=None,
        optimize="auto",
        progbar=False,
        **contract_opts,
    ):
        """Compute the norm of this tensor network by expanding it in terms of
        generalized loops of tensors.
        """
        from quimb.tensor.belief_propagation import (
            combine_local_contractions,
            gen_region_counts,
        )

        if isinstance(gloops, int):
            max_size = gloops
            gloops = None
        else:
            max_size = None

        if gloops is None:
            gloops = tuple(self.gen_gloops_sites(max_size=max_size))
        else:
            gloops = tuple(gloops)

        psi = self.copy()

        # make all tree like norms 1.0 -> region intersections
        # which are tree like can thus be ignored
        nfactor = psi.normalize_simple(gauges)

        if autoreduce:
            neighbors = self.get_site_neighbor_map()
        else:
            neighbors = None

        region_counts = gen_region_counts(
            itertools.chain(gloops, ((site,) for site in psi.sites)),
            autocomplete=autocomplete,
        )

        if progbar:
            region_counts = Progbar(region_counts)

        zvals = []
        for region, C in region_counts:
            if autoreduce:
                # check if we can map cluster to a smaller generalized loop
                region = gloop_remove_dangling(region, neighbors)

                if not region:
                    # region is tree like -> contributes 1.0
                    continue

            tags = tuple(map(psi.site_tag, region))
            kr = psi.select(tags, which="any", virtual=False)
            kr.gauge_simple_insert(gauges)

            lni = (kr.H | kr).contract(optimize=optimize, **contract_opts)
            zvals.append((lni, C))

        return (combine_local_contractions(zvals) * nfactor) ** 0.5

    def compute_local_expectation_gloop_expand(
        self,
        terms,
        gloops=None,
        *,
        gauges=None,
        combine="prod",
        normalized=True,
        autocomplete=True,
        autoreduce=True,
        grow_from="all",
        strict_size=False,
        optimize="auto",
        info=None,
        return_all=False,
        executor=None,
        progbar=False,
        **contract_opts,
    ):
        """Contract many local expectations using generalized loop expansion.

        Parameters
        ----------
        terms : dict[node or tuple[node], array_like]
            The terms to compute the expectation of, with keys being the
            site(s) and values being the local operators as plain arrays.
        gloops : None, int, or sequence[sequence[node]], optional
            The generalized loops to use. If an integer, generate loops up to
            and including this size. If ``None`` the maximum loop size is set
            as the smallest non-trivial loop found. If an explicit set of
            loops is given, only these loops are considered.
        gauges : dict[str, array_like], optional
            The store of gauge bonds, the keys being indices and the values
            being the vectors. Only bonds present in this dictionary will be
            used.
        combine : {"prod", "sum"}, optional
            How to combine the local expectations of each cluster.
        normalized : bool, "prod", "local" or "separate", "global", optional
            Whether and how to normalize the result.

            - True: uses "prod" if combine="prod", "local" if combine="sum".
            - "prod": shorthand for `combine="prod"` and `normalized=True`.
            - "local": each cluster within a term is normalized separately.
            - "separate": the expectation and normalization are computed
              separately and then combined for each term.
            - "global": a single normalization factor is computed and used for
              all terms.

        autocomplete : bool, optional
            Whether to automatically complete the local region graph by adding
            all required intersecting regions automatically.
        autoreduce : bool, optional
            Whether to reduce any clusters with dangling bonds (which can
            still be generated as intersecting regions of loops) to generalized
            loops with dangling bonds removed. Should only be used at a BP
            fixed point.
        grow_from : {"all", "any"}, optional
            If auto generating loops, whether generate only those which contain
            *all* sites in the term, or just *any*. The loops generated by
            "any" are a superset of those generated by "all", giving a more
            accuracte estimate.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer to use.
        info : dict, optional
            A dictionary to store intermediate results in to avoid recomputing
            them. This is useful when computing various expectations with
            different sets of loops. This should only be reused when both the
            tensor network and gauges remain the same.
        return_all : bool, optional
            Whether to return all results, or just the summed expectation.
        executor : Executor, optional
            If supplied compute the terms in parallel using this executor.
        progbar : bool, optional
            Whether to show a progress bar.
        contract_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract`.

        Returns
        -------
        expecs : float or dict[node or tuple[node], float]
            If ``return_all==False``, return the summed expectation value of
            the given terms. Otherwise, return a dictionary mapping each term's
            location to the expectation value.
        """
        if info is None:
            info = {}

        if normalized == "global":
            nfactor = self.norm_gloop_expand(
                gloops=gloops,
                autocomplete=autocomplete,
                autoreduce=autoreduce,
                gauges=gauges,
                optimize=optimize,
                **contract_opts,
            )
            tn = self / nfactor
            normalized = False
        else:
            tn = self

        return _compute_expecs_maybe_in_parallel(
            fn=_tn_local_expectation_gloop_expand,
            tn=tn,
            terms=terms,
            gloops=gloops,
            combine=combine,
            normalized=normalized,
            gauges=gauges,
            autocomplete=autocomplete,
            autoreduce=autoreduce,
            grow_from=grow_from,
            strict_size=strict_size,
            optimize=optimize,
            info=info,
            return_all=return_all,
            executor=executor,
            progbar=progbar,
            **contract_opts,
        )

    def partial_trace(
        self,
        keep,
        max_bond,
        optimize,
        flatten=True,
        reduce=False,
        normalized=True,
        symmetrized="auto",
        rehearse=False,
        method="contract_compressed",
        **contract_compressed_opts,
    ):
        """Partially trace this tensor network state, keeping only the sites in
        ``keep``, using compressed contraction.

        Parameters
        ----------
        keep : iterable of hashable
            The sites to keep.
        max_bond : int
            The maximum bond dimensions to use while compressed contracting.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer to use, should specifically generate
            contractions paths designed for compressed contraction.
        flatten : {False, True, 'all'}, optional
            Whether to force 'flattening' (contracting all physical indices) of
            the tensor network before  contraction, whilst this makes the TN
            generally more complex to contract, the accuracy is usually
            improved. If ``'all'`` also flatten the tensors in ``keep``.
        reduce : bool, optional
            Whether to first 'pull' the physical indices off their respective
            tensors using QR reduction. Experimental.
        normalized : bool, optional
            Whether to normalize the reduced density matrix at the end.
        symmetrized : {'auto', True, False}, optional
            Whether to symmetrize the reduced density matrix at the end. This
            should be unecessary if ``flatten`` is set to ``True``.
        rehearse : {False, 'tn', 'tree', True}, optional
            Whether to perform the computation or not::

                - False: perform the computation.
                - 'tn': return the tensor network without running the path
                  optimizer.
                - 'tree': run the path optimizer and return the
                  ``cotengra.ContractonTree``..
                - True: run the path optimizer and return the ``PathInfo``.

        contract_compressed_opts : dict, optional
            Additional keyword arguments to pass to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract_compressed`.

        Returns
        -------
        rho : array_like
            The reduce density matrix of sites in ``keep``.
        """
        if symmetrized == "auto":
            symmetrized = not flatten

        # form the partial trace
        k_inds = tuple(map(self.site_ind, keep))

        k = self.copy()
        if reduce:
            k.reduce_inds_onto_bond(*k_inds, tags="__BOND__", drop_tags=True)

        # b = k.conj().reindex_(dict(zip(k_inds, b_inds)))
        # tn = (b | k)
        bra_ind_id = "_bra{}"
        b_inds = tuple(map(bra_ind_id.format, keep))
        tn = k.make_reduced_density_matrix(keep, bra_ind_id=bra_ind_id)
        output_inds = k_inds + b_inds

        if flatten:
            for site in self.gen_site_coos():
                if (site not in keep) or (flatten == "all"):
                    # check if site exists still to permit e.g. local methods
                    # to use this same logic
                    tag = tn.site_tag(site)
                    if tag in tn.tag_map:
                        tn ^= tag

        tn.fuse_multibonds_()

        if method == "contract_compressed":
            if reduce:
                output_inds = None
                tn, tn_reduced = tn.partition("__BOND__", inplace=True)

            if rehearse:
                return _handle_rehearse(
                    rehearse, tn, optimize, output_inds=output_inds
                )

            t_rho = tn.contract_compressed(
                optimize,
                max_bond=max_bond,
                output_inds=output_inds,
                **contract_compressed_opts,
            )

            if reduce:
                t_rho |= tn_reduced

            rho = t_rho.to_dense(k_inds, b_inds)

        elif method == "contract_around":
            tn.contract_around_(
                tuple(map(self.site_tag, keep)),
                "any",
                max_bond=max_bond,
                **contract_compressed_opts,
            )

            if rehearse:
                return _handle_rehearse(
                    rehearse, tn, optimize, output_inds=output_inds
                )

            rho = tn.to_dense(
                k_inds,
                b_inds,
                optimize=optimize,
            )

        else:
            raise ValueError(f"Unknown method: {method}.")

        if symmetrized:
            rho = (rho + dag(rho)) / 2

        if normalized:
            rho = rho / do("trace", rho)

        return rho

    def local_expectation(
        self,
        G,
        where,
        max_bond,
        optimize,
        flatten=True,
        normalized=True,
        symmetrized="auto",
        reduce=False,
        rehearse=False,
        **contract_compressed_opts,
    ):
        """Compute the local expectation of operator ``G`` at site(s) ``where``
        by approximately contracting the full overlap tensor network.

        Parameters
        ----------
        G : array_like
            The local operator to compute the expectation of.
        where : node or sequence of nodes
            The sites to compute the expectation for.
        max_bond : int
            The maximum bond dimensions to use while compressed contracting.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer to use, should specifically generate
            contractions paths designed for compressed contraction.
        method : {'rho', 'rho-reduced'}, optional
            The method to use to compute the expectation value.
        flatten : bool, optional
            Whether to force 'flattening' (contracting all physical indices) of
            the tensor network before  contraction, whilst this makes the TN
            generally more complex to contract, the accuracy is usually much
            improved.
        normalized : bool, optional
            If computing via `partial_trace`, whether to normalize the reduced
            density matrix at the end.
        symmetrized : {'auto', True, False}, optional
            If computing via `partial_trace`, whether to symmetrize the reduced
            density matrix at the end. This should be unecessary if ``flatten``
            is set to ``True``.
        rehearse : {False, 'tn', 'tree', True}, optional
            Whether to perform the computation or not::

                - False: perform the computation.
                - 'tn': return the tensor network without running the path
                  optimizer.
                - 'tree': run the path optimizer and return the
                  ``cotengra.ContractonTree``..
                - True: run the path optimizer and return the ``PathInfo``.

        contract_compressed_opts : dict, optional
            Additional keyword arguments to pass to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract_compressed`.

        Returns
        -------
        expec : float
        """
        rho = self.partial_trace(
            keep=where,
            max_bond=max_bond,
            optimize=optimize,
            flatten=flatten,
            reduce=reduce,
            normalized=normalized,
            symmetrized=symmetrized,
            rehearse=rehearse,
            **contract_compressed_opts,
        )
        if rehearse:
            return rho

        return do("tensordot", rho, G, axes=((0, 1), (1, 0)))

    def compute_local_expectation(
        self,
        terms,
        max_bond,
        optimize,
        *,
        flatten=True,
        normalized=True,
        symmetrized="auto",
        reduce=False,
        return_all=False,
        rehearse=False,
        executor=None,
        progbar=False,
        **contract_compressed_opts,
    ):
        """Compute the local expectations of many local operators, by
        approximately contracting the full overlap tensor network.

        Parameters
        ----------
        terms : dict[node or (node, node), array_like]
            The terms to compute the expectation of, with keys being the sites
            and values being the local operators.
        max_bond : int
            The maximum bond dimension to use during contraction.
        optimize : str or PathOptimizer
            The compressed contraction path optimizer to use.
        method : {'rho', 'rho-reduced'}, optional
            The method to use to compute the expectation value.

                - 'rho': compute the expectation value via the reduced density
                  matrix.
                - 'rho-reduced': compute the expectation value via the reduced
                  density matrix, having reduced the physical indices onto the
                  bonds first.

        flatten : bool, optional
            Whether to force 'flattening' (contracting all physical indices) of
            the tensor network before  contraction, whilst this makes the TN
            generally more complex to contract, the accuracy can often be much
            improved.
        normalized : bool, optional
            Whether to locally normalize the result.
        symmetrized : {'auto', True, False}, optional
            Whether to symmetrize the reduced density matrix at the end. This
            should be unecessary if ``flatten`` is set to ``True``.
        return_all : bool, optional
            Whether to return all results, or just the summed expectation. If
            ``rehease is not False``, this is ignored and a dict is always
            returned.
        rehearse : {False, 'tn', 'tree', True}, optional
            Whether to perform the computations or not::

                - False: perform the computation.
                - 'tn': return the tensor networks of each local expectation,
                  without running the path optimizer.
                - 'tree': run the path optimizer and return the
                  ``cotengra.ContractonTree`` for each local expectation.
                - True: run the path optimizer and return the ``PathInfo`` for
                  each local expectation.

        executor : Executor, optional
            If supplied compute the terms in parallel using this executor.
        progbar : bool, optional
            Whether to show a progress bar.
        contract_compressed_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract_compressed`.

        Returns
        -------
        expecs : float or dict[node or (node, node), float]
            If ``return_all==False``, return the summed expectation value of
            the given terms. Otherwise, return a dictionary mapping each term's
            location to the expectation value.
        """
        return _compute_expecs_maybe_in_parallel(
            fn=_tn_local_expectation,
            tn=self,
            terms=terms,
            return_all=return_all,
            executor=executor,
            progbar=progbar,
            max_bond=max_bond,
            optimize=optimize,
            normalized=normalized,
            symmetrized=symmetrized,
            reduce=reduce,
            flatten=flatten,
            rehearse=rehearse,
            **contract_compressed_opts,
        )

    compute_local_expectation_rehearse = functools.partialmethod(
        compute_local_expectation, rehearse=True
    )

    compute_local_expectation_tn = functools.partialmethod(
        compute_local_expectation, rehearse="tn"
    )

    def sample_configuration_cluster(
        self,
        gauges=None,
        max_distance=0,
        fillin=0,
        max_iterations=100,
        tol=5e-6,
        optimize="auto-hq",
        seed=None,
    ):
        """Sample a configuration for this state using the simple update or
        cluster style environement approximation. The algorithms proceeds as
        a decimation:

        1. Compute every remaining site's local density matrix.
        2. The site with the largest bias is sampled.
        3. The site is projected into this sampled local config.
        4. The state is regauged given the projection.
        5. Repeat until all sites are sampled.

        Parameters
        ----------
        gauges : dict[str, array_like], optional
            The store of gauge bonds, the keys being indices and the values
            being the vectors. Only bonds present in this dictionary will be
            gauged.
        max_distance : int, optional
            The maximum distance to consider when computing the local density
            matrix for each site. Zero meaning on the site itself, 1 meaning
            the site and its immediate neighbors, etc.
        fillin : bool or int, optional
            When selecting the local tensors, whether and how many times to
            'fill-in' corner tensors attached multiple times to the local
            region. On a lattice this fills in the corners. See
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.select_local`.
        max_iterations : int, optional
            The maximum number of iterations to perform when gauging the state.
        tol : float, optional
            The tolerance to converge to when gauging the state.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer to use.
        seed : None, int or np.random.Generator, optional
            A random seed or random number generator to use.

        Returns
        -------
        config : dict[Node, int]
            The sampled configuration.
        omega : float
            The probability of the sampled configuration in terms of the
            approximate distribution induced by the cluster scheme.
        """
        import numpy as np

        rng = np.random.default_rng(seed)

        psi = self.copy()
        gauges = gauges.copy() if gauges is not None else {}

        # do an initial equilibration of the bond gauges
        psi.gauge_all_simple_(
            max_iterations=max_iterations,
            tol=tol,
            gauges=gauges,
        )

        config = {}
        omega = 1.0
        remaining = set(psi.sites)

        while remaining:
            probs = {}

            for i in remaining:
                # get the approx local density matrix for each remaining site
                rhoi = psi.partial_trace_cluster(
                    where=(i,),
                    gauges=gauges,
                    max_distance=max_distance,
                    fillin=fillin,
                    optimize=optimize,
                    normalized=False,
                )
                # extract the diagonal
                rhoi = do("to_numpy", rhoi)
                pi = np.diag(rhoi).real
                # normalize and store
                probs[i] = pi / pi.sum()

            # choose site with maximum bias to sample enxt
            i = max(probs, key=lambda i: np.max(probs[i]))
            remaining.remove(i)
            # get the local prob distribution at that site
            pmax = probs[i]
            # sample a local config according to the distribution
            xi = rng.choice(pmax.size, p=pmax)
            config[i] = xi
            # track local probability
            omega *= pmax[xi]
            # project the site into the sampled state
            psi.isel_({psi.site_ind(i): xi})
            # get the local tid, to efficiently restart the gauging
            tids = psi._get_tids_from_tags(i)
            # regauge, given the projected site
            psi.gauge_all_simple_(
                max_iterations=max_iterations,
                tol=tol,
                gauges=gauges,
                touched_tids=tids,
            )

        return config, omega


class TensorNetworkGenOperator(TensorNetworkGen):
    """A tensor network which notionally has a single tensor and two outer
    indices per 'site', though these could be labelled arbitrarily and could
    also be linked in an arbitrary geometry by bonds. By convention, if
    converted to a dense matrix, the 'upper' indices would be on the left and
    the 'lower' indices on the right.
    """

    _EXTRA_PROPS = (
        "_sites",
        "_site_tag_id",
        "_upper_ind_id",
        "_lower_ind_id",
    )

    @property
    def upper_ind_id(self):
        """The string specifier for the upper phyiscal indices."""
        return self._upper_ind_id

    def upper_ind(self, site):
        """Get the upper physical index name of ``site``."""
        return self.upper_ind_id.format(site)

    def reindex_upper_sites(self, new_id, where=None, inplace=False):
        """Modify the upper site indices for all or some tensors in this
        operator tensor network (without changing the ``upper_ind_id``).

        Parameters
        ----------
        new_id : str
            A string with a format placeholder to accept a site, e.g. "up{}".
        where : None or sequence
            Which sites to update the index labels on. If ``None`` (default)
            all sites.
        inplace : bool
            Whether to reindex in place.
        """
        if where is None:
            where = self.gen_sites_present()

        return self.reindex(
            {self.upper_ind(x): new_id.format(x) for x in where},
            inplace=inplace,
        )

    reindex_upper_sites_ = functools.partialmethod(
        reindex_upper_sites, inplace=True
    )

    @upper_ind_id.setter
    def upper_ind_id(self, new_id):
        if new_id == self._lower_ind_id:
            raise ValueError(
                "Setting the same upper and upper index ids will"
                " make the two ambiguous."
            )

        if self._upper_ind_id != new_id:
            self.reindex_upper_sites_(new_id)
            self._upper_ind_id = new_id
            self._upper_inds = None

    @property
    def upper_inds(self):
        """Return a tuple of all upper indices."""
        if getattr(self, "_upper_inds", None) is None:
            self._upper_inds = tuple(map(self.upper_ind, self.gen_site_coos()))
        return self._upper_inds

    @property
    def upper_inds_present(self):
        """Return a tuple of all upper indices still present in the tensor
        network.
        """
        return tuple(map(self.upper_ind, self.gen_sites_present()))

    @property
    def lower_ind_id(self):
        """The string specifier for the lower phyiscal indices."""
        return self._lower_ind_id

    def lower_ind(self, site):
        """Get the lower physical index name of ``site``."""
        return self.lower_ind_id.format(site)

    def reindex_lower_sites(self, new_id, where=None, inplace=False):
        """Modify the lower site indices for all or some tensors in this
        operator tensor network (without changing the ``lower_ind_id``).

        Parameters
        ----------
        new_id : str
            A string with a format placeholder to accept a site, e.g. "up{}".
        where : None or sequence
            Which sites to update the index labels on. If ``None`` (default)
            all sites.
        inplace : bool
            Whether to reindex in place.
        """
        if where is None:
            where = self.gen_sites_present()

        return self.reindex(
            {self.lower_ind(x): new_id.format(x) for x in where},
            inplace=inplace,
        )

    reindex_lower_sites_ = functools.partialmethod(
        reindex_lower_sites, inplace=True
    )

    @lower_ind_id.setter
    def lower_ind_id(self, new_id):
        if new_id == self._upper_ind_id:
            raise ValueError(
                "Setting the same upper and lower index ids will"
                " make the two ambiguous."
            )

        if self._lower_ind_id != new_id:
            self.reindex_lower_sites_(new_id)
            self._lower_ind_id = new_id
            self._lower_inds = None

    @property
    def lower_inds(self):
        """Return a tuple of all lower indices."""
        if getattr(self, "_lower_inds", None) is None:
            self._lower_inds = tuple(map(self.lower_ind, self.gen_site_coos()))
        return self._lower_inds

    @property
    def lower_inds_present(self):
        """Return a tuple of all lower indices still present in the tensor
        network.
        """
        return tuple(map(self.lower_ind, self.gen_sites_present()))

    def to_dense(self, *inds_seq, to_qarray=False, **contract_opts):
        """Contract this tensor network 'operator' into a dense array.

        Parameters
        ----------
        inds_seq : sequence of sequences of str
            How to group the site indices into the dense array. By default,
            use a single group ordered like ``sites``, but only containing
            those sites which are still present.
        to_qarray : bool
            Whether to turn the dense array into a ``qarray``, if the backend
            would otherwise be ``'numpy'``.
        contract_opts
            Options to pass to
            :meth:`~quimb.tensor.tensor_core.TensorNewtork.contract`.

        Returns
        -------
        array
        """
        if not inds_seq:
            inds_seq = (self.upper_inds_present, self.lower_inds_present)

        return TensorNetwork.to_dense(
            self, *inds_seq, to_qarray=to_qarray, **contract_opts
        )

    to_qarray = functools.partialmethod(to_dense, to_qarray=True)

    def phys_dim(self, site=None, which="upper"):
        """Get the physical dimension of ``site``."""
        if site is None:
            site = next(iter(self.gen_sites_present()))

        if which == "upper":
            return self[site].ind_size(self.upper_ind(site))

        if which == "lower":
            return self[site].ind_size(self.lower_ind(site))

    gate_upper = functools.partialmethod(tensor_network_ag_gate, which="upper")
    gate_upper_ = functools.partialmethod(
        tensor_network_ag_gate,
        which="upper",
        inplace=True,
    )
    gate_lower = functools.partialmethod(tensor_network_ag_gate, which="lower")
    gate_lower_ = functools.partialmethod(
        tensor_network_ag_gate,
        which="lower",
        inplace=True,
    )

    def gate_upper_with_op_lazy(
        self,
        A,
        transpose=False,
        inplace=False,
    ):
        r"""Act lazily with the operator tensor network ``A``, which should
        have matching structure, on this operator tensor network (``B``), like
        ``A @ B``. The returned tensor network will have the same structure as
        this one, but with the operator gated in lazily, i.e. uncontracted.

        .. math::

            B \rightarrow A B

        or (if ``transpose=True``):

        .. math::

            B \rightarrow A^T B

        Parameters
        ----------
        A : TensorNetworkGenOperator
            The operator tensor network to gate with, or apply to this tensor
            network.
        transpose : bool, optional
            Whether to contract the lower or upper indices of ``A`` with the
            upper indices of ``B``. If ``False`` (the default), the lower
            indices of ``A`` will be contracted with the upper indices of
            ``B``, if ``True`` the upper indices of ``A`` will be
            contracted with the upper indices of ``B``, which is like applying
            the transpose first.
        inplace : bool, optional
            Whether to perform the gate operation inplace on this tensor
            network.

        Returns
        -------
        TensorNetworkGenOperator
        """
        return tensor_network_apply_op_op(
            A=A,
            B=self,
            which_A="upper" if transpose else "lower",
            which_B="upper",
            contract=False,
            inplace=inplace,
        )

    gate_upper_with_op_lazy_ = functools.partialmethod(
        gate_upper_with_op_lazy, inplace=True
    )

    def gate_lower_with_op_lazy(
        self,
        A,
        transpose=False,
        inplace=False,
    ):
        r"""Act lazily 'from the right' with the operator tensor network ``A``,
        which should have matching structure, on this operator tensor network
        (``B``), like ``B @ A``. The returned tensor network will have the same
        structure as this one, but with the operator gated in lazily, i.e.
        uncontracted.

        .. math::

            B \rightarrow B A

        or (if ``transpose=True``):

        .. math::

            B \rightarrow B A^T

        Parameters
        ----------
        A : TensorNetworkGenOperator
            The operator tensor network to gate with, or apply to this tensor
            network.
        transpose : bool, optional
            Whether to contract the upper or lower indices of ``A`` with the
            lower indices of this TN. If ``False`` (the default), the upper
            indices of ``A`` will be contracted with the lower indices of
            ``B``, if ``True`` the lower indices of ``A`` will be contracted
            with the lower indices of this TN, which is like applying the
            transpose first.
        inplace : bool, optional
            Whether to perform the gate operation inplace on this tensor
            network.

        Returns
        -------
        TensorNetworkGenOperator
        """
        return tensor_network_apply_op_op(
            B=self,
            A=A,
            which_A="lower" if transpose else "upper",
            which_B="lower",
            contract=False,
            inplace=inplace,
        )

    gate_lower_with_op_lazy_ = functools.partialmethod(
        gate_lower_with_op_lazy, inplace=True
    )

    def gate_sandwich_with_op_lazy(
        self,
        A,
        inplace=False,
    ):
        r"""Act lazily with the operator tensor network ``A``, which should
        have matching structure, on this operator tensor network (``B``), like
        :math:`B \rightarrow A B A^\dagger`. The returned tensor network will
        have the same structure as this one, but with the operator gated in
        lazily, i.e. uncontracted.

        Parameters
        ----------
        A : TensorNetworkGenOperator
            The operator tensor network to gate with, or apply to this tensor
            network.
        inplace : bool, optional
            Whether to perform the gate operation inplace on this tensor

        Returns
        -------
        TensorNetworkGenOperator
        """
        B = self if inplace else self.copy()
        B.gate_upper_with_op_lazy_(A)
        B.gate_lower_with_op_lazy_(A.conj(), transpose=True)
        return B

    gate_sandwich_with_op_lazy_ = functools.partialmethod(
        gate_sandwich_with_op_lazy, inplace=True
    )


def _handle_rehearse(
    rehearse,
    tn: TensorNetwork,
    optimize,
    **kwargs,
):
    if rehearse is True:
        tree = tn.contraction_tree(optimize, **kwargs)
        return {
            "tn": tn,
            "tree": tree,
            "W": tree.contraction_width(log=2),
            "C": tree.contraction_cost(log=10),
        }
    if rehearse == "tn":
        return tn
    if rehearse == "tree":
        return tn.contraction_tree(optimize, **kwargs)
    if rehearse == "info":
        return tn.contraction_info(optimize, **kwargs)


def _compute_expecs_maybe_in_parallel(
    fn,
    tn,
    terms,
    return_all=False,
    executor=None,
    progbar=False,
    **kwargs,
):
    """Unified helper function for the various methods that compute many
    expectations, possibly in parallel, possibly with a progress bar.
    """
    if not isinstance(terms, dict):
        terms = dict(terms.items())

    if executor is None:
        results = (fn(tn, G, where, **kwargs) for where, G in terms.items())
    else:
        if hasattr(executor, "scatter"):
            tn = executor.scatter(tn)

        futures = [
            executor.submit(fn, tn, G, where, **kwargs)
            for where, G in terms.items()
        ]
        results = (future.result() for future in futures)

    if progbar:
        results = Progbar(results, total=len(terms))

    expecs = dict(zip(terms.keys(), results))

    if return_all or kwargs.get("rehearse", False):
        return expecs

    return functools.reduce(add, expecs.values())


def _tn_local_expectation(tn: TensorNetworkGenVector, *args, **kwargs):
    """Define as function for pickleability."""
    return tn.local_expectation(*args, **kwargs)


def _tn_local_expectation_cluster(tn: TensorNetworkGenVector, *args, **kwargs):
    """Define as function for pickleability."""
    return tn.local_expectation_cluster(*args, **kwargs)


def _tn_local_expectation_exact(tn: TensorNetworkGenVector, *args, **kwargs):
    """Define as function for pickleability."""
    return tn.local_expectation_exact(*args, **kwargs)


def _tn_local_expectation_sloop_expand(
    tn: TensorNetworkGenVector, *args, **kwargs
):
    """Define as function for pickleability."""
    return tn.local_expectation_sloop_expand(*args, **kwargs)


def _tn_local_expectation_gloop_expand(
    tn: TensorNetworkGenVector, *args, **kwargs
):
    """Define as function for pickleability."""
    return tn.local_expectation_gloop_expand(*args, **kwargs)


def _combine_expansion_expectations(
    expecs, counts, norms, combine="prod", normalized=True
):
    # now we combine the local expectations and normalizations
    if normalized == "prod":
        # allow this as legacy shorthand for combine="prod"
        combine = "prod"
        normalized = True

    if combine == "prod":
        from .belief_propagation import combine_local_contractions

        vals = [(e, C) for e, C in zip(expecs, counts)]
        if normalized:
            # local and separate normalization are equivalent for prod
            vals.extend((n, -C) for n, C in zip(norms, counts))

        expec = combine_local_contractions(vals)

    elif combine == "sum":
        if normalized is True:
            # default to "local" normalization
            normalized = "local"

        if normalized == "local":
            expec = sum(C * e / n for C, e, n in zip(counts, expecs, norms))
        elif normalized == "separate":
            expec = sum(C * e for C, e in zip(counts, expecs))
            norm = sum(C * n for C, n in zip(counts, norms))
            expec = expec / norm
        elif not normalized:
            expec = sum(C * e for C, e in zip(counts, expecs))
        else:
            raise ValueError(
                f"normalized={normalized} should be one of "
                "{True, False, 'local', 'separate'} "
                " when combine='sum'."
            )
    else:
        raise ValueError(f"combine={combine} should be 'prod' or 'sum'.")

    return expec
