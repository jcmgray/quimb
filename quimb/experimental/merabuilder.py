"""Tools for constructing MERA for arbitrary geometry.

TODO::

    - [ ] 2D, 3D MERA classes
    - [ ] general strategies for arbitrary geometries
    - [ ] layer_tag? and hanling of other attributes
    - [ ] handle dangling case
    - [ ] invariant generators?

 DONE::

    - [x] layer_gate methods for arbitrary geometry
    - [x] 1D: generic way to handle finite and open boundary conditions
    - [x] hook into other arbgeom infrastructure for computing rdms etc

"""
import itertools
import functools
from quimb.tensor.tensor_core import (
    Tensor,
    IsoTensor,
    oset_union,
    prod,
)
from quimb.tensor.tensor_arbgeom import (
    TensorNetworkGenVector,
    oset,
    tags_to_oset,
    rand_uuid,
    _compute_expecs_maybe_in_parallel,
    _tn_local_expectation,
)
from quimb.tensor.tensor_1d import TensorNetwork1DVector
from quimb.utils import partition


class TensorNetworkGenIso(TensorNetworkGenVector):
    """A class for building generic 'isometric' or MERA like tensor network
    states with arbitrary geometry. After supplying the underyling `sites` of
    the problem - which can be an arbitrary sequence of hashable objects - one
    places either unitaries, isometries or tree tensors layered above groups of
    sites. The isometric and tree tensors effectively coarse grain blocks into
    a single new site, and the unitaries generally 'disentangle' between
    blocks.
    """

    _EXTRA_PROPS = (
        "_site_tag_id",
        "_sites",
        "_site_ind_id",
        "_layer_ind_id",
    )

    @classmethod
    def empty(
        cls,
        sites,
        phys_dim=2,
        site_tag_id="I{}",
        site_ind_id="k{}",
        layer_ind_id="l{}",
    ):
        new = object.__new__(cls)
        new.phys_dim = phys_dim
        new._sites = tuple(sites)
        new._site_tag_id = site_tag_id
        new._site_ind_id = site_ind_id
        new._layer_ind_id = layer_ind_id

        new._open_upper_sites = oset(new._sites)
        new._open_lower_sites = oset(new._sites)

        super().__init__(new, ())
        return new

    @property
    def layer_ind_id(self):
        return self._layer_ind_id

    def layer_ind(self, site):
        return self._layer_ind_id.format(site)

    def layer_gate_raw(
        self,
        G,
        where,
        iso=True,
        new_sites=None,
        tags=None,
        all_site_tags=None,
    ):
        """Build out this MERA by placing either a new unitary, isometry or
        tree tensor, given by ``G``, at the sites given by ``where``. This
        handles propagating the lightcone of tags and marking the correct
        indices of the ``IsoTensor`` as ``left_inds``.

        Parameters
        ----------
        G : array_like
            The raw array to place at the sites. Its shape determines whether
            it is a unitary or isometry/tree. It should have ``k + len(where)``
            dimensions. For a unitary ``k == len(where)``. If it is an
            isometry/tree, ``k`` will generally be ``1``, or ``0`` to 'cap' the
            MERA. The rightmost indices are those attached to the current open
            layer indices.
        where : sequence of hashable
            The sites to layer the tensor above.
        iso : bool, optional
            Whether to declare the tensor as an unitary/isometry by marking
            the left indices. If ``iso = False`` (a 'tree' tensor) then one
            should have ``k <= 1``. Once you have such a 'tree' tensor you
            cannot place isometries or unitaries above it. It will also have
            the lightcone tags of every site. Technically one could place
            'PEPS' style tensor with ``iso = False`` and ``k > 1`` but some
            methods might break.
        new_sites : sequence of hashable, optional
            Which sites to make new open sites. If not given, defaults to the
            first ``k`` sites in ``where``.
        tags : sequence of str, optional
            Custom tags to add to the new tensor, in addition to the
            automatically generated site tags.
        all_site_tags : sequence of str, optional
            For performance, supply all site tags to avoid recomputing them.
        """
        if all_site_tags is None:
            all_site_tags = oset(map(self.site_tag, self.gen_site_coos()))

        # work out 'lower' tensor indices
        nbelow = len(where)
        below_ix = []
        reindex_map = {}
        tags = tags_to_oset(tags)
        for site in where:
            if site in self._open_lower_sites:
                # this is the first tensor placed above site
                below_ix.append(self.site_ind(site))
                self._open_lower_sites.remove(site)
                tags.add(self.site_tag(site))
            else:
                # tensor is being placed above existing tensor
                current_lower_ix = self.layer_ind(site)
                new_lower_ix = rand_uuid()
                reindex_map[current_lower_ix] = new_lower_ix
                below_ix.append(new_lower_ix)

        # work out 'upper' tensor indices
        nabove = len(G.shape) - nbelow
        if new_sites is None:
            new_sites = where[:nabove]
        above_ix = [self.layer_ind(site) for site in new_sites]

        for site in where[nabove:]:
            # if tensor is not a unitary then some upper sites are removed
            self._open_upper_sites.remove(site)

        # want to propagate just site tags from tensors below
        old_tags = oset_union(t.tags for t in self._inds_get(*reindex_map))

        if iso and "TREE" in old_tags:
            raise ValueError(
                "You can't place isometric tensors above tree tensors."
            )

        if not iso:
            # tensor is in lightcone of all sites
            tags |= all_site_tags
            tags.add("TREE")
            left_inds = None
            if nabove > 1:
                import warnings

                warnings.warn(
                    "You are placing a tensor which is neither "
                    "isometric/unitary or a tree. Some methods might break."
                )
        else:
            # just want site tags present on tensors below
            tags |= old_tags & all_site_tags
            if nbelow == nabove:
                tags.add("UNI")
            else:
                tags.add("ISO")
            if nabove == 0:
                tags.add("CAP")
            left_inds = below_ix

        # rewire and add tensor
        self.reindex_(reindex_map)
        self |= IsoTensor(
            data=G,
            inds=below_ix + above_ix,
            left_inds=left_inds,
            tags=tags,
        )

    def layer_gate_fill_fn(
        self,
        fill_fn,
        operation,
        where,
        max_bond,
        new_sites=None,
        tags=None,
        all_site_tags=None,
    ):
        """Build out this MERA by placing either a new unitary, isometry or
        tree tensor at sites ``where``, generating the data array using
        ``fill_fn`` and maximum bond dimension ``max_bond``.

        Parameters
        ----------
        fill_fn : callable
            A function with signature ``fill_fn(shape) -> array_like``.
        operation : {"iso", "uni", "cap", "tree", "treecap"}
            The type of tensor to place.
        where : sequence of hashable
            The sites to layer the tensor above.
        max_bond : int
            The maximum bond dimension of the tensor. This only applies for
            isometries and trees and when the product of the lower dimensions
            is greater than ``max_bond``.
        new_sites : sequence of hashable, optional
            Which sites to make new open sites. If not given, defaults to the
            first ``k`` sites in ``where``.
        tags : sequence of str, optional
            Custom tags to add to the new tensor, in addition to the
            automatically generated site tags.
        all_site_tags : sequence of str, optional
            For performance, supply all site tags to avoid recomputing them.

        See Also
        --------
        layer_gate_raw
        """
        shape = []
        for site in where:
            if site in self._open_lower_sites:
                shape.append(self.phys_dim)
            else:
                shape.append(self.ind_size(self.layer_ind(site)))

        if operation == "uni":
            # unitary, map is shape preserving
            shape = (*shape, *shape)
        elif operation in ("iso", "tree"):
            current_size = prod(shape)
            new_size = min(current_size, max_bond)
            shape = (*shape, new_size)
        elif operation in ("cap", "treecap"):
            # no new sitess
            shape = tuple(shape)
        else:
            raise ValueError(
                f"Unknown operation: '{operation}'. Should be one of: "
                "'uni', 'iso', 'cap', 'tree', or 'treecap'."
            )

        G = fill_fn(shape)
        self.layer_gate_raw(
            G,
            where,
            new_sites=new_sites,
            tags=tags,
            all_site_tags=all_site_tags,
            iso="tree" not in operation,
        )

    def partial_trace(
        self,
        keep,
        optimize="auto-hq",
        rehearse=False,
        preserve_tensor=False,
        **contract_opts,
    ):
        """Partial trace out all sites except those in ``keep``, making use of
        the lightcone structure of the MERA.

        Parameters
        ----------
        keep : sequence of hashable
            The sites to keep.
        optimize : str or PathOptimzer, optional
            The contraction ordering strategy to use.
        rehearse : {False, "tn", "tree"}, optional
            Whether to rehearse the contraction rather than actually performing
            it. If:

            - ``False``: perform the contraction and return the reduced density
              matrix,
            - "tn": just the lightcone tensor network is returned,
            - "tree": just the contraction tree that will be used is returned.

        contract_opts
            Additional options to pass to
            :func:`~quimb.tensor.tensor_core.tensor_contract`.

        Returns
        -------
        array_like
            The reduced density matrix on sites ``keep``.
        """
        tags = tuple(map(self.site_tag, keep))
        k = self.select_any(tags, virtual=False)

        kix = tuple(map(self.site_ind, keep))
        bix = tuple(f"b{site}" for site in keep)
        b = k.reindex(dict(zip(kix, bix))).conj_()
        tn = b | k

        if rehearse == "tn":
            return tn

        contract_opts["optimize"] = optimize

        if rehearse == "tree":
            return tn.contraction_tree(output_inds=bix + kix, **contract_opts)

        t = tn.contract(output_inds=bix + kix, **contract_opts)
        if preserve_tensor:
            return t

        return t.to_dense(bix, kix)

    def local_expectation(
        self,
        G,
        where,
        optimize="auto-hq",
        rehearse=False,
        **contract_opts,
    ):
        """Compute the expectation value of a local operator ``G`` at sites
        ``where``. This is done by contracting the lightcone tensor network
        to form the reduced density matrix, before taking the trace with
        ``G``.

        Parameters
        ----------
        G : array_like
            The local operator to compute the expectation value of.
        where : sequence of hashable
            The sites to compute the expectation value at.
        optimize : str or PathOptimzer, optional
            The contraction ordering strategy to use.
        rehearse : {False, "tn", "tree"}, optional
            Whether to rehearse the contraction rather than actually performing
            it. See :meth:`~quimb.tensor.mera.MERA.partial_trace` for details.
        contract_opts
            Additional options to pass to
            :func:`~quimb.tensor.tensor_core.tensor_contract`.

        Returns
        -------
        float
            The expectation value of ``G`` at sites ``where``.

        See Also
        --------
        partial_trace
        """
        t_rho = self.partial_trace(
            keep=where,
            optimize=optimize,
            rehearse=rehearse,
            preserve_tensor=True,
            **contract_opts,
        )

        if rehearse:
            # returned t_rho is the tree or whole TN etc.
            return t_rho

        # make sure G is compatible shape (could be supplied in 'matrix' form)
        if G.shape != t_rho.shape:
            # n.b. both are hermitian so no 'transpose' needed here
            G = G.reshape(t_rho.shape)

        # make gate tensor
        nphys = t_rho.ndim // 2
        bix, kix = t_rho.inds[:nphys], t_rho.inds[nphys:]
        t_G = Tensor(G, inds=kix + bix)

        return t_rho @ t_G

    def compute_local_expectation(
        self,
        terms,
        optimize="auto-hq",
        return_all=False,
        rehearse=False,
        executor=None,
        progbar=False,
        **contract_opts,
    ):
        """Compute the expectation value of a collection of local operators
        ``terms`` at sites ``where``. This is done by contracting the lightcone
        tensor network to form the reduced density matrices, before taking the
        trace with each ``G`` in ``terms``.

        Parameters
        ----------
        terms : dict[tuple[hashable], array_like]
            The local operators to compute the expectation value of, keyed by
            the sites they act on.
        optimize : str or PathOptimzer, optional
            The contraction ordering strategy to use.
        return_all : bool, optional
            Whether to return all the expectation values, or just the sum.
        rehearse : {False, "tn", "tree"}, optional
            Whether to rehearse the contraction rather than actually performing
            it. See :meth:`~quimb.tensor.mera.MERA.partial_trace` for details.
        executor : Executor, optional
            The executor to use for parallelism.
        progbar : bool, optional
            Whether to show a progress bar.
        contract_opts
            Additional options to pass to
            :func:`~quimb.tensor.tensor_core.tensor_contract`.
        """
        return _compute_expecs_maybe_in_parallel(
            fn=_tn_local_expectation,
            tn=self,
            terms=terms,
            return_all=return_all,
            executor=executor,
            progbar=progbar,
            optimize=optimize,
            rehearse=rehearse,
            **contract_opts,
        )

    def expand_bond_dimension(
        self,
        new_bond_dim,
        rand_strength=0.0,
        inds_to_expand=None,
        inplace=False,
    ):
        """Expand the maxmimum bond dimension of this isometric tensor network
        to ``new_bond_dim``. Unlike
        :meth:`~quimb.tensor.tensor_core.TensorNetwork.expand_bond_dimension`
        this proceeds from the physical indices upwards, and only increases a
        bonds size if ``new_bond_dim`` is larger than product of the lower
        indices dimensions.

        Parameters
        ----------
        new_bond_dim : int
            The new maximum bond dimension to expand to.
        rand_strength : float, optional
            The strength of random noise to add to the new array entries,
            if any.
        inds_to_expand : sequence of str, optional
            The indices to expand, if not all.
        inplace : bool, optional
            Whether to expand this tensor network in place, or return a new
            one.

        Returns
        -------
        TensorNetworkGenIso
        """
        if inds_to_expand is not None:
            return super().expand_bond_dimension(
                new_bond_dim=new_bond_dim,
                rand_strength=rand_strength,
                inds_to_expand=inds_to_expand,
                inplace=inplace,
            )

        tn = self if inplace else self.copy()

        tids_done = oset()
        inds_done = oset(tn.site_inds)
        tids_todo = tn._get_tids_from_inds(inds_done, "any")

        # XXX: switch this logic to get_tree_span('CAP')? to
        # ensure topologically sorted order?
        while tids_todo:
            tid = tids_todo.popleft()
            t = tn.tensor_map[tid]

            if t.left_inds is not None:
                below_inds = oset(t.left_inds)
                above_inds = oset(t.inds) - below_inds
            else:
                below_inds, above_inds = oset(), oset()
                for ix in t.inds:
                    (below_inds if ix in inds_done else above_inds).add(ix)

            if len(above_inds) == 0:
                # top piece
                continue

            elif len(above_inds) == 1:
                # isometry, bond can expand
                (ix,) = above_inds
                cur_sz = t.ind_size(ix)
                rem_inds_sz = t.size // cur_sz

                # don't expand beyond product of lower index sizes
                new_sz = min(rem_inds_sz, new_bond_dim)
                if new_sz > cur_sz:
                    tn.expand_bond_dimension_(
                        new_bond_dim=new_sz,
                        rand_strength=rand_strength,
                        inds_to_expand=ix,
                    )

            elif len(above_inds) == len(below_inds):
                # unitary gate, maintain bond sizes
                for bix, aix in zip(below_inds, above_inds):
                    tn.expand_bond_dimension_(
                        new_bond_dim=t.ind_size(bix),
                        rand_strength=rand_strength,
                        inds_to_expand=aix,
                    )

            else:
                raise NotImplementedError

            tids_done.add(tid)
            inds_done.update(above_inds)
            for tid_above in tn._get_tids_from_inds(above_inds, "any"):
                if tid_above not in tids_done:
                    tids_todo.add(tid_above)

        return tn

    expand_bond_dimension_ = functools.partialmethod(
        expand_bond_dimension, inplace=True
    )


def calc_1d_unis_isos(sites, block_size, cyclic, group_from_right):
    """Given ``sites``, assumed to be in a 1D order, though not neccessarily
    contiguous, calculate unitary and isometry groupings::

               │         │ <- new grouped site
        ┐   ┌─────┐   ┌─────┐   ┌
        │   │ ISO │   │ ISO │   │
        ┘   └─────┘   └─────┘   └
        │   │..│..│   │..│..│   │
        ┌───┐  │  ┌───┐  │  ┌───┐
        │UNI│  │  │UNI│  │  │UNI│
        └───┘  │  └───┘  │  └───┘
        │   │ ... │   │ ... │   │
            ^^^^^^^ <- isometry groupings of size, block_size
        ^^^^^     ^^^^^ <- unitary groupings of size 2

    Parameters
    ----------
    sites : sequence of hashable
        The sites to apply a layer to.
    block_size : int
        How many sites to group together per isometry block. Note that
        currently the unitaries will only ever act on blocks of size 2 across
        isometry block boundaries.
    cyclic : bool
        Whether to apply disentangler / unitaries across the boundary. The
        isometries will never be applied across the boundary, but since they
        always form a tree such a bipartition is natural.
    group_from_right : bool
        Wether to group the sites starting from the left or right. This only
        matters if ``block_size`` does not divide the number of sites.
        Alternating between left and right more evenly tiles the unitaries and
        isometries, especially at lower layers.

    Returns
    -------
    unis : list[tuple]
        The unitary groupings.
    isos : list[tuple]
        The isometry groupings.
    """
    sites = tuple(sites)
    nsites = len(sites)

    # track this so we know neighboring sites
    ranks = {s: i for i, s in enumerate(sites)}

    # first we linearly partition the sites to form the isometry groups
    size = block_size * (nsites // block_size)
    if group_from_right:
        grouped = sites[-size:]
    else:
        grouped = sites[:size]
    isos = list(partition(block_size, grouped))

    # then we disentangle at the edges of the
    # isometries to form the unitaries
    unis = set()
    for iso in isos:
        # n.b. only when the groups are not adjacent (e.g. because the number
        # of sites doesn't divide) will there be a left (right) disentangler
        # which is not also a right (left) disentangler. In that case a site
        # can see 2 unitaries rather than the usual 1 unitary and 1 isometry:
        #            │
        #     ┐   ┌─────┐   ┌
        #     │   │ ISO │   │
        #     ┘   └─────┘   └
        #     │   │..│..│   │
        #     ┌───┐  │  ┌───┐
        #     │UNI│  │  │UNI│
        #     └───┘  │  └───┘
        #     │   │     │   │
        #     sl  si    sf  sr

        # attempt left disentangle
        si = iso[0]
        ri = ranks[si]
        if cyclic or ri > 0:
            sl = sites[ri - 1]
            unis.add((sl, si))

        # attempt right disentangle
        sf = iso[-1]
        rf = ranks[sf]
        if cyclic or rf < nsites - 1:
            sr = sites[(rf + 1) % nsites]
            unis.add((sf, sr))

    return sorted(unis), isos


class MERA(TensorNetwork1DVector, TensorNetworkGenIso):
    """Replacement class for ``MERA`` which uses the new infrastructure and
    thus has methods like ``compute_local_expectation``.
    """

    _EXTRA_PROPS = tuple(
        sorted(
            set(TensorNetwork1DVector._EXTRA_PROPS)
            | set(TensorNetworkGenIso._EXTRA_PROPS)
        )
    )
    _CONTRACT_STRUCTURED = False

    def __init__(self, *args, **kwargs):
        self._num_layers = None
        super().__init__(*args, **kwargs)

    @classmethod
    def from_fill_fn(
        cls,
        fill_fn,
        L,
        D,
        phys_dim=2,
        block_size=2,
        cyclic=True,
        uni_fill_fn=None,
        iso_fill_fn=None,
        cap_fill_fn=None,
        **kwargs,
    ):
        """Create a 1D MERA using ``fill_fn(shape) -> array_like`` to fill the
        tensors.

        Parameters
        ----------
        fill_fn : callable
            A function which takes a shape and returns an array_like of that
            shape. You can override this specfically for the unitaries,
            isometries and cap tensors using the kwargs ``uni_fill_fn``,
            ``iso_fill_fn`` and ``cap_fill_fn``.
        L : int
            The number of sites.
        D : int
            The maximum bond dimension.
        phys_dim : int, optional
            The dimension of the physical indices.
        block_size : int, optional
            The size of the isometry blocks. Binary MERA is the default,
            ternary MERA is ``block_size=3``.
        cyclic : bool, optional
            Whether to apply disentangler / unitaries across the boundary. The
            isometries will never be applied across the boundary, but since
            they always form a tree such a bipartition is natural.
        uni_fill_fn : callable, optional
            A function which takes a shape and returns an array_like of that
            shape. This is used to fill the unitary tensors. If ``None`` then
            ``fill_fn`` is used.
        iso_fill_fn : callable, optional
            A function which takes a shape and returns an array_like of that
            shape. This is used to fill the isometry tensors. If ``None`` then
            ``fill_fn`` is used.
        cap_fill_fn : callable, optional
            A function which takes a shape and returns an array_like of that
            shape. This is used to fill the cap tensors. If ``None`` then
            ``fill_fn`` is used.
        kwargs
            Supplied to ``TensorNetworkGenIso.__init__``.
        """
        mera = cls.empty(sites=range(L), phys_dim=phys_dim, **kwargs)
        mera._L = L

        if uni_fill_fn is None:
            uni_fill_fn = fill_fn
        if iso_fill_fn is None:
            iso_fill_fn = fill_fn
        if cap_fill_fn is None:
            cap_fill_fn = iso_fill_fn

        for lyr in itertools.count():
            remaining_sites = sorted(mera._open_upper_sites)

            if len(remaining_sites) <= block_size + 1:
                # can terminate with a 'cap'
                mera.layer_gate_fill_fn(
                    cap_fill_fn,
                    "cap",
                    remaining_sites,
                    D,
                    tags=f"LAYER{lyr}",
                )
                break

            # else add a disentangling and grouping layer
            uni_groups, iso_groups = calc_1d_unis_isos(
                remaining_sites,
                block_size,
                cyclic,
                group_from_right=lyr % 2,
            )
            for uni_sites in uni_groups:
                mera.layer_gate_fill_fn(
                    uni_fill_fn,
                    "uni",
                    uni_sites,
                    D,
                    tags=f"LAYER{lyr}",
                )
            for iso_sites in iso_groups:
                mera.layer_gate_fill_fn(
                    iso_fill_fn,
                    "iso",
                    iso_sites,
                    D,
                    tags=f"LAYER{lyr}",
                )

        mera._num_layers = lyr + 1

        return mera

    @classmethod
    def rand(
        cls,
        L,
        D,
        seed=None,
        block_size=2,
        phys_dim=2,
        cyclic=True,
        isometrize_method="svd",
        **kwargs,
    ):
        """Return a random (optionally isometrized) MERA.

        Parameters
        ----------
        L : int
            The number of sites.
        D : int
            The maximum bond dimension.
        seed : int, optional
            A random seed.
        block_size : int, optional
            The size of the isometry blocks. Binary MERA is the default,
            ternary MERA is ``block_size=3``.
        phys_dim : int, optional
            The dimension of the physical indices.
        cyclic : bool, optional
            Whether to apply disentangler / unitaries across the boundary. The
            isometries will never be applied across the boundary, but since
            they always form a tree such a bipartition is natural.
        isometrize_method : str or None, optional
            If given, the method to use to isometrize the MERA. If ``None``
            then the MERA is not isometrized.
        """
        import numpy as np

        rng = np.random.default_rng(seed)
        mera = cls.from_fill_fn(
            lambda shape: rng.normal(size=shape),
            L,
            D,
            block_size,
            phys_dim,
            cyclic,
            **kwargs,
        )
        if isometrize_method is not None:
            mera.isometrize_(isometrize_method)
        return mera

    @property
    def num_layers(self):
        return self._num_layers


def TTN_randtree_rand(
    sites,
    D,
    phys_dim=2,
    group_size=2,
    iso=False,
    seed=None,
    **kwargs,
):
    """Return a randomly constructed tree tensor network.

    Parameters
    ----------
    sites : list of hashable
        The sites of the tensor network.
    D : int
        The maximum bond dimension.
    phys_dim : int, optional
        The dimension of the physical indices.
    group_size : int, optional
        How many sites to group together in each tensor.
    iso : bool, optional
        Whether to build the tree with an isometric flow towards the top.
    seed : int, optional
        A random seed.
    kwargs
        Supplied to ``TensorNetworkGenIso.empty``.

    Returns
    -------
    ttn : TensorNetworkGenIso
        The tree tensor network.
    """
    import numpy as np

    sites = list(sites)

    rng = np.random.default_rng(seed)
    tn = TensorNetworkGenIso.empty(sites, phys_dim=phys_dim, **kwargs)

    while len(sites) > group_size + 1:
        # randomly pick two sites to merge
        merge = sorted(
            sites.pop(rng.integers(len(sites))) for _ in range(group_size)
        )
        tn.layer_gate_fill_fn(
            lambda shape: rng.normal(size=shape),
            "iso" if iso else "tree",
            merge,
            max_bond=D,
        )
        sites.append(merge[0])

    tn.layer_gate_fill_fn(
        lambda shape: rng.normal(size=shape),
        "cap" if iso else "treecap",
        sites,
        max_bond=D,
    )

    return tn
