"""Classes and algorithms related to 1D tensor networks.
"""

import re
import operator
import functools
from math import log2
from numbers import Integral

import scipy.sparse.linalg as spla
from autoray import do, dag, reshape, conj, get_dtype_name, transpose

from ..utils import (
    check_opt, print_multi_line, ensure_dict, partition_all, deprecated
)
import quimb as qu
from .tensor_core import (
    Tensor,
    TensorNetwork,
    rand_uuid,
    bonds,
    bonds_size,
    oset,
    tags_to_oset,
    get_tags,
    PTensor,
)
from .tensor_arbgeom import tensor_network_align, tensor_network_apply_op_vec
from ..linalg.base_linalg import norm_trace_dense
from . import array_ops as ops


align_TN_1D = deprecated(
    tensor_network_align, 'align_TN_1D', 'tensor_network_align')


def expec_TN_1D(*tns, compress=None, eps=1e-15):
    """Compute the expectation of several 1D TNs, using transfer matrix
    compression if any are periodic.

    Parameters
    ----------
    tns : sequence of TensorNetwork1D
        The MPS and MPO to find expectation of. Should start and begin with
        an MPS e.g. ``(MPS, MPO, ...,  MPS)``.
    compress : {None, False, True}, optional
        Whether to perform transfer matrix compression on cyclic systems. If
        set to ``None`` (the default), decide heuristically.
    eps : float, optional
        The accuracy of the transfer matrix compression.

    Returns
    -------
    x : float
        The expectation value.
    """
    expec_tn = functools.reduce(operator.or_, tensor_network_align(*tns))

    # if OBC or <= 0.0 specified use exact contraction
    cyclic = any(tn.cyclic for tn in tns)
    if not cyclic:
        compress = False

    n = expec_tn.L
    isflat = all(isinstance(tn, TensorNetwork1DFlat) for tn in tns)

    # work out whether to compress, could definitely be improved ...
    if compress is None and isflat:
        # compression only worth it for long, high bond dimension TNs.
        total_bd = qu.prod(tn.bond_size(0, 1) for tn in tns)
        compress = (n >= 100) and (total_bd >= 1000)

    if compress:
        expec_tn.replace_section_with_svd(1, n, eps=eps, inplace=True)
        return expec_tn ^ all

    return expec_tn ^ ...


_VALID_GATE_CONTRACT = {False, True, 'swap+split',
                        'split-gate', 'swap-split-gate', 'auto-split-gate'}
_VALID_GATE_PROPAGATE = {'sites', 'register', False, True}
_TWO_BODY_ONLY = _VALID_GATE_CONTRACT - {True, False}


def maybe_factor_gate_into_tensor(G, dp, ng, where):
    # allow gate to be a matrix as long as it factorizes into tensor
    shape_matches_2d = (ops.ndim(G) == 2) and (G.shape[1] == dp ** ng)
    shape_matches_nd = all(d == dp for d in G.shape)

    if shape_matches_2d:
        G = ops.asarray(G)
        if ng >= 2:
            G = reshape(G, [dp] * 2 * ng)

    elif not shape_matches_nd:
        raise ValueError(
            f"Gate with shape {G.shape} doesn't match sites {where}.")

    return G


def gate_TN_1D(tn, G, where, contract=False, tags=None,
               propagate_tags='sites', inplace=False,
               cur_orthog=None, **compress_opts):
    r"""Act with the gate ``g`` on sites ``where``, maintaining the outer
    indices of the 1D tensor netowork::


        contract=False       contract=True
            . .                    . .             <- where
        o-o-o-o-o-o-o        o-o-o-GGG-o-o-o
        | | | | | | |        | | | / \ | | |
            GGG
            | |


        contract='split-gate'        contract='swap-split-gate'
              . .                          . .                      <- where
          o-o-o-o-o-o-o                o-o-o-o-o-o-o
          | | | | | | |                | | | | | | |
              G~G                          G~G
              | |                          \ /
                                            X
                                           / \

        contract='swap+split'
                . .            <- where
          o-o-o-G=G-o-o-o
          | | | | | | | |

    Note that the sites in ``where`` do not have to be contiguous. By default,
    site tags will be propagated to the gate tensors, identifying a
    'light cone'.

    Parameters
    ----------
    tn : TensorNetwork1DVector
        The 1D vector-like tensor network, for example, and MPS.
    G : array
        A square array to act with on sites ``where``. It should have twice the
        number of dimensions as the number of sites. The second half of these
        will be contracted with the MPS, and the first half indexed with the
        correct ``site_ind_id``. Sites are read left to right from the shape.
        A two-dimensional array is permissible if each dimension factorizes
        correctly.
    where : int or sequence of int
        Where the gate should act.
    contract : {False, 'split-gate', 'swap-split-gate',
                'auto-split-gate', True, 'swap+split'}, optional
        Whether to contract the gate into the 1D tensor network. If,

            - False: leave the gate uncontracted, the default
            - 'split-gate': like False, but split the gate if it is two-site.
            - 'swap-split-gate': like 'split-gate', but decompose the gate as
              if a swap had first been applied
            - 'auto-split-gate': automatically select between the above three
              options, based on the rank of the gate.
            - True: contract the gate into the tensor network, if the gate acts
              on more than one site, this will produce an ever larger tensor.
            - 'swap+split': Swap sites until they are adjacent, then contract
              the gate and split the resulting tensor, then swap the sites back
              to their original position. In this way an MPS structure can be
              explicitly maintained at the cost of rising bond-dimension.

    tags : str or sequence of str, optional
        Tag the new gate tensor with these tags.
    propagate_tags : {'sites', 'register', False, True}, optional
        Add any tags from the sites to the new gate tensor (only matters if
        ``contract=False`` else tags are merged anyway):

            - If ``'sites'``, then only propagate tags matching e.g. 'I{}' and
              ignore all others. I.e. just propagate the lightcone.
            - If ``'register'``, then only propagate tags matching the sites of
              where this gate was actually applied. I.e. ignore the lightcone,
              just keep track of which 'registers' the gate was applied to.
            - If ``False``, propagate nothing.
            - If ``True``, propagate all tags.

    inplace, bool, optional
        Perform the gate in place.
    compress_opts
        Supplied to :meth:`~quimb.tensor.tensor_core.Tensor.split`
        if ``contract='swap+split'`` or
        :meth:`~quimb.tensor.tensor_1d.MatrixProductState.gate_with_auto_swap`
        if ``contract='swap+split'``.

    Returns
    -------
    TensorNetwork1DVector

    See Also
    --------
    MatrixProductState.gate_split

    Examples
    --------
    >>> p = MPS_rand_state(3, 7)
    >>> p.gate_(spin_operator('X'), where=1, tags=['GX'])
    >>> p
    <MatrixProductState(tensors=4, L=3, max_bond=7)>

    >>> p.outer_inds()
    ('k0', 'k1', 'k2')
    """
    check_opt('contract', contract, _VALID_GATE_CONTRACT)
    check_opt('propagate_tags', propagate_tags, _VALID_GATE_PROPAGATE)

    psi = tn if inplace else tn.copy()

    if isinstance(where, Integral):
        where = (where,)
    ng = len(where)  # number of sites the gate acts on

    dp = psi.phys_dim(where[0])
    tags = tags_to_oset(tags)

    if (ng > 2) and contract in _TWO_BODY_ONLY:
        raise ValueError(f"Can't use `contract='{contract}'` for >2 sites.")

    G = maybe_factor_gate_into_tensor(G, dp, ng, where)

    if contract == 'swap+split' and ng > 1:
        psi.gate_with_auto_swap(G, where, cur_orthog=cur_orthog,
                                inplace=True, **compress_opts)
        return psi

    bnds = [rand_uuid() for _ in range(ng)]
    site_ix = [psi.site_ind(i) for i in where]
    gate_ix = site_ix + bnds

    psi.reindex_(dict(zip(site_ix, bnds)))

    # get the sites that used to have the physical indices
    site_tids = psi._get_tids_from_inds(bnds, which='any')

    # convert the gate into a tensor - check if it is parametrized
    if isinstance(G, ops.PArray):
        if (ng >= 2) and (contract is not False):
            raise ValueError(
                "For a parametrized gate acting on more than one site "
                "``contract`` must be false to preserve the array shape.")

        TG = PTensor.from_parray(G, gate_ix, tags=tags, left_inds=bnds)
    else:
        TG = Tensor(G, gate_ix, tags=tags, left_inds=bnds)

    # handle 'swap+split' only for ``ng == 1``
    if contract in (True, 'swap+split'):
        # pop the sites, contract, then re-add
        pts = [psi._pop_tensor(tid) for tid in site_tids]
        psi |= TG.contract(*pts)
        return psi

    # if not contracting the gate into the network, work out which tags to
    # 'propagate' forward from the tensors being acted on to the gate tensors
    if propagate_tags:
        if propagate_tags == 'register':
            old_tags = oset(map(psi.site_tag, where))
        else:
            old_tags = get_tags(psi.tensor_map[tid] for tid in site_tids)

        if propagate_tags == 'sites':
            # use regex to take tags only matching e.g. 'I0', 'I13'
            rex = re.compile(psi.site_tag_id.format(r"\d+"))
            old_tags = oset(filter(rex.match, old_tags))

        TG.modify(tags=TG.tags | old_tags)

    if ng == 1:
        psi |= TG
        return psi

    # check if we should split multi-site gates (which may result in an easier
    #     tensor network to contract if we use compression)
    if contract in ('split-gate', 'auto-split-gate'):
        #  | |       | |
        #  GGG  -->  G~G
        #  | |       | |
        ts_gate_norm = TG.split(TG.inds[::2], get='tensors', **compress_opts)

    # sometimes it is worth performing the decomposition *across* the gate,
    #     effectively introducing a SWAP
    if contract in ('swap-split-gate', 'auto-split-gate'):
        #            \ /
        #  | |        X
        #  GGG  -->  / \
        #  | |       G~G
        #            | |
        ts_gate_swap = TG.split(TG.inds[::3], get='tensors', **compress_opts)

    # like 'split-gate' but check the rank for swapped indices also, and if no
    #     rank reduction, simply don't swap
    if contract == 'auto-split-gate':
        #            | |      \ /
        #  | |       | |       X           | |
        #  GGG  -->  G~G  or  / \   or ... GGG
        #  | |       | |      G~G          | |
        #            | |      | |
        norm_rank = bonds_size(*ts_gate_norm)
        swap_rank = bonds_size(*ts_gate_swap)

        if swap_rank < norm_rank:
            contract = 'swap-split-gate'
        elif norm_rank < dp**ng:
            contract = 'split-gate'
        else:
            # else no rank reduction available - leave as ``contract=False``.
            contract = False

    if contract == 'swap-split-gate':
        ts_gate = ts_gate_swap
    elif contract == 'split-gate':
        ts_gate = ts_gate_norm
    else:
        ts_gate = (TG,)

    # if we are splitting the gate then only add site tags on the tensors
    # directly 'above' the site
    if contract in ('split-gate', 'swap-split-gate'):
        if propagate_tags == 'register':
            ts_gate[0].drop_tags(psi.site_tag(where[1]))
            ts_gate[1].drop_tags(psi.site_tag(where[0]))

    for t in ts_gate:
        psi |= t
    return psi


def superop_TN_1D(tn_super, tn_op,
                  upper_ind_id='k{}',
                  lower_ind_id='b{}',
                  so_outer_upper_ind_id=None,
                  so_inner_upper_ind_id=None,
                  so_inner_lower_ind_id=None,
                  so_outer_lower_ind_id=None):
    r"""Take a tensor network superoperator and act with it on a
    tensor network operator, maintaining the original upper and lower
    indices of the operator::


         outer_upper_ind_id                           upper_ind_id
            | | | ... |                               | | | ... |
            +----------+                              +----------+
            | tn_super +---+                          | tn_super +---+
            +----------+   |     upper_ind_id         +----------+   |
            | | | ... |    |      | | | ... |         | | | ... |    |
         inner_upper_ind_id|     +-----------+       +-----------+   |
                           |  +  |   tn_op   |   =   |   tn_op   |   |
         inner_lower_ind_id|     +-----------+       +-----------+   |
            | | | ... |    |      | | | ... |         | | | ... |    |
            +----------+   |      lower_ind_id        +----------+   |
            | tn_super +---+                          | tn_super +---+
            +----------+                              +----------+
            | | | ... | <--                           | | | ... |
         outer_lower_ind_id                           lower_ind_id


    Parameters
    ----------
    tn_super : TensorNetwork
        The superoperator in the form of a 1D-like tensor network.
    tn_op : TensorNetwork
        The operator to be acted on in the form of a 1D-like tensor network.
    upper_ind_id : str, optional
        Current id of the upper operator indices, e.g. usually ``'k{}'``.
    lower_ind_id : str, optional
        Current id of the lower operator indices, e.g. usually ``'b{}'``.
    so_outer_upper_ind_id : str, optional
        Current id of the superoperator's upper outer indices, these will be
        reindexed to form the new effective operators upper indices.
    so_inner_upper_ind_id : str, optional
        Current id of the superoperator's upper inner indices, these will be
        joined with those described by ``upper_ind_id``.
    so_inner_lower_ind_id : str, optional
        Current id of the superoperator's lower inner indices, these will be
        joined with those described by ``lower_ind_id``.
    so_outer_lower_ind_id : str, optional
        Current id of the superoperator's lower outer indices, these will be
        reindexed to form the new effective operators lower indices.

    Returns
    -------
    KAK : TensorNetwork
        The tensornetwork of the superoperator acting on the operator.
    """
    n = tn_op.L

    if so_outer_upper_ind_id is None:
        so_outer_upper_ind_id = getattr(tn_super, 'outer_upper_ind_id', 'kn{}')
    if so_inner_upper_ind_id is None:
        so_inner_upper_ind_id = getattr(tn_super, 'inner_upper_ind_id', 'k{}')
    if so_inner_lower_ind_id is None:
        so_inner_lower_ind_id = getattr(tn_super, 'inner_lower_ind_id', 'b{}')
    if so_outer_lower_ind_id is None:
        so_outer_lower_ind_id = getattr(tn_super, 'outer_lower_ind_id', 'bn{}')

    reindex_map = {}
    for i in range(n):
        upper_bnd = rand_uuid()
        lower_bnd = rand_uuid()
        reindex_map[upper_ind_id.format(i)] = upper_bnd
        reindex_map[lower_ind_id.format(i)] = lower_bnd
        reindex_map[so_inner_upper_ind_id.format(i)] = upper_bnd
        reindex_map[so_inner_lower_ind_id.format(i)] = lower_bnd
        reindex_map[so_outer_upper_ind_id.format(i)] = upper_ind_id.format(i)
        reindex_map[so_outer_lower_ind_id.format(i)] = lower_ind_id.format(i)

    return tn_super.reindex(reindex_map) & tn_op.reindex(reindex_map)


class TensorNetwork1D(TensorNetwork):
    """Base class for tensor networks with a one-dimensional structure.
    """

    _NDIMS = 1
    _EXTRA_PROPS = ('_site_tag_id', '_L')
    _CONTRACT_STRUCTURED = True

    def _compatible_1d(self, other):
        """Check whether ``self`` and ``other`` are compatible 2D tensor
        networks such that they can remain a 2D tensor network when combined.
        """
        return (
            isinstance(other, TensorNetwork1D) and
            all(getattr(self, e) == getattr(other, e)
                for e in TensorNetwork1D._EXTRA_PROPS)
        )

    def __and__(self, other):
        new = super().__and__(other)
        if self._compatible_1d(other):
            new.view_as_(TensorNetwork1D, like=self)
        return new

    def __or__(self, other):
        new = super().__or__(other)
        if self._compatible_1d(other):
            new.view_as_(TensorNetwork1D, like=self)
        return new

    @property
    def L(self):
        """The number of sites.
        """
        return self._L

    @property
    def nsites(self):
        """The number of sites.
        """
        return self._L

    def gen_site_coos(self):
        return tuple(i for i in range(self.L) if
                     self.site_tag(i) in self.tag_map)

    @property
    def site_tag_id(self):
        """The string specifier for tagging each site of this 1D TN.
        """
        return self._site_tag_id

    def site_tag(self, i):
        """The name of the tag specifiying the tensor at site ``i``.
        """
        if not isinstance(i, str):
            i = i % self.L
        return self.site_tag_id.format(i)

    def slice2sites(self, tag_slice):
        """Take a slice object, and work out its implied start, stop and step,
        taking into account cyclic boundary conditions.

        Examples
        --------
        Normal slicing:

            >>> p = MPS_rand_state(10, bond_dim=7)
            >>> p.slice2sites(slice(5))
            (0, 1, 2, 3, 4)

            >>> p.slice2sites(slice(4, 8))
            (4, 5, 6, 7)

        Slicing from end backwards:

            >>> p.slice2sites(slice(..., -3, -1))
            (9, 8)

        Slicing round the end:

            >>> p.slice2sites(slice(7, 12))
            (7, 8, 9, 0, 1)

            >>> p.slice2sites(slice(-3, 2))
            (7, 8, 9, 0, 1)

        If the start point is > end point (*before* modulo n), then step needs
        to be negative to return anything.
        """
        if tag_slice.start is None:
            start = 0
        elif tag_slice.start is ...:
            if tag_slice.step == -1:
                start = self.L - 1
            else:
                start = -1
        else:
            start = tag_slice.start

        if tag_slice.stop in (..., None):
            stop = self.L
        else:
            stop = tag_slice.stop

        step = 1 if tag_slice.step is None else tag_slice.step

        return tuple(s % self.L for s in range(start, stop, step))

    def maybe_convert_coo(self, x):
        """Check if ``x`` is an integer and convert to the
        corresponding site tag if so.
        """
        if isinstance(x, Integral):
            return (self.site_tag(x),)

        if isinstance(x, slice):
            return tuple(map(self.site_tag, self.slice2sites(x)))

        return x

    def _get_tids_from_tags(self, tags, which='all'):
        """This is the function that lets single integers be used for many
        'tag' based functions.
        """
        tags = self.maybe_convert_coo(tags)
        return super()._get_tids_from_tags(tags, which=which)

    def retag_sites(self, new_id, where=None, inplace=False):
        """Modify the site tags for all or some tensors in this 1D TN
        (without changing the ``site_tag_id``).
        """
        if where is None:
            where = self.gen_site_coos()

        return self.retag({self.site_tag(i): new_id.format(i) for i in where},
                          inplace=inplace)

    @site_tag_id.setter
    def site_tag_id(self, new_id):
        if self._site_tag_id != new_id:
            self.retag_sites(new_id, inplace=True)
            self._site_tag_id = new_id

    @property
    def site_tags(self):
        """An ordered tuple of the actual site tags.
        """
        return tuple(map(self.site_tag, self.gen_site_coos()))

    @property
    def sites(self):
        return tuple(self.gen_site_coos())

    @functools.wraps(tensor_network_align)
    def align(self, *args, inplace=False, **kwargs):
        return tensor_network_align(self, *args, inplace=inplace, **kwargs)

    align_ = functools.partialmethod(align, inplace=True)

    def contract_structured(
        self,
        tag_slice,
        structure_bsz=5,
        inplace=False,
        **opts
    ):
        """Perform a structured contraction, translating ``tag_slice`` from a
        ``slice`` or `...` to a cumulative sequence of tags.

        Parameters
        ----------
        tag_slice : slice or ...
            The range of sites, or `...` for all.
        inplace : bool, optional
            Whether to perform the contraction inplace.

        Returns
        -------
        TensorNetwork, Tensor or scalar
            The result of the contraction, still a ``TensorNetwork`` if the
            contraction was only partial.

        See Also
        --------
        contract, contract_tags, contract_cumulative
        """
        # check for all sites
        if tag_slice is ...:
            # else slice over all sites
            tag_slice = slice(0, self.L)

        # filter sites by the slice, but also which sites are present at all
        tags_seq = filter(self.tag_map.__contains__,
                          map(self.site_tag, self.slice2sites(tag_slice)))

        # partition sites into `structure_bsz` groups
        if structure_bsz > 1:
            tags_seq = partition_all(structure_bsz, tags_seq)

        # contract each block of sites cumulatively
        return self.contract_cumulative(tags_seq, inplace=inplace, **opts)

    def __repr__(self):
        """Insert length and max bond into standard print.
        """
        s = super().__repr__()
        extra = f', L={self.L}, max_bond={self.max_bond()}'
        s = f'{s[:-2]}{extra}{s[-2:]}'
        return s

    def __str__(self):
        """Insert length and max bond into standard print.
        """
        s = super().__str__()
        extra = f', L={self.L}, max_bond={self.max_bond()}'
        s = f'{s[:-1]}{extra}{s[-1:]}'
        return s


class TensorNetwork1DVector(TensorNetwork1D,
                            TensorNetwork):
    """1D Tensor network which overall is like a vector with a single type of
    site ind.
    """

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_site_ind_id',
        '_L',
    )

    def reindex_all(self, new_id, inplace=False):
        """Reindex all physical sites and change the ``site_ind_id``.
        """
        tn = self if inplace else self.copy()
        tn.site_ind_id = new_id
        return tn

    reindex_all_ = functools.partialmethod(reindex_all, inplace=True)

    def reindex_sites(self, new_id, where=None, inplace=False):
        """Update the physical site index labels to a new string specifier.
        Note that this doesn't change the stored id string with the TN.

        Parameters
        ----------
        new_id : str
            A string with a format placeholder to accept an int, e.g. "ket{}".
        where : None or slice
            Which sites to update the index labels on. If ``None`` (default)
            all sites.
        inplace : bool
            Whether to reindex in place.
        """
        if where is None:
            indices = self.gen_site_coos()
        elif isinstance(where, slice):
            indices = self.slice2sites(where)
        else:
            indices = where

        return self.reindex({self.site_ind(i): new_id.format(i)
                             for i in indices}, inplace=inplace)

    reindex_sites_ = functools.partialmethod(reindex_sites, inplace=True)

    def _get_site_ind_id(self):
        return self._site_ind_id

    def _set_site_ind_id(self, new_id):
        if self._site_ind_id != new_id:
            self.reindex_sites_(new_id)
            self._site_ind_id = new_id

    site_ind_id = property(_get_site_ind_id, _set_site_ind_id,
                           doc="The string specifier for the physical indices")

    def site_ind(self, i):
        """Get the physical index name of site ``i``.
        """
        if not isinstance(i, str):
            i = i % self.L
        return self.site_ind_id.format(i)

    @property
    def site_inds(self):
        """An ordered tuple of the actual physical indices.
        """
        return tuple(map(self.site_ind, self.gen_site_coos()))

    def to_dense(self, *inds_seq, **contract_opts):
        """Return the dense ket version of this 1D vector, i.e. a
        ``qarray`` with shape (-1, 1).
        """
        if not inds_seq:
            # just use list of site indices
            return do('reshape', TensorNetwork.to_dense(
                self, self.site_inds, **contract_opts
            ), (-1, 1))

        return TensorNetwork.to_dense(self, *inds_seq, **contract_opts)

    def phys_dim(self, i=None):
        if i is None:
            i = next(iter(self.gen_site_coos()))
        return self.ind_size(self.site_ind(i))

    @functools.wraps(gate_TN_1D)
    def gate(self, *args, inplace=False, **kwargs):
        return gate_TN_1D(self, *args, inplace=inplace, **kwargs)

    gate_ = functools.partialmethod(gate, inplace=True)

    @functools.wraps(expec_TN_1D)
    def expec(self, *args, **kwargs):
        return expec_TN_1D(self, *args, **kwargs)

    def correlation(self, A, i, j, B=None, **expec_opts):
        """Correlation of operator ``A`` between ``i`` and ``j``.

        Parameters
        ----------
        A : array
            The operator to act with, can be multi site.
        i : int or sequence of int
            The first site(s).
        j : int or sequence of int
            The second site(s).
        expec_opts
            Supplied to :func:`~quimb.tensor.tensor_1d.expec_TN_1D`.

        Returns
        -------
        C : float
            The correlation ``<A(i)> + <A(j)> - <A(ij)>``.

        Examples
        --------
        >>> ghz = (MPS_computational_state('0000') +
        ...        MPS_computational_state('1111')) / 2**0.5
        >>> ghz.correlation(pauli('Z'), 0, 1)
        1.0
        >>> ghz.correlation(pauli('Z'), 0, 1, B=pauli('X'))
        0.0
        """
        if B is None:
            B = A

        bra = self.H

        pA = self.gate(A, i, contract=True)
        cA = expec_TN_1D(bra, pA, **expec_opts)

        pB = self.gate(B, j, contract=True)
        cB = expec_TN_1D(bra, pB, **expec_opts)

        pAB = pA.gate_(B, j, contract=True)
        cAB = expec_TN_1D(bra, pAB, **expec_opts)

        return cAB - cA * cB


class TensorNetwork1DOperator(TensorNetwork1D,
                              TensorNetwork):

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_upper_ind_id',
        '_lower_ind_id',
        '_L',
    )

    def reindex_lower_sites(self, new_id, where=None, inplace=False):
        """Update the lower site index labels to a new string specifier.

        Parameters
        ----------
        new_id : str
            A string with a format placeholder to accept an int, e.g.
            ``"ket{}"``.
        where : None or slice
            Which sites to update the index labels on. If ``None`` (default)
            all sites.
        inplace : bool
            Whether to reindex in place.
        """
        if where is None:
            start = 0
            stop = self.L
        else:
            start = 0 if where.start is None else where.start
            stop = self.L if where.stop is ... else where.stop

        return self.reindex({self.lower_ind(i): new_id.format(i)
                             for i in range(start, stop)}, inplace=inplace)

    reindex_lower_sites_ = functools.partialmethod(
        reindex_lower_sites, inplace=True)

    def reindex_upper_sites(self, new_id, where=None, inplace=False):
        """Update the upper site index labels to a new string specifier.

        Parameters
        ----------
        new_id : str
            A string with a format placeholder to accept an int, e.g. "ket{}".
        where : None or slice
            Which sites to update the index labels on. If ``None`` (default)
            all sites.
        inplace : bool
            Whether to reindex in place.
        """
        if where is None:
            start = 0
            stop = self.L
        else:
            start = 0 if where.start is None else where.start
            stop = self.L if where.stop is ... else where.stop

        return self.reindex({self.upper_ind(i): new_id.format(i)
                             for i in range(start, stop)}, inplace=inplace)

    reindex_upper_sites_ = functools.partialmethod(
        reindex_upper_sites, inplace=True)

    def _get_lower_ind_id(self):
        return self._lower_ind_id

    def _set_lower_ind_id(self, new_id):
        if new_id == self._upper_ind_id:
            raise ValueError("Setting the same upper and lower index ids will"
                             " make the two ambiguous.")

        if self._lower_ind_id != new_id:
            self.reindex_lower_sites_(new_id)
            self._lower_ind_id = new_id

    lower_ind_id = property(
        _get_lower_ind_id, _set_lower_ind_id,
        doc="The string specifier for the lower phyiscal indices")

    def lower_ind(self, i):
        """The name of the lower ('ket') index at site ``i``.
        """
        return self.lower_ind_id.format(i)

    @property
    def lower_inds(self):
        """An ordered tuple of the actual lower physical indices.
        """
        return tuple(map(self.lower_ind, self.gen_site_coos()))

    def _get_upper_ind_id(self):
        return self._upper_ind_id

    def _set_upper_ind_id(self, new_id):
        if new_id == self._lower_ind_id:
            raise ValueError("Setting the same upper and lower index ids will"
                             " make the two ambiguous.")

        if self._upper_ind_id != new_id:
            self.reindex_upper_sites_(new_id)
            self._upper_ind_id = new_id

    upper_ind_id = property(_get_upper_ind_id, _set_upper_ind_id,
                            doc="The string specifier for the upper phyiscal "
                            "indices")

    def upper_ind(self, i):
        """The name of the upper ('bra') index at site ``i``.
        """
        return self.upper_ind_id.format(i)

    @property
    def upper_inds(self):
        """An ordered tuple of the actual upper physical indices.
        """
        return tuple(map(self.upper_ind, self.gen_site_coos()))

    def to_dense(self, *inds_seq, **contract_opts):
        """Return the dense matrix version of this 1D operator, i.e. a
        ``qarray`` with shape (d, d).
        """
        if not inds_seq:
            inds_seq = (self.upper_inds, self.lower_inds)

        return TensorNetwork.to_dense(self, *inds_seq, **contract_opts)

    def phys_dim(self, i=None, which='upper'):
        """Get a physical index size of this 1D operator.
        """
        if i is None:
            i = next(iter(self.gen_site_coos()))

        if which == 'upper':
            return self[i].ind_size(self.upper_ind(i))

        if which == 'lower':
            return self[i].ind_size(self.lower_ind(i))


def set_default_compress_mode(opts, cyclic=False):
    opts.setdefault('cutoff_mode', 'rel' if cyclic else 'rsum2')


class TensorNetwork1DFlat(TensorNetwork1D,
                          TensorNetwork):
    """1D Tensor network which has a flat structure.
    """

    _EXTRA_PROPS = ('_site_tag_id', '_L')

    def _left_decomp_site(self, i, bra=None, **split_opts):
        T1, T2 = self[i], self[i + 1]
        rix, lix = T1.filter_bonds(T2)

        set_default_compress_mode(split_opts, self.cyclic)
        Q, R = T1.split(lix, get='tensors', right_inds=rix, **split_opts)
        R = R @ T2

        Q.transpose_like_(T1)
        R.transpose_like_(T2)

        self[i].modify(data=Q.data)
        self[i + 1].modify(data=R.data)

        if bra is not None:
            bra[i].modify(data=Q.data.conj())
            bra[i + 1].modify(data=R.data.conj())

    def _right_decomp_site(self, i, bra=None, **split_opts):
        T1, T2 = self[i], self[i - 1]
        lix, rix = T1.filter_bonds(T2)

        set_default_compress_mode(split_opts, self.cyclic)
        L, Q = T1.split(lix, get='tensors', right_inds=rix, **split_opts)
        L = T2 @ L

        L.transpose_like_(T2)
        Q.transpose_like_(T1)

        self[i - 1].modify(data=L.data)
        self[i].modify(data=Q.data)

        if bra is not None:
            bra[i - 1].modify(data=L.data.conj())
            bra[i].modify(data=Q.data.conj())

    def left_canonize_site(self, i, bra=None):
        r"""Left canonize this TN's ith site, inplace::

                i                i
               -o-o-            ->-s-
            ... | | ...  ==> ... | | ...

        Parameters
        ----------
        i : int
            Which site to canonize. The site at i + 1 also absorbs the
            non-isometric part of the decomposition of site i.
        bra : None or matching TensorNetwork to self, optional
            If set, also update this TN's data with the conjugate canonization.
        """
        self._left_decomp_site(i, bra=bra, method='qr')

    def right_canonize_site(self, i, bra=None):
        r"""Right canonize this TN's ith site, inplace::

                  i                i
               -o-o-            -s-<-
            ... | | ...  ==> ... | | ...

        Parameters
        ----------
        i : int
            Which site to canonize. The site at i - 1 also absorbs the
            non-isometric part of the decomposition of site i.
         bra : None or matching TensorNetwork to self, optional
            If set, also update this TN's data with the conjugate canonization.
        """
        self._right_decomp_site(i, bra=bra, method='lq')

    def left_canonize(self, stop=None, start=None, normalize=False, bra=None):
        r"""Left canonize all or a portion of this TN. If this is a MPS,
        this implies that::

                          i              i
            >->->->->->->-o-o-         +-o-o-
            | | | | | | | | | ...  =>  | | | ...
            >->->->->->->-o-o-         +-o-o-

        Parameters
        ----------
        start : int, optional
            If given, the site to start left canonizing at.
        stop : int, optional
            If given, the site to stop left canonizing at.
        normalize : bool, optional
            Whether to normalize the state, only works for OBC.
        bra : MatrixProductState, optional
            If supplied, simultaneously left canonize this MPS too, assuming it
            to be the conjugate state.
        """
        if start is None:
            start = -1 if self.cyclic else 0
        if stop is None:
            stop = self.L - 1

        for i in range(start, stop):
            self.left_canonize_site(i, bra=bra)

        if normalize:
            factor = self[-1].norm()
            self[-1] /= factor
            if bra is not None:
                bra[-1] /= factor

    def right_canonize(self, stop=None, start=None, normalize=False, bra=None):
        r"""Right canonize all or a portion of this TN. If this is a MPS,
        this implies that::

                   i                           i
                -o-o-<-<-<-<-<-<-<          -o-o-+
             ... | | | | | | | | |   ->  ... | | |
                -o-o-<-<-<-<-<-<-<          -o-o-+


        Parameters
        ----------
        start : int, optional
            If given, the site to start right canonizing at.
        stop : int, optional
            If given, the site to stop right canonizing at.
        normalize : bool, optional
            Whether to normalize the state.
        bra : MatrixProductState, optional
            If supplied, simultaneously right canonize this MPS too, assuming
            it to be the conjugate state.
        """
        if start is None:
            start = self.L - (0 if self.cyclic else 1)
        if stop is None:
            stop = 0

        for i in range(start, stop, -1):
            self.right_canonize_site(i, bra=bra)

        if normalize:
            factor = self[0].norm()
            self[0] /= factor
            if bra is not None:
                bra[0] /= factor

    def canonize_cyclic(self, i, bra=None, method='isvd', inv_tol=1e-10):
        """Bring this MatrixProductState into (possibly only approximate)
        canonical form at site(s) ``i``.

        Parameters
        ----------
        i :  int or slice
            The site or range of sites to make canonical.
        bra : MatrixProductState, optional
            Simultaneously canonize this state as well, assuming it to be the
            co-vector.
        method : {'isvd', 'svds', ...}, optional
            How to perform the lateral compression.
        inv_tol : float, optional
            Tolerance with which to invert the gauge.
        """
        if isinstance(i, Integral):
            start, stop = i, i + 1
        elif isinstance(i, slice):
            start, stop = i.start, i.stop
        else:
            start, stop = min(i), max(i) + 1
            if tuple(i) != tuple(range(start, stop)):
                raise ValueError("Parameter ``i`` should be an integer or "
                                 f"contiguous block of integers, got {i}.")

        k = self.copy()
        b = k.H
        k.add_tag('_KET')
        b.add_tag('_BRA')
        kb = k & b

        # approximate the rest of the chain with a separable transfer operator
        kbc = kb.replace_section_with_svd(start, stop, eps=0.0, which='!any',
                                          method=method, max_bond=1,
                                          ltags='_LEFT', rtags='_RIGHT')

        EL = kbc['_LEFT'].squeeze()
        # explicitly symmetrize to hermitian
        EL.modify(data=(EL.data + dag(EL.data)) / 2)
        # split into upper 'ket' part and lower 'bra' part, symmetric
        EL_lix, = EL.bonds(kbc[k.site_tag(start), '_BRA'])
        _, x = EL.split(EL_lix, method='eigh', cutoff=-1, get='arrays')

        ER = kbc['_RIGHT'].squeeze()
        # explicitly symmetrize to hermitian
        ER.modify(data=(ER.data + dag(ER.data)) / 2)
        # split into upper 'ket' part and lower 'bra' part, symmetric
        ER_lix, = ER.bonds(kbc[k.site_tag(stop - 1), '_BRA'])
        _, y = ER.split(ER_lix, method='eigh', cutoff=-1, get='arrays')

        self.insert_gauge(x, start - 1, start, tol=inv_tol)
        self.insert_gauge(y, stop, stop - 1, tol=inv_tol)

        if bra is not None:
            for i in (start - 1, start, stop, stop - 1):
                bra[i].modify(data=self[i].data.conj())

    def shift_orthogonality_center(self, current, new, bra=None):
        """Move the orthogonality center of this MPS.

        Parameters
        ----------
        current : int
            The current orthogonality center.
        new : int
            The target orthogonality center.
        bra : MatrixProductState, optional
            If supplied, simultaneously move the orthogonality center of this
            MPS too, assuming it to be the conjugate state.
        """
        if new > current:
            for i in range(current, new):
                self.left_canonize_site(i, bra=bra)
        else:
            for i in range(current, new, -1):
                self.right_canonize_site(i, bra=bra)

    def canonize(self, where, cur_orthog='calc', bra=None):
        r"""Mixed canonize this TN. If this is a MPS, this implies that::

                          i                      i
            >->->->->- ->-o-<- -<-<-<-<-<      +-o-+
            | | | | |...| | |...| | | | |  ->  | | |
            >->->->->- ->-o-<- -<-<-<-<-<      +-o-+

        You can also supply a set of indices to orthogonalize around, and a
        current location of the orthogonality center for efficiency::

                  current                              where
                  .......                              .....
            >->->-c-c-c-c-<-<-<-<-<-<      >->->->->->-w-w-w-<-<-<-<
            | | | | | | | | | | | | |  ->  | | | | | | | | | | | | |
            >->->-c-c-c-c-<-<-<-<-<-<      >->->->->->-w-w-w-<-<-<-<
               cmin     cmax                           i   j

        This would only move ``cmin`` to ``i`` and ``cmax`` to ``j`` if
        necessary.

        Parameters
        ----------
        where : int or sequence of int
            Which site(s) to orthogonalize around. If a sequence of int then
            make sure that section from min(where) to max(where) is orthog.
        cur_orthog : int, sequence of int, or 'calc'
            If given, the current site(s), so as to shift the orthogonality
            ceneter as efficiently as possible. If 'calc', calculate the
            current orthogonality center.
        bra : MatrixProductState, optional
            If supplied, simultaneously mixed canonize this MPS too, assuming
            it to be the conjugate state.
        """
        if isinstance(where, int):
            i = j = where
        else:
            i, j = min(where), max(where)

        if cur_orthog == 'calc':
            cur_orthog = self.calc_current_orthog_center()

        if cur_orthog is not None:
            if isinstance(cur_orthog, int):
                cmin = cmax = cur_orthog
            else:
                cmin, cmax = min(cur_orthog), max(cur_orthog)

            if cmax > j:
                self.shift_orthogonality_center(cmax, j, bra=bra)
            if cmin < i:
                self.shift_orthogonality_center(cmin, i, bra=bra)

        else:
            self.left_canonize(i, bra=bra)
            self.right_canonize(j, bra=bra)

        return self

    def left_compress_site(self, i, bra=None, **compress_opts):
        """Left compress this 1D TN's ith site, such that the site is then
        left unitary with its right bond (possibly) reduced in dimension.

        Parameters
        ----------
        i : int
            Which site to compress.
        bra : None or matching TensorNetwork to self, optional
            If set, also update this TN's data with the conjugate compression.
        compress_opts
            Supplied to :meth:`Tensor.split`.
        """
        compress_opts.setdefault('absorb', 'right')
        self._left_decomp_site(i, bra=bra, **compress_opts)

    def right_compress_site(self, i, bra=None, **compress_opts):
        """Right compress this 1D TN's ith site, such that the site is then
        right unitary with its left bond (possibly) reduced in dimension.

        Parameters
        ----------
        i : int
            Which site to compress.
        bra : None or matching TensorNetwork to self, optional
            If set, update this TN's data with the conjugate compression.
        compress_opts
            Supplied to :meth:`Tensor.split`.
        """
        compress_opts.setdefault('absorb', 'left')
        self._right_decomp_site(i, bra=bra, **compress_opts)

    def left_compress(self, start=None, stop=None, bra=None, **compress_opts):
        """Compress this 1D TN, from left to right, such that it becomes
        left-canonical (unless ``absorb != 'right'``).

        Parameters
        ----------
        start : int, optional
            Site to begin compressing on.
        stop : int, optional
            Site to stop compressing at (won't itself be an isometry).
        bra : None or TensorNetwork like this one, optional
            If given, update this TN as well, assuming it to be the conjugate.
        compress_opts
            Supplied to :meth:`Tensor.split`.
        """
        if start is None:
            start = -1 if self.cyclic else 0
        if stop is None:
            stop = self.L - 1

        for i in range(start, stop):
            self.left_compress_site(i, bra=bra, **compress_opts)

    def right_compress(self, start=None, stop=None, bra=None, **compress_opts):
        """Compress this 1D TN, from right to left, such that it becomes
        right-canonical (unless ``absorb != 'left'``).

        Parameters
        ----------
        start : int, optional
            Site to begin compressing on.
        stop : int, optional
            Site to stop compressing at (won't itself be an isometry).
        bra : None or TensorNetwork like this one, optional
            If given, update this TN as well, assuming it to be the conjugate.
        compress_opts
            Supplied to :meth:`Tensor.split`.
        """
        if start is None:
            start = self.L - (0 if self.cyclic else 1)
        if stop is None:
            stop = 0

        for i in range(start, stop, -1):
            self.right_compress_site(i, bra=bra, **compress_opts)

    def compress(self, form=None, **compress_opts):
        """Compress this 1D Tensor Network, possibly into canonical form.

        Parameters
        ----------
        form : {None, 'flat', 'left', 'right'} or int
            Output form of the TN. ``None`` left canonizes the state first for
            stability reasons, then right_compresses (default). ``'flat'``
            tries to distribute the singular values evenly -- state will not
            be canonical. ``'left'`` and ``'right'`` put the state into left
            and right canonical form respectively with a prior opposite sweep,
            or an int will put the state into mixed canonical form at that
            site.
        compress_opts
            Supplied to :meth:`Tensor.split`.
        """
        if form is None:
            form = 'right'

        if isinstance(form, Integral):
            self.right_canonize()
            self.left_compress(**compress_opts)
            self.right_canonize(stop=form)

        elif form == 'left':
            self.right_canonize(bra=compress_opts.get('bra', None))
            self.left_compress(**compress_opts)
        elif form == 'right':
            self.left_canonize(bra=compress_opts.get('bra', None))
            self.right_compress(**compress_opts)

        elif form == 'flat':
            compress_opts['absorb'] = 'both'
            self.right_compress(stop=self.L // 2, **compress_opts)
            self.left_compress(stop=self.L // 2, **compress_opts)

        else:
            raise ValueError(f"Form specifier {form} not understood, should be"
                             " either 'left', 'right', 'flat' or an int "
                             "specifiying a new orthog center.")

    def compress_site(self, i, canonize=True, cur_orthog='calc', bra=None,
                      **compress_opts):
        r"""Compress the bonds adjacent to site ``i``, by default first setting
        the orthogonality center to that site::

                 i                     i
            -o-o-o-o-o-    -->    ->->~o~<-<-
             | | | | |             | | | | |

        Parameters
        ----------
        i : int
            Which site to compress around
        canonize : bool, optional
            Whether to first set the orthogonality center to site ``i``.
        cur_orthog : int, optional
            If given, the known current orthogonality center, to speed up the
            mixed canonization.
        bra : MatrixProductState, optional
            The conjugate state to also apply the compression to.
        compress_opts
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_split`.
        """
        if canonize:
            self.canonize(i, cur_orthog=cur_orthog, bra=bra)

        if self.cyclic or i > 0:
            self.left_compress_site(i - 1, bra=bra, **compress_opts)

        if self.cyclic or i < self.L - 1:
            self.right_compress_site(i + 1, bra=bra, **compress_opts)

    def bond(self, i, j):
        """Get the name of the index defining the bond between sites i and j.
        """
        bond, = self[i].bonds(self[j])
        return bond

    def bond_size(self, i, j):
        """Return the size of the bond between site ``i`` and ``j``.
        """
        b_ix = self.bond(i, j)
        return self[i].ind_size(b_ix)

    def bond_sizes(self):
        bnd_szs = [self.bond_size(i, i + 1) for i in range(self.L - 1)]
        if self.cyclic:
            bnd_szs.append(self.bond_size(-1, 0))
        return bnd_szs

    def amplitude(self, b):
        """Compute the amplitude of configuration ``b``.

        Parameters
        ----------
        b : sequence of int
            The configuration to compute the amplitude of.

        Returns
        -------
        c_b : scalar
        """
        if len(b) != self.nsites:
            raise ValueError(f"Bit-string {b} length does not "
                             f"match MPS length {self.nsites}.")

        selector = {self.site_ind(i): int(xi) for i, xi in enumerate(b)}
        mps_b = self.isel(selector)
        return mps_b ^ ...

    def singular_values(self, i, cur_orthog=None, method='svd'):
        r"""Find the singular values associated with the ith bond::

            ....L....   i
            o-o-o-o-o-l-o-o-o-o-o-o-o-o-o-o-o
            | | | | |   | | | | | | | | | | |
                   i-1  ..........R..........

        Leaves the 1D TN in mixed canoncial form at bond ``i``.

        Parameters
        ----------
        i : int
            Which bond, or equivalently, the number of sites in the
            left partition.
        cur_orthog : int
            If given, the known current orthogonality center, to speed up the
            mixed canonization, e.g. if sweeping this function from left to
            right would use ``i - 1``.

        Returns
        -------
        svals : 1d-array
            The singular values.
        """
        if not (0 < i < self.L):
            raise ValueError(f"Need 0 < i < {self.L}, got i={i}.")

        self.canonize(i, cur_orthog)

        Tm1 = self[i]
        left_inds = Tm1.bonds(self[i - 1])
        return Tm1.singular_values(left_inds, method=method)

    def expand_bond_dimension(
        self,
        new_bond_dim,
        rand_strength=0.0,
        bra=None,
        inplace=True,
    ):
        """Expand the bond dimensions of this 1D tensor network to at least
        ``new_bond_dim``.

        Parameters
        ----------
        new_bond_dim : int
            Minimum bond dimension to expand to.
        inplace : bool, optional
            Whether to perform the expansion in place.
        bra : MatrixProductState, optional
            Mirror the changes to ``bra`` inplace, treating it as the conjugate
            state.
        rand_strength : float, optional
            If ``rand_strength > 0``, fill the new tensor entries with gaussian
            noise of strength ``rand_strength``.

        Returns
        -------
        MatrixProductState
        """
        tn = super().expand_bond_dimension(
            new_bond_dim=new_bond_dim,
            rand_strength=rand_strength,
            inplace=inplace,
        )

        if bra is not None:
            for coo in tn.gen_site_coos():
                bra[coo].modify(data=tn[coo].data.conj())

        return tn

    def count_canonized(self):
        if self.cyclic:
            return 0, 0

        ov = self.H & self
        num_can_l = 0
        num_can_r = 0

        def isidentity(x):
            d = x.shape[0]
            if get_dtype_name(x) in ('float32', 'complex64'):
                rtol, atol = 1e-5, 1e-6
            else:
                rtol, atol = 1e-9, 1e-11
            idtty = do('eye', d, dtype=x.dtype, like=x)
            return do('allclose', x, idtty, rtol=rtol, atol=atol)

        for i in range(self.L - 1):
            ov ^= slice(max(0, i - 1), i + 1)
            x = ov[i].data
            if isidentity(x):
                num_can_l += 1
            else:
                break

        for j in reversed(range(i + 1, self.L)):
            ov ^= slice(j, min(self.L, j + 2))
            x = ov[j].data
            if isidentity(x):
                num_can_r += 1
            else:
                break

        return num_can_l, num_can_r

    def calc_current_orthog_center(self):
        """Calculate the site(s) of the current orthogonality center.

        Returns
        -------
        int or (int, int)
            The site, or min/max, around which this MPS is orthogonal.
        """
        lo, ro = self.count_canonized()
        i, j = lo, self.L - ro - 1
        return i if i == j else i, j

    def as_cyclic(self, inplace=False):
        """Convert this flat, 1D, TN into cyclic form by adding a dummy bond
        between the first and last sites.
        """
        tn = self if inplace else self.copy()

        # nothing to do
        if tn.cyclic:
            return tn

        tn.new_bond(0, -1)
        tn.cyclic = True
        return tn

    def show(self, max_width=None):
        l1 = ""
        l2 = ""
        l3 = ""
        num_can_l, num_can_r = self.count_canonized()
        for i in range(self.L - 1):
            bdim = self.bond_size(i, i + 1)
            strl = len(str(bdim))
            l1 += f" {bdim}"
            l2 += (">" if i < num_can_l else
                   "<" if i >= self.L - num_can_r else
                   "●") + ("─" if bdim < 100 else "━") * strl
            l3 += "│" + " " * strl
            strl = len(str(bdim))

        l1 += " "
        l2 += "<" if num_can_r > 0 else "●"
        l3 += "│"

        if self.cyclic:
            bdim = self.bond_size(0, self.L - 1)
            bnd_str = ("─" if bdim < 100 else "━") * strl
            l1 = f" {bdim}{l1}{bdim} "
            l2 = f"+{bnd_str}{l2}{bnd_str}+"
            l3 = f" {' ' * strl}{l3}{' ' * strl} "

        print_multi_line(l1, l2, l3, max_width=max_width)


class MatrixProductState(TensorNetwork1DVector,
                         TensorNetwork1DFlat,
                         TensorNetwork1D,
                         TensorNetwork):
    """Initialise a matrix product state, with auto labelling and tagging.

    Parameters
    ----------
    arrays : sequence of arrays
        The tensor arrays to form into a MPS.
    shape : str, optional
        String specifying layout of the tensors. E.g. 'lrp' (the default)
        indicates the shape corresponds left-bond, right-bond, physical index.
        End tensors have either 'l' or 'r' dropped from the string.
    site_ind_id : str
        A string specifiying how to label the physical site indices. Should
        contain a ``'{}'`` placeholder. It is used to generate the actual
        indices like: ``map(site_ind_id.format, range(len(arrays)))``.
    site_tag_id : str
        A string specifiying how to tag the tensors at each site. Should
        contain a ``'{}'`` placeholder. It is used to generate the actual tags
        like: ``map(site_tag_id.format, range(len(arrays)))``.
    tags : str or sequence of str, optional
        Global tags to attach to all tensors.
    bond_name : str, optional
        The base name of the bond indices, onto which uuids will be added.
    """

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_site_ind_id',
        'cyclic',
        '_L',
    )

    def __init__(self, arrays, *, shape='lrp', tags=None, bond_name="",
                 site_ind_id='k{}', site_tag_id='I{}', **tn_opts):

        # short-circuit for copying MPSs
        if isinstance(arrays, MatrixProductState):
            super().__init__(arrays)
            return

        arrays = tuple(arrays)

        self._L = len(arrays)

        # process site indices
        self._site_ind_id = site_ind_id
        site_inds = map(site_ind_id.format, range(self.L))

        # process site tags
        self._site_tag_id = site_tag_id
        site_tags = map(site_tag_id.format, range(self.L))
        if tags is not None:
            # mix in global tags
            tags = tags_to_oset(tags)
            site_tags = (tags | oset((st,)) for st in site_tags)

        self.cyclic = (ops.ndim(arrays[0]) == 3)

        # transpose arrays to 'lrp' order.
        def gen_orders():
            lp_ord = tuple(shape.replace('r', "").find(x) for x in 'lp')
            lrp_ord = tuple(shape.find(x) for x in 'lrp')
            rp_ord = tuple(shape.replace('l', "").find(x) for x in 'rp')
            yield lp_ord if not self.cyclic else lrp_ord
            for _ in range(self.L - 2):
                yield lrp_ord
            yield rp_ord if not self.cyclic else lrp_ord

        def gen_inds():
            cyc_bond = (rand_uuid(base=bond_name),) if self.cyclic else ()

            nbond = rand_uuid(base=bond_name)
            yield cyc_bond + (nbond, next(site_inds))
            pbond = nbond
            for _ in range(self.L - 2):
                nbond = rand_uuid(base=bond_name)
                yield (pbond, nbond, next(site_inds))
                pbond = nbond
            yield (pbond,) + cyc_bond + (next(site_inds),)

        def gen_tensors():
            for array, site_tag, inds, order in zip(arrays, site_tags,
                                                    gen_inds(), gen_orders()):
                yield Tensor(transpose(array, order), inds=inds, tags=site_tag)

        super().__init__(gen_tensors(), virtual=True, **tn_opts)

    @classmethod
    def from_dense(cls, psi, dims, site_ind_id='k{}',
                   site_tag_id='I{}', **split_opts):
        """Create a ``MatrixProductState`` directly from a dense vector

        Parameters
        ----------
        psi : array_like
            The dense state to convert to MPS from.
        dims : sequence of int
            Physical subsystem dimensions of each site.
        site_ind_id : str, optional
            How to index the physical sites, see
            :class:`~quimb.tensor.tensor_1d.MatrixProductState`.
        site_tag_id : str, optional
            How to tag the physical sites, see
            :class:`~quimb.tensor.tensor_1d.MatrixProductState`.
        split_opts
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_split` to
            in order to partition the dense vector into tensors.

        Returns
        -------
        MatrixProductState

        Examples
        --------

            >>> dims = [2, 2, 2, 2, 2, 2]
            >>> psi = rand_ket(prod(dims))
            >>> mps = MatrixProductState.from_dense(psi, dims)
            >>> mps.show()
             2 4 8 4 2
            o-o-o-o-o-o
            | | | | | |
        """
        set_default_compress_mode(split_opts)

        L = len(dims)
        inds = [site_ind_id.format(i) for i in range(L)]

        T = Tensor(reshape(ops.asarray(psi), dims), inds=inds)

        def gen_tensors():
            #           split
            #       <--  : yield
            #            : :
            #     OOOOOOO--O-O-O
            #     |||||||  | | |
            #     .......
            #    left_inds
            TM = T
            for i in range(L - 1, 0, -1):
                TM, TR = TM.split(left_inds=inds[:i], get='tensors',
                                  rtags=site_tag_id.format(i), **split_opts)
                yield TR
            TM.add_tag(site_tag_id.format(0))
            yield TM

        tn = TensorNetwork(gen_tensors())
        return cls.from_TN(tn, cyclic=False, L=L,
                           site_ind_id=site_ind_id,
                           site_tag_id=site_tag_id)

    def add_MPS(self, other, inplace=False, compress=False, **compress_opts):
        """Add another MatrixProductState to this one.
        """
        if self.L != other.L:
            raise ValueError("Can't add MPS with another of different length.")

        new_mps = self if inplace else self.copy()

        for i in new_mps.gen_site_coos():
            t1, t2 = new_mps[i], other[i]

            if set(t1.inds) != set(t2.inds):
                # Need to use bonds to match indices
                reindex_map = {}

                if i > 0 or self.cyclic:
                    pair = ((i - 1) % self.L, i)
                    reindex_map[other.bond(*pair)] = new_mps.bond(*pair)

                if i < new_mps.L - 1 or self.cyclic:
                    pair = (i, (i + 1) % self.L)
                    reindex_map[other.bond(*pair)] = new_mps.bond(*pair)

                t2 = t2.reindex(reindex_map)

            t1.direct_product_(t2, sum_inds=new_mps.site_ind(i))

        if compress:
            new_mps.compress(**compress_opts)

        return new_mps

    add_MPS_ = functools.partialmethod(add_MPS, inplace=True)

    def permute_arrays(self, shape='lrp'):
        """Permute the indices of each tensor in this MPS to match ``shape``.
        This doesn't change how the overall object interacts with other tensor
        networks but may be useful for extracting the underlying arrays
        consistently. This is an inplace operation.

        Parameters
        ----------
        shape : str, optional
            A permutation of ``'lrp'`` specifying the desired order of the
            left, right, and physical indices respectively.
        """
        for i in self.sites:
            inds = {'p': self.site_ind(i)}
            if self.cyclic or i > 0:
                inds['l'] = self.bond(i, (i - 1) % self.L)
            if self.cyclic or i < self.L - 1:
                inds['r'] = self.bond(i, (i + 1) % self.L)
            inds = [inds[s] for s in shape if s in inds]
            self[i].transpose_(*inds)

    def __add__(self, other):
        """MPS addition.
        """
        return self.add_MPS(other, inplace=False)

    def __iadd__(self, other):
        """In-place MPS addition.
        """
        return self.add_MPS(other, inplace=True)

    def __sub__(self, other):
        """MPS subtraction.
        """
        return self.add_MPS(other * -1, inplace=False)

    def __isub__(self, other):
        """In-place MPS subtraction.
        """
        return self.add_MPS(other * -1, inplace=True)

    def normalize(self, bra=None, eps=1e-15, insert=None):
        """Normalize this MPS, optional with co-vector ``bra``. For periodic
        MPS this uses transfer matrix SVD approximation with precision ``eps``
        in order to be efficient. Inplace.

        Parameters
        ----------
        bra : MatrixProductState, optional
            If given, normalize this MPS with the same factor.
        eps : float, optional
            If cyclic, precision to approximation transfer matrix with.
            Default: 1e-14.
        insert : int, optional
            Insert the corrective normalization on this site, random if
            not given.

        Returns
        -------
        old_norm : float
            The old norm ``self.H @ self``.
        """
        norm = expec_TN_1D(self.H, self, eps=eps)

        if insert is None:
            insert = -1

        self[insert].modify(data=self[insert].data / norm ** 0.5)
        if bra is not None:
            bra[insert].modify(data=bra[insert].data / norm ** 0.5)

        return norm

    def gate_split(self, G, where, inplace=False, **compress_opts):
        r"""Apply a two-site gate and then split resulting tensor to retrieve a
        MPS form::

            -o-o-A-B-o-o-
             | | | | | |            -o-o-GGG-o-o-           -o-o-X~Y-o-o-
             | | GGG | |     ==>     | | | | | |     ==>     | | | | | |
             | | | | | |                 i j                     i j
                 i j

        As might be found in TEBD.

        Parameters
        ----------
        G : array
            The gate, with shape ``(d**2, d**2)`` for physical dimension ``d``.
        where : (int, int)
            Indices of the sites to apply the gate to.
        compress_opts
            Supplied to :func:`~quimb.tensor.tensor_split`.

        See Also
        --------
        gate, gate_with_auto_swap
        """
        tn = self if inplace else self.copy()

        i, j = where

        Ti, Tj = tn[i], tn[j]
        ix_i, ix_j = tn.site_ind(i), tn.site_ind(j)

        # Make Tensor of gate
        d = tn.phys_dim(i)
        TG = Tensor(reshape(ops.asarray(G), (d, d, d, d)),
                    inds=("_tmpi", "_tmpj", ix_i, ix_j))

        # Contract gate into the two sites
        TG = TG.contract(Ti, Tj)
        TG.reindex_({"_tmpi": ix_i, "_tmpj": ix_j})

        # Split the tensor
        _, left_ix = Ti.filter_bonds(Tj)
        set_default_compress_mode(compress_opts, self.cyclic)
        nTi, nTj = TG.split(left_inds=left_ix, get='tensors', **compress_opts)

        # make sure the new data shape matches and reinsert
        Ti.modify(data=nTi.transpose_like_(Ti).data)
        Tj.modify(data=nTj.transpose_like_(Tj).data)

        return tn

    gate_split_ = functools.partialmethod(gate_split, inplace=True)

    def swap_sites_with_compress(self, i, j, cur_orthog=None,
                                 inplace=False, **compress_opts):
        """Swap sites ``i`` and ``j`` by contracting, then splitting with the
        physical indices swapped.

        Parameters
        ----------
        i : int
            The first site to swap.
        j : int
            The second site to swap.
        cur_orthog : int, sequence of int, or 'calc'
            If known, the current orthogonality center.
        inplace : bond, optional
            Perform the swaps inplace.
        compress_opts
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_split`.
        """
        i, j = sorted((i, j))
        if i + 1 != j:
            raise ValueError("Sites aren't adjacent.")

        mps = self if inplace else self.copy()
        mps.canonize((i, j), cur_orthog)

        # get site tensors and indices
        ix_i, ix_j = map(mps.site_ind, (i, j))
        Ti, Tj = mps[i], mps[j]
        _, unshared = Ti.filter_bonds(Tj)

        # split the contracted tensor, swapping the site indices
        Tij = Ti @ Tj
        lix = [i for i in unshared if i != ix_i] + [ix_j]
        set_default_compress_mode(compress_opts, self.cyclic)
        sTi, sTj = Tij.split(lix, get='tensors', **compress_opts)

        # reindex and transpose the tensors to directly update original tensors
        sTi.reindex_({ix_j: ix_i})
        sTj.reindex_({ix_i: ix_j})
        sTi.transpose_like_(Ti)
        sTj.transpose_like_(Tj)

        Ti.modify(data=sTi.data)
        Tj.modify(data=sTj.data)

        return mps

    def swap_site_to(self, i, f, cur_orthog=None,
                     inplace=False, **compress_opts):
        r"""Swap site ``i`` to site ``f``, compressing the bond after each
        swap::

                  i       f
            0 1 2 3 4 5 6 7 8 9      0 1 2 4 5 6 7 3 8 9
            o-o-o-x-o-o-o-o-o-o      o-o-o-o-o-o-o-x-o-o
            | | | | | | | | | |  ->  | | | | | | | | | |


        Parameters
        ----------
        i : int
            The site to move.
        f : int
            The new location for site ``i``.
        cur_orthog : int, sequence of int, or 'calc'
            If known, the current orthogonality center.
        inplace : bond, optional
            Perform the swaps inplace.
        compress_opts
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_split`.
        """
        mps = self if inplace else self.copy()

        if i == f:
            return mps
        if i < f:
            js = range(i, f)
        if f < i:
            js = range(i - 1, f - 1, -1)

        for j in js:
            mps.swap_sites_with_compress(
                j, j + 1, inplace=True, cur_orthog=cur_orthog, **compress_opts)
            cur_orthog = (j, j + 1)

        return mps

    def gate_with_auto_swap(self, G, where, inplace=False,
                            cur_orthog=None, **compress_opts):
        """Perform a two site gate on this MPS by, if necessary, swapping and
        compressing the sites until they are adjacent, using ``gate_split``,
        then unswapping the sites back to their original position.

        Parameters
        ----------
        G : array
            The gate, with shape ``(d**2, d**2)`` for physical dimension ``d``.
        where : (int, int)
            Indices of the sites to apply the gate to.
        cur_orthog : int, sequence of int, or 'calc'
            If known, the current orthogonality center.
        inplace : bond, optional
            Perform the swaps inplace.
        compress_opts
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_split`.

        See Also
        --------
        gate, gate_split
        """
        mps = self if inplace else self.copy()

        i, j = sorted(where)
        need2swap = i + 1 != j

        # move j site adjacent to i site
        if need2swap:
            mps.swap_site_to(j, i + 1, cur_orthog=cur_orthog,
                             inplace=True, **compress_opts)
            cur_orthog = (i + 1, i + 2)

        # make sure sites are orthog center, then apply and split
        mps.canonize((i, i + 1), cur_orthog)
        mps.gate_split_(G, (i, i + 1), **compress_opts)

        # move j site back to original position
        if need2swap:
            mps.swap_site_to(i + 1, j, cur_orthog=(i, i + 1),
                             inplace=True, **compress_opts, )

        return mps

    def magnetization(self, i, direction='Z', cur_orthog=None):
        """Compute the magnetization at site ``i``.
        """
        if self.cyclic:
            msg = ("``magnetization`` currently makes use of orthogonality for"
                   " efficiencies sake, for cyclic systems is it still "
                   "possible to compute as a normal expectation.")
            raise NotImplementedError(msg)

        self.canonize(i, cur_orthog)

        # +-k-+
        # | O |
        # +-b-+

        Tk = self[i]
        ind1, ind2 = self.site_ind(i), '__tmp__'
        Tb = Tk.H.reindex({ind1: ind2})

        O_data = qu.spin_operator(direction, S=(self.phys_dim(i) - 1) / 2)
        TO = Tensor(O_data, inds=(ind1, ind2))

        return Tk.contract(TO, Tb)

    def schmidt_values(self, i, cur_orthog=None, method='svd'):
        r"""Find the schmidt values associated with the bipartition of this
        MPS between sites on either site of ``i``. In other words, ``i`` is the
        number of sites in the left hand partition::

            ....L....   i
            o-o-o-o-o-S-o-o-o-o-o-o-o-o-o-o-o
            | | | | |   | | | | | | | | | | |
                   i-1  ..........R..........

        The schmidt values, ``S``, are the singular values associated with the
        ``(i - 1, i)`` bond, squared, provided the MPS is mixed canonized at
        one of those sites.

        Parameters
        ----------
        i : int
            The number of sites in the left partition.
        cur_orthog : int
            If given, the known current orthogonality center, to speed up the
            mixed canonization.

        Returns
        -------
        S : 1d-array
            The schmidt values.
        """
        if self.cyclic:
            raise NotImplementedError

        return self.singular_values(i, cur_orthog, method=method)**2

    def entropy(self, i, cur_orthog=None, method='svd'):
        """The entropy of bipartition between the left block of ``i`` sites and
        the rest.

        Parameters
        ----------
        i : int
            The number of sites in the left partition.
        cur_orthog : int
            If given, the known current orthogonality center, to speed up the
            mixed canonization.

        Returns
        -------
        float
        """
        if self.cyclic:
            msg = ("For cyclic systems, try explicitly computing the entropy "
                   "of the (compressed) reduced density matrix.")
            raise NotImplementedError(msg)

        S = self.schmidt_values(i, cur_orthog=cur_orthog, method=method)
        S = S[S > 0.0]
        return do('sum', -S * do('log2', S))

    def schmidt_gap(self, i, cur_orthog=None, method='svd'):
        """The schmidt gap of bipartition between the left block of ``i`` sites
        and the rest.

        Parameters
        ----------
        i : int
            The number of sites in the left partition.
        cur_orthog : int
            If given, the known current orthogonality center, to speed up the
            mixed canonization.

        Returns
        -------
        float
        """
        if self.cyclic:
            raise NotImplementedError

        S = self.schmidt_values(i, cur_orthog=cur_orthog, method=method)

        if len(S) == 1:
            return S[0]

        return S[0] - S[1]

    def partial_trace(self, keep, upper_ind_id="b{}", rescale_sites=True):
        r"""Partially trace this matrix product state, producing a matrix
        product operator.

        Parameters
        ----------
        keep : sequence of int or slice
            Indicies of the sites to keep.
        upper_ind_id : str, optional
            The ind id of the (new) 'upper' inds, i.e. the 'bra' inds.
        rescale_sites : bool, optional
            If ``True`` (the default), then the kept sites will be rescaled to
            ``(0, 1, 2, ...)`` etc. rather than keeping their original site
            numbers.

        Returns
        -------
        rho : MatrixProductOperator
            The density operator in MPO form.
        """
        p_bra = self.copy()
        p_bra.reindex_sites_(upper_ind_id, where=keep)
        rho = self.H & p_bra
        # now have e.g:
        #     | |     |   |
        # o-o-o-o-o-o-o-o-o
        # | |     | |   |
        # o-o-o-o-o-o-o-o-o
        #     | |     |   |

        if isinstance(keep, slice):
            keep = self.slice2sites(keep)

        keep = sorted(keep)

        for i in self.gen_site_coos():
            if i in keep:
                #      |
                #     -o-             |
                # ... -o- ... -> ... -O- ...
                #     i|             i|
                rho ^= self.site_tag(i)
            else:
                #        |
                #     -o-o-              |
                # ...  |    ... -> ... -OO- ...
                #     -o-o-              |i+1
                #      i |i+1
                if i < self.L - 1:
                    rho >>= [self.site_tag(i), self.site_tag(i + 1)]
                else:
                    rho >>= [self.site_tag(i), self.site_tag(max(keep))]

                rho.drop_tags(self.site_tag(i))

        # if single site a single tensor is produced
        if isinstance(rho, Tensor):
            rho = TensorNetwork([rho])

        if rescale_sites:
            # e.g. [3, 4, 5, 7, 9] -> [0, 1, 2, 3, 4]
            retag, reind = {}, {}
            for new, old in enumerate(keep):
                retag[self.site_tag(old)] = self.site_tag(new)
                reind[self.site_ind(old)] = self.site_ind(new)
                reind[upper_ind_id.format(old)] = upper_ind_id.format(new)

            rho.retag_(retag)
            rho.reindex_(reind)
            L = len(keep)
        else:
            L = self.L

        # transpose upper and lower tags to match other MPOs
        rho.view_as_(
            MatrixProductOperator,
            cyclic=self.cyclic, L=L, site_tag_id=self.site_tag_id,
            lower_ind_id=upper_ind_id, upper_ind_id=self.site_ind_id, )

        rho.fuse_multibonds(inplace=True)
        return rho

    def ptr(self, keep, upper_ind_id="b{}", rescale_sites=True):
        """Alias of :meth:`~quimb.tensor.MatrixProductState.partial_trace`.
        """
        return self.partial_trace(keep, upper_ind_id,
                                  rescale_sites=rescale_sites)

    def bipartite_schmidt_state(self, sz_a, get='ket', cur_orthog=None):
        r"""Compute the reduced state for a bipartition of an OBC MPS, in terms
        of the minimal left/right schmidt basis::

                A            B
            .........     ...........
            >->->->->--s--<-<-<-<-<-<    ->   +-s-+
            | | | | |     | | | | | |         |   |
           k0 k1...                          kA   kB

        Parameters
        ----------
        sz_a : int
            The number of sites in subsystem A, must be ``0 < sz_a < N``.
        get : {'ket', 'rho', 'ket-dense', 'rho-dense'}, optional
            Get the:

            - 'ket': vector form as tensor.
            - 'rho': density operator form, i.e. vector outer product
            - 'ket-dense': like 'ket' but return ``qarray``.
            - 'rho-dense': like 'rho' but return ``qarray``.

        cur_orthog : int, optional
            If given, take as the current orthogonality center so as to
            efficienctly move it a minimal distance.
        """
        if self.cyclic:
            raise NotImplementedError("MPS must have OBC.")

        s = do('diag', self.singular_values(sz_a, cur_orthog=cur_orthog))

        if 'dense' in get:
            kd = qu.qarray(s.reshape(-1, 1))
            if 'ket' in get:
                return kd
            elif 'rho' in get:
                return kd @ kd.H

        else:
            k = Tensor(s, (self.site_ind('A'), self.site_ind('B')))
            if 'ket' in get:
                return k
            elif 'rho' in get:
                return k & k.reindex({'kA': 'bA', 'kB': 'bB'})

    @staticmethod
    def _do_lateral_compress(mps, kb, section, leave_short, ul, ll, heps,
                             hmethod, hmax_bond, verbosity, compressed,
                             **compress_opts):

        #           section
        #   ul -o-o-o-o-o-o-o-o-o-       ul -\       /-
        #       | | | | | | | | |   ==>       0~~~~~0
        #   ll -o-o-o-o-o-o-o-o-o-       ll -/   :   \-
        #                                      hmax_bond

        if leave_short:
            # if section is short doesn't make sense to lateral compress
            #     work out roughly when this occurs by comparing bond size
            left_sz = mps.bond_size(section[0] - 1, section[0])
            right_sz = mps.bond_size(section[-1], section[-1] + 1)

            if mps.phys_dim() ** len(section) <= left_sz * right_sz:
                if verbosity >= 1:
                    print(f"Leaving lateral compress of section '{section}' as"
                          f" it is too short: length={len(section)}, eff "
                          f"size={left_sz * right_sz}.")
                return

        if verbosity >= 1:
            print(f"Laterally compressing section {section}. Using options: "
                  f"eps={heps}, method={hmethod}, max_bond={hmax_bond}")

        section_tags = map(mps.site_tag, section)
        kb.replace_with_svd(section_tags, (ul, ll), heps, inplace=True,
                            ltags='_LEFT', rtags='_RIGHT', method=hmethod,
                            max_bond=hmax_bond, **compress_opts)

        compressed.append(section)

    @staticmethod
    def _do_vertical_decomp(mps, kb, section, sysa, sysb, compressed, ul, ur,
                            ll, lr, vmethod, vmax_bond, veps, verbosity,
                            **compress_opts):
        if section == sysa:
            label = 'A'
        elif section == sysb:
            label = 'B'
        else:
            return

        section_tags = [mps.site_tag(i) for i in section]

        if section in compressed:

            #                    ----U----             |  <- vmax_bond
            #  -\      /-            /             ----U----
            #    L~~~~R     ==>      \       ==>
            #  -/      \-            /             ----D----
            #                    ----D----             |  <- vmax_bond

            # try and choose a sensible method
            if vmethod is None:
                left_sz = mps.bond_size(section[0] - 1, section[0])
                right_sz = mps.bond_size(section[-1], section[-1] + 1)
                if left_sz * right_sz <= 2**13:
                    # cholesky is not rank revealing
                    vmethod = 'eigh' if vmax_bond else 'cholesky'
                else:
                    vmethod = 'isvd'

            if verbosity >= 1:
                print(f"Performing vertical decomposition of section {label}, "
                      f"using options: eps={veps}, method={vmethod}, "
                      f"max_bond={vmax_bond}.")

            # do vertical SVD
            kb.replace_with_svd(
                section_tags, (ul, ur), right_inds=(ll, lr), eps=veps,
                ltags='_UP', rtags='_DOWN', method=vmethod, inplace=True,
                max_bond=vmax_bond, **compress_opts)

            # cut joined bond by reindexing to upper- and lower- ind_id.
            kb.cut_between((mps.site_tag(section[0]), '_UP'),
                           (mps.site_tag(section[0]), '_DOWN'),
                           f"_tmp_ind_u{label}",
                           f"_tmp_ind_l{label}")

        else:
            # just unfold and fuse physical indices:
            #                              |
            #   -A-A-A-A-A-A-A-        -AAAAAAA-
            #    | | | | | | |   ===>
            #   -A-A-A-A-A-A-A-        -AAAAAAA-
            #                              |

            if verbosity >= 1:
                print(f"Just vertical unfolding section {label}.")

            kb, sec = kb.partition(section_tags, inplace=True)
            sec_l, sec_u = sec.partition('_KET', inplace=True)
            T_UP = (sec_u ^ all)
            T_UP.add_tag('_UP')
            T_UP.fuse_({f"_tmp_ind_u{label}":
                        [mps.site_ind(i) for i in section]})
            T_DN = (sec_l ^ all)
            T_DN.add_tag('_DOWN')
            T_DN.fuse_({f"_tmp_ind_l{label}":
                        [mps.site_ind(i) for i in section]})
            kb |= T_UP
            kb |= T_DN

    def partial_trace_compress(self, sysa, sysb, eps=1e-8,
                               method=('isvd', None), max_bond=(None, 1024),
                               leave_short=True, renorm=True,
                               lower_ind_id='b{}', verbosity=0,
                               **compress_opts):
        r"""Perform a compressed partial trace using singular value
        lateral then vertical decompositions of transfer matrix products::


                    .....sysa......     ...sysb....
            o-o-o-o-A-A-A-A-A-A-A-A-o-o-B-B-B-B-B-B-o-o-o-o-o-o-o-o-o
            | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

                                      ==> form inner product

                    ...............     ...........
            o-o-o-o-A-A-A-A-A-A-A-A-o-o-B-B-B-B-B-B-o-o-o-o-o-o-o-o-o
            | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
            o-o-o-o-A-A-A-A-A-A-A-A-o-o-B-B-B-B-B-B-o-o-o-o-o-o-o-o-o

                                      ==> lateral SVD on each section

                      .....sysa......     ...sysb....
                      /\             /\   /\         /\
              ... ~~~E  A~~~~~~~~~~~A  E~E  B~~~~~~~B  E~~~ ...
                      \/             \/   \/         \/

                                      ==> vertical SVD and unfold on A & B

                              |                 |
                      /-------A-------\   /-----B-----\
              ... ~~~E                 E~E             E~~~ ...
                      \-------A-------/   \-----B-----/
                              |                 |

        With various special cases including OBC or end spins included in
        subsytems.


        Parameters
        ----------
        sysa :  sequence of int
            The sites, which should be contiguous, defining subsystem A.
        sysb :  sequence of int
            The sites, which should be contiguous, defining subsystem B.
        eps : float or (float, float), optional
            Tolerance(s) to use when compressing the subsystem transfer
            matrices and vertically decomposing.
        method : str or (str, str), optional
            Method(s) to use for laterally compressing the state then
            vertially compressing subsytems.
        max_bond : int or (int, int), optional
            The maximum bond to keep for laterally compressing the state then
            vertially compressing subsytems.
        leave_short : bool, optional
            If True (the default), don't try to compress short sections.
        renorm : bool, optional
            If True (the default), renomalize the state so that ``tr(rho)==1``.
        lower_ind_id : str, optional
            The index id to create for the new density matrix, the upper_ind_id
            is automatically taken as the current site_ind_id.
        compress_opts : dict, optional
            If given, supplied to ``partial_trace_compress`` to govern how
            singular values are treated. See ``tensor_split``.
        verbosity : {0, 1}, optional
            How much information to print while performing the compressed
            partial trace.

        Returns
        -------
        rho_ab : TensorNetwork
            Density matrix tensor network with
            ``outer_inds = ('k0', 'k1', 'b0', 'b1')`` for example.
        """
        N = self.L

        if (len(sysa) + len(sysb) == N) and not self.cyclic:
            return self.bipartite_schmidt_state(len(sysa), get='rho')

        # parse horizontal and vertical svd tolerances and methods
        try:
            heps, veps = eps
        except (ValueError, TypeError):
            heps = veps = eps
        try:
            hmethod, vmethod = method
        except (ValueError, TypeError):
            hmethod = vmethod = method
        try:
            hmax_bond, vmax_bond = max_bond
        except (ValueError, TypeError):
            hmax_bond = vmax_bond = max_bond

        # the sequence of sites in each of the 'environment' sections
        envm = range(max(sysa) + 1, min(sysb))
        envl = range(0, min(sysa))
        envr = range(max(sysb) + 1, N)

        # spread norm, and if not cyclic put in mixed canonical form, taking
        # care that the orthogonality centre is in right place to use identity
        k = self.copy()
        k.left_canonize()
        k.right_canonize(max(sysa) + (bool(envm) or bool(envr)))

        # form the inner product
        b = k.conj()
        k.add_tag('_KET')
        b.add_tag('_BRA')
        kb = k | b

        # label the various partitions
        names = ('_ENVL', '_SYSA', '_ENVM', '_SYSB', '_ENVR')
        for name, where in zip(names, (envl, sysa, envm, sysb, envr)):
            if where:
                kb.add_tag(name, where=map(self.site_tag, where), which='any')

        if self.cyclic:
            # can combine right and left envs
            sections = [envm, sysa, sysb, (*envr, *envl)]
        else:
            sections = [envm]
            # if either system includes end, can ignore and use identity
            if 0 not in sysa:
                sections.append(sysa)
            if N - 1 not in sysb:
                sections.append(sysb)

        # ignore empty sections
        sections = list(filter(len, sections))

        # figure out the various indices
        ul_ur_ll_lrs = []
        for section in sections:

            #          ...section[i]....
            #   ul[i] -o-o-o-o-o-o-o-o-o- ur[i]
            #          | | | | | | | | |
            #   ll[i] -o-o-o-o-o-o-o-o-o- lr[i]

            st_left = self.site_tag(section[0] - 1)
            st_right = self.site_tag(section[0])
            ul, = bonds(kb['_KET', st_left], kb['_KET', st_right])
            ll, = bonds(kb['_BRA', st_left], kb['_BRA', st_right])

            st_left = self.site_tag(section[-1])
            st_right = self.site_tag(section[-1] + 1)
            ur, = bonds(kb['_KET', st_left], kb['_KET', st_right])
            lr, = bonds(kb['_BRA', st_left], kb['_BRA', st_right])

            ul_ur_ll_lrs.append((ul, ur, ll, lr))

        # lateral compress sections if long
        compressed = []
        for section, (ul, _, ll, _) in zip(sections, ul_ur_ll_lrs):
            self._do_lateral_compress(self, kb, section, leave_short, ul, ll,
                                      heps, hmethod, hmax_bond, verbosity,
                                      compressed, **compress_opts)

        # vertical compress and unfold system sections only
        for section, (ul, ur, ll, lr) in zip(sections, ul_ur_ll_lrs):
            self._do_vertical_decomp(self, kb, section, sysa, sysb, compressed,
                                     ul, ur, ll, lr, vmethod, vmax_bond, veps,
                                     verbosity, **compress_opts)

        if not self.cyclic:
            # check if either system is at end, and thus reduces to identities
            #
            #  A-A-A-A-A-A-A-m-m-m-            \-m-m-m-
            #  | | | | | | | | | |  ...  ==>     | | |  ...
            #  A-A-A-A-A-A-A-m-m-m-            /-m-m-m-
            #
            if 0 in sysa:
                # get neighbouring tensor
                if envm:
                    try:
                        TU = TD = kb['_ENVM', '_LEFT']
                    except KeyError:
                        # didn't lateral compress
                        TU = kb['_ENVM', '_KET', self.site_tag(envm[0])]
                        TD = kb['_ENVM', '_BRA', self.site_tag(envm[0])]
                else:
                    TU = kb['_SYSB', '_UP']
                    TD = kb['_SYSB', '_DOWN']
                ubnd, = kb['_KET', self.site_tag(sysa[-1])].bonds(TU)
                lbnd, = kb['_BRA', self.site_tag(sysa[-1])].bonds(TD)

                # delete the A system
                kb.delete('_SYSA')
                kb.reindex_({ubnd: "_tmp_ind_uA", lbnd: "_tmp_ind_lA"})
            else:
                # or else replace the left or right envs with identites since
                #
                #  >->->->-A-A-A-A-           +-A-A-A-A-
                #  | | | | | | | |  ...  ==>  | | | | |
                #  >->->->-A-A-A-A-           +-A-A-A-A-
                #
                kb.replace_with_identity('_ENVL', inplace=True)

            if N - 1 in sysb:
                # get neighbouring tensor
                if envm:
                    try:
                        TU = TD = kb['_ENVM', '_RIGHT']
                    except KeyError:
                        # didn't lateral compress
                        TU = kb['_ENVM', '_KET', self.site_tag(envm[-1])]
                        TD = kb['_ENVM', '_BRA', self.site_tag(envm[-1])]
                else:
                    TU = kb['_SYSA', '_UP']
                    TD = kb['_SYSA', '_DOWN']
                ubnd, = kb['_KET', self.site_tag(sysb[0])].bonds(TU)
                lbnd, = kb['_BRA', self.site_tag(sysb[0])].bonds(TD)

                # delete the B system
                kb.delete('_SYSB')
                kb.reindex_({ubnd: "_tmp_ind_uB", lbnd: "_tmp_ind_lB"})
            else:
                kb.replace_with_identity('_ENVR', inplace=True)

        kb.reindex_({
            '_tmp_ind_uA': self.site_ind('A'),
            '_tmp_ind_lA': lower_ind_id.format('A'),
            '_tmp_ind_uB': self.site_ind('B'),
            '_tmp_ind_lB': lower_ind_id.format('B'),
        })

        if renorm:
            # normalize
            norm = kb.trace(['kA', 'kB'], ['bA', 'bB'])

            ts = []
            tags = kb.tags

            # check if we have system A
            if '_SYSA' in tags:
                ts.extend(kb[sysa[0]])

            # check if we have system B
            if '_SYSB' in tags:
                ts.extend(kb[sysb[0]])

            # If we dont' have either (OBC with both at ends) use middle envm
            if len(ts) == 0:
                ts.extend(kb[envm[0]])

            nt = len(ts)

            if verbosity > 0:
                print(f"Renormalizing for norm {norm} among {nt} tensors.")

            # now spread the norm out among tensors
            for t in ts:
                t.modify(data=t.data / norm**(1 / nt))

        return kb

    def logneg_subsys(self, sysa, sysb, compress_opts=None,
                      approx_spectral_opts=None, verbosity=0,
                      approx_thresh=2**12):
        r"""Compute the logarithmic negativity between subsytem blocks, e.g.::

                               sysa         sysb
                             .........       .....
            ... -o-o-o-o-o-o-A-A-A-A-A-o-o-o-B-B-B-o-o-o-o-o-o-o- ...
                 | | | | | | | | | | | | | | | | | | | | | | | |

        Parameters
        ----------
        sysa :  sequence of int
            The sites, which should be contiguous, defining subsystem A.
        sysb :  sequence of int
            The sites, which should be contiguous, defining subsystem B.
        eps : float, optional
            Tolerance to use when compressing the subsystem transfer matrices.
        method : str or (str, str), optional
            Method(s) to use for laterally compressing the state then
            vertially compressing subsytems.
        compress_opts : dict, optional
            If given, supplied to ``partial_trace_compress`` to govern how
            singular values are treated. See ``tensor_split``.
        approx_spectral_opts
            Supplied to :func:`~quimb.approx_spectral_function`.

        Returns
        -------
        ln : float
            The logarithmic negativity.

        See Also
        --------
        MatrixProductState.partial_trace_compress, approx_spectral_function
        """
        if not self.cyclic and (len(sysa) + len(sysb) == self.L):
            # pure bipartition with OBC
            psi = self.bipartite_schmidt_state(len(sysa), get='ket-dense')
            d = round(psi.shape[0]**0.5)
            return qu.logneg(psi, [d, d])

        compress_opts = ensure_dict(compress_opts)
        approx_spectral_opts = ensure_dict(approx_spectral_opts)

        # set the default verbosity for each method
        compress_opts.setdefault('verbosity', verbosity)
        approx_spectral_opts.setdefault('verbosity', verbosity)

        # form the compressed density matrix representation
        rho_ab = self.partial_trace_compress(sysa, sysb, **compress_opts)

        # view it as an operator
        rho_ab_pt_lo = rho_ab.aslinearoperator(['kA', 'bB'], ['bA', 'kB'])

        if rho_ab_pt_lo.shape[0] <= approx_thresh:
            tr_norm = norm_trace_dense(rho_ab_pt_lo.to_dense(), isherm=True)
        else:
            # estimate its spectrum and sum the abs(eigenvalues)
            tr_norm = qu.approx_spectral_function(
                rho_ab_pt_lo, abs, **approx_spectral_opts)

        # clip below 0
        return max(0, log2(tr_norm))

    def measure(
        self,
        site,
        remove=False,
        outcome=None,
        renorm=True,
        cur_orthog=None,
        get=None,
        inplace=False,
    ):
        r"""Measure this MPS at ``site``, including projecting the state.
        Optionally remove the site afterwards, yielding an MPS with one less
        site. In either case the orthogonality center of the returned MPS is
        ``min(site, new_L - 1)``.

        Parameters
        ----------
        site : int
            The site to measure.
        remove : bool, optional
            Whether to remove the site completely after projecting the
            measurement. If ``True``, sites greater than ``site`` will be
            retagged and reindex one down, and the MPS will have one less site.
            E.g::

                0-1-2-3-4-5-6
                       / / /  - measure and remove site 3
                0-1-2-4-5-6
                              - reindex sites (4, 5, 6) to (3, 4, 5)
                0-1-2-3-4-5

        outcome : None or int, optional
            Specify the desired outcome of the measurement. If ``None``, it
            will be randomly sampled according to the local density matrix.
        renorm : bool, optional
            Whether to renormalize the state post measurement.
        cur_orthog : None or int, optional
            If you already know the orthogonality center, you can supply it
            here for efficiencies sake.
        get : {None, 'outcome'}, optional
            If ``'outcome'``, simply return the outcome, and don't perform any
            projection.
        inplace : bool, optional
            Whether to perform the measurement in place or not.

        Returns
        -------
        outcome : int
            The measurement outcome, drawn from ``range(phys_dim)``.
        psi : MatrixProductState
            The measured state, if ``get != 'outcome'``.
        """
        if self.cyclic:
            raise ValueError('Not supported on cyclic MPS yet.')

        tn = self if inplace else self.copy()
        L = tn.L
        d = self.phys_dim(site)

        # make sure MPS is canonicalized
        if cur_orthog is not None:
            tn.shift_orthogonality_center(cur_orthog, site)
        else:
            tn.canonize(site)

        # local tensor and physical dim
        t = tn[site]
        ind = tn.site_ind(site)

        # diagonal of reduced density matrix = probs
        tii = t.contract(t.H, output_inds=(ind,))
        p = do('real', tii.data)

        if outcome is None:
            # sample an outcome
            outcome = do('random.choice', do('arange', d, like=p), p=p)

        if get == 'outcome':
            return outcome

        # project the outcome and renormalize
        t.isel_({ind: outcome})

        if renorm:
            t.modify(data=t.data / p[outcome]**0.5)

        if remove:
            # contract the projected tensor into neighbor
            if site == L - 1:
                tn ^= slice(site - 1, site + 1)
            else:
                tn ^= slice(site, site + 2)

            # adjust structure for one less spin
            for i in range(site + 1, L):
                tn[i].reindex_({tn.site_ind(i): tn.site_ind(i - 1)})
                tn[i].retag_({tn.site_tag(i): tn.site_tag(i - 1)})
            tn._L = L - 1
        else:
            # simply re-expand tensor dimensions (with zeros)
            t.new_ind(ind, size=d, axis=-1)

        return outcome, tn

    measure_ = functools.partialmethod(measure, inplace=True)


class MatrixProductOperator(TensorNetwork1DOperator,
                            TensorNetwork1DFlat,
                            TensorNetwork1D,
                            TensorNetwork):
    """Initialise a matrix product operator, with auto labelling and tagging.

    Parameters
    ----------
    arrays : sequence of arrays
        The tensor arrays to form into a MPO.
    shape : str, optional
        String specifying layout of the tensors. E.g. 'lrud' (the default)
        indicates the shape corresponds left-bond, right-bond, 'up' physical
        index, 'down' physical index.
        End tensors have either 'l' or 'r' dropped from the string.
    upper_ind_id : str
        A string specifiying how to label the upper physical site indices.
        Should contain a ``'{}'`` placeholder. It is used to generate the
        actual indices like: ``map(upper_ind_id.format, range(len(arrays)))``.
    lower_ind_id : str
        A string specifiying how to label the lower physical site indices.
        Should contain a ``'{}'`` placeholder. It is used to generate the
        actual indices like: ``map(lower_ind_id.format, range(len(arrays)))``.
    site_tag_id : str
        A string specifiying how to tag the tensors at each site. Should
        contain a ``'{}'`` placeholder. It is used to generate the actual tags
        like: ``map(site_tag_id.format, range(len(arrays)))``.
    tags : str or sequence of str, optional
        Global tags to attach to all tensors.
    bond_name : str, optional
        The base name of the bond indices, onto which uuids will be added.
    """

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_upper_ind_id',
        '_lower_ind_id',
        'cyclic',
        '_L',
    )

    def __init__(self, arrays, shape='lrud', site_tag_id='I{}', tags=None,
                 upper_ind_id='k{}', lower_ind_id='b{}', bond_name="",
                 **tn_opts):
        # short-circuit for copying
        if isinstance(arrays, MatrixProductOperator):
            super().__init__(arrays)
            return

        arrays = tuple(arrays)

        self._L = len(arrays)

        # process site indices
        self._upper_ind_id = upper_ind_id
        self._lower_ind_id = lower_ind_id
        upper_inds = map(upper_ind_id.format, range(self.L))
        lower_inds = map(lower_ind_id.format, range(self.L))

        # process site tags
        self._site_tag_id = site_tag_id
        site_tags = map(site_tag_id.format, range(self.L))
        if tags is not None:
            if isinstance(tags, str):
                tags = (tags,)
            else:
                tags = tuple(tags)

            site_tags = tuple((st,) + tags for st in site_tags)

        self.cyclic = (ops.ndim(arrays[0]) == 4)

        # transpose arrays to 'lrud' order.
        def gen_orders():
            lud_ord = tuple(shape.replace('r', "").find(x) for x in 'lud')
            rud_ord = tuple(shape.replace('l', "").find(x) for x in 'rud')
            lrud_ord = tuple(map(shape.find, 'lrud'))
            yield rud_ord if not self.cyclic else lrud_ord
            for _ in range(self.L - 2):
                yield lrud_ord
            yield lud_ord if not self.cyclic else lrud_ord

        def gen_inds():
            cyc_bond = (rand_uuid(base=bond_name),) if self.cyclic else ()

            nbond = rand_uuid(base=bond_name)
            yield (*cyc_bond, nbond, next(upper_inds), next(lower_inds))
            pbond = nbond
            for _ in range(self.L - 2):
                nbond = rand_uuid(base=bond_name)
                yield (pbond, nbond, next(upper_inds), next(lower_inds))
                pbond = nbond
            yield (pbond, *cyc_bond, next(upper_inds), next(lower_inds))

        def gen_tensors():
            for array, site_tag, inds, order in zip(arrays, site_tags,
                                                    gen_inds(), gen_orders()):

                yield Tensor(transpose(array, order), inds=inds, tags=site_tag)

        super().__init__(gen_tensors(), virtual=True, **tn_opts)

    def add_MPO(self, other, inplace=False, compress=False, **compress_opts):
        """Add another MatrixProductState to this one.
        """
        if self.L != other.L:
            raise ValueError("Can't add MPO with another of different length."
                             f"Got lengths {self.L} and {other.L}")

        summed = self if inplace else self.copy()

        for i in summed.gen_site_coos():
            t1, t2 = summed[i], other[i]

            if set(t1.inds) != set(t2.inds):
                # Need to use bonds to match indices
                reindex_map = {}

                if i > 0 or self.cyclic:
                    pair = ((i - 1) % self.L, i)
                    reindex_map[other.bond(*pair)] = summed.bond(*pair)

                if i < summed.L - 1 or self.cyclic:
                    pair = (i, (i + 1) % self.L)
                    reindex_map[other.bond(*pair)] = summed.bond(*pair)

                t2 = t2.reindex(reindex_map)

            sum_inds = (summed.upper_ind(i), summed.lower_ind(i))
            t1.direct_product_(t2, sum_inds=sum_inds)

        if compress:
            summed.compress(**compress_opts)

        return summed

    add_MPO_ = functools.partialmethod(add_MPO, inplace=True)

    _apply_mps = tensor_network_apply_op_vec

    def _apply_mpo(self, other, compress=False, **compress_opts):
        A, B = self.copy(), other.copy()

        # align the indices and combine into a ladder
        A.upper_ind_id = B.upper_ind_id
        B.upper_ind_id = "__tmp{}__"
        A.lower_ind_id = "__tmp{}__"
        AB = A | B

        # contract each pair of tensors at each site
        for i in range(A.L):
            AB ^= A.site_tag(i)

        # convert back to MPO and fuse the double bonds
        AB.view_as_(
            MatrixProductOperator,
            upper_ind_id=A.upper_ind_id,
            lower_ind_id=B.lower_ind_id,
            cyclic=self.cyclic,
        )

        AB.fuse_multibonds_()

        # optionally compress
        if compress:
            AB.compress(**compress_opts)

        return AB

    def apply(self, other, compress=False, **compress_opts):
        r"""Act with this MPO on another MPO or MPS, such that the resulting
        object has the same tensor network structure/indices as ``other``.

        For an MPS::

                   | | | | | | | | | | | | | | | | | |
             self: A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A
                   | | | | | | | | | | | | | | | | | |
            other: x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x

                                   -->

                   | | | | | | | | | | | | | | | | | |   <- other.site_ind_id
              out: y=y=y=y=y=y=y=y=y=y=y=y=y=y=y=y=y=y

        For an MPO::

                   | | | | | | | | | | | | | | | | | |
             self: A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A
                   | | | | | | | | | | | | | | | | | |
            other: B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B-B
                   | | | | | | | | | | | | | | | | | |

                                   -->

                   | | | | | | | | | | | | | | | | | |   <- other.upper_ind_id
              out: C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C=C
                   | | | | | | | | | | | | | | | | | |   <- other.lower_ind_id

        The resulting TN will have the same structure/indices as ``other``, but
        probably with larger bonds (depending on compression).


        Parameters
        ----------
        other : MatrixProductOperator or MatrixProductState
            The object to act on.
        compress : bool, optional
            Whether to compress the resulting object.
        compress_opts
            Supplied to :meth:`TensorNetwork1DFlat.compress`.

        Returns
        -------
        MatrixProductOperator or MatrixProductState
        """
        if isinstance(other, MatrixProductState):
            return self._apply_mps(other, compress=compress, **compress_opts)
        elif isinstance(other, MatrixProductOperator):
            return self._apply_mpo(other, compress=compress, **compress_opts)
        else:
            raise TypeError("Can only Dot with a MatrixProductOperator or a "
                            f"MatrixProductState, got {type(other)}")

    dot = apply

    def permute_arrays(self, shape='lrud'):
        """Permute the indices of each tensor in this MPO to match ``shape``.
        This doesn't change how the overall object interacts with other tensor
        networks but may be useful for extracting the underlying arrays
        consistently. This is an inplace operation.

        Parameters
        ----------
        shape : str, optional
            A permutation of ``'lrud'`` specifying the desired order of the
            left, right, upper and lower (down) indices respectively.
        """
        for i in self.sites:
            inds = {'u': self.upper_ind(i), 'd': self.lower_ind(i)}
            if self.cyclic or i > 0:
                inds['l'] = self.bond(i, (i - 1) % self.L)
            if self.cyclic or i < self.L - 1:
                inds['r'] = self.bond(i, (i + 1) % self.L)
            inds = [inds[s] for s in shape if s in inds]
            self[i].transpose_(*inds)

    def trace(self, left_inds=None, right_inds=None):
        """Take the trace of this MPO.
        """
        if left_inds is None:
            left_inds = map(self.upper_ind, self.gen_site_coos())
        if right_inds is None:
            right_inds = map(self.lower_ind, self.gen_site_coos())

        return super().trace(left_inds, right_inds)

    def partial_transpose(self, sysa, inplace=False):
        """Perform the partial transpose on this MPO by swapping the bra and
        ket indices on sites in ``sysa``.

        Parameters
        ----------
        sysa : sequence of int or int
            The sites to transpose indices on.
        inplace : bool, optional
            Whether to perform the partial transposition inplace.

        Returns
        -------
        MatrixProductOperator
        """
        tn = self if inplace else self.copy()

        if isinstance(sysa, Integral):
            sysa = (sysa,)

        tmp_ind_id = "__tmp_{}__"

        tn.reindex_({tn.upper_ind(i): tmp_ind_id.format(i) for i in sysa})
        tn.reindex_({tn.lower_ind(i): tn.upper_ind(i) for i in sysa})
        tn.reindex_({tmp_ind_id.format(i): tn.lower_ind(i) for i in sysa})
        return tn

    def __add__(self, other):
        """MPO addition.
        """
        return self.add_MPO(other, inplace=False)

    def __iadd__(self, other):
        """In-place MPO addition.
        """
        return self.add_MPO(other, inplace=True)

    def __sub__(self, other):
        """MPO subtraction.
        """
        return self.add_MPO(-1 * other, inplace=False)

    def __isub__(self, other):
        """In-place MPO subtraction.
        """
        return self.add_MPO(-1 * other, inplace=True)

    @property
    def lower_inds(self):
        """An ordered tuple of the actual lower physical indices.
        """
        return tuple(map(self.lower_ind, self.gen_site_coos()))

    def rand_state(self, bond_dim, **mps_opts):
        """Get a random vector matching this MPO.
        """
        return qu.tensor.MPS_rand_state(
            self.L, bond_dim=bond_dim,
            phys_dim=[self.phys_dim(i) for i in self.sites],
            dtype=self.dtype, cyclic=self.cyclic, **mps_opts
        )

    def identity(self, **mpo_opts):
        """Get a identity matching this MPO.
        """
        return qu.tensor.MPO_identity_like(self, **mpo_opts)

    def show(self, max_width=None):
        l1 = ""
        l2 = ""
        l3 = ""
        num_can_l, num_can_r = self.count_canonized()
        for i in range(self.L - 1):
            bdim = self.bond_size(i, i + 1)
            strl = len(str(bdim))
            l1 += f"│{bdim}"
            l2 += (">" if i < num_can_l else
                   "<" if i >= self.L - num_can_r else
                   "●") + ("─" if bdim < 100 else "━") * strl
            l3 += "│" + " " * strl

        l1 += "│"
        l2 += "<" if num_can_r > 0 else "●"
        l3 += "│"

        if self.cyclic:
            bdim = self.bond_size(0, self.L - 1)
            bnd_str = ("─" if bdim < 100 else "━") * strl
            l1 = f" {bdim}{l1}{bdim} "
            l2 = f"+{bnd_str}{l2}{bnd_str}+"
            l3 = f" {' ' * strl}{l3}{' ' * strl} "

        print_multi_line(l1, l2, l3, max_width=max_width)


class Dense1D(TensorNetwork1DVector,
              TensorNetwork1D,
              TensorNetwork):
    """Mimics other 1D tensor network structures, but really just keeps the
    full state in a single tensor. This allows e.g. applying gates in the same
    way for quantum circuit simulation as lazily represented hilbert spaces.

    Parameters
    ----------
    array : array_like
        The full hilbert space vector - assumed to be made of equal hilbert
        spaces each of size ``phys_dim`` and will be reshaped as such.
    phys_dim : int, optional
        The hilbert space size of each site, default: 2.
    tags : sequence of str, optional
        Extra tags to add to the tensor network.
    site_ind_id : str, optional
        String formatter describing how to label the site indices.
    site_tag_id : str, optional
        String formatter describing how to label the site tags.
    tn_opts
        Supplied to :class:`~quimb.tensor.tensor_core.TensorNetwork`.
    """

    _EXTRA_PROPS = (
        '_site_ind_id',
        '_site_tag_id',
        '_L',
    )

    def __init__(self, array, phys_dim=2, tags=None,
                 site_ind_id='k{}', site_tag_id='I{}', **tn_opts):

        # copy short-circuit
        if isinstance(array, Dense1D):
            super().__init__(array)
            return

        # work out number of sites and sub-dimensions etc.
        self._L = qu.infer_size(array, base=phys_dim)
        dims = [phys_dim] * self.L
        data = ops.asarray(array).reshape(*dims)

        # process site indices
        self._site_ind_id = site_ind_id
        site_inds = [self.site_ind(i) for i in range(self.L)]

        # process site tags
        self._site_tag_id = site_tag_id
        site_tags = oset(self.site_tag(i) for i in range(self.L))

        if tags is not None:
            # mix in global tags
            site_tags = tags_to_oset(tags) | site_tags

        T = Tensor(data=data, inds=site_inds, tags=site_tags)

        super().__init__([T], virtual=True, **tn_opts)

    @classmethod
    def rand(cls, n, phys_dim=2, dtype=float, **dense1d_opts):
        """Create a random dense vector 'tensor network'.
        """
        array = qu.randn(phys_dim ** n, dtype=dtype)
        array /= qu.norm(array, 'fro')
        return cls(array, **dense1d_opts)


class SuperOperator1D(
    TensorNetwork1D,
    TensorNetwork,
):
    r"""A 1D tensor network super-operator class::

        0   1   2       n-1
        |   |   |        |     <-- outer_upper_ind_id
        O===O===O==     =O
        |\  |\  |\       |\     <-- inner_upper_ind_id
          )   )   ) ...    )   <-- K (size of local Kraus sum)
        |/  |/  |/       |/     <-- inner_lower_ind_id
        O===O===O==     =O
        |   | : |        |     <-- outer_lower_ind_id
              :
             chi (size of entangling bond dim)

    Parameters
    ----------
    arrays : sequence of arrays
        The data arrays defining the superoperator, this should be a sequence
        of 2n arrays, such that the first two correspond to the upper and lower
        operators acting on site 0 etc. The arrays should be 5 dimensional
        unless OBC conditions are desired, in which case the first two and last
        two should be 4-dimensional. The dimensions of array can be should
        match the ``shape`` option.

    """

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_outer_upper_ind_id',
        '_inner_upper_ind_id',
        '_inner_lower_ind_id',
        '_outer_lower_ind_id',
        'cyclic',
        '_L',
    )

    def __init__(
        self, arrays,
        shape='lrkud',
        site_tag_id='I{}',
        outer_upper_ind_id='kn{}',
        inner_upper_ind_id='k{}',
        inner_lower_ind_id='b{}',
        outer_lower_ind_id='bn{}',
        tags=None,
        tags_upper=None,
        tags_lower=None,
        **tn_opts,
    ):
        # short-circuit for copying
        if isinstance(arrays, SuperOperator1D):
            super().__init__(arrays)
            return

        arrays = tuple(arrays)
        self._L = len(arrays) // 2

        # process indices
        self._outer_upper_ind_id = outer_upper_ind_id
        self._inner_upper_ind_id = inner_upper_ind_id
        self._inner_lower_ind_id = inner_lower_ind_id
        self._outer_lower_ind_id = outer_lower_ind_id

        outer_upper_inds = map(outer_upper_ind_id.format, self.gen_site_coos())
        inner_upper_inds = map(inner_upper_ind_id.format, self.gen_site_coos())
        inner_lower_inds = map(inner_lower_ind_id.format, self.gen_site_coos())
        outer_lower_inds = map(outer_lower_ind_id.format, self.gen_site_coos())

        # process tags
        self._site_tag_id = site_tag_id
        tags = tags_to_oset(tags)
        tags_upper = tags_to_oset(tags_upper)
        tags_lower = tags_to_oset(tags_lower)

        def gen_tags():
            for site_tag in self.site_tags:
                yield (site_tag,) + tags + tags_upper
                yield (site_tag,) + tags + tags_lower

        self.cyclic = (ops.ndim(arrays[0]) == 5)

        # transpose arrays to 'lrkud' order
        #        u
        #        |
        #     l--O--r
        #        |\
        #        d k
        def gen_orders():
            lkud_ord = tuple(shape.replace('r', "").find(x) for x in 'lkud')
            rkud_ord = tuple(shape.replace('l', "").find(x) for x in 'rkud')
            lrkud_ord = tuple(map(shape.find, 'lrkud'))
            yield rkud_ord if not self.cyclic else lrkud_ord
            yield rkud_ord if not self.cyclic else lrkud_ord
            for _ in range(self.L - 2):
                yield lrkud_ord
                yield lrkud_ord
            yield lkud_ord if not self.cyclic else lrkud_ord
            yield lkud_ord if not self.cyclic else lrkud_ord

        def gen_inds():
            #                    |<- outer_upper_ind
            # cycU_ix or pU_ix --O-- nU_ix
            #                   /|<- inner_upper_ind
            #           k_ix ->(
            #                   \|<- inner_lower_ind
            # cycL_ix or pL_ix --O-- nL_ix
            #                    |<- outer_lower_ind
            if self.cyclic:
                cycU_ix, cycL_ix = (rand_uuid(),), (rand_uuid(),)
            else:
                cycU_ix, cycL_ix = (), ()
            nU_ix, nL_ix, k_ix = rand_uuid(), rand_uuid(), rand_uuid()
            yield (*cycU_ix, nU_ix, k_ix,
                   next(outer_upper_inds), next(inner_upper_inds))
            yield (*cycL_ix, nL_ix, k_ix,
                   next(outer_lower_inds), next(inner_lower_inds))
            pU_ix, pL_ix = nU_ix, nL_ix
            for _ in range(self.L - 2):
                nU_ix, nL_ix, k_ix = rand_uuid(), rand_uuid(), rand_uuid()
                yield (pU_ix, nU_ix, k_ix,
                       next(outer_upper_inds), next(inner_upper_inds))
                yield (pL_ix, nL_ix, k_ix,
                       next(outer_lower_inds), next(inner_lower_inds))
                pU_ix, pL_ix = nU_ix, nL_ix
            k_ix = rand_uuid()
            yield (pU_ix, *cycU_ix, k_ix,
                   next(outer_upper_inds), next(inner_upper_inds))
            yield (pL_ix, *cycL_ix, k_ix,
                   next(outer_lower_inds), next(inner_lower_inds))

        def gen_tensors():
            for array, tags, inds, order in zip(arrays, gen_tags(),
                                                gen_inds(), gen_orders()):
                yield Tensor(transpose(array, order), inds=inds, tags=tags)

        super().__init__(gen_tensors(), virtual=True, **tn_opts)

    @classmethod
    def rand(cls, n, K, chi, phys_dim=2, herm=True,
             cyclic=False, dtype=complex, **superop_opts):

        def gen_arrays():
            for i in range(n):
                shape = []
                if cyclic or (i != 0):
                    shape += [chi]
                if cyclic or (i != n - 1):
                    shape += [chi]
                shape += [K, phys_dim, phys_dim]
                data = qu.randn(shape=shape, dtype=dtype)
                yield data
                if herm:
                    yield data.conj()
                else:
                    yield qu.randn(shape=shape, dtype=dtype)

        arrays = map(ops.sensibly_scale, gen_arrays())

        return cls(arrays, **superop_opts)

    @property
    def outer_upper_ind_id(self):
        return self._outer_upper_ind_id

    @property
    def inner_upper_ind_id(self):
        return self._inner_upper_ind_id

    @property
    def inner_lower_ind_id(self):
        return self._inner_lower_ind_id

    @property
    def outer_lower_ind_id(self):
        return self._outer_lower_ind_id


class TNLinearOperator1D(spla.LinearOperator):
    r"""A 1D tensor network linear operator like::

                 start                 stop - 1
                   .                     .
                 :-O-O-O-O-O-O-O-O-O-O-O-O-:                 --+
                 : | | | | | | | | | | | | :                   |
                 :-H-H-H-H-H-H-H-H-H-H-H-H-:    acting on    --V
                 : | | | | | | | | | | | | :                   |
                 :-O-O-O-O-O-O-O-O-O-O-O-O-:                 --+
        left_inds^                         ^right_inds

    Like :class:`~quimb.tensor.tensor_core.TNLinearOperator`, but performs a
    structured contract from one end to the other than can handle very long
    chains possibly more efficiently by contracting in blocks from one end.


    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to turn into a ``LinearOperator``.
    left_inds : sequence of str
        The left indicies.
    right_inds : sequence of str
        The right indicies.
    start : int
        Index of starting site.
    stop : int
        Index of stopping site (does not include this site).
    ldims : tuple of int, optional
        If known, the dimensions corresponding to ``left_inds``.
    rdims : tuple of int, optional
        If known, the dimensions corresponding to ``right_inds``.

    See Also
    --------
    TNLinearOperator
    """

    def __init__(self, tn, left_inds, right_inds, start, stop,
                 ldims=None, rdims=None, is_conj=False, is_trans=False):
        self.tn = tn
        self.start, self.stop = start, stop

        if ldims is None or rdims is None:
            ind_sizes = tn.ind_sizes()
            ldims = tuple(ind_sizes[i] for i in left_inds)
            rdims = tuple(ind_sizes[i] for i in right_inds)

        self.left_inds, self.right_inds = left_inds, right_inds
        self.ldims, ld = ldims, qu.prod(ldims)
        self.rdims, rd = rdims, qu.prod(rdims)
        self.tags = self.tn.tags

        # conjugate inputs/ouputs rather all tensors if necessary
        self.is_conj = is_conj
        self.is_trans = is_trans
        self._conj_linop = None
        self._adjoint_linop = None
        self._transpose_linop = None

        super().__init__(dtype=self.tn.dtype, shape=(ld, rd))

    def _matvec(self, vec):
        in_data = reshape(vec, self.rdims)

        if self.is_conj:
            in_data = conj(in_data)

        if self.is_trans:
            i, f, s = self.start, self.stop, 1
        else:
            i, f, s = self.stop - 1, self.start - 1, -1

        # add the vector to the right of the chain
        tnc = self.tn | Tensor(in_data, self.right_inds, tags=['_VEC'])
        tnc.view_like_(self.tn)
        # tnc = self.tn.copy()
        # tnc |= Tensor(in_data, self.right_inds, tags=['_VEC'])

        # absorb it into the rightmost site
        tnc ^= ['_VEC', self.tn.site_tag(i)]

        # then do a structured contract along the whole chain
        out_T = tnc ^ slice(i, f, s)

        out_data = out_T.transpose_(*self.left_inds).data.ravel()
        if self.is_conj:
            out_data = conj(out_data)

        return out_data

    def _matmat(self, mat):
        d = mat.shape[-1]
        in_data = reshape(mat, (*self.rdims, d))

        if self.is_conj:
            in_data = conj(in_data)

        if self.is_trans:
            i, f, s = self.start, self.stop, 1
        else:
            i, f, s = self.stop - 1, self.start - 1, -1

        # add the vector to the right of the chain
        in_ix = (*self.right_inds, '_mat_ix')

        tnc = self.tn | Tensor(in_data, inds=in_ix, tags=['_VEC'])
        tnc.view_like_(self.tn)
        # tnc = self.tn.copy()
        # tnc |= Tensor(in_data, inds=in_ix, tags=['_VEC'])

        # absorb it into the rightmost site
        tnc ^= ['_VEC', self.tn.site_tag(i)]

        # then do a structured contract along the whole chain
        out_T = tnc ^ slice(i, f, s)

        out_ix = (*self.left_inds, '_mat_ix')
        out_data = reshape(out_T.transpose_(*out_ix).data, (-1, d))
        if self.is_conj:
            out_data = conj(out_data)

        return out_data

    def copy(self, conj=False, transpose=False):

        if transpose:
            inds = (self.right_inds, self.left_inds)
            dims = (self.rdims, self.ldims)
            is_trans = not self.is_trans
        else:
            inds = (self.left_inds, self.right_inds)
            dims = (self.ldims, self.rdims)
            is_trans = self.is_trans

        if conj:
            is_conj = not self.is_conj
        else:
            is_conj = self.is_conj

        return TNLinearOperator1D(self.tn, *inds, self.start, self.stop, *dims,
                                  is_conj=is_conj, is_trans=is_trans)

    def conj(self):
        if self._conj_linop is None:
            self._conj_linop = self.copy(conj=True)
        return self._conj_linop

    def _transpose(self):
        if self._transpose_linop is None:
            self._transpose_linop = self.copy(transpose=True)
        return self._transpose_linop

    def _adjoint(self):
        """Hermitian conjugate of this TNLO.
        """
        # cache the adjoint
        if self._adjoint_linop is None:
            self._adjoint_linop = self.copy(conj=True, transpose=True)
        return self._adjoint_linop

    def to_dense(self):
        T = self.tn ^ slice(self.start, self.stop)

        if self.is_conj:
            T = T.conj()

        return T.to_dense(self.left_inds, self.right_inds)

    @property
    def A(self):
        return self.to_dense()
