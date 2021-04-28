"""Core tensor network tools.
"""
import os
import copy
import functools

import numpy as np

from ..utils import (check_opt, oset)
from .drawing import draw_tn

from .tensor_core import Tensor, TensorNetwork, _parse_split_opts, oset_union, tags_to_oset, rand_uuid, _parse_split_opts
from .tensor_core import tensor_contract as _tensor_contract
from .block_tools import apply, get_smudge_balance
from .block_interface import dispatch_settings

# --------------------------------------------------------------------------- #
#                                Tensor Funcs                                 #
# --------------------------------------------------------------------------- #

def _core_contract(T1, T2):
    conc = [ind for ind in T1.inds if ind in T2.inds]
    ax1 = [T1.inds.index(ind) for ind in conc]
    ax2 = [T2.inds.index(ind) for ind in conc]
    o_array = np.tensordot(T1.data, T2.data, (ax1, ax2))
    o_ix = tuple([ind for ind in T1.inds+T2.inds if ind not in conc])
    o_tags = oset.union(T1.tags, T2.tags)
    if len(o_ix) == 0:
        return o_array
    else:
        return T1.__class__(data=o_array, inds=o_ix, tags=o_tags)

def tensor_contract(*tensors, output_inds=None, **contract_opts):
    path_info = _tensor_contract(*tensors, get='path-info', **contract_opts)
    tensors = list(tensors)
    for conc in path_info.contraction_list:
        pos1, pos2 = sorted(conc[0])
        T2 = tensors.pop(pos2)
        T1 = tensors.pop(pos1)
        out = _core_contract(T1, T2)
        tensors.append(out)

    if not isinstance(out, (float, complex)):
        _output_inds = out.inds
        if output_inds is None:
            output_inds = _output_inds
        else:
            output_inds = tuple(output_inds)
        if output_inds!=_output_inds:
            out.transpose_(*output_inds)
    return out

def tensor_split(
    T,
    left_inds,
    method='svd',
    get=None,
    absorb='both',
    max_bond=None,
    cutoff=1e-10,
    cutoff_mode='rel',
    renorm=None,
    ltags=None,
    rtags=None,
    stags=None,
    bond_ind=None,
    right_inds=None,
    qpn_info = None,
):
    if left_inds is None:
        left_inds = oset(T.inds) - oset(right_inds)
    else:
        left_inds = tags_to_oset(left_inds)

    if right_inds is None:
        right_inds = oset(T.inds) - oset(left_inds)

    _left_inds = [T.inds.index(ind) for ind in left_inds]
    _right_inds = [T.inds.index(ind) for ind in right_inds]

    if get == 'values':
        raise NotImplementedError

    opts = _parse_split_opts(
        method, cutoff, absorb, max_bond, cutoff_mode, renorm)

    # ``s`` itself will be None unless ``absorb=None`` is specified
    if method == "svd":
        left, s, right = T.data.tensor_svd(_left_inds, right_idx=_right_inds, **opts)
    elif method == "qr":
        mod = {"right":"qr", "left":"lq"}[absorb]
        s = None
        left, right = T.data.tensor_qr(_left_inds, right_idx=_right_inds, mod=mod)
    else:
        raise NotImplementedError


    if get == 'arrays':
        if absorb is None:
            return left, s, right
        return left, right

    if bond_ind is None:
        if absorb is None:
            bond_ind = (rand_uuid(), rand_uuid())
        else:
            bond_ind = (rand_uuid(),)
    else:
        if absorb is None:
            if isinstance(bond_ind, str):
                bond_ind = (bond_ind, rand_uuid())
            else:
                if len(bond_ind) != 2:
                    raise ValueError("for absorb=None, bond_ind must be a tuple/list of two strings")
        else:
            if isinstance(bond_ind, str):
                bond_ind = (bond_ind, )

    ltags = T.tags | tags_to_oset(ltags)
    rtags = T.tags | tags_to_oset(rtags)

    Tl = T.__class__(data=left, inds=(*left_inds, bond_ind[0]), tags=ltags)
    Tr = T.__class__(data=right, inds=(bond_ind[-1], *right_inds), tags=rtags)

    if absorb is None:
        stags = T.tags | tags_to_oset(stags)
        Ts = T.__class__(data=s, inds=bond_ind, tags=stags)
        tensors = (Tl, Ts, Tr)
    else:
        tensors = (Tl, Tr)

    if get == 'tensors':
        return tensors

    return BlockTensorNetwork(tensors, check_collisions=False)

def tensor_canonize_bond(T1, T2, absorb='right', **split_opts):
    check_opt('absorb', absorb, ('left', 'both', 'right'))

    if absorb == 'both':
        split_opts.setdefault('cutoff', 0.0)
        return tensor_compress_bond(T1, T2, **split_opts)

    split_opts.setdefault('method', 'qr')
    shared_ix, left_env_ix = T1.filter_bonds(T2)

    if absorb == "right":
        new_T1, tRfact = T1.split(left_env_ix, get='tensors', absorb=absorb, **split_opts)
        new_T2 = T2.contract(tRfact)
    else:
        tLfact, new_T2 = T2.split(shared_ix, get="tensors", absorb=absorb, **split_opts)
        new_T1 = T1.contract(tLfact)

    T1.modify(data=new_T1.data, inds=new_T1.inds)
    T2.modify(data=new_T2.data, inds=new_T2.inds)

def tensor_compress_bond(
    T1,
    T2,
    reduced=True,
    absorb='both',
    info=None,
    **compress_opts
):
    shared_ix, left_env_ix = T1.filter_bonds(T2)
    if not shared_ix:
        raise ValueError("The tensors specified don't share an bond.")

    if reduced:
        # a) -> b)
        T1_L, T1_R = T1.split(left_inds=left_env_ix, right_inds=shared_ix, absorb="right",
                              get='tensors', method='qr')
        T2_L, T2_R = T2.split(left_inds=shared_ix, absorb="left", get='tensors', method='qr')
        # b) -> c)
        M = (T1_R @ T2_L)
        M.drop_tags()
        # c) -> d)
        M_L, *s, M_R = M.split(left_inds=T1_L.bonds(M), get='tensors',
                               absorb=absorb, **compress_opts)

        # make sure old bond being used
        ns_ix, = M_L.bonds(M_R)
        M_L.reindex_({ns_ix: shared_ix[0]})
        M_R.reindex_({ns_ix: shared_ix[0]})

        # d) -> e)
        T1C = T1_L.contract(M_L)
        T2C = M_R.contract(T2_R)
    else:
        T12 = T1 @ T2
        T1C, *s, T2C = T12.split(left_inds=left_env_ix, get='tensors',
                                 absorb=absorb, **compress_opts)
        T1C.transpose_like_(T1)
        T2C.transpose_like_(T2)

    # update with the new compressed data
    T1.modify(data=T1C.data, inds=T1C.inds)
    T2.modify(data=T2C.data, inds=T2C.inds)

    if s and info is not None:
        info['singular_values'], = s


def tensor_balance_bond(t1, t2, smudge=1e-6):
    ix, = t1.bonds(t2)
    t1H = t1.H.reindex_({ix: ix+'*'})
    t2H = t2.H.reindex_({ix: ix+'*'})
    out1 = _core_contract(t1H, t1)
    out2 = _core_contract(t2H, t2)
    s1, s2 = get_smudge_balance(out1, out2, ix, smudge)
    t1.multiply_index_diagonal_(ix, s1, location="back")
    t2.multiply_index_diagonal_(ix, s2, location="front")

# --------------------------------------------------------------------------- #
#                                Tensor Class                                 #
# --------------------------------------------------------------------------- #

class BlockTensor(Tensor):

    __slots__ = ('_data', '_inds', '_tags', '_left_inds', '_owners')

    def _apply_function(self, fn):
        self._data = apply(self.data, fn)

    def expand_ind(self, ind, size):
        raise NotImplementedError

    def new_ind(self, name, size=1, axis=0):
        raise NotImplementedError

    @property
    def shape(self):
        """Return the "inflated" shape composed of maximal size for each leg
        """
        return self.data.shape

    def astype(self, dtype, inplace=False):
        raise NotImplementedError

    def ind_size(self, ind):
        ax = self.inds.index(ind)
        return self.get_bond_info(ax)

    def conj(self, inplace=False):
        """Conjugate this tensors data (does nothing to indices).
        """
        t = self if inplace else self.copy()
        t.modify(data=t.data.conj())
        return t

    conj_ = functools.partialmethod(conj, inplace=True)

    @property
    def H(self):
        t = self.copy()
        t.modify(data=t.data.dagger, inds=t.inds[::-1])
        return t

    def transpose(self, *output_inds, inplace=False):
        t = self if inplace else self.copy()

        output_inds = tuple(output_inds)  # need to re-use this.

        if set(t.inds) != set(output_inds):
            raise ValueError("'output_inds' must be permutation of the current"
                             f" tensor indices, but {set(t.inds)} != "
                             f"{set(output_inds)}")

        current_ind_map = {ind: i for i, ind in enumerate(t.inds)}
        out_shape = tuple(current_ind_map[i] for i in output_inds)

        t.modify(data=np.transpose(t.data, out_shape), inds=output_inds)
        return t

    transpose_ = functools.partialmethod(transpose, inplace=True)

    def trace(self, ind1, ind2, inplace=False):
        raise NotImplementedError

    def sum_reduce(self, ind, inplace=False):
        raise NotImplementedError

    def collapse_repeated(self, inplace=False):
        raise NotImplementedError

    def contract(self, *others, output_inds=None, **opts):
        return tensor_contract(self, *others, output_inds=output_inds, **opts)

    def direct_product(self, other, sum_inds=(), inplace=False):
        raise NotImplementedError

    def split(self, *args, **kwargs):
        return tensor_split(self, *args, **kwargs)

    def distance(self, other, **contract_opts):
        raise NotImplementedError

    def entropy(self, left_inds, method='svd'):
        raise NotImplementedError

    def fuse(self, fuse_map, inplace=False):
        raise NotImplementedError

    def unfuse(self, unfuse_map, shape_map, inplace=False):
        raise NotImplementedError

    def to_dense(self, *inds_seq, to_qarray=True):
        raise NotImplementedError

    def squeeze(self, include=None, inplace=False):
        raise NotImplementedError

    def norm(self):
        """Frobenius norm of this tensor.
        """
        return self.data.norm()

    def symmetrize(self, ind1, ind2, inplace=False):
        raise NotImplementedError

    def unitize(self, left_inds=None, inplace=False, method='qr'):
        raise NotImplementedError

    def randomize(self, dtype=None, inplace=False, **randn_opts):
        raise NotImplementedError

    def flip(self, ind, inplace=False):
        raise NotImplementedError

    def multiply_index_diagonal(self, ind, x, inplace=False, location="front"):
        if location not in ["front", "back"]:
            raise ValueError("invalid for the location of the diagonal")
        t = self if inplace else self.copy()
        ax = t.inds.index(ind)
        if isinstance(x, Tensor):
            x = x.data
        if location=="front":
            out = np.tensordot(x, t.data, axes=((1,), (ax,)))
            transpose_order = list(range(1, ax+1)) + [0] + list(range(ax+1, t.ndim))
        else:
            out = np.tensordot(t.data, x, axes=((ax,),(0,)))
            transpose_order = list(range(ax)) + [t.ndim-1] + list(range(ax, t.ndim-1))
        data = np.transpose(out, transpose_order)
        t.modify(data=data)
        return t

    multiply_index_diagonal_ = functools.partialmethod(
        multiply_index_diagonal, inplace=True)

    def almost_equals(self, other, **kwargs):
        raise NotImplementedError

    def __and__(self, other):
        """Combine with another ``Tensor`` or ``TensorNetwork`` into a new
        ``TensorNetwork``.
        """
        return BlockTensorNetwork((self, other))

    def __or__(self, other):
        """Combine virtually (no copies made) with another ``Tensor`` or
        ``TensorNetwork`` into a new ``TensorNetwork``.
        """
        return BlockTensorNetwork((self, other), virtual=True)


    def draw(self, *args, **kwargs):
        """Plot a graph of this tensor and its indices.
        """
        draw_tn(BlockTensorNetwork((self,)), *args, **kwargs)

    graph = draw

# --------------------------------------------------------------------------- #
#                            Tensor Network Class                             #
# --------------------------------------------------------------------------- #

class BlockTensorNetwork(TensorNetwork):

    __slots__ = ('_inner_inds', '_outer_inds', '_tid_counter')
    _EXTRA_PROPS = ()
    _CONTRACT_STRUCTURED = False

    def replace_with_identity(self, where, which='any', inplace=False):
        raise NotImplementedError

    def replace_with_svd(self, where, left_inds, eps, *, which='any',
                         right_inds=None, method='isvd', max_bond=None,
                         absorb='both', cutoff_mode='rel', renorm=None,
                         ltags=None, rtags=None, keep_tags=True,
                         start=None, stop=None, inplace=False):
        raise NotImplementedError

    def replace_section_with_svd(self, start, stop, eps,
                                 **replace_with_svd_opts):
        raise NotImplementedError

    def convert_to_zero(self):
        raise NotImplementedError

    def contract_between(self, tags1, tags2, **contract_opts):
        tid1, = self._get_tids_from_tags(tags1, which='all')
        tid2, = self._get_tids_from_tags(tags2, which='all')

        # allow no-op for same tensor specified twice ('already contracted')
        if tid1 == tid2:
            return

        T1 = self._pop_tensor(tid1)
        T2 = self._pop_tensor(tid2)
        T12 = tensor_contract(T1, T2, **contract_opts)
        self.add_tensor(T12, tid=tid2, virtual=True)

    def contract_ind(self, ind, **contract_opts):
        """Contract tensors connected by ``ind``.
        """
        tids = self._get_tids_from_inds(ind)
        ts = [self._pop_tensor(tid) for tid in tids]
        self |= tensor_contract(*ts, **contract_opts)

    def _compress_between_tids(
        self,
        tid1,
        tid2,
        canonize_distance=None,
        canonize_opts=None,
        equalize_norms=False,
        **compress_opts
    ):
        Tl = self.tensor_map[tid1]
        Tr = self.tensor_map[tid2]

        if canonize_distance:
            raise NotImplementedError

        tensor_compress_bond(Tl, Tr, **compress_opts)

        if equalize_norms:
            self.strip_exponent(tid1, equalize_norms)
            self.strip_exponent(tid2, equalize_norms)

    def _canonize_between_tids(
        self,
        tid1,
        tid2,
        equalize_norms=False,
        **canonize_opts,
    ):
        Tl = self.tensor_map[tid1]
        Tr = self.tensor_map[tid2]
        tensor_canonize_bond(Tl, Tr, **canonize_opts)

        if equalize_norms:
            self.strip_exponent(tid1, equalize_norms)
            self.strip_exponent(tid2, equalize_norms)

    def new_bond(self, tags1, tags2, **opts):
        raise NotImplementedError

    def insert_gauge(self, U, where1, where2, Uinv=None, tol=1e-10):
        raise NotImplementedError

    # ----------------------- contracting the network ----------------------- #
    def contract_tags(self, tags, inplace=False, which='any', **opts):
        """Contract the tensors that match any or all of ``tags``.

        Parameters
        ----------
        tags : sequence of str
            The list of tags to filter the tensors by. Use ``...``
            (``Ellipsis``) to contract all.
        inplace : bool, optional
            Whether to perform the contraction inplace.
        which : {'all', 'any'}
            Whether to require matching all or any of the tags.

        Returns
        -------
        TensorNetwork, Tensor or scalar
            The result of the contraction, still a ``TensorNetwork`` if the
            contraction was only partial.

        See Also
        --------
        contract, contract_cumulative, contract_structured
        """
        untagged_tn, tagged_ts = self.partition_tensors(
            tags, inplace=inplace, which=which)

        if not tagged_ts:
            raise ValueError("No tags were found - nothing to contract. "
                             "(Change this to a no-op maybe?)")

        contracted = tensor_contract(*tagged_ts, **opts)

        if untagged_tn is None:
            return contracted

        untagged_tn.add_tensor(contracted, virtual=True)
        return untagged_tn

    def contract(self, tags=..., inplace=False, **opts):
        if tags is all:
            return tensor_contract(*self, **opts)

        # this checks whether certain TN classes have a manually specified
        #     contraction pattern (e.g. 1D along the line)
        if self._CONTRACT_STRUCTURED:
            if (tags is ...) or isinstance(tags, slice):
                return self.contract_structured(tags, inplace=inplace, **opts)

        # else just contract those tensors specified by tags.
        return self.contract_tags(tags, inplace=inplace, **opts)

    contract_ = functools.partialmethod(contract, inplace=True)

    def __matmul__(self, other):
        """Overload "@" to mean full contraction with another network.
        """
        return BlockTensorNetwork((self, other)) ^ ...

    def aslinearoperator(self, left_inds, right_inds, ldims=None, rdims=None,
                         backend=None, optimize='auto'):
        raise NotImplementedError

    def to_dense(self, *inds_seq, to_qarray=True, **contract_opts):
        raise NotImplementedError

    def distance(self, *args, **kwargs):
        raise NotImplementedError

    def fit(
        self,
        tn_target,
        method='als',
        tol=1e-9,
        inplace=False,
        progbar=False,
        **fitting_opts
    ):
        raise NotImplementedError

    # --------------- information about indices and dimensions -------------- #

    def squeeze(self, fuse=False, inplace=False):
        raise NotImplementedError

    def unitize(self, mode='error', inplace=False, method='qr'):
        raise NotImplementedError

    def balance_bonds(self, inplace=False):
        tn = self if inplace else self.copy()

        for ix, tids in tn.ind_map.items():
            if len(tids) != 2:
                continue
            tid1, tid2 = tids
            t1, t2 = [tn.tensor_map[x] for x in (tid1, tid2)]
            tensor_balance_bond(t1, t2)

        return tn

    balance_bonds_ = functools.partialmethod(balance_bonds, inplace=True)

    def fuse_multibonds(self, inplace=False):
        raise NotImplementedError

    def rank_simplify(
        self,
        output_inds=None,
        equalize_norms=False,
        cache=None,
        inplace=False,
    ):
        raise NotImplementedError

    def diagonal_reduce(
        self,
        output_inds=None,
        atol=1e-12,
        cache=None,
        inplace=False,
    ):
        raise NotImplementedError
