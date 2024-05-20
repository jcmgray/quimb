"""Core tensor network tools."""

import functools

import numpy as np
from opt_einsum.contract import _tensordot, _transpose, parse_backend

from ...utils import check_opt, oset
from ..tensor_core import (
    Tensor,
    TensorNetwork,
    _parse_split_opts,
    rand_uuid,
    tags_to_oset,
    tensor_balance_bond,
    tensor_canonize_bond,
    tensor_compress_bond,
    tensor_contract,
    tensor_split,
)
from .block_tools import (
    add_with_smudge,
    get_smudge_balance,
    inv_with_smudge,
)

# --------------------------------------------------------------------------- #
#                                Tensor Funcs                                 #
# --------------------------------------------------------------------------- #


def _launch_block_expression(
    expr, tensors, backend="auto", preserve_tensor=False, **kwargs
):
    if len(tensors) == 1:
        return tensors[0]
    evaluate_constants = kwargs.pop("evaluate_constants", False)
    if evaluate_constants:
        raise NotImplementedError

    contraction_list = expr.contraction_list
    tensors = list(tensors)
    operands = [Ta.data for Ta in tensors]
    backend = parse_backend(operands, backend)
    # Start contraction loop
    for _, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, _, _ = contraction
        tmp_operands = [tensors.pop(x) for x in inds]
        # Call tensordot (check if should prefer einsum, but only if available)
        input_str, results_index = einsum_str.split("->")
        input_left, input_right = input_str.split(",")
        contract_out = (oset(input_left) | oset(input_right)) - (
            oset(input_left) & oset(input_right)
        )

        if contract_out == oset(results_index):
            Ta, Tb = tmp_operands
            tensor_result = "".join(
                s for s in input_left + input_right if s not in idx_rm
            )
            # Find indices to contract over
            left_pos, right_pos = [], []
            for s in idx_rm:
                left_pos.append(input_left.find(s))
                right_pos.append(input_right.find(s))

            # Contract!
            new_view = _tensordot(
                Ta.data,
                Tb.data,
                axes=(tuple(left_pos), tuple(right_pos)),
                backend=backend,
            )

            o_ix = [ind for ind in Ta.inds if ind not in Tb.inds] + [
                ind for ind in Tb.inds if ind not in Ta.inds
            ]

            # Build a new view if needed
            if tensor_result != results_index:
                transpose = tuple(map(tensor_result.index, results_index))
                new_view = _transpose(
                    new_view, axes=transpose, backend=backend
                )
                o_ix = [o_ix[ix] for ix in transpose]

            o_tags = oset.union(Ta.tags, Tb.tags)
            if len(o_ix) != 0 or preserve_tensor:
                new_view = Ta.__class__(data=new_view, inds=o_ix, tags=o_tags)
        # Call einsum
        else:
            raise NotImplementedError(
                "Generic Einsum Operations not supported"
            )
        # Append new items and dereference what we can
        tensors.append(new_view)
        del tmp_operands, new_view
    return tensors[0]


def flip_pattern(pattern):
    string_inv = {"+": "-", "-": "+"}
    return "".join([string_inv[ix] for ix in pattern])


# --------------------------------------------------------------------------- #
#                                Tensor Class                                 #
# --------------------------------------------------------------------------- #


class BlockTensor(Tensor):
    __slots__ = ("_data", "_inds", "_tags", "_left_inds", "_owners")

    def expand_ind(self, ind, size):
        raise NotImplementedError

    def new_ind(self, name, size=1, axis=0):
        raise NotImplementedError

    @property
    def symmetry(self):
        return self.data.dq.__name__

    @property
    def net_symmetry(self):
        return self.data.dq

    @property
    def shape(self):
        """Return the "inflated" shape composed of maximal size for each leg"""
        return self.data.shape

    def astype(self, dtype, inplace=False):
        raise NotImplementedError

    def bond_info(self, ind):
        ax = self.inds.index(ind)
        return self.data.get_bond_info(ax)

    @property
    def H(self):
        t = self.copy()
        t.modify(data=t.data.dagger, inds=t.inds[::-1])
        return t

    def trace(self, ind1, ind2, inplace=False):
        """Trace index ``ind1`` with ``ind2``, removing both."""
        t = self if inplace else self.copy()

        old_inds, new_inds = [], []
        for ix in t.inds:
            if ix in (ind1, ind2):
                old_inds.append(ind1)
            else:
                old_inds.append(ix)
                new_inds.append(ix)
        old_inds, new_inds = tuple(old_inds), tuple(new_inds)
        ax1 = self.inds.index(ind1)
        ax2 = self.inds.index(ind2)
        new_data = t.data.trace(ax1, ax2)
        t.modify(data=new_data, inds=new_inds, left_inds=None)
        return t

    def sum_reduce(self, ind, inplace=False):
        raise NotImplementedError

    def collapse_repeated(self, inplace=False):
        raise NotImplementedError

    def direct_product(self, other, sum_inds=(), inplace=False):
        raise NotImplementedError

    def distance(self, other, **contract_opts):
        raise NotImplementedError

    def entropy(self, left_inds, method="svd"):
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
        """Frobenius norm of this tensor."""
        return self.data.norm()

    def symmetrize(self, ind1, ind2, inplace=False):
        raise NotImplementedError

    def unitize(self, left_inds=None, inplace=False, method="qr"):
        raise NotImplementedError

    def randomize(self, dtype=None, inplace=False, **randn_opts):
        raise NotImplementedError

    def flip(self, ind, inplace=False):
        raise NotImplementedError

    def multiply_index_diagonal(
        self,
        ind,
        x,
        inplace=False,
        location="front",
        flip_pattern=False,
        sqrt=False,
        inverse=False,
        smudge=0,
    ):
        if location not in ["front", "back"]:
            raise ValueError("invalid for the location of the diagonal")
        t = self if inplace else self.copy()
        ax = t.inds.index(ind)
        iax = {"front": 1, "back": 0}[location]
        if isinstance(x, Tensor):
            x = x.data
        if flip_pattern:
            x = x.copy()
            x.pattern = flip_pattern(x.pattern)
        if x.pattern[iax] == t.data.pattern[ax]:
            raise ValueError("Symmetry relations not compatible")
        if sqrt:
            x = sqrt(x)
        if inverse:
            x = inv_with_smudge(x, smudge)
        elif smudge != 0:
            x = add_with_smudge(x, smudge)

        if location == "front":
            out = np.tensordot(x, t.data, axes=((iax,), (ax,)))
            transpose_order = (
                list(range(1, ax + 1)) + [0] + list(range(ax + 1, t.ndim))
            )
        else:
            out = np.tensordot(t.data, x, axes=((ax,), (iax,)))
            transpose_order = (
                list(range(ax)) + [t.ndim - 1] + list(range(ax, t.ndim - 1))
            )
        data = np.transpose(out, transpose_order)
        data.shape = t.data.shape
        t.modify(data=data)
        return t

    multiply_index_diagonal_ = functools.partialmethod(
        multiply_index_diagonal, inplace=True
    )

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

    _EXTRA_PROPS = ()


@tensor_split.register(BlockTensor)
def tensor_split_block_tensor(
    T,
    left_inds,
    method="svd",
    get=None,
    absorb="both",
    max_bond=None,
    cutoff=1e-10,
    cutoff_mode="rel",
    renorm=None,
    ltags=None,
    rtags=None,
    stags=None,
    bond_ind=None,
    right_inds=None,
    qpn_info=None,
):
    if left_inds is None:
        left_inds = oset(T.inds) - oset(right_inds)
    else:
        left_inds = tags_to_oset(left_inds)

    if right_inds is None:
        right_inds = oset(T.inds) - oset(left_inds)

    _left_inds = [T.inds.index(ind) for ind in left_inds]
    _right_inds = [T.inds.index(ind) for ind in right_inds]

    if get == "values":
        raise NotImplementedError

    opts = _parse_split_opts(
        method, cutoff, absorb, max_bond, cutoff_mode, renorm
    )

    # ``s`` itself will be None unless ``absorb=None`` is specified
    if method == "svd":
        left, s, right = T.data.tensor_svd(
            _left_inds, right_idx=_right_inds, **opts
        )
    elif method == "qr":
        if absorb == "left":
            mod = "lq"
        else:
            mod = "qr"
        s = None
        left, right = T.data.tensor_qr(
            _left_inds, right_idx=_right_inds, mod=mod
        )
    else:
        raise NotImplementedError

    if get == "arrays":
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
                    raise ValueError(
                        "for absorb=None, bond_ind must be a tuple/list of two strings"
                    )
        else:
            if isinstance(bond_ind, str):
                bond_ind = (bond_ind,)

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

    if get == "tensors":
        return tensors

    return BlockTensorNetwork(tensors, check_collisions=False)


@tensor_canonize_bond.register(BlockTensor)
def tensor_canonize_bond_block_tensor(T1, T2, absorb="right", **split_opts):
    check_opt("absorb", absorb, ("left", "both", "right"))

    if absorb == "both":
        split_opts.setdefault("cutoff", 0.0)
        return tensor_compress_bond(T1, T2, **split_opts)

    split_opts.setdefault("method", "qr")
    shared_ix, left_env_ix = T1.filter_bonds(T2)

    if absorb == "right":
        new_T1, tRfact = T1.split(
            left_env_ix, get="tensors", absorb=absorb, **split_opts
        )
        new_T2 = tRfact.contract(T2)
    else:
        tLfact, new_T2 = T2.split(
            shared_ix, get="tensors", absorb=absorb, **split_opts
        )
        new_T1 = T1.contract(tLfact)

    T1.modify(data=new_T1.data, inds=new_T1.inds)
    T2.modify(data=new_T2.data, inds=new_T2.inds)


@tensor_compress_bond.register(BlockTensor)
def tensor_compress_bond_block_tensor(
    T1, T2, reduced=True, absorb="both", info=None, **compress_opts
):
    shared_ix, left_env_ix = T1.filter_bonds(T2)
    if not shared_ix:
        raise ValueError("The tensors specified don't share an bond.")

    if reduced:
        # a) -> b)
        T1_L, T1_R = T1.split(
            left_inds=left_env_ix,
            right_inds=shared_ix,
            absorb="right",
            get="tensors",
            method="qr",
        )
        T2_L, T2_R = T2.split(
            left_inds=shared_ix, absorb="left", get="tensors", method="qr"
        )
        # b) -> c)
        M = T1_R @ T2_L
        M.drop_tags()
        # c) -> d)
        M_L, *s, M_R = M.split(
            left_inds=T1_L.bonds(M),
            get="tensors",
            absorb=absorb,
            **compress_opts,
        )

        # make sure old bond being used
        (ns_ix,) = M_L.bonds(M_R)
        M_L.reindex_({ns_ix: shared_ix[0]})
        M_R.reindex_({ns_ix: shared_ix[0]})

        # d) -> e)
        T1C = T1_L.contract(M_L)
        T2C = M_R.contract(T2_R)
    else:
        T12 = T1 @ T2
        T1C, *s, T2C = T12.split(
            left_inds=left_env_ix,
            get="tensors",
            absorb=absorb,
            **compress_opts,
        )
        T1C.transpose_like_(T1)
        T2C.transpose_like_(T2)

    # update with the new compressed data
    T1.modify(data=T1C.data, inds=T1C.inds)
    T2.modify(data=T2C.data, inds=T2C.inds)

    if s and info is not None:
        (info["singular_values"],) = s


@tensor_balance_bond.register(BlockTensor)
def tensor_balance_bond_block_tensor(t1, t2, smudge=1e-6):
    (ix,) = t1.bonds(t2)
    t1H = t1.H.reindex_({ix: ix + "*"})
    t2H = t2.H.reindex_({ix: ix + "*"})
    out1 = tensor_contract(t1H, t1.copy(), inplace=False)
    out2 = tensor_contract(t2H, t2.copy(), inplace=False)
    s1, s2 = get_smudge_balance(out1, out2, ix, smudge)
    t1.multiply_index_diagonal_(ix, s1, location="back")
    t2.multiply_index_diagonal_(ix, s2, location="front")


# --------------------------------------------------------------------------- #
#                            Tensor Network Class                             #
# --------------------------------------------------------------------------- #


class BlockTensorNetwork(TensorNetwork):
    _EXTRA_PROPS = ()
    _CONTRACT_STRUCTURED = False

    def trace(self, left_inds, right_inds, **contract_opts):
        """Trace over ``left_inds`` joined with ``right_inds``"""
        tn = self.copy()
        _left_inds = []
        _right_inds = []
        out = None
        for u, l in zip(left_inds, right_inds):
            (T1,) = tn._inds_get(u)
            (T2,) = tn._inds_get(l)
            if T1 is T2:
                out = T1.trace(u, l, inplace=True)
            else:
                _left_inds.append(u)
                _right_inds.append(l)
        if _left_inds:
            tn.reindex_({u: l for u, l in zip(_left_inds, _right_inds)})
            return tn.contract_tags(..., inplace=True, **contract_opts)
        else:
            if tn.outer_inds():
                return tn
            else:
                return out.data

    def replace_with_identity(self, where, which="any", inplace=False):
        raise NotImplementedError

    def replace_with_svd(
        self,
        where,
        left_inds,
        eps,
        *,
        which="any",
        right_inds=None,
        method="isvd",
        max_bond=None,
        absorb="both",
        cutoff_mode="rel",
        renorm=None,
        ltags=None,
        rtags=None,
        keep_tags=True,
        start=None,
        stop=None,
        inplace=False,
    ):
        raise NotImplementedError

    def replace_section_with_svd(
        self, start, stop, eps, **replace_with_svd_opts
    ):
        raise NotImplementedError

    def convert_to_zero(self):
        raise NotImplementedError

    def new_bond(self, tags1, tags2, **opts):
        raise NotImplementedError

    def insert_gauge(self, U, where1, where2, Uinv=None, tol=1e-10):
        raise NotImplementedError

    def __matmul__(self, other):
        """Overload "@" to mean full contraction with another network."""
        return BlockTensorNetwork((self, other)) ^ ...

    def aslinearoperator(
        self,
        left_inds,
        right_inds,
        ldims=None,
        rdims=None,
        backend=None,
        optimize="auto",
    ):
        raise NotImplementedError

    def to_dense(self, *inds_seq, to_qarray=True, **contract_opts):
        raise NotImplementedError

    def distance(self, *args, **kwargs):
        raise NotImplementedError

    def fit(
        self,
        tn_target,
        method="als",
        tol=1e-9,
        inplace=False,
        progbar=False,
        **fitting_opts,
    ):
        raise NotImplementedError

    # --------------- information about indices and dimensions -------------- #

    def squeeze(self, fuse=False, inplace=False):
        raise NotImplementedError

    def unitize(self, mode="error", inplace=False, method="qr"):
        raise NotImplementedError

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
