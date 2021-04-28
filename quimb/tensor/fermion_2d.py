"""Classes and algorithms related to 2D tensor networks.
"""
import re
import functools
from operator import add
from itertools import product, cycle, starmap
from collections import defaultdict

import opt_einsum as oe

from ..utils import check_opt, pairwise
from .tensor_core import (
    bonds,
    rand_uuid,
    oset,
    tags_to_oset,
    oset_union,
)
from .tensor_2d import (
    Rotator2D,
    TensorNetwork2D,
    TensorNetwork2DVector,
    TensorNetwork2DFlat,
    TensorNetwork2DOperator,
    PEPS,
    PEPO,
    is_lone_coo,
    gen_long_range_path,
    calc_plaquette_sizes,
    calc_plaquette_map)
from .tensor_block import BlockTensorNetwork
from .fermion import FermionTensor, FermionTensorNetwork, tensor_contract

INVERSE_CUTOFF = 1e-10

class FermionTensorNetwork2D(FermionTensorNetwork, TensorNetwork2D):
    _EXTRA_PROPS = (
        '_site_tag_id',
        '_row_tag_id',
        '_col_tag_id',
        '_Lx',
        '_Ly',
    )

    def _compatible_2d(self, other):
        """Check whether ``self`` and ``other`` are compatible 2D tensor
        networks such that they can remain a 2D tensor network when combined.
        """
        return (
            isinstance(other, FermionTensorNetwork2D) and
            all(getattr(self, e) == getattr(other, e)
                for e in FermionTensorNetwork2D._EXTRA_PROPS)
        )

    def __and__(self, other):
        new = super().__and__(other)
        if self._compatible_2d(other):
            new.view_as_(FermionTensorNetwork2D, like=self)
        return new

    def __or__(self, other):
        new = super().__or__(other)
        if self._compatible_2d(other):
            new.view_as_(FermionTensorNetwork2D, like=self)
        return new

    def flatten(self, fuse_multibonds=True, inplace=False):
        raise NotImplementedError

    def reorder(self, direction, layer_tags=None, inplace=False):
        Lx, Ly = self._Lx, self._Ly
        tid_map = dict()
        current_position = 0
        if direction == "row":
            iterator = product(range(Lx), range(Ly))
        elif direction == "col":
            iterator = product(range(Ly), range(Lx))
        else:
            raise KeyError("direction not supported")

        for i, j in iterator:
            x, y = (i, j) if direction=="row" else (j, i)
            site_tag = self.site_tag(x, y)
            tids = self._get_tids_from_tags(site_tag)
            if len(tids) == 1:
                tid, = tids
                if tid not in tid_map:
                    tid_map[tid] = current_position
                    current_position +=1
            else:
                if layer_tags is None:
                    _tags = [self.tensor_map[ix].tags for ix in tids]
                    _tmp_tags = _tags[0].copy()
                    for itag in _tags[1:]:
                        _tmp_tags &= itag
                    _layer_tags = sorted([list(i-_tmp_tags)[0] for i in _tags])
                else:
                    _layer_tags = layer_tags
                for tag in _layer_tags:
                    tid, = self._get_tids_from_tags((site_tag, tag))
                    if tid not in tid_map:
                        tid_map[tid] = current_position
                        current_position += 1

        return self._reorder_from_tid(tid_map, inplace)

    def _contract_boundary_full_bond(
        self,
        xrange,
        yrange,
        from_which,
        max_bond,
        cutoff=0.0,
        method='eigh',
        renorm=True,
        optimize='auto-hq',
        opposite_envs=None,
        contract_boundary_opts=None,
    ):
        raise NotImplementedError

    def compute_environments(
        self,
        from_which,
        xrange=None,
        yrange=None,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode='mps',
        layer_tags=None,
        dense=False,
        compress_opts=None,
        envs=None,
        **contract_boundary_opts
    ):
        direction = {"left": "col",
                     "right": "col",
                     "top": "row",
                     "bottom": "row"}[from_which]
        tn = self.reorder(direction, layer_tags=layer_tags)

        r2d = Rotator2D(tn, xrange, yrange, from_which)
        sweep, row_tag = r2d.vertical_sweep, r2d.row_tag
        contract_boundary_fn = r2d.get_contract_boundary_fn()

        if envs is None:
            envs = {}

        if mode == 'full-bond':
            # set shared storage for opposite env contractions
            contract_boundary_opts.setdefault('opposite_envs', {})

        envs[from_which, sweep[0]] = FermionTensorNetwork([])
        first_row = row_tag(sweep[0])
        envs['mid', sweep[0]] = tn.select(first_row).copy()
        if dense:
            tn ^= first_row
        envs[from_which, sweep[1]] = tn.select(first_row).copy()

        for i in sweep[2:]:
            iprevprev = i - 2 * sweep.step
            iprev = i - sweep.step
            envs['mid', iprev] = tn.select(row_tag(iprev)).copy()
            if dense:
                tn ^= (row_tag(iprevprev), row_tag(iprev))
            else:
                contract_boundary_fn(
                    iprevprev, iprev,
                    max_bond=max_bond,
                    cutoff=cutoff,
                    mode=mode,
                    canonize=canonize,
                    layer_tags=layer_tags,
                    compress_opts=compress_opts,
                    **contract_boundary_opts,
                )

            envs[from_which, i] = tn.select(first_row).copy()

        return envs

    compute_bottom_environments = functools.partialmethod(
        compute_environments, from_which='bottom')

    compute_top_environments = functools.partialmethod(
        compute_environments, from_which='top')

    compute_left_environments = functools.partialmethod(
        compute_environments, from_which='left')

    compute_right_environments = functools.partialmethod(
        compute_environments, from_which='right')

    def _compute_plaquette_environments_row_first(
        self,
        x_bsz,
        y_bsz,
        max_bond=None,
        cutoff=1e-10,
        canonize=True,
        layer_tags=None,
        second_dense=None,
        row_envs=None,
        **compute_environment_opts
    ):
        if second_dense is None:
            second_dense = x_bsz < 2

        # first we contract from either side to produce column environments
        if row_envs is None:
            row_envs = self.compute_row_environments(
                max_bond=max_bond, cutoff=cutoff, canonize=canonize,
                layer_tags=layer_tags, **compute_environment_opts)

        # next we form vertical strips and contract from both top and bottom
        #     for each column
        col_envs = dict()
        for i in range(self.Lx - x_bsz + 1):
            #
            #      ●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●
            #     ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲
            #     o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o     ┬
            #     | | | | | | | | | | | | | | | | | | | |     ┊ x_bsz
            #     o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o     ┴
            #     ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱
            #      ●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●━━━●
            #
            row_i = FermionTensorNetwork((
                row_envs['bottom', i],
                *[row_envs['mid', i+x] for x in range(x_bsz)],
                row_envs['top', i + x_bsz - 1],
            )).view_as_(FermionTensorNetwork2D, like=self)
            #
            #           y_bsz
            #           <-->               second_dense=True
            #       ●──      ──●
            #       │          │            ╭──     ──╮
            #       ●── .  . ──●            │╭─ . . ─╮│     ┬
            #       │          │     or     ●         ●     ┊ x_bsz
            #       ●── .  . ──●            │╰─ . . ─╯│     ┴
            #       │          │            ╰──     ──╯
            #       ●──      ──●
            #     'left'    'right'       'left'    'right'
            #
            col_envs[i] = row_i.compute_col_environments(
                xrange=(max(i - 1, 0), min(i + x_bsz, self.Lx - 1)),
                max_bond=max_bond, cutoff=cutoff,
                canonize=canonize, layer_tags=layer_tags,
                dense=second_dense, **compute_environment_opts)

        # then range through all the possible plaquettes, selecting the correct
        # boundary tensors from either the column or row environments
        plaquette_envs = dict()
        for i0, j0 in product(range(self.Lx - x_bsz + 1),
                              range(self.Ly - y_bsz + 1)):

            # we want to select bordering tensors from:
            #
            #       L──A──A──R    <- A from the row environments
            #       │  │  │  │
            #  i0+1 L──●──●──R
            #       │  │  │  │    <- L, R from the column environments
            #  i0   L──●──●──R
            #       │  │  │  │
            #       L──B──B──R    <- B from the row environments
            #
            #         j0  j0+1
            #
            env_ij = FermionTensorNetwork((
                col_envs[i0]['left', j0],
                *[col_envs[i0]['mid', ix] for ix in range(j0, j0+y_bsz)],
                col_envs[i0]['right', j0 + y_bsz - 1]
            ), check_collisions=False)

            ij_tags = (self.site_tag(i0 +ix, j0 + iy) for ix in range(x_bsz) for iy in range(y_bsz))
            tid_lst = []
            for ij in ij_tags:
                tid_lst += list(env_ij._get_tids_from_tags(ij))
            position = range(len(env_ij.tensor_map)-len(tid_lst), len(env_ij.tensor_map))
            reorder_map = {i:j for i, j in zip(tid_lst, position)}
            env_ij._reorder_from_tid(reorder_map, inplace=True)
            plaquette_envs[(i0, j0), (x_bsz, y_bsz)] = env_ij

        return plaquette_envs

    def _compute_plaquette_environments_col_first(
        self,
        x_bsz,
        y_bsz,
        max_bond=None,
        cutoff=1e-10,
        canonize=True,
        layer_tags=None,
        second_dense=None,
        col_envs=None,
        **compute_environment_opts
    ):
        if second_dense is None:
            second_dense = y_bsz < 2

        # first we contract from either side to produce column environments
        if col_envs is None:
            col_envs = self.compute_col_environments(
                max_bond=max_bond, cutoff=cutoff, canonize=canonize,
                layer_tags=layer_tags, **compute_environment_opts)

        # next we form vertical strips and contract from both top and bottom
        #     for each column
        row_envs = dict()
        for j in range(self.Ly - y_bsz + 1):
            #
            #        y_bsz
            #        <-->
            #
            #      ╭─╱o─╱o─╮
            #     ●──o|─o|──●
            #     ┃╭─|o─|o─╮┃
            #     ●──o|─o|──●
            #     ┃╭─|o─|o─╮┃
            #     ●──o|─o|──●
            #     ┃╭─|o─|o─╮┃
            #     ●──o╱─o╱──●
            #     ┃╭─|o─|o─╮┃
            #     ●──o╱─o╱──●
            #
            col_j = FermionTensorNetwork((
                col_envs['left', j],
                *[col_envs['mid', j+jn] for jn in range(y_bsz)],
                col_envs['right', j + y_bsz - 1],
            )).view_as_(FermionTensorNetwork2D, like=self)
            #
            #        y_bsz
            #        <-->        second_dense=True
            #     ●──●──●──●      ╭──●──╮
            #     │  │  │  │  or  │ ╱ ╲ │    'top'
            #        .  .           . .                  ┬
            #                                            ┊ x_bsz
            #        .  .           . .                  ┴
            #     │  │  │  │  or  │ ╲ ╱ │    'bottom'
            #     ●──●──●──●      ╰──●──╯
            #
            row_envs[j] = col_j.compute_row_environments(
                yrange=(max(j - 1, 0), min(j + y_bsz, self.Ly - 1)),
                max_bond=max_bond, cutoff=cutoff, canonize=canonize,
                layer_tags=layer_tags, dense=second_dense,
                **compute_environment_opts)

        # then range through all the possible plaquettes, selecting the correct
        # boundary tensors from either the column or row environments
        plaquette_envs = dict()
        for i0, j0 in product(range(self.Lx - x_bsz + 1),
                              range(self.Ly - y_bsz + 1)):

            # we want to select bordering tensors from:
            #
            #          A──A──A──A    <- A from the row environments
            #          │  │  │  │
            #     i0+1 L──●──●──R
            #          │  │  │  │    <- L, R from the column environments
            #     i0   L──●──●──R
            #          │  │  │  │
            #          B──B──B──B    <- B from the row environments
            #
            #            j0  j0+1
            #
            env_ij = FermionTensorNetwork((
                row_envs[j0]['bottom', i0],
                *[row_envs[j0]['mid', ix] for ix in range(i0, i0+x_bsz)],
                row_envs[j0]['top', i0 + x_bsz - 1]
            ), check_collisions=False)

            ij_tags = (self.site_tag(i0 +ix, j0 + iy) for ix in range(x_bsz) for iy in range(y_bsz))
            tid_lst = []
            for ij in ij_tags:
                tid_lst += list(env_ij._get_tids_from_tags(ij))
            position = range(len(env_ij.tensor_map)-len(tid_lst), len(env_ij.tensor_map))
            reorder_map = {i:j for i, j in zip(tid_lst, position)}
            env_ij._reorder_from_tid(reorder_map, inplace=True)
            plaquette_envs[(i0, j0), (x_bsz, y_bsz)] = env_ij

        return plaquette_envs

def gate_string_split_(TG, where, string, original_ts, bonds_along,
                       reindex_map, site_ix, info, **compress_opts):
    # by default this means singuvalues are kept in the string 'blob' tensor
    compress_opts.setdefault('absorb', 'right')
    loc_info = dict([t.get_fermion_info() for t in original_ts])
    # the outer, neighboring indices of each tensor in the string
    neighb_inds = []

    # tensors we are going to contract in the blob, reindex some to attach gate
    contract_ts = []
    fermion_info = []
    qpn_infos = []

    for t, coo in zip(original_ts, string):
        neighb_inds.append(tuple(ix for ix in t.inds if ix not in bonds_along))
        contract_ts.append(t.reindex_(reindex_map) if coo in where else t)
        fermion_info.append(t.get_fermion_info())
        qpn_infos.append(t.data.dq)

    blob = tensor_contract(*contract_ts, TG, inplace=True)
    regauged = []
    work_site = blob.get_fermion_info()[1]
    fs = blob.fermion_owner[0]

    # one by one extract the site tensors again from each end
    inner_ts = [None] * len(string)
    i = 0
    j = len(string) - 1

    while True:
        lix = neighb_inds[i]
        if i > 0:
            lix += (bonds_along[i - 1],)

        # the original bond we are restoring
        bix = bonds_along[i]

        # split the blob!
        qpn_info = [blob.data.dq - qpn_infos[i], qpn_infos[i]]
        lix = tuple(oset(blob.inds)-oset(lix))
        blob, *maybe_svals, inner_ts[i] = blob.split(
            left_inds=lix, get='tensors', bond_ind=bix, qpn_info=qpn_info, **compress_opts)

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`

        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            #coo_pair = tuple(sorted((string[i], string[i + 1])))
            coo_pair = (string[i], string[i+1])
            info['singular_values', coo_pair] = s

            # regauge the blob but record so as to unguage later
            if i != j - 1:
                blob.multiply_index_diagonal_(bix, s, location="back")
                regauged.append((i + 1, bix, "back", s))

        # move inwards along string, terminate if two ends meet
        i += 1
        if i == j:
            inner_ts[i] = blob
            break

        # extract at end of string
        lix = neighb_inds[j]
        if j < len(string) - 1:
            lix += (bonds_along[j],)

        # the original bond we are restoring
        bix = bonds_along[j - 1]

        # split the blob!
        qpn_info = [qpn_infos[j], blob.data.dq - qpn_infos[j]]
        inner_ts[j], *maybe_svals, blob= blob.split(
            left_inds=lix, get='tensors', bond_ind=bix, qpn_info=qpn_info, **compress_opts)

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            coo_pair = (string[j-1], string[j])
            info['singular_values', coo_pair] = s

            # regauge the blob but record so as to ungauge later
            if j != i + 1:
                blob.multiply_index_diagonal_(bix, s, location="front")
                regauged.append((j - 1, bix, "front", s))

        # move inwards along string, terminate if two ends meet
        j -= 1
        if j == i:
            inner_ts[j] = blob
            break
    # SVD funcs needs to be modify and make sure S has even parity
    for i, bix, location, s in regauged:
        snew = inv_with_smudge(s, INVERSE_CUTOFF, gauge_smudge=0)
        t = inner_ts[i]
        t.multiply_index_diagonal_(bix, snew, location=location)

    for to, tn in zip(original_ts, inner_ts):
        tn.transpose_like_(to)
        to.modify(data=tn.data, inds=tn.inds)

    for i, (tid, _) in enumerate(fermion_info):
        if i==0:
            fs.replace_tensor(work_site, original_ts[i], tid=tid, virtual=True)
        else:
            fs.insert_tensor(work_site+i, original_ts[i], tid=tid, virtual=True)

    fs._reorder_from_dict(dict(fermion_info))

def gate_string_reduce_split_(TG, where, string, original_ts, bonds_along,
                       reindex_map, site_ix, info, **compress_opts):
    compress_opts.setdefault('absorb', 'right')

    # indices to reduce, first and final include physical indices for gate
    inds_to_reduce = [(bonds_along[0], site_ix[0])]
    for b1, b2 in pairwise(bonds_along):
        inds_to_reduce.append((b1, b2))
    inds_to_reduce.append((bonds_along[-1], site_ix[-1]))

    # tensors that remain on the string sites and those pulled into string
    outer_ts, inner_ts = [], []
    fermion_info = []
    fs = TG.fermion_owner[0]
    tid_lst = []
    for coo, rix, t in zip(string, inds_to_reduce, original_ts):
        tq, tr = t.split(left_inds=None, right_inds=rix,
                         method='qr', get='tensors', absorb="right")
        fermion_info.append(t.get_fermion_info())
        outer_ts.append(tq)
        inner_ts.append(tr.reindex_(reindex_map) if coo in where else tr)

    for tq, tr, t in zip(outer_ts, inner_ts, original_ts):
        isite = t.get_fermion_info()[1]
        fs.replace_tensor(isite, tr, virtual=True)
        fs.insert_tensor(isite+1, tq, virtual=True)

    blob = tensor_contract(*inner_ts, TG, inplace=True)
    work_site = blob.get_fermion_info()[1]
    regauged = []

    # extract the new reduced tensors sequentially from each end
    i = 0
    j = len(string) - 1

    while True:

        # extract at beginning of string
        lix = bonds(blob, outer_ts[i])
        if i == 0:
            lix.add(site_ix[0])
        else:
            lix.add(bonds_along[i - 1])

        # the original bond we are restoring
        bix = bonds_along[i]

        # split the blob!
        lix = tuple(oset(blob.inds)-oset(lix))
        blob, *maybe_svals, inner_ts[i] = blob.split(
            left_inds=lix, get='tensors', bond_ind=bix, **compress_opts)

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            coo_pair = (string[i], string[i + 1])
            info['singular_values', coo_pair] = s

            # regauge the blob but record so as to unguage later
            if i != j - 1:
                blob.multiply_index_diagonal_(bix, s, location="back")
                regauged.append((i + 1, bix, "back", s))

        # move inwards along string, terminate if two ends meet
        i += 1
        if i == j:
            inner_ts[i] = blob
            break

        # extract at end of string
        lix = bonds(blob, outer_ts[j])
        if j == len(string) - 1:
            lix.add(site_ix[-1])
        else:
            lix.add(bonds_along[j])

        # the original bond we are restoring
        bix = bonds_along[j - 1]

        # split the blob!
        inner_ts[j], *maybe_svals, blob = blob.split(
            left_inds=lix, get='tensors', bond_ind=bix, **compress_opts)
        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            coo_pair = (string[j - 1], string[j])
            info['singular_values', coo_pair] = s

            # regauge the blob but record so as to unguage later
            if j != i + 1:
                blob.multiply_index_diagonal_(bix, s, location="front")
                regauged.append((j - 1, bix, "front", s))

        # move inwards along string, terminate if two ends meet
        j -= 1
        if j == i:
            inner_ts[j] = blob
            break

    for i, (tid, _) in enumerate(fermion_info):
        if i==0:
            fs.replace_tensor(work_site, inner_ts[i], tid=tid, virtual=True)
        else:
            fs.insert_tensor(work_site+i, inner_ts[i], tid=tid, virtual=True)
    new_ts = [
        tensor_contract(ts, tr, inplace=True).transpose_like_(to)
        for to, ts, tr in zip(original_ts, outer_ts, inner_ts)
    ]

    for i, bix, location, s in regauged:
        snew = inv_with_smudge(s, INVERSE_CUTOFF, gauge_smudge=0)
        t = new_ts[i]
        t.multiply_index_diagonal_(bix, snew, location=location)

    for (tid, _), to, t in zip(fermion_info, original_ts, new_ts):
        site = t.get_fermion_info()[1]
        to.modify(data=t.data)
        fs.replace_tensor(site, to, tid=tid, virtual=True)

    fs._reorder_from_dict(dict(fermion_info))


class FermionTensorNetwork2DVector(FermionTensorNetwork2D,
                                   FermionTensorNetwork,
                                   TensorNetwork2DVector):
    """Mixin class  for a 2D square lattice vector TN, i.e. one with a single
    physical index per site.
    """

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_row_tag_id',
        '_col_tag_id',
        '_Lx',
        '_Ly',
        '_site_ind_id',
    )

    def to_dense(self, *inds_seq, **contract_opts):
        raise NotImplementedError

    def make_norm(
        self,
        mangle_append='*',
        layer_tags=('KET', 'BRA'),
        return_all=False,
    ):
        ket = self.copy()
        ket.add_tag(layer_tags[0])

        bra = ket.retag({layer_tags[0]: layer_tags[1]})
        bra = bra.H
        if mangle_append:
            bra.mangle_inner_(mangle_append)
        norm = ket & bra

        if return_all:
            return norm, ket, bra
        return norm

    def gate(
        self,
        G,
        where,
        contract=False,
        tags=None,
        inplace=False,
        info=None,
        long_range_use_swaps=False,
        long_range_path_sequence=None,
        **compress_opts
    ):
        check_opt("contract", contract, (False, True, 'split', 'reduce-split'))

        psi = self if inplace else self.copy()

        if is_lone_coo(where):
            where = (where,)
        else:
            where = tuple(where)
        ng = len(where)

        tags = tags_to_oset(tags)

        # allow a matrix to be reshaped into a tensor if it factorizes
        #     i.e. (4, 4) assumed to be two qubit gate -> (2, 2, 2, 2)

        site_ix = [psi.site_ind(i, j) for i, j in where]
        # new indices to join old physical sites to new gate
        bnds = [rand_uuid() for _ in range(ng)]
        reindex_map = dict(zip(site_ix, bnds))

        TG = FermionTensor(G.copy(), inds=site_ix+bnds, tags=tags, left_inds=site_ix) # [bnds first, then site_ix]

        if contract is False:
            #
            #       │   │      <- site_ix
            #       GGGGG
            #       │╱  │╱     <- bnds
            #     ──●───●──
            #      ╱   ╱
            #
            psi.reindex_(reindex_map)
            psi |= TG
            return psi

        elif (contract is True) or (ng == 1):
            #
            #       │╱  │╱
            #     ──GGGGG──
            #      ╱   ╱
            #
            psi.reindex_(reindex_map)
            input_tids = psi._get_tids_from_inds(bnds, which='any')
            isite = [psi.tensor_map[itid].get_fermion_info()[1] for itid in input_tids]

            psi.fermion_space.add_tensor(TG, virtual=True)

            # get the sites that used to have the physical indices
            site_tids = psi._get_tids_from_inds(bnds, which='any')

            # pop the sites, contract, then re-add
            pts = [psi._pop_tensor(tid) for tid in site_tids]
            out = tensor_contract(*pts, TG, inplace=True)
            psi.fermion_space.move(out.get_fermion_info()[0], min(isite))
            psi |= out
            return psi

        # following are all based on splitting tensors to maintain structure
        ij_a, ij_b = where

        # parse the argument specifying how to find the path between
        # non-nearest neighbours
        if long_range_path_sequence is not None:
            # make sure we can index
            long_range_path_sequence = tuple(long_range_path_sequence)
            # if the first element is a str specifying move sequence, e.g.
            #     ('v', 'h')
            #     ('av', 'bv', 'ah', 'bh')  # using swaps
            manual_lr_path = not isinstance(long_range_path_sequence[0], str)
            # otherwise assume a path has been manually specified, e.g.
            #     ((1, 2), (2, 2), (2, 3), ... )
            #     (((1, 1), (1, 2)), ((4, 3), (3, 3)), ...)  # using swaps
        else:
            manual_lr_path = False

        psi.fermion_space.add_tensor(TG, virtual=True)
        # check if we are not nearest neighbour and need to swap first
        if long_range_use_swaps:
            raise NotImplementedError

        string = tuple(gen_long_range_path(
                 *where, sequence=long_range_path_sequence))

        # the tensors along this string, which will be updated
        original_ts = [psi[coo] for coo in string]

        # the len(string) - 1 indices connecting the string
        bonds_along = [next(iter(bonds(t1, t2)))
                       for t1, t2 in pairwise(original_ts)]

        if contract == 'split':
            #
            #       │╱  │╱          │╱  │╱
            #     ──GGGGG──  ==>  ──G┄┄┄G──
            #      ╱   ╱           ╱   ╱
            #
            gate_string_split_(
                TG, where, string, original_ts, bonds_along,
                reindex_map, site_ix, info, **compress_opts)

        elif contract == 'reduce-split':
            #
            #       │   │             │ │
            #       GGGGG             GGG               │ │
            #       │╱  │╱   ==>     ╱│ │  ╱   ==>     ╱│ │  ╱          │╱  │╱
            #     ──●───●──       ──>─●─●─<──       ──>─GGG─<──  ==>  ──G┄┄┄G──
            #      ╱   ╱           ╱     ╱           ╱     ╱           ╱   ╱
            #    <QR> <LQ>                            <SVD>
            #
            gate_string_reduce_split_(
                TG, where, string, original_ts, bonds_along,
                reindex_map, site_ix, info, **compress_opts)
        return psi

    gate_ = functools.partialmethod(gate, inplace=True)

    def compute_local_expectation(
        self,
        terms,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode='mps',
        layer_tags=('KET', 'BRA'),
        normalized=False,
        autogroup=True,
        contract_optimize='auto-hq',
        return_all=False,
        plaquette_envs=None,
        plaquette_map=None,
        **plaquette_env_options,
    ):
        norm, ket, bra = self.make_norm(return_all=True, layer_tags=layer_tags)
        plaquette_env_options["max_bond"] = max_bond
        plaquette_env_options["cutoff"] = cutoff
        plaquette_env_options["canonize"] = canonize
        plaquette_env_options["mode"] = mode
        plaquette_env_options["layer_tags"] = layer_tags

        # factorize both local and global phase on the operator tensors
        new_terms = dict()
        for where, op in terms.items():
            if is_lone_coo(where):
                _where = (where,)
            else:
                _where = tuple(where)
            ng = len(_where)
            site_ix = [bra.site_ind(i, j) for i, j in _where]
            bnds = [rand_uuid() for _ in range(ng)]
            TG = FermionTensor(op.copy(), inds=site_ix+bnds, left_inds=site_ix)
            new_terms[where] = bra.fermion_space.move_past(TG).data

        if plaquette_envs is None:
            plaquette_envs = dict()
            for x_bsz, y_bsz in calc_plaquette_sizes(terms.keys(), autogroup):
                plaquette_envs.update(norm.compute_plaquette_environments(
                    x_bsz=x_bsz, y_bsz=y_bsz, **plaquette_env_options))

        if plaquette_map is None:
            # work out which plaquettes to use for which terms
            plaquette_map = calc_plaquette_map(plaquette_envs)

        # now group the terms into just the plaquettes we need
        plaq2coo = defaultdict(list)
        for where, G in new_terms.items():
            p = plaquette_map[where]
            plaq2coo[p].append((where, G))

        expecs = dict()
        for p in plaq2coo:
            # site tags for the plaquette
            tn = plaquette_envs[p]
            if normalized:
                norm_i0j0 = tn.contract(all, optimize=contract_optimize)
            else:
                norm_i0j0 = None

            for where, G in plaq2coo[p]:
                newtn = tn.copy()
                if is_lone_coo(where):
                    _where = (where,)
                else:
                    _where = tuple(where)
                ng = len(_where)
                site_ix = [bra.site_ind(i, j) for i, j in _where]
                bnds = [rand_uuid() for _ in range(ng)]
                reindex_map = dict(zip(site_ix, bnds))
                TG = FermionTensor(G.copy(), inds=site_ix+bnds, left_inds=site_ix)
                ntsr = len(newtn.tensor_map)
                fs = newtn.fermion_space
                tids = newtn._get_tids_from_inds(site_ix, which='any')
                for tid_ in tids:
                    tsr = newtn.tensor_map[tid_]
                    if layer_tags[0] in tsr.tags:
                        tsr.reindex_(reindex_map)
                newtn.add_tensor(TG, virtual=True)
                expec_ij = newtn.contract(all, optimize=contract_optimize)
                expecs[where] = expec_ij, norm_i0j0

        if return_all:
            return expecs

        if normalized:
            return functools.reduce(add, (e / n for e, n in expecs.values()))

        return functools.reduce(add, (e for e, _ in expecs.values()))

class FermionTensorNetwork2DOperator(FermionTensorNetwork2D,
                                     FermionTensorNetwork,
                                     TensorNetwork2DOperator):

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_row_tag_id',
        '_col_tag_id',
        '_Lx',
        '_Ly',
        '_upper_ind_id',
        '_lower_ind_id',
    )

    def to_dense(self, *inds_seq, **contract_opts):
        raise NotImplementedError


class FermionTensorNetwork2DFlat(FermionTensorNetwork2D,
                                 FermionTensorNetwork,
                                 TensorNetwork2DFlat):
    """Mixin class for a 2D square lattice tensor network with a single tensor
    per site, for example, both PEPS and PEPOs.
    """

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_row_tag_id',
        '_col_tag_id',
        '_Lx',
        '_Ly',
    )

    def expand_bond_dimension(self, new_bond_dim, inplace=True, bra=None,
                              rand_strength=0.0):
        raise NotImplementedError


class FPEPS(FermionTensorNetwork2DVector,
            FermionTensorNetwork2DFlat,
            FermionTensorNetwork2D,
            FermionTensorNetwork,
            PEPS):

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_row_tag_id',
        '_col_tag_id',
        '_Lx',
        '_Ly',
        '_site_ind_id',
    )

    def __init__(self, arrays, *, shape='urdlp', tags=None,
                 site_ind_id='k{},{}', site_tag_id='I{},{}',
                 row_tag_id='ROW{}', col_tag_id='COL{}', **tn_opts):

        if isinstance(arrays, FPEPS):
            super().__init__(arrays)
            return

        tags = tags_to_oset(tags)
        self._site_ind_id = site_ind_id
        self._site_tag_id = site_tag_id
        self._row_tag_id = row_tag_id
        self._col_tag_id = col_tag_id

        arrays = tuple(tuple(x for x in xs) for xs in arrays)
        self._Lx = len(arrays)
        self._Ly = len(arrays[0])
        tensors = []

        # cache for both creating and retrieving indices
        ix = defaultdict(rand_uuid)

        for i, j in product(range(self.Lx), range(self.Ly)):
            array = arrays[i][j]

            # figure out if we need to transpose the arrays from some order
            #     other than up right down left physical
            array_order = shape
            if i == self.Lx - 1:
                array_order = array_order.replace('u', '')
            if j == self.Ly - 1:
                array_order = array_order.replace('r', '')
            if i == 0:
                array_order = array_order.replace('d', '')
            if j == 0:
                array_order = array_order.replace('l', '')

            # allow convention of missing bonds to be singlet dimensions
            if len(array.shape) != len(array_order):
                raise TypeError("input array does not ahve right shape of (Lx, Ly)")

            transpose_order = tuple(
                array_order.find(x) for x in 'urdlp' if x in array_order
            )
            if transpose_order != tuple(range(len(array_order))):
                array = np.transpose(array, transpose_order)

            # get the relevant indices corresponding to neighbours
            inds = []
            if 'u' in array_order:
                inds.append(ix[(i + 1, j), (i, j)])
            if 'r' in array_order:
                inds.append(ix[(i, j), (i, j + 1)])
            if 'd' in array_order:
                inds.append(ix[(i, j), (i - 1, j)])
            if 'l' in array_order:
                inds.append(ix[(i, j - 1), (i, j)])
            inds.append(self.site_ind(i, j))

            # mix site, row, column and global tags

            ij_tags = tags | oset((self.site_tag(i, j),
                                   self.row_tag(i),
                                   self.col_tag(j)))

            # create the site tensor!
            tensors.append(FermionTensor(data=array, inds=inds, tags=ij_tags))

        super().__init__(tensors, virtual=True, **tn_opts)

    @classmethod
    def rand(cls, Lx, Ly, bond_dim, phys_dim=2,
             dtype=float, seed=None, **peps_opts):
        raise NotImplementedError

    def add_PEPS(self, other, inplace=False):
        raise NotImplementedError

class FPEPO(FermionTensorNetwork2DOperator,
            FermionTensorNetwork2DFlat,
            FermionTensorNetwork2D,
            FermionTensorNetwork,
            PEPO):

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_row_tag_id',
        '_col_tag_id',
        '_Lx',
        '_Ly',
        '_upper_ind_id',
        '_lower_ind_id',
    )

    def __init__(self, arrays, *, shape='urdlbk', tags=None,
                 upper_ind_id='k{},{}', lower_ind_id='b{},{}',
                 site_tag_id='I{},{}', row_tag_id='ROW{}', col_tag_id='COL{}',
                 **tn_opts):

        if isinstance(arrays, FPEPO):
            super().__init__(arrays)
            return

        tags = tags_to_oset(tags)
        self._upper_ind_id = upper_ind_id
        self._lower_ind_id = lower_ind_id
        self._site_tag_id = site_tag_id
        self._row_tag_id = row_tag_id
        self._col_tag_id = col_tag_id

        arrays = tuple(tuple(x for x in xs) for xs in arrays)
        self._Lx = len(arrays)
        self._Ly = len(arrays[0])
        tensors = []

        # cache for both creating and retrieving indices
        ix = defaultdict(rand_uuid)

        for i, j in product(range(self.Lx), range(self.Ly)):
            array = arrays[i][j]

            # figure out if we need to transpose the arrays from some order
            #     other than up right down left physical
            array_order = shape
            if i == self.Lx - 1:
                array_order = array_order.replace('u', '')
            if j == self.Ly - 1:
                array_order = array_order.replace('r', '')
            if i == 0:
                array_order = array_order.replace('d', '')
            if j == 0:
                array_order = array_order.replace('l', '')

            # allow convention of missing bonds to be singlet dimensions
            if len(array.shape) != len(array_order):
                raise ValueError("Input arrays do not have right shape (Lx, Ly)")

            transpose_order = tuple(
                array_order.find(x) for x in 'urdlbk' if x in array_order
            )
            if transpose_order != tuple(range(len(array_order))):
                array = np.transpose(array, transpose_order)

            # get the relevant indices corresponding to neighbours
            inds = []
            if 'u' in array_order:
                inds.append(ix[(i + 1, j), (i, j)])
            if 'r' in array_order:
                inds.append(ix[(i, j), (i, j + 1)])
            if 'd' in array_order:
                inds.append(ix[(i, j), (i - 1, j)])
            if 'l' in array_order:
                inds.append(ix[(i, j - 1), (i, j)])
            inds.append(self.lower_ind(i, j))
            inds.append(self.upper_ind(i, j))

            # mix site, row, column and global tags
            ij_tags = tags | oset((self.site_tag(i, j),
                                   self.row_tag(i),
                                   self.col_tag(j)))

            # create the site tensor!
            tensors.append(FermionTensor(data=array, inds=inds, tags=ij_tags))

        super().__init__(tensors, virtual=True, **tn_opts)

    @classmethod
    def rand(cls, Lx, Ly, bond_dim, phys_dim=2, herm=False,
             dtype=float, seed=None, **pepo_opts):
        raise NotImplementedError

    def add_PEPO(self, other, inplace=False):
        """Add this PEPO with another.
        """
        raise NotImplementedError
