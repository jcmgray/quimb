"""Classes and algorithms related to Fermionic 2D tensor networks."""

import functools
from operator import add
from itertools import product
from collections import defaultdict
import numpy as np

from ...utils import check_opt, pairwise
from ..tensor_core import bonds, rand_uuid, oset, tags_to_oset
from ..tensor_2d import (
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
    calc_plaquette_map,
)
from .fermion_core import FermionTensor, FermionTensorNetwork, tensor_contract
from .block_gen import rand_all_blocks, ones_single_block
from .block_tools import inv_with_smudge

INVERSE_CUTOFF = 1e-10


class FermionTensorNetwork2D(FermionTensorNetwork, TensorNetwork2D):
    """A subclass of ``quimb.tensor.tensor_2d.TensorNetwork2D`` that overrides methods
    that depend on ordering of the tensors. Reorder method is added to aid row/column-wise
    operations. Environments are now computed as an entire FermionTensorNetwork so that the
    plaquettes are placed correctly

    """

    _EXTRA_PROPS = (
        "_site_tag_id",
        "_x_tag_id",
        "_y_tag_id",
        "_Lx",
        "_Ly",
    )

    def _compatible_2d(self, other):
        """Check whether ``self`` and ``other`` are compatible 2D tensor
        networks such that they can remain a 2D tensor network when combined.
        """
        return isinstance(other, FermionTensorNetwork2D) and all(
            getattr(self, e) == getattr(other, e)
            for e in FermionTensorNetwork2D._EXTRA_PROPS
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
        r"""Reorder all tensors either row/column-wise

        If ``direction == 'row'`` then::

                     |  |  |  |  |  |  |
        Row 0:      ─●─>●─>●─>●─>●─>●─>●─    then Row 1
                     |  |  |  |  |  |  |
        Row 1:      ─●─>●─>●─>●─>●─>●─>●─    then Row 2
                     |  |  |  |  |  |  |
        Row 2:      ─●─>●─>●─>●─>●─>●─>●─
                     |  |  |  |  |  |  |

        If ``direction == 'col'`` then::

                     v  v  v  v  v  v  v
                    ─●──●──●──●──●──●──●─
                     v  v  v  v  v  v  v
                    ─●──●──●──●──●──●──●─
                     v  v  v  v  v  v  v
                    ─●──●──●──●──●──●──●─
                     v  v  v  v  v  v  v

        Parameters
        ----------
        direction : {"row", "col"}
            The direction to reorder the entire network
        layer_tags : optional
            The relative order within a single coordinate
        inplace : bool, optional
            Whether to perform the operation inplace
        """
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
            x, y = (i, j) if direction == "row" else (j, i)
            site_tag = self.site_tag(x, y)
            tids = self._get_tids_from_tags(site_tag)
            if len(tids) == 1:
                (tid,) = tids
                if tid not in tid_map:
                    tid_map[tid] = current_position
                    current_position += 1
            else:
                if layer_tags is None:
                    _tags = [self.tensor_map[ix].tags for ix in tids]
                    _tmp_tags = _tags[0].copy()
                    for itag in _tags[1:]:
                        _tmp_tags &= itag
                    _layer_tags = sorted(
                        [list(i - _tmp_tags)[0] for i in _tags]
                    )
                else:
                    _layer_tags = layer_tags
                for tag in _layer_tags:
                    (tid,) = self._get_tids_from_tags((site_tag, tag))
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
        method="eigh",
        renorm=True,
        optimize="auto-hq",
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
        mode="mps",
        layer_tags=None,
        dense=False,
        compress_opts=None,
        envs=None,
        **contract_boundary_opts,
    ):
        """Compute the ``self.Lx`` 1D boundary tensor networks describing
        the environments of rows and columns. The returned tensor network
        also contains the original plaquettes
        """
        direction = {
            "ymin": "col",
            "ymax": "col",
            "xmax": "row",
            "xmin": "row",
        }[from_which]
        tn = self.reorder(direction, layer_tags=layer_tags)

        r2d = Rotator2D(tn, xrange, yrange, from_which)
        sweep, x_tag = r2d.sweep, r2d.x_tag

        if envs is None:
            envs = {}

        if mode == "full-bond":
            # set shared storage for opposite env contractions
            contract_boundary_opts.setdefault("opposite_envs", {})

        envs[from_which, sweep[0]] = FermionTensorNetwork([])
        first_row = x_tag(sweep[0])
        envs["mid", sweep[0]] = tn.select(first_row).copy()
        if len(sweep) == 1:
            return envs
        if dense:
            tn ^= first_row
        envs[from_which, sweep[1]] = tn.select(first_row).copy()

        for i in sweep[2:]:
            iprevprev = i - 2 * sweep.step
            iprev = i - sweep.step
            envs["mid", iprev] = tn.select(x_tag(iprev)).copy()
            if dense:
                tn ^= (x_tag(iprevprev), x_tag(iprev))
            else:
                tn.contract_boundary_from_(
                    xrange=(
                        (iprevprev, iprev) if r2d.plane == "x" else r2d.xrange
                    ),
                    yrange=(
                        (iprevprev, iprev) if r2d.plane == "y" else r2d.yrange
                    ),
                    from_which=from_which,
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

    compute_xmin_environments = functools.partialmethod(
        compute_environments, from_which="xmin"
    )

    compute_xmax_environments = functools.partialmethod(
        compute_environments, from_which="xmax"
    )

    compute_ymin_environments = functools.partialmethod(
        compute_environments, from_which="ymin"
    )

    compute_ymax_environments = functools.partialmethod(
        compute_environments, from_which="ymax"
    )

    def _compute_plaquette_environments_x_first(
        self,
        x_bsz,
        y_bsz,
        max_bond=None,
        cutoff=1e-10,
        canonize=True,
        layer_tags=None,
        second_dense=None,
        x_envs=None,
        **compute_environment_opts,
    ):
        if second_dense is None:
            second_dense = x_bsz < 2

        if x_envs is None:
            x_envs = self.compute_x_environments(
                max_bond=max_bond,
                cutoff=cutoff,
                canonize=canonize,
                layer_tags=layer_tags,
                **compute_environment_opts,
            )

        # next we form vertical strips and contract in both y directions
        y_envs = dict()
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
            tn_xi = FermionTensorNetwork(
                (
                    x_envs["xmax", i],
                    *[x_envs["mid", i + x] for x in range(x_bsz)],
                    x_envs["xmin", i + x_bsz - 1],
                )
            ).view_as_(FermionTensorNetwork2D, like=self)
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
            #     'ymin'    'ymax'        'ymin'    'ymax'
            #
            y_envs[i] = tn_xi.compute_y_environments(
                xrange=(max(i - 1, 0), min(i + x_bsz, self.Lx - 1)),
                max_bond=max_bond,
                cutoff=cutoff,
                canonize=canonize,
                layer_tags=layer_tags,
                dense=second_dense,
                **compute_environment_opts,
            )

        # then range through all the possible plaquettes, selecting the correct
        # boundary tensors from either the x or y environments
        plaquette_envs = dict()
        for i0, j0 in product(
            range(self.Lx - x_bsz + 1), range(self.Ly - y_bsz + 1)
        ):
            # we want to select bordering tensors from:
            #
            #       L──A──A──R    <- A from the x environments
            #       │  │  │  │
            #  i0+1 L──●──●──R
            #       │  │  │  │    <- L, R from the y environments
            #  i0   L──●──●──R
            #       │  │  │  │
            #       L──B──B──R    <- B from the x environments
            #
            #         j0  j0+1
            #
            env_ij = FermionTensorNetwork(
                (
                    y_envs[i0]["ymin", j0],
                    *[y_envs[i0]["mid", ix] for ix in range(j0, j0 + y_bsz)],
                    y_envs[i0]["ymax", j0 + y_bsz - 1],
                ),
                check_collisions=False,
            )

            plaquette_envs[(i0, j0), (x_bsz, y_bsz)] = env_ij

        return plaquette_envs

    def _compute_plaquette_environments_y_first(
        self,
        x_bsz,
        y_bsz,
        max_bond=None,
        cutoff=1e-10,
        canonize=True,
        layer_tags=None,
        second_dense=None,
        y_envs=None,
        **compute_environment_opts,
    ):
        if second_dense is None:
            second_dense = y_bsz < 2

        # first we contract from either side to produce y environments
        if y_envs is None:
            y_envs = self.compute_y_environments(
                max_bond=max_bond,
                cutoff=cutoff,
                canonize=canonize,
                layer_tags=layer_tags,
                **compute_environment_opts,
            )

        # next we form vertical strips and contract from both x directions
        x_envs = dict()
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
            tn_yj = FermionTensorNetwork(
                (
                    y_envs["ymin", j],
                    *[y_envs["mid", j + jn] for jn in range(y_bsz)],
                    y_envs["ymax", j + y_bsz - 1],
                )
            ).view_as_(FermionTensorNetwork2D, like=self)
            #
            #        y_bsz
            #        <-->        second_dense=True
            #     ●──●──●──●      ╭──●──╮
            #     │  │  │  │  or  │ ╱ ╲ │    'xmax'
            #        .  .           . .                  ┬
            #                                            ┊ x_bsz
            #        .  .           . .                  ┴
            #     │  │  │  │  or  │ ╲ ╱ │    'xmin'
            #     ●──●──●──●      ╰──●──╯
            #
            x_envs[j] = tn_yj.compute_x_environments(
                yrange=(max(j - 1, 0), min(j + y_bsz, self.Ly - 1)),
                max_bond=max_bond,
                cutoff=cutoff,
                canonize=canonize,
                layer_tags=layer_tags,
                dense=second_dense,
                **compute_environment_opts,
            )

        # then range through all the possible plaquettes, selecting the correct
        # boundary tensors from either the x or y environments
        plaquette_envs = dict()
        for i0, j0 in product(
            range(self.Lx - x_bsz + 1), range(self.Ly - y_bsz + 1)
        ):
            # we want to select bordering tensors from:
            #
            #          A──A──A──A    <- A from the x environments
            #          │  │  │  │
            #     i0+1 L──●──●──R
            #          │  │  │  │    <- L, R from the y environments
            #     i0   L──●──●──R
            #          │  │  │  │
            #          B──B──B──B    <- B from the x environments
            #
            #            j0  j0+1
            #
            env_ij = FermionTensorNetwork(
                (
                    x_envs[j0]["xmin", i0],
                    *[x_envs[j0]["mid", ix] for ix in range(i0, i0 + x_bsz)],
                    x_envs[j0]["xmax", i0 + x_bsz - 1],
                ),
                check_collisions=False,
            )

            plaquette_envs[(i0, j0), (x_bsz, y_bsz)] = env_ij

        return plaquette_envs


def gate_string_split_(
    TG,
    where,
    string,
    original_ts,
    bonds_along,
    reindex_map,
    site_ix,
    info,
    **compress_opts,
):
    # by default this means singuvalues are kept in the string 'blob' tensor
    compress_opts.setdefault("absorb", "right")
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
        lix = tuple(oset(blob.inds) - oset(lix))
        blob, *maybe_svals, inner_ts[i] = blob.split(
            left_inds=lix,
            get="tensors",
            bond_ind=bix,
            qpn_info=qpn_info,
            **compress_opts,
        )

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`

        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            coo_pair = (string[i], string[i + 1])
            info["singular_values", coo_pair] = s

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
        inner_ts[j], *maybe_svals, blob = blob.split(
            left_inds=lix,
            get="tensors",
            bond_ind=bix,
            qpn_info=qpn_info,
            **compress_opts,
        )

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            coo_pair = (string[j - 1], string[j])
            info["singular_values", coo_pair] = s

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
        t = inner_ts[i]
        t.multiply_index_diagonal_(bix, s, location=location, inverse=True)

    revert_index_map = {v: k for k, v in reindex_map.items()}
    for to, tn in zip(original_ts, inner_ts):
        to.reindex_(revert_index_map)
        tn.transpose_like_(to)
        to.modify(data=tn.data)

    for i, (tid, _) in enumerate(fermion_info):
        if i == 0:
            fs.replace_tensor(work_site, original_ts[i], tid=tid, virtual=True)
        else:
            fs.insert_tensor(
                work_site + i, original_ts[i], tid=tid, virtual=True
            )

    fs._reorder_from_dict(dict(fermion_info))


def gate_string_reduce_split_(
    TG,
    where,
    string,
    original_ts,
    bonds_along,
    reindex_map,
    site_ix,
    info,
    **compress_opts,
):
    compress_opts.setdefault("absorb", "right")

    # indices to reduce, first and final include physical indices for gate
    inds_to_reduce = [(bonds_along[0], site_ix[0])]
    for b1, b2 in pairwise(bonds_along):
        inds_to_reduce.append((b1, b2))
    inds_to_reduce.append((bonds_along[-1], site_ix[-1]))

    # tensors that remain on the string sites and those pulled into string
    outer_ts, inner_ts = [], []
    fermion_info = []
    fs = TG.fermion_owner[0]

    for coo, rix, t in zip(string, inds_to_reduce, original_ts):
        tq, tr = t.split(
            left_inds=None,
            right_inds=rix,
            method="qr",
            get="tensors",
            absorb="right",
        )
        fermion_info.append(t.get_fermion_info())
        outer_ts.append(tq)
        inner_ts.append(tr.reindex_(reindex_map) if coo in where else tr)

    for tq, tr, t in zip(outer_ts, inner_ts, original_ts):
        isite = t.get_fermion_info()[1]
        fs.replace_tensor(isite, tr, virtual=True)
        fs.insert_tensor(isite + 1, tq, virtual=True)

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
        lix = tuple(oset(blob.inds) - oset(lix))
        blob, *maybe_svals, inner_ts[i] = blob.split(
            left_inds=lix, get="tensors", bond_ind=bix, **compress_opts
        )

        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            coo_pair = (string[i], string[i + 1])
            info["singular_values", coo_pair] = s

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
            left_inds=lix, get="tensors", bond_ind=bix, **compress_opts
        )
        # if singular values are returned (``absorb=None``) check if we should
        #     return them via ``info``, e.g. for ``SimpleUpdate`
        if maybe_svals and info is not None:
            s = next(iter(maybe_svals)).data
            coo_pair = (string[j - 1], string[j])
            info["singular_values", coo_pair] = s

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
        if i == 0:
            fs.replace_tensor(work_site, inner_ts[i], tid=tid, virtual=True)
        else:
            fs.insert_tensor(work_site + i, inner_ts[i], tid=tid, virtual=True)
    new_ts = [
        tensor_contract(ts, tr, inplace=True).transpose_like_(to)
        for to, ts, tr in zip(original_ts, outer_ts, inner_ts)
    ]

    for i, bix, location, s in regauged:
        t = new_ts[i]
        t.multiply_index_diagonal_(bix, s, location=location, inverse=True)

    for (tid, _), to, t in zip(fermion_info, original_ts, new_ts):
        site = t.get_fermion_info()[1]
        to.modify(data=t.data)
        fs.replace_tensor(site, to, tid=tid, virtual=True)

    fs._reorder_from_dict(dict(fermion_info))


class FermionTensorNetwork2DVector(
    FermionTensorNetwork2D, FermionTensorNetwork, TensorNetwork2DVector
):
    """Mixin class  for a 2D square lattice vector TN, i.e. one with a single
    physical index per site.
    """

    _EXTRA_PROPS = (
        "_site_tag_id",
        "_x_tag_id",
        "_y_tag_id",
        "_Lx",
        "_Ly",
        "_site_ind_id",
    )

    def to_dense(self, *inds_seq, **contract_opts):
        raise NotImplementedError

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
        **compress_opts,
    ):
        check_opt("contract", contract, (False, True, "split", "reduce-split"))

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

        TG = FermionTensor(
            G.copy(), inds=site_ix + bnds, tags=tags, left_inds=site_ix
        )  # [bnds first, then site_ix]

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
            input_tids = psi._get_tids_from_inds(bnds, which="any")
            isite = [
                psi.tensor_map[itid].get_fermion_info()[1]
                for itid in input_tids
            ]

            psi.fermion_space.add_tensor(TG, virtual=True)

            # get the sites that used to have the physical indices
            site_tids = psi._get_tids_from_inds(bnds, which="any")

            # pop the sites, contract, then re-add
            pts = [psi._pop_tensor(tid) for tid in site_tids]
            out = tensor_contract(*pts, TG, inplace=True)
            psi |= out
            psi.fermion_space.move(out.get_fermion_info()[0], min(isite))
            return psi

        # parse the argument specifying how to find the path between
        # non-nearest neighbours
        if long_range_path_sequence is not None:
            long_range_path_sequence = tuple(long_range_path_sequence)

        if long_range_use_swaps:
            raise NotImplementedError

        psi.fermion_space.add_tensor(TG, virtual=True)
        # check if we are not nearest neighbour and need to swap first

        string = tuple(
            gen_long_range_path(*where, sequence=long_range_path_sequence)
        )

        # the tensors along this string, which will be updated
        original_ts = [psi[coo] for coo in string]

        # the len(string) - 1 indices connecting the string
        bonds_along = [
            next(iter(bonds(t1, t2))) for t1, t2 in pairwise(original_ts)
        ]

        if contract == "split":
            #
            #       │╱  │╱          │╱  │╱
            #     ──GGGGG──  ==>  ──G┄┄┄G──
            #      ╱   ╱           ╱   ╱
            #
            gate_string_split_(
                TG,
                where,
                string,
                original_ts,
                bonds_along,
                reindex_map,
                site_ix,
                info,
                **compress_opts,
            )

        elif contract == "reduce-split":
            #
            #       │   │             │ │
            #       GGGGG             GGG               │ │
            #       │╱  │╱   ==>     ╱│ │  ╱   ==>     ╱│ │  ╱          │╱  │╱
            #     ──●───●──       ──>─●─●─<──       ──>─GGG─<──  ==>  ──G┄┄┄G──
            #      ╱   ╱           ╱     ╱           ╱     ╱           ╱   ╱
            #    <QR> <LQ>                            <SVD>
            #
            gate_string_reduce_split_(
                TG,
                where,
                string,
                original_ts,
                bonds_along,
                reindex_map,
                site_ix,
                info,
                **compress_opts,
            )
        return psi

    gate_ = functools.partialmethod(gate, inplace=True)

    def compute_local_expectation(
        self,
        terms,
        max_bond=None,
        *,
        cutoff=1e-10,
        canonize=True,
        mode="mps",
        layer_tags=("KET", "BRA"),
        normalized=False,
        autogroup=True,
        contract_optimize="auto-hq",
        return_all=False,
        plaquette_envs=None,
        plaquette_map=None,
        **plaquette_env_options,
    ):
        norm, _, bra = self.make_norm(return_all=True, layer_tags=layer_tags)

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
            TG = FermionTensor(
                op.copy(), inds=site_ix + bnds, left_inds=site_ix
            )
            new_terms[where] = bra.fermion_space.move_past(TG).data

        if plaquette_envs is None:
            plaquette_envs = dict()
            for x_bsz, y_bsz in calc_plaquette_sizes(terms.keys(), autogroup):
                plaquette_envs.update(
                    norm.compute_plaquette_environments(
                        x_bsz=x_bsz, y_bsz=y_bsz, **plaquette_env_options
                    )
                )

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
                TG = FermionTensor(
                    G.copy(), inds=site_ix + bnds, left_inds=site_ix
                )
                tids = newtn._get_tids_from_inds(site_ix, which="any")
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


class FermionTensorNetwork2DOperator(
    FermionTensorNetwork2D, FermionTensorNetwork, TensorNetwork2DOperator
):
    _EXTRA_PROPS = (
        "_site_tag_id",
        "_x_tag_id",
        "_y_tag_id",
        "_Lx",
        "_Ly",
        "_upper_ind_id",
        "_lower_ind_id",
    )

    def to_dense(self, *inds_seq, **contract_opts):
        raise NotImplementedError


class FermionTensorNetwork2DFlat(
    FermionTensorNetwork2D, FermionTensorNetwork, TensorNetwork2DFlat
):
    """Mixin class for a 2D square lattice tensor network with a single tensor
    per site, for example, both PEPS and PEPOs.
    """

    _EXTRA_PROPS = (
        "_site_tag_id",
        "_x_tag_id",
        "_y_tag_id",
        "_Lx",
        "_Ly",
    )

    def expand_bond_dimension(
        self, new_bond_dim, inplace=True, bra=None, rand_strength=0.0
    ):
        raise NotImplementedError


class FPEPS(
    FermionTensorNetwork2DVector,
    FermionTensorNetwork2DFlat,
    FermionTensorNetwork2D,
    FermionTensorNetwork,
    PEPS,
):
    _EXTRA_PROPS = (
        "_site_tag_id",
        "_x_tag_id",
        "_y_tag_id",
        "_Lx",
        "_Ly",
        "_site_ind_id",
    )

    def __init__(
        self,
        arrays,
        *,
        shape="urdlp",
        tags=None,
        site_ind_id="k{},{}",
        site_tag_id="I{},{}",
        x_tag_id="X{}",
        y_tag_id="Y{}",
        **tn_opts,
    ):
        if isinstance(arrays, FPEPS):
            super().__init__(arrays)
            return

        tags = tags_to_oset(tags)
        self._site_ind_id = site_ind_id
        self._site_tag_id = site_tag_id
        self._x_tag_id = x_tag_id
        self._y_tag_id = y_tag_id

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
                array_order = array_order.replace("u", "")
            if j == self.Ly - 1:
                array_order = array_order.replace("r", "")
            if i == 0:
                array_order = array_order.replace("d", "")
            if j == 0:
                array_order = array_order.replace("l", "")

            # allow convention of missing bonds to be singlet dimensions
            if len(array.shape) != len(array_order):
                raise TypeError(
                    "input array does not ahve right shape of (Lx, Ly)"
                )

            transpose_order = tuple(
                array_order.find(x) for x in "urdlp" if x in array_order
            )
            if transpose_order != tuple(range(len(array_order))):
                array = np.transpose(array, transpose_order)

            # get the relevant indices corresponding to neighbours
            inds = []
            if "u" in array_order:
                inds.append(ix[(i + 1, j), (i, j)])
            if "r" in array_order:
                inds.append(ix[(i, j), (i, j + 1)])
            if "d" in array_order:
                inds.append(ix[(i, j), (i - 1, j)])
            if "l" in array_order:
                inds.append(ix[(i, j - 1), (i, j)])
            inds.append(self.site_ind(i, j))

            # mix site, x, y and global tags

            ij_tags = tags | oset(
                (self.site_tag(i, j), self.x_tag(i), self.y_tag(j))
            )

            # create the site tensor!
            tensors.append(FermionTensor(data=array, inds=inds, tags=ij_tags))

        super().__init__(tensors, virtual=True, **tn_opts)

    @classmethod
    def rand(
        cls,
        Lx,
        Ly,
        bond_dim,
        symmetry_infos,
        dq_infos,
        phys_dim=2,
        seed=None,
        dtype=float,
        **peps_opts,
    ):
        r"""Construct a random 2d FPEPS with given quantum particle number distribution

        Parameters
        ----------
        Lx : int
            The number of rows.
        Ly : int
            The number of columns.
        bond_dim: int
            Virtual bond dimension for each virtual block
        symmetry_infos : dict[tuple[int], list/tuple]
            A dictionary mapping the site coordinates to the allowed quantum particle
            numbers in each dimension ordered by up, right, down, left and physical,
            which will be supplied to ``rand_all_blocks``
        dq_infos: dict[tuple[ix], int or tuple/list of integers]
            A dictionary mapping the site coordinates to the net quantum particle numbers
            on that site, which will be supplied to ``rand_all_blocks``
        phys_dim: int
            Physical bond dimension for each physical block
        seed : int, optional
            A random seed.
        dtype : {'float64', 'complex128', 'float32', 'complex64'}, optional
            The underlying data type.
        pepes_opts
            Supplied to :class:`~quimb.tensor.fermion_2d.FPEPS`.

        Returns
        -------
        FPEPS
        """
        if seed is not None:
            np.random.seed(seed)
        pattern_map = {"d": "+", "l": "+", "p": "+", "u": "-", "r": "-"}

        arrays = [[None for _ in range(Ly)] for _ in range(Lx)]
        for i, j in product(range(Lx), range(Ly)):
            shape = []
            pattern = ""
            if i != Lx - 1:  # bond up
                shape.append(bond_dim)
                pattern += pattern_map["u"]
            if j != Ly - 1:  # bond right
                shape.append(bond_dim)
                pattern += pattern_map["r"]
            if i != 0:  # bond down
                shape.append(bond_dim)
                pattern += pattern_map["d"]
            if j != 0:  # bond left
                shape.append(bond_dim)
                pattern += pattern_map["l"]
            shape.append(phys_dim)
            pattern += pattern_map["p"]
            symmetry_info = symmetry_infos[i, j]
            arrays[i][j] = rand_all_blocks(
                shape,
                symmetry_info,
                dtype=dtype,
                pattern=pattern,
                dq=dq_infos[i, j],
            )
        return FPEPS(arrays, **peps_opts)

    @classmethod
    def gen_site_prod_state(cls, Lx, Ly, phys_infos, phys_dim=1, **peps_opts):
        r"""Construct a 2d FPEPS as site product state

        Parameters
        ----------
        Lx : int
            The number of rows.
        Ly : int
            The number of columns.
        phys_infos: dict[tuple[int], int or tuple/list]
            A dictionary mapping the site coordinates to the specified single quantum
            particle state
        phys_dim: int
            Physical bond dimension for the physical block
        pepes_opts
            Supplied to :class:`~quimb.tensor.fermion_2d.FPEPS`.

        Returns
        -------
        FPEPS
        """
        pattern_map = {"d": "+", "l": "+", "p": "+", "u": "-", "r": "-"}
        arrays = [[None for _ in range(Ly)] for _ in range(Lx)]
        for i, j in product(range(Lx), range(Ly)):
            shape = []
            pattern = ""
            if i != Lx - 1:  # bond up
                shape.append(1)
                pattern += pattern_map["u"]
            if j != Ly - 1:  # bond right
                shape.append(1)
                pattern += pattern_map["r"]
            if i != 0:  # bond down
                shape.append(1)
                pattern += pattern_map["d"]
            if j != 0:  # bond left
                shape.append(1)
                pattern += pattern_map["l"]
            shape.append(phys_dim)
            pattern += pattern_map["p"]
            arrays[i][j] = ones_single_block(
                shape, pattern, phys_infos[i, j], ind=len(shape) - 1
            )
        return FPEPS(arrays, **peps_opts)

    def add_PEPS(self, other, inplace=False):
        raise NotImplementedError


class FPEPO(
    FermionTensorNetwork2DOperator,
    FermionTensorNetwork2DFlat,
    FermionTensorNetwork2D,
    FermionTensorNetwork,
    PEPO,
):
    _EXTRA_PROPS = (
        "_site_tag_id",
        "_x_tag_id",
        "_y_tag_id",
        "_Lx",
        "_Ly",
        "_upper_ind_id",
        "_lower_ind_id",
    )

    def __init__(
        self,
        arrays,
        *,
        shape="urdlbk",
        tags=None,
        upper_ind_id="k{},{}",
        lower_ind_id="b{},{}",
        site_tag_id="I{},{}",
        x_tag_id="X{}",
        y_tag_id="Y{}",
        **tn_opts,
    ):
        if isinstance(arrays, FPEPO):
            super().__init__(arrays)
            return

        tags = tags_to_oset(tags)
        self._upper_ind_id = upper_ind_id
        self._lower_ind_id = lower_ind_id
        self._site_tag_id = site_tag_id
        self._x_tag_id = x_tag_id
        self._y_tag_id = y_tag_id

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
                array_order = array_order.replace("u", "")
            if j == self.Ly - 1:
                array_order = array_order.replace("r", "")
            if i == 0:
                array_order = array_order.replace("d", "")
            if j == 0:
                array_order = array_order.replace("l", "")

            # allow convention of missing bonds to be singlet dimensions
            if len(array.shape) != len(array_order):
                raise ValueError(
                    "Input arrays do not have right shape (Lx, Ly)"
                )

            transpose_order = tuple(
                array_order.find(x) for x in "urdlbk" if x in array_order
            )
            if transpose_order != tuple(range(len(array_order))):
                array = np.transpose(array, transpose_order)

            # get the relevant indices corresponding to neighbours
            inds = []
            if "u" in array_order:
                inds.append(ix[(i + 1, j), (i, j)])
            if "r" in array_order:
                inds.append(ix[(i, j), (i, j + 1)])
            if "d" in array_order:
                inds.append(ix[(i, j), (i - 1, j)])
            if "l" in array_order:
                inds.append(ix[(i, j - 1), (i, j)])
            inds.append(self.lower_ind(i, j))
            inds.append(self.upper_ind(i, j))

            # mix site, x, y and global tags
            ij_tags = tags | oset(
                (self.site_tag(i, j), self.x_tag(i), self.y_tag(j))
            )

            # create the site tensor!
            tensors.append(FermionTensor(data=array, inds=inds, tags=ij_tags))

        super().__init__(tensors, virtual=True, **tn_opts)

    @classmethod
    def rand(
        cls,
        Lx,
        Ly,
        bond_dim,
        phys_dim=2,
        herm=False,
        dtype=float,
        seed=None,
        **pepo_opts,
    ):
        raise NotImplementedError

    def add_PEPO(self, other, inplace=False):
        """Add this PEPO with another."""
        raise NotImplementedError
