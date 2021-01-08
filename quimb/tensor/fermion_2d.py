from .fermion import FermionTensorNetwork, FermionTensor
from .tensor_2d import TensorNetwork2D, TensorNetwork2DVector, PEPS
from .tensor_core import (
    rand_uuid,
    oset,
    tags_to_oset
)
from collections import defaultdict
from itertools import product
import numpy as np


class FermionTensorNetwork2D(FermionTensorNetwork,TensorNetwork2D):

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

    def compute_row_environments(self, dense=False, **compress_opts):
        layer_tags = compress_opts.get("layer_tags", None)
        reorder_tags = compress_opts.pop("reorder_tags", layer_tags)
        env_bottom = self.reorder_right_row(layer_tags=reorder_tags)
        env_top = env_bottom.copy()

        row_envs = dict()

        # upwards pass
        row_envs['below', 0] = FermionTensorNetwork([])
        first_row = self.row_tag(0)
        if dense:
            env_bottom ^= first_row
        row_envs['mid', 0] = env_bottom.select(first_row).simple_copy()
        row_envs['below', 1] = env_bottom.select(first_row).simple_copy()
        for i in range(2, env_bottom.Lx):
            below_row = env_bottom.row_tag(i-1)
            row_envs["mid", i-1] = env_bottom.select(below_row).simple_copy()
            if dense:
                env_bottom ^= (self.row_tag(i - 2), self.row_tag(i - 1))
            else:
                env_bottom.contract_boundary_from_bottom_(
                    (i - 2, i - 1), **compress_opts)
            row_envs['below', i] = env_bottom.select(first_row).simple_copy()

        last_row = env_bottom.row_tag(self.Lx-1)
        row_envs['mid', self.Lx-1] = env_bottom.select(last_row).simple_copy()
        # downwards pass
        row_envs['above', self.Lx - 1] = FermionTensorNetwork([])
        last_row = self.row_tag(self.Lx - 1)
        if dense:
            env_top ^= last_row
        row_envs['above', self.Lx - 2] = env_top.select(last_row).simple_copy()
        for i in range(env_top.Lx - 3, -1, -1):
            if dense:
                env_top ^= (self.row_tag(i + 1), self.row_tag(i + 2))
            else:
                env_top.contract_boundary_from_top_(
                    (i + 1, i + 2), **compress_opts)
            row_envs['above', i] = env_top.select(last_row).simple_copy()

        return row_envs

    def compute_col_environments(self, dense=False, **compress_opts):
        layer_tags = compress_opts.get("layer_tags", None)
        reorder_tags = compress_opts.pop("reorder_tags", layer_tags)
        env_left = self.reorder_upward_column(layer_tags=reorder_tags)
        env_right = env_left.copy()
        col_envs = dict()

        # upwards pass
        col_envs['left', 0] = FermionTensorNetwork([])
        first_col = self.col_tag(0)
        if dense:
            env_left ^= first_col
        col_envs['mid', 0] = env_left.select(first_col).simple_copy()
        col_envs['left', 1] = env_left.select(first_col).simple_copy()

        for i in range(2, env_left.Ly):
            left_col = env_left.col_tag(i-1)
            col_envs["mid", i-1] = env_left.select(left_col).simple_copy()
            if dense:
                env_left ^= (self.col_tag(i - 2), self.col_tag(i - 1))
            else:
                env_left.contract_boundary_from_left_(
                    (i - 2, i - 1), **compress_opts)
            col_envs['left', i] = env_left.select(first_col).simple_copy()

        last_col = env_left.col_tag(self.Ly-1)
        col_envs['mid', self.Ly-1] = env_left.select(last_col).simple_copy()
        # downwards pass
        col_envs['right', self.Ly - 1] = FermionTensorNetwork([])
        last_col = self.col_tag(self.Ly - 1)
        if dense:
            env_right ^= last_col
        col_envs['right', self.Ly - 2] = env_right.select(last_col).simple_copy()
        for i in range(env_right.Ly - 3, -1, -1):
            if dense:
                env_right ^= (self.col_tag(i + 1), self.col_tag(i + 2))
            else:
                env_right.contract_boundary_from_right_(
                    (i + 1, i + 2), **compress_opts)
            col_envs['right', i] = env_right.select(last_col).simple_copy()

        return col_envs

    def _compute_plaquette_environments_row_first(
        self,
        x_bsz,
        y_bsz,
        second_dense=None,
        row_envs=None,
        **compute_environment_opts
    ):
        if second_dense is None:
            second_dense = x_bsz < 2

        # first we contract from either side to produce column environments
        if row_envs is None:
            row_envs = self.compute_row_environments(
                **compute_environment_opts)

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
                row_envs['below', i],
                *[row_envs['mid', i+x] for x in range(x_bsz)],
                row_envs['above', i + x_bsz - 1],
            ), check_collisions=False).view_as_(FermionTensorNetwork2D, like=self)
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
                dense=second_dense, **compute_environment_opts)

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
        second_dense=None,
        col_envs=None,
        **compute_environment_opts
    ):
        if second_dense is None:
            second_dense = y_bsz < 2

        # first we contract from either side to produce column environments
        if col_envs is None:
            col_envs = self.compute_col_environments(
                **compute_environment_opts)

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
                *[col_envs['mid', j+y] for y in range(y_bsz)],
                col_envs['right', j + y_bsz - 1],
            ), check_collisions=False).view_as_(FermionTensorNetwork2D, like=self)
            #
            #        y_bsz
            #        <-->        second_dense=True
            #     ●──●──●──●      ╭──●──╮
            #     │  │  │  │  or  │ ╱ ╲ │    'above'
            #        .  .           . .                  ┬
            #                                            ┊ x_bsz
            #        .  .           . .                  ┴
            #     │  │  │  │  or  │ ╲ ╱ │    'below'
            #     ●──●──●──●      ╰──●──╯
            #
            row_envs[j] = col_j.compute_row_environments(
                yrange=(max(j - 1, 0), min(j + y_bsz, self.Ly - 1)),
                dense=second_dense, **compute_environment_opts)

        # then range through all the possible plaquettes, selecting the correct
        # boundary tensors from either the column or row environments
        plaquette_envs = dict()
        for i0, j0 in product(range(self.Lx - x_bsz + 1),
                              range(self.Ly - y_bsz + 1)):


            env_ij = FermionTensorNetwork((
                row_envs[j0]['below', i0],
                *[row_envs[j0]['mid', ix] for ix in range(i0, i0+x_bsz)],
                row_envs[j0]['above', i0 + x_bsz - 1]
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


    def reorder(self, direction="ru", layer_tags=None, inplace=False):
        Lx, Ly = self._Lx, self._Ly
        row_wise = (direction[0] in ["r", "l"])
        iter_dic = {"u": range(Lx),
                    "d": range(Lx)[::-1],
                    "r": range(Ly),
                    "l": range(Ly)[::-1]}
        iterator = product(iter_dic[direction[1]], iter_dic[direction[0]])
        position = 0
        tid_map = dict()
        for i, j in iterator:
            x, y = (i, j) if row_wise else (j, i)
            site_tag = self.site_tag(x, y)
            tid = self._get_tids_from_tags(site_tag)
            if len(tid)==1:
                tid,  = tid
                if tid not in tid_map:
                    tid_map[tid] = position
                    position += 1
            else:
                if layer_tags is None:
                    _tags = [self.tensor_map[ix].tags for ix in tid]
                    _tmp_tags = _tags[0].copy()
                    for itag in _tags[1:]:
                        _tmp_tags &= itag
                    _layer_tags = sorted([list(i-_tmp_tags)[0] for i in _tags])
                else:
                    _layer_tags = layer_tags
                for tag in _layer_tags:
                    tid, = self._get_tids_from_tags((site_tag, tag))
                    if tid not in tid_map:
                        tid_map[tid] = position
                        position += 1

        return self._reorder_from_tid(tid_map, inplace)

    def reorder_upward_column(self, direction="right", layer_tags=None, inplace=False):
        direction = "u" + direction[0]
        return self.reorder(direction=direction, layer_tags=layer_tags, inplace=inplace)

    def reorder_downward_column(self, direction="right", layer_tags=None, inplace=False):
        direction = "d" + direction[0]
        return self.reorder(direction=direction, layer_tags=layer_tags, inplace=inplace)

    def reorder_right_row(self, direction="upward", layer_tags=None, inplace=False):
        direction = "r" + direction[0]
        return self.reorder(direction=direction, layer_tags=layer_tags, inplace=inplace)

    def reorder_left_row(self, direction="upward", layer_tags=None, inplace=False):
        direction = "l" + direction[0]
        return self.reorder(direction=direction, layer_tags=layer_tags, inplace=inplace)


class FPEPS(FermionTensorNetwork2D,
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
                 row_tag_id='ROW{}', col_tag_id='COL{}',
                 order_iterator=None, **tn_opts):

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

        if order_iterator is None:
            order_iterator = product(range(self.Lx), range(self.Ly))
        for i, j in order_iterator:
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
            if array.ndim != len(array_order):
                raise ValueError("array shape not matching array order")

            transpose_order = tuple(
                array_order.find(x) for x in 'urdlp' if x in array_order
            )

            if transpose_order != tuple(range(len(array_order))):
                array = array.transpose(transpose_order)

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
        super().__init__(tensors, check_collisions=False, **tn_opts)

    @classmethod
    def rand(cls, Lx, Ly, bond_dim, phys_dim=2,
             dtype=float, seed=None, parity=None,
             **peps_opts):
        """Create a random (un-normalized) PEPS.

        Parameters
        ----------
        Lx : int
            The number of rows.
        Ly : int
            The number of columns.
        bond_dim : int
            The bond dimension.
        physical : int, optional
            The physical index dimension.
        dtype : dtype, optional
            The dtype to create the arrays with, default is real double.
        seed : int, optional
            A random seed.
        parity: int or int array of (0,1), optional
            parity for each site, default is random parity for all sites
        peps_opts
            Supplied to :class:`~quimb.tensor.tensor_2d.PEPS`.

        Returns
        -------
        psi : PEPS
        """
        if seed is not None:
            np.random.seed(seed)

        arrays = [[None for _ in range(Ly)] for _ in range(Lx)]

        from pyblock3.algebra.fermion import SparseFermionTensor
        from pyblock3.algebra.symmetry import SZ, BondInfo

        if isinstance(parity, np.ndarray):
            if not parity.shape != (Lx, Ly):
                raise ValueError("parity array shape not matching (Lx, Ly)")
        elif isinstance(parity, int):
            parity = np.ones((Lx, Ly), dtype=int) * (parity % 2)
        elif parity is None:
            parity = np.random.randint(0,2,Lx*Ly).reshape(Lx, Ly)
        else:
            raise TypeError("parity type not recoginized")

        vir_info = BondInfo({SZ(0): bond_dim, SZ(1): bond_dim})
        phy_info = BondInfo({SZ(0): phys_dim, SZ(1): phys_dim})

        for i, j in product(range(Lx), range(Ly)):

            shape = []
            if i != Lx - 1:  # bond up
                shape.append(vir_info)
            if j != Ly - 1:  # bond right
                shape.append(vir_info)
            if i != 0:  # bond down
                shape.append(vir_info)
            if j != 0:  # bond left
                shape.append(vir_info)

            shape.append(phy_info)
            dq = SZ(parity[i][j])

            arrays[i][j] = SparseFermionTensor.random(shape, dq=dq, dtype=dtype).to_flat()


        return cls(arrays, **peps_opts)


class FermionTensorNetwork2DVector(TensorNetwork2DVector,
                                   FermionTensorNetwork2D,
                                   FermionTensorNetwork):


    def to_dense(self, *inds_seq, **contract_opts):
        raise NotImplementedError


    def make_norm(
        self,
        mangle_append='*',
        layer_tags=('KET', 'BRA'),
        return_all=False,
    ):
        """Make the norm tensor network of this 2D vector.

        Parameters
        ----------
        mangle_append : {str, False or None}, optional
            How to mangle the inner indices of the bra.
        layer_tags : (str, str), optional
            The tags to identify the top and bottom.
        return_all : bool, optional
            Return the norm, the ket and the bra.
        """
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
        **compress_opts
    ):
        check_opt("contract", contract, (False, True, 'split', 'reduce-split'))

        psi = self if inplace else self.copy()

        if is_lone_coo(where):
            where = (where,)
        else:
            where = tuple(where)
        ng = len(where)

        dp = psi.phys_dim(*where[0])
        tags = tags_to_oset(tags)

        # allow a matrix to be reshaped into a tensor if it factorizes
        #     i.e. (4, 4) assumed to be two qubit gate -> (2, 2, 2, 2)

        site_ix = [psi.site_ind(i, j) for i, j in where]
        # new indices to join old physical sites to new gate
        bnds = [rand_uuid() for _ in range(ng)]
        reindex_map = dict(zip(site_ix, bnds))

        TG = Tensor(G, inds=site_ix + bnds, tags=tags, left_inds=bnds)

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

            # get the sites that used to have the physical indices
            site_tids = psi._get_tids_from_inds(bnds, which='any')

            # pop the sites, contract, then re-add
            pts = [psi._pop_tensor(tid) for tid in site_tids]
            psi |= tensor_contract(*pts, TG)

            return psi

        # following are all based on splitting tensors to maintain structure
        ij_a, ij_b = where

        long_range_path_sequence = None
        manual_lr_path = False
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


    def compute_norm(
        self,
        layer_tags=('KET', 'BRA'),
        **contract_opts,
    ):
        """Compute the norm of this vector via boundary contraction.
        """
        raise NotImplementedError
        norm = self.make_norm(layer_tags=layer_tags)
        return norm.contract_boundary(layer_tags=layer_tags, **contract_opts)

    def compute_local_expectation(
        self,
        terms,
        normalized=False,
        autogroup=True,
        contract_optimize='auto-hq',
        return_all=False,
        plaquette_envs=None,
        plaquette_map=None,
        **plaquette_env_options,
    ):

        raise NotImplementedError

    def normalize(
        self,
        balance_bonds=False,
        equalize_norms=False,
        inplace=False,
        **boundary_contract_opts,
    ):
        raise NotImplementedError
