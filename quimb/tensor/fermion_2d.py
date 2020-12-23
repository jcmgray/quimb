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

    def contract_boundary_from_bottom(
        self,
        xrange,
        yrange=None,
        canonize=True,
        compress_sweep='left',
        layer_tags=None,
        inplace=False,
        **compress_opts
    ):
        tn = self if inplace else self.copy()
        Lx, Ly = self._Lx, self._Ly
        if yrange is None: yrange = (0, Ly-1)
        for i in range(min(xrange), max(xrange)):
            for j in range(min(yrange), max(yrange) + 1):
                tag1, tag2 = tn.site_tag(i, j), tn.site_tag(i + 1, j)
                if layer_tags is not None:
                    for p in range(len(layer_tags)-1):
                        tn.contract_between((tag1, layer_tags[p]), (tag1, layer_tags[p+1]))
                        tn.contract_between((tag2, layer_tags[p]), (tag2, layer_tags[p+1]))
                tn.contract_between(tag1, tag2)
            self.compress_row(i, sweep=compress_sweep,
                              yrange=yrange, **compress_opts)
        return tn

    def contract_boundary_from_top(
        self,
        xrange,
        yrange=None,
        canonize=True,
        compress_sweep='left',
        layer_tags=None,
        inplace=False,
        **compress_opts
    ):
        tn = self if inplace else self.copy()
        Lx, Ly = self._Lx, self._Ly
        if yrange is None: yrange = (0, Ly-1)
        for i in range(max(xrange), min(xrange), -1):
            for j in range(min(yrange), max(yrange) + 1):
                tag1, tag2 = tn.site_tag(i, j), tn.site_tag(i - 1, j)
                if layer_tags is not None:
                    for p in range(len(layer_tags)-1):
                        tn.contract_between((tag1, layer_tags[p]), (tag1, layer_tags[p+1]))
                        tn.contract_between((tag2, layer_tags[p]), (tag2, layer_tags[p+1]))
                tn.contract_between(tag1, tag2)

            self.compress_row(i, sweep=compress_sweep,
                              yrange=yrange, **compress_opts)
        return tn

    def contract_boundary_from_right(
        self,
        yrange,
        xrange=None,
        canonize=True,
        compress_sweep='down',
        layer_tags=None,
        inplace=False,
        **compress_opts
    ):
        tn = self if inplace else self.copy()
        Lx, Ly = self._Lx, self._Ly
        if xrange is None: xrange = (0, Lx-1)
        for j in range(max(yrange), min(yrange), -1):
            for i in range(min(xrange), max(xrange) + 1):
                tag1, tag2 = tn.site_tag(i, j), tn.site_tag(i, j-1)
                if layer_tags is not None:
                    for p in range(len(layer_tags)-1):
                        tn.contract_between((tag1, layer_tags[p]), (tag1, layer_tags[p+1]))
                        tn.contract_between((tag2, layer_tags[p]), (tag2, layer_tags[p+1]))
                tn.contract_between(tag1, tag2)

            self.compress_column(j, sweep=compress_sweep,
                                 xrange=xrange, **compress_opts)
        return tn

    def contract_boundary_from_left(
        self,
        yrange,
        xrange=None,
        canonize=True,
        compress_sweep='down',
        layer_tags=None,
        inplace=False,
        **compress_opts
    ):
        tn = self if inplace else self.copy()
        Lx, Ly = self._Lx, self._Ly
        if xrange is None: xrange = (0, Lx-1)
        for j in range(min(yrange), max(yrange)):
            for i in range(min(xrange), max(xrange) + 1):
                tag1, tag2 = tn.site_tag(i, j), tn.site_tag(i, j+1)
                if layer_tags is not None:
                    for p in range(len(layer_tags)-1):
                        tn.contract_between((tag1, layer_tags[p]), (tag1, layer_tags[p+1]))
                        tn.contract_between((tag2, layer_tags[p]), (tag2, layer_tags[p+1]))
                tn.contract_between(tag1, tag2)

            self.compress_column(j, sweep=compress_sweep,
                                 xrange=xrange, **compress_opts)
        return tn

    def compute_row_environments(self, layer_tags=None, **compress_opts):
        Lx = self._Lx
        env_bottom = self.reorder_right_row(layer_tags=layer_tags)
        env_top = env_bottom.copy()
        row_envs = dict()

        first_row = env_bottom.row_tag(0)
        row_envs["below", 0] = FermionTensorNetwork([])
        row_envs['below', 1] = env_bottom.select(first_row).simple_copy()
        row_envs['mid', 0] = env_bottom.select(first_row).simple_copy()

        for i in range(2, Lx):
            below_row = env_bottom.row_tag(i-1)
            row_envs["mid", i-1] = env_bottom.select(below_row).simple_copy()
            env_bottom.contract_boundary_from_bottom((i-2, i-1), layer_tags=layer_tags, inplace=True, **compress_opts)
            row_envs['below', i] = env_bottom.select(below_row).simple_copy()

        last_row = env_bottom.row_tag(Lx-1)
        row_envs['mid', Lx-1] = env_bottom.select(last_row).simple_copy()

        row_envs['above', Lx-1] = FermionTensorNetwork([])
        row_envs['above', Lx-2] = env_top.select(last_row).simple_copy()

        for i in range(Lx-3, -1, -1):
            env_top.contract_boundary_from_top((i+1, i+2), layer_tags=layer_tags, inplace=True, **compress_opts)
            row_envs['above', i] = env_top.select(last_row).simple_copy()

        return row_envs

    def compute_col_environments(self, layer_tags=None, **compress_opts):
        Ly = self._Ly
        env_left = self.reorder_upward_column(layer_tags=layer_tags)
        env_right = env_left.copy()
        col_envs = dict()

        first_col = env_left.col_tag(0)
        col_envs["left", 0] = FermionTensorNetwork([])
        col_envs['left', 1] = env_left.select(first_col).simple_copy()
        col_envs['mid', 0] = env_left.select(first_col).simple_copy()

        for i in range(2, Ly):
            left_row = env_left.col_tag(i-1)
            col_envs["mid", i-1] = env_left.select(left_row).simple_copy()
            env_left.contract_boundary_from_left((i-2, i-1), layer_tags=layer_tags, inplace=True, **compress_opts)
            col_envs['left', i] = env_left.select(left_row).simple_copy()

        last_col = env_left.col_tag(Ly-1)
        col_envs['mid', Ly-1] = env_left.select(last_col).simple_copy()

        col_envs['right', Ly-1] = FermionTensorNetwork([])
        col_envs['right', Ly-2] = env_right.select(last_col).simple_copy()

        for i in range(Ly-3, -1, -1):
            env_right.contract_boundary_from_right((i+1, i+2), layer_tags=layer_tags, inplace=True, **compress_opts)
            col_envs['right', i] = env_right.select(last_col).simple_copy()

        return col_envs

    def _reorder_from_tid(self, tid_map, inplace=False):
        tn = self if inplace else self.copy()
        for tid, site in tid_map.items():
            tn.fermion_space.move(tid, site)
        return tn

    def reorder(self, direction="ru", layer_tags=None, inplace=False):
        Lx, Ly = self._Lx, self._Ly
        row_wise = (direction[0] in ["r", "l"])
        iter_dic = {"u": range(Lx),
                    "d": range(Lx)[::-1],
                    "r": range(Ly),
                    "l": range(Ly)[::-1]}
        if row_wise:
            iterator = product(iter_dic[direction[1]], iter_dic[direction[0]])
        else:
            iterator = product(iter_dic[direction[0]], iter_dic[direction[1]])
        position = 0
        tid_map = dict()
        for i, j in iterator:
            x, y = (i, j) if row_wise else (j, i)
            site_tag = self.site_tag(x, y)
            if layer_tags is None:
                tid,  = self._get_tids_from_tags(site_tag)
                tid_map[tid] = position
                position += 1
            else:
                for tag in layer_tags:
                    tid, = self._get_tids_from_tags((site_tag, tag))
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
