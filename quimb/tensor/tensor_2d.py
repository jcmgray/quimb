"""Classes and algorithms related to 2D tensor networks.
"""

from itertools import product
from collections import defaultdict

from autoray import do

from ..gen.rand import randn, seed_rand
from ..utils import print_multi_line
from .tensor_core import (
    Tensor,
    tags2set,
    rand_uuid,
    TensorNetwork,
)
from .array_ops import sensibly_scale


class TensorNetwork2D(TensorNetwork):
    r"""Mixin class for tensor networks with a square lattice two-dimensional
    structure, indexed by ``[{row},{column}]`` so that::

                     'COL{j}'
                        v

        i=Lx-1 ●--●--●--●--●--●--   --●
               |  |  |  |  |  |       |
                     ...
               |  |  |  |  |  | 'I{i},{j}' = 'I3,5' e.g.
        i=3    ●--●--●--●--●--●--
               |  |  |  |  |  |       |
        i=2    ●--●--●--●--●--●--   --●    <== 'ROW{i}'
               |  |  |  |  |  |  ...  |
        i=1    ●--●--●--●--●--●--   --●
               |  |  |  |  |  |       |
        i=0    ●--●--●--●--●--●--   --●

             j=0, 1, 2, 3, 4, 5    j=Ly-1

    This implies the following conventions:

        * the 'up' bond is coordinates ``(i, j), (i + 1, j)``
        * the 'down' bond is coordinates ``(i, j), (i - 1, j)``
        * the 'right' bond is coordinates ``(i, j), (i, j + 1)``
        * the 'left' bond is coordinates ``(i, j), (i, j - 1)``

    """

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_row_tag_id',
        '_col_tag_id',
        '_Lx',
        '_Ly',
    )

    @property
    def Lx(self):
        """The number of rows.
        """
        return self._Lx

    @property
    def Ly(self):
        """The number of columns.
        """
        return self._Ly

    @property
    def site_tag_id(self):
        """The string specifier for tagging each site of this 2D TN.
        """
        return self._site_tag_id

    def site_tag(self, i, j):
        """The name of the tag specifiying the tensor at site ``(i, j)``.
        """
        if not isinstance(i, str):
            i = i % self.Lx
        if not isinstance(j, str):
            j = j % self.Ly
        return self.site_tag_id.format(i, j)

    @property
    def row_tag_id(self):
        """The string specifier for tagging each row of this 2D TN.
        """
        return self._row_tag_id

    def row_tag(self, i):
        if not isinstance(i, str):
            i = i % self.Lx
        return self.row_tag_id.format(i)

    @property
    def col_tag_id(self):
        """The string specifier for tagging each column of this 2D TN.
        """
        return self._col_tag_id

    def col_tag(self, j):
        if not isinstance(j, str):
            j = j % self.Ly
        return self.col_tag_id.format(j)

    @property
    def site_tags(self):
        """All of the site tags.
        """
        return tuple(self.site_tag(i, j)
                     for i in range(self.Lx) for j in range(self.Ly))

    def maybe_convert_coo(self, x):
        """Check if ``x`` is a tuple of two ints and convert to the
        corresponding site tag if so.
        """
        if not isinstance(x, str):
            try:
                i, j = map(int, x)
                return self.site_tag(i, j)
            except (ValueError, TypeError):
                pass
        return x

    def _get_tids_from_tags(self, tags, which='all'):
        """This is the function that lets coordinates such as ``(i, j)`` be
        used for many 'tag' based functions.
        """
        tags = self.maybe_convert_coo(tags)
        return super()._get_tids_from_tags(tags, which=which)

    def gen_site_coos(self):
        """Generate coordinates for all the sites in this 2D TN.
        """
        for i in range(self.Lx):
            for j in range(self.Ly):
                yield (i, j)

    def gen_bond_coos(self):
        """Generate pairs of coordinates for all the bonds in this 2D TN.
        """
        return gen_2d_bond_pairs(self.Lx, self.Ly)

    def canonize_row(self, i, around=(0, 1)):
        r"""Canonize a row of this 2D tensor network around sites ``around``::

             |  |  |  |  |         |  |  |  |  |
            -●--●--●--●--●-       -●--●--●--●--●-
             |  |  |  |  |         |  |  |  |  |
            -●--●--●--●--●-  ==>  ->--○--○--<--<- row=i
             |  |  |  |  |         |  |  |  |  |
            -●--●--●--●--●-       -●--●--●--●--●-
             |  |  |  |  |         |  |  |  |  |
                .  .                  .  .
               around                around

        Does not imply an orthogonal form in the same way as in 1D.
        """
        # sweep to the right
        for j in range(min(around)):
            self.canonize_between((i, j), (i, j + 1))

        # sweep to the left
        for j in range(self.Ly - 2, max(around), -1):
            self.canonize_between((i, j + 1), (i, j))

    def compress_row(self, i, sweep='right', **compress_opts):
        r"""Compress a row of this 2D tensor network::

                   |  |  |  |  |         |  |  |  |  |
                  -●--●--●--●--●-       -●--●--●--●--●-
                   |  |  |  |  |         |  |  |  |  |
            row=i -●==●==●==●==●-  ==>  ->-->-->--○==●-  ==>  ...
                   |  |  |  |  |         |  |  |  |  |
                  -●--●--●--●--●-       -●--●--●--●--●-
                   |  |  |  |  |         |  |  |  |  |

        Sweeping either to the right or to the left.
        """
        js, absorb = {
            'right': (range(self.Ly - 1), 'right'),
            'left': (range(self.Ly - 2, -1, -1), 'left'),
        }[sweep]

        for j in js:
            self.compress_between((i, j), (i, j + 1),
                                  absorb=absorb, **compress_opts)

    def __getitem__(self, key):
        """Key based tensor selection, checking for integer based shortcut.
        """
        return super().__getitem__(self.maybe_convert_coo(key))

    def show(self):
        """Print a unicode schematic of this PEPS and its bond dimensions.
        """
        show_2d(self)


class TensorNetwork2DVector(TensorNetwork2D,
                            TensorNetwork):
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

    @property
    def site_ind_id(self):
        return self._site_ind_id

    def site_ind(self, i, j):
        if not isinstance(i, str):
            i = i % self.Lx
        if not isinstance(j, str):
            j = j % self.Ly
        return self.site_ind_id.format(i, j)

    @property
    def site_inds(self):
        """All of the site inds.
        """
        return tuple(self.site_ind(i, j) for i, j in self.gen_site_coos())

    def to_dense(self, *inds_seq, **contract_opts):
        """Return the dense ket version of this 2D vector, i.e. a ``qarray``
        with shape (-1, 1).
        """
        if not inds_seq:
            # just use list of site indices
            return do('reshape', TensorNetwork.to_dense(
                self, self.site_inds, **contract_opts
            ), (-1, 1))

        return TensorNetwork.to_dense(self, *inds_seq, **contract_opts)

    def phys_dim(self, i=0, j=0):
        """Get the size of the physical indices / a specific physical index.
        """
        return self.ind_size(self.site_ind(i, j))


class TensorNetwork2DOperator(TensorNetwork2D,
                              TensorNetwork):
    """Mixin class  for a 2D square lattice TN operator, i.e. one with both
    'upper' and 'lower' site (physical) indices.
    """

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_row_tag_id',
        '_col_tag_id',
        '_Lx',
        '_Ly',
        '_upper_ind_id',
        '_lower_ind_id',
    )

    @property
    def lower_ind_id(self):
        return self._lower_ind_id

    def lower_ind(self, i, j):
        if not isinstance(i, str):
            i = i % self.Lx
        if not isinstance(j, str):
            j = j % self.Ly
        return self.lower_ind_id.format(i, j)

    @property
    def lower_inds(self):
        """All of the lower inds.
        """
        return tuple(self.lower_ind(i, j) for i, j in self.gen_site_coos())

    @property
    def upper_ind_id(self):
        return self._upper_ind_id

    def upper_ind(self, i, j):
        if not isinstance(i, str):
            i = i % self.Lx
        if not isinstance(j, str):
            j = j % self.Ly
        return self.upper_ind_id.format(i, j)

    @property
    def upper_inds(self):
        """All of the upper inds.
        """
        return tuple(self.upper_ind(i, j) for i, j in self.gen_site_coos())

    def to_dense(self, *inds_seq, **contract_opts):
        """Return the dense matrix version of this 2D operator, i.e. a
        ``qarray`` with shape (d, d).
        """
        if not inds_seq:
            inds_seq = (self.upper_inds, self.lower_inds)

        return TensorNetwork.to_dense(self, *inds_seq, **contract_opts)

    def phys_dim(self, i=0, j=0, which='upper'):
        """Get a physical index size of this 2D operator.
        """
        if which == 'upper':
            return self[i, j].ind_size(self.upper_ind(i))

        if which == 'lower':
            return self[i, j].ind_size(self.lower_ind(i))


class TensorNetwork2DFlat(TensorNetwork2D,
                          TensorNetwork):
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

    def bond(self, coo1, coo2):
        """Get the name of the index defining the bond between sites at
        ``coo1`` and ``coo2``.
        """
        b_ix, = self[coo1].bonds(self[coo2])
        return b_ix

    def bond_size(self, coo1, coo2):
        """Return the size of the bond between sites at ``coo1`` and ``coo2``.
        """
        b_ix = self.bond(coo1, coo2)
        return self[coo1].ind_size(b_ix)


class PEPS(TensorNetwork2DVector,
           TensorNetwork2DFlat,
           TensorNetwork2D,
           TensorNetwork):
    r"""Projected Entangled Pair States object::


             ●----●----●----●----●----●--
            /|   /|   /|   /|   /|   /|
             |    |    |    |    |    |
             ●----●----●----●----●----●--
            /|   /|   /|   /|   /|   /|
             |    |    |    |    |    |   ...
             ●----●----●----●----●----●--
            /|   /|   /|   /|   /|   /|
             |    |    |    |    |    |
             ●----●----●----●----●----●--
            /|   /|   /|   /|   /|   /|
                        ...

    Parameters
    ----------
    arrays : sequence of sequence of array
        The core tensor data arrays.
    shape : str, optional
        Which order the dimensions of the arrays are stored in, the default
        ``'urdlp'`` stands for ('up', 'right', 'down', 'left', 'physical').
        Arrays on the edge of lattice are assumed to be missing the
        corresponding dimension.
    tags : set[str], optional
        Extra global tags to add to the tensor network.
    site_ind_id : str, optional
        String specifier for naming convention of site indices.
    site_tag_id : str, optional
        String specifier for naming convention of site tags.
    row_tag_id : str, optional
        String specifier for naming convention of row tags.
    col_tag_id : str, optional
        String specifier for naming convention of column tags.
    """

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

        if isinstance(arrays, PEPS):
            super().__init__(arrays)
            return

        tags = tags2set(tags)
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
                array = do('squeeze', array)

            transpose_order = tuple(
                array_order.find(x) for x in 'urdlp' if x in array_order
            )
            if transpose_order != tuple(range(len(array_order))):
                array = do('transpose', array, transpose_order)

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
            ij_tags = tags.union({self.site_tag(i, j),
                                  self.row_tag(i),
                                  self.col_tag(j)})

            # create the site tensor!
            tensors.append(Tensor(data=array, inds=inds, tags=ij_tags))

        super().__init__(tensors, check_collisions=False, **tn_opts)

    @classmethod
    def rand(cls, Lx, Ly, bond_dim, phys_dim=2,
             dtype=float, seed=None, **peps_opts):
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
        peps_opts
            Supplied to :class:`~quimb.tensor.tensor_2d.PEPS`.

        Returns
        -------
        psi : PEPS
        """
        if seed is not None:
            seed_rand(seed)

        arrays = [[None for _ in range(Ly)] for _ in range(Lx)]

        for i, j in product(range(Lx), range(Ly)):

            shape = []
            if i != Lx - 1:  # bond up
                shape.append(bond_dim)
            if j != Ly - 1:  # bond right
                shape.append(bond_dim)
            if i != 0:  # bond down
                shape.append(bond_dim)
            if j != 0:  # bond left
                shape.append(bond_dim)
            shape.append(phys_dim)

            arrays[i][j] = sensibly_scale(randn(shape, dtype=dtype))

        return cls(arrays, **peps_opts)

    def show(self):
        """Print a unicode schematic of this PEPS and its bond dimensions.
        """
        show_2d(self, show_lower=True)


def gen_2d_bond_pairs(Lx, Ly, cyclic=(False, False)):
    r"""Utility for generating all the bond coordinates for a grid::

                 ...

          |       |       |       |
          |       |       |       |
        (1,0)---(1,1)---(1,2)---(1,3)---
          |       |       |       |
          |       |       |       |
        (1,0)---(1,1)---(1,2)---(1,3)---  ...
          |       |       |       |
          |       |       |       |
        (0,0)---(0,1)---(0,2)---(0,3)---

    Including with cyclic boundary conditions in either or both directions.
    """
    for i in range(Lx):
        for j in range(Ly):
            if i + 1 < Lx:
                yield (i, j), (i + 1, j)
            if j + 1 < Ly:
                yield (i, j), (i, j + 1)

            if cyclic[0] and i == Lx - 1:
                yield (0, j), (i, j)

            if cyclic[1] and j == Ly - 1:
                yield (i, 0), (i, j)


def show_2d(tn_2d, show_lower=False, show_upper=False):
    """Base function for printing a unicode schematic of flat 2D TNs.
    """

    lb = '╱' if show_lower else ' '
    ub = '╱' if show_upper else ' '

    line0 = ' ' + (f' {ub}{{:^3}}' * (tn_2d.Ly - 1)) + f' {ub}'
    bszs = [tn_2d.bond_size((0, j), (0, j + 1)) for j in range(tn_2d.Ly - 1)]

    lines = [line0.format(*bszs)]

    for i in range(tn_2d.Lx - 1):
        lines.append(' ●' + ('━━━━●' * (tn_2d.Ly - 1)))

        # vertical bonds
        lines.append(f'{lb}┃{{:<3}}' * tn_2d.Ly)
        bszs = [tn_2d.bond_size((i, j), (i + 1, j)) for j in range(tn_2d.Ly)]
        lines[-1] = lines[-1].format(*bszs)

        # horizontal bonds below
        lines.append(' ┃' + (f'{ub}{{:^3}}┃' * (tn_2d.Ly - 1)) + f'{ub}')
        bszs = [tn_2d.bond_size((i + 1, j), (i + 1, j + 1))
                for j in range(tn_2d.Ly - 1)]
        lines[-1] = lines[-1].format(*bszs)

    lines.append(' ●' + ('━━━━●' * (tn_2d.Ly - 1)))
    lines.append(f'{lb}    ' * tn_2d.Ly)

    print_multi_line(*lines)


def contract_2d_one_layer_boundary(
    tn,
    max_bond,
    horizontal_canonize=True,
    alternate=True,
    **compress_opts,
):
    """Contract a 2D tensor network assumed to consist of one layer::

            ●--●--●--●--●
            |  |  |  |  |
            ●--●--●--●--●
            |  |  |  |  |
            ●--●--●--●--●

    I.e. representing a scalar with a single tensor per site.

    Parameters
    ----------
    tn : TensorNetwork2D
        The tensor network to contract.
    max_bond : int
        The maximum bond to use in the boundary 'MPS'.
    horizontal_canonize : bool, optional
        Whether to canonicalize along each newly formed double row before
        compressing.
    alternate : bool, optional
        Whether to alternate between sweeps to the right and sweeps to the
        left.
    compress_opts
        Supplied to :meth:`~quimb.tensor.tensor_core.compress_between`.

    See Also
    --------
    contract_2d_two_layer_boundary
    """
    compress_opts['max_bond'] = max_bond

    tn = tn.copy()

    for i in range(tn.Lx - 2):
        #
        #     |  |  |  |  |
        #     ●--●--●--●--●       |  |  |  |  |
        #     |  |  |  |  |  -->  ●==●==●==●==●
        #     ●--●--●--●--●
        #
        for j in range(tn.Ly):
            tn.contract_between((i, j), (i + 1, j))

        if (not alternate) or (i % 2 == 0):
            if horizontal_canonize:
                #
                #     |  |  |  |  |
                #     ●==●==<==<==<
                #
                tn.canonize_row(i, around=(0, 1))
            #
            #     |  |  |  |  |  -->  |  |  |  |  |  -->  |  |  |  |  |
            #     >--●==●==●==●  -->  >-->--●==●==●  -->  >-->-->--●==●
            #     .  .           -->     .  .        -->        .  .
            tn.compress_row(i, sweep='right', **compress_opts)
        else:
            if horizontal_canonize:
                #
                #     |  |  |  |  |
                #     >==>==>==●==●
                #
                tn.canonize_row(i, around=(tn.Ly - 2, tn.Ly - 1))
            #
            #     |  |  |  |  |  <--  |  |  |  |  |  <--  |  |  |  |  |
            #     ●==●--<--<--<  <--  ●==●==●--<--<  <--  ●==●==●==●--<
            #        .  .        <--        .  .     <--           .  .
            #
            tn.compress_row(i, sweep='left', **compress_opts)

    # once we are down to two layers just contract fully:
    #
    #     ●--●--●--●--●--
    #     |  |  |  |  |   ...
    #     ●==●==●==●==●==
    #
    return tn.contract(all, optimize='auto-hq')


def contract_2d_two_layer_boundary(
    tn,
    max_bond,
    upper_tag='KET',
    lower_tag='BRA',
    horizontal_canonize=True,
    **compress_opts,
):
    """Contract a 2D tensor network assumed to consist of two layers::

              /   /   /   /   /   /   /
            -●---●---●---●---●---●---●-
            /|  /|  /|  /|  /|  /|  /|
          -●---●---●---●---●---●---●-|
          /|-○/|-○/|-○/|-○/|-○/|-○/|-○-
        -●---●---●---●---●---●---●-|/
        /|-○/|-○/|-○/|-○/|-○/|-○/|-○-
         |/  |/  |/  |/  |/  |/  |/
        -○---○---○---○---○---○---○-
        /   /   /   /   /   /   /

    I.e. representing a scalar with a pair or upper and lower tensors per site.
    This has, for the same ``max_bond``, better scaling than
    :func:`~quimb.tensor.tensor_2d.contract_2d_one_layer_boundary` if used on
    the same tensor network but flattened.

    Parameters
    ----------
    tn : TensorNetwork2D
        The tensor network to contract.
    max_bond : int
        The maximum bond to use in the boundary 'MPO'.
    upper_tag : str, optional
        The tag assumed to denote the upper layer of tensors.
    upper_tag : str, optional
        The tag assumed to denote the layer layer of tensors.
    horizontal_canonize : bool, optional
        Whether to canonicalize along each newly formed double row before
        compressing.
    compress_opts
        Supplied to :meth:`~quimb.tensor.tensor_core.compress_between`.

    See Also
    --------
    contract_2d_one_layer_boundary
    """
    compress_opts['max_bond'] = max_bond

    tn = tn.copy()

    # initial row 0 'MPO' formation
    for j in range(tn.Ly):
        #
        #    | || || || || |       | || || || || |
        #    ●-○●-○●-○●-○●-○       ●-○●-○●-○●-○●-○
        #    | || || || || |  ==>   \| \| \| \| \|
        #    ●-○●-○●-○●-○●-○         ●--●--●--●--●
        #
        tn ^= tn.site_tag(0, j)

    # row by row
    for i in range(tn.Lx - 3):
        if horizontal_canonize:
            #
            #    | || || || || |
            #    ●-○●-○●-○●-○●-○
            #     \| \| \| \| \|
            #      ●==●==<==<==<  <--
            #
            tn.canonize_row(i)
        #
        #    | || || || || |
        #    ●-○●-○●-○●-○●-○
        #     \| \| \| \| \|
        #      >-->-->--●==<  -->
        #
        tn.compress_row(i, **compress_opts)

        for j in range(tn.Ly):
            #
            #    | || || || || |
            #    | ○--○--○--○--○
            #    |/ |/ |/ |/ |/
            #    ●==●==●==●==●
            #
            tn.contract_between(
                [tn.site_tag(i, j)],
                [tn.site_tag(i + 1, j), lower_tag]
            )

        if horizontal_canonize:
            #
            #    | || || || || |
            #    | ○--○--○--○--○
            #    |/ |/ |/ |/ |/
            #    ●==●==<==<==<    <--
            #
            tn.canonize_row(i)
        #
        #    | || || || || |
        #    | ○--○--○--○--○
        #    |/ |/ |/ |/ |/
        #    >-->-->--●==<    -->
        #
        tn.compress_row(i, **compress_opts)

        #
        #     \| \| \| \| \|
        #      ●==●==●==●==●
        #
        for j in range(tn.Ly):
            tn ^= (i + 1, j)

    # once we are down to a few rows just contract all together
    return tn.contract(all, optimize='auto-hq')
