import functools
from numbers import Integral
from itertools import product, starmap
from collections import defaultdict

from autoray import do

from ..gen.rand import randn, seed_rand
from . import array_ops as ops
from .tensor_core import (
    Tensor,
    TensorNetwork,
    oset,
    tags_to_oset,
    rand_uuid,
)


def gen_3d_bonds(Lx, Ly, Lz, steppers, coo_filter=None):
    """Convenience function for tiling pairs of bond coordinates on a 3D
    lattice given a function like ``lambda i, j, k: (i + 1, j + 1, k + 1)``.

    Parameters
    ----------
    Lx : int
        The number of x-slices.
    Ly : int
        The number of y-slices.
    Lz : int
        The number of z-slices.
    steppers : callable or sequence of callable
        Function(s) that take args ``(i, j, k)`` and generate another
        coordinate, thus defining a bond.
    coo_filter : callable
        Function that takes args ``(i, j, k)`` and only returns ``True`` if
        this is to be a valid starting coordinate.

    Yields
    ------
    bond : tuple[tuple[int, int, int], tuple[int, int, int]]
        A pair of coordinates.

    Examples
    --------

    Generate nearest neighbor bonds:

        >>> for bond in gen_3d_bonds(2, 2, 2, [lambda i, j, k: (i + 1, j, k),
        ...                                    lambda i, j, k: (i, j + 1, k),
        ...                                    lambda i, j, k: (i, j, k + 1)]):
        ...     print(bond)
        ((0, 0, 0), (1, 0, 0))
        ((0, 0, 0), (0, 1, 0))
        ((0, 0, 0), (0, 0, 1))
        ((0, 0, 1), (1, 0, 1))
        ((0, 0, 1), (0, 1, 1))
        ((0, 1, 0), (1, 1, 0))
        ((0, 1, 0), (0, 1, 1))
        ((0, 1, 1), (1, 1, 1))
        ((1, 0, 0), (1, 1, 0))
        ((1, 0, 0), (1, 0, 1))
        ((1, 0, 1), (1, 1, 1))
        ((1, 1, 0), (1, 1, 1))

    """

    if callable(steppers):
        steppers = (steppers,)

    for i, j, k in product(range(Lx), range(Ly), range(Lz)):
        if (coo_filter is None) or coo_filter(i, j, k):
            for stepper in steppers:
                i2, j2, k2 = stepper(i, j, k)
                if (0 <= i2 < Lx) and (0 <= j2 < Ly) and (0 <= k2 < Lz):
                    yield (i, j, k), (i2, j2, k2)


class TensorNetwork3D(TensorNetwork):

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_x_tag_id',
        '_y_tag_id',
        '_z_tag_id',
        '_Lx',
        '_Ly',
        '_Lz',
    )

    def _compatible_3d(self, other):
        """Check whether ``self`` and ``other`` are compatible 3D tensor
        networks such that they can remain a 3D tensor network when combined.
        """
        return (
            isinstance(other, TensorNetwork3D) and
            all(getattr(self, e) == getattr(other, e)
                for e in TensorNetwork3D._EXTRA_PROPS)
        )

    def __and__(self, other):
        new = super().__and__(other)
        if self._compatible_3d(other):
            new.view_as_(TensorNetwork3D, like=self)
        return new

    def __or__(self, other):
        new = super().__or__(other)
        if self._compatible_3d(other):
            new.view_as_(TensorNetwork3D, like=self)
        return new

    @property
    def Lx(self):
        """The number of x-slices.
        """
        return self._Lx

    @property
    def Ly(self):
        """The number of y-slices.
        """
        return self._Ly

    @property
    def Lz(self):
        """The number of z-slices.
        """
        return self._Lz

    @property
    def nsites(self):
        """The total number of sites.
        """
        return self._Lx * self._Ly * self._Lz

    @property
    def site_tag_id(self):
        """The string specifier for tagging each site of this 3D TN.
        """
        return self._site_tag_id

    def site_tag(self, i, j, k):
        """The name of the tag specifiying the tensor at site ``(i, j, k)``.
        """
        if not isinstance(i, str):
            i = i % self.Lx
        if not isinstance(j, str):
            j = j % self.Ly
        if not isinstance(k, str):
            k = k % self.Lz
        return self.site_tag_id.format(i, j, k)

    @property
    def x_tag_id(self):
        """The string specifier for tagging each x-slice of this 3D TN.
        """
        return self._x_tag_id

    def x_tag(self, i):
        if not isinstance(i, str):
            i = i % self.Lx
        return self.x_tag_id.format(i)

    @property
    def x_tags(self):
        """A tuple of all of the ``Lx`` different x-slice tags.
        """
        return tuple(map(self.x_tag, range(self.Lx)))

    @property
    def y_tag_id(self):
        """The string specifier for tagging each y-slice of this 3D TN.
        """
        return self._y_tag_id

    def y_tag(self, j):
        if not isinstance(j, str):
            j = j % self.Ly
        return self.y_tag_id.format(j)

    @property
    def y_tags(self):
        """A tuple of all of the ``Ly`` different y-slice tags.
        """
        return tuple(map(self.y_tag, range(self.Ly)))

    @property
    def z_tag_id(self):
        """The string specifier for tagging each z-slice of this 3D TN.
        """
        return self._z_tag_id

    def z_tag(self, k):
        if not isinstance(k, str):
            k = k % self.Lz
        return self.z_tag_id.format(k)

    @property
    def z_tags(self):
        """A tuple of all of the ``Lz`` different z-slice tags.
        """
        return tuple(map(self.z_tag, range(self.Lz)))

    @property
    def site_tags(self):
        """All of the ``Lx * Ly`` site tags.
        """
        return tuple(starmap(self.site_tag, self.gen_site_coos()))

    def maybe_convert_coo(self, coo):
        """Check if ``coo`` is a tuple of three ints and convert to the
        corresponding site tag if so.
        """
        if not isinstance(coo, str):
            try:
                i, j, k = map(int, coo)
                return self.site_tag(i, j, k)
            except (ValueError, TypeError):
                pass
        return coo

    def _get_tids_from_tags(self, tags, which='all'):
        """This is the function that lets coordinates such as ``(i, j, k)`` be
        used for many 'tag' based functions.
        """
        tags = self.maybe_convert_coo(tags)
        return super()._get_tids_from_tags(tags, which=which)

    def gen_site_coos(self):
        """Generate coordinates for all the sites in this 3D TN.
        """
        return product(range(self.Lx), range(self.Ly), range(self.Lz))

    def gen_bond_coos(self):
        """Generate pairs of coordinates for all the bonds in this 3D TN.
        """
        return gen_3d_bonds(self.Lx, self.Ly, self.Lz, steppers=[
            lambda i, j, k: (i + 1, j, k),
            lambda i, j, k: (i, j + 1, k),
            lambda i, j, k: (i, j, k + 1),
        ])

    def __getitem__(self, key):
        """Key based tensor selection, checking for integer based shortcut.
        """
        return super().__getitem__(self.maybe_convert_coo(key))

    def __repr__(self):
        """Insert number of slices into standard print.
        """
        s = super().__repr__()
        extra = (f', Lx={self.Lx}, Ly={self.Ly}, Lz={self.Lz}, '
                 f'max_bond={self.max_bond()}')
        s = f'{s[:-2]}{extra}{s[-2:]}'
        return s

    def __str__(self):
        """Insert number of slices into standard print.
        """
        s = super().__repr__()
        extra = (f', Lx={self.Lx}, Ly={self.Ly}, Lz={self.Lz}'
                 f'max_bond={self.max_bond()}')
        s = f'{s[:-2]}{extra}{s[-2:]}'
        return s

    def flatten(self, fuse_multibonds=True, inplace=False):
        """Contract all tensors corresponding to each site into one.
        """
        tn = self if inplace else self.copy()

        for i, j, k in self.gen_site_coos():
            tn ^= (i, j, k)

        if fuse_multibonds:
            tn.fuse_multibonds_()

        return tn.view_as_(TensorNetwork3DFlat, like=self)

    flatten_ = functools.partialmethod(flatten, inplace=True)


def is_lone_coo(where):
    """Check if ``where`` has been specified as a single coordinate triplet.
    """
    return (len(where) == 3) and (isinstance(where[0], Integral))


class TensorNetwork3DVector(TensorNetwork3D,
                            TensorNetwork):
    """Mixin class  for a 3D square lattice vector TN, i.e. one with a single
    physical index per site.
    """

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_x_tag_id',
        '_y_tag_id',
        '_z_tag_id',
        '_Lx',
        '_Ly',
        '_Lz',
        '_site_ind_id',
    )

    @property
    def site_ind_id(self):
        return self._site_ind_id

    def site_ind(self, i, j, k):
        if not isinstance(i, str):
            i = i % self.Lx
        if not isinstance(j, str):
            j = j % self.Ly
        if not isinstance(k, str):
            k = k % self.Lz
        return self.site_ind_id.format(i, j, k)

    def reindex_sites(self, new_id, where=None, inplace=False):
        if where is None:
            where = self.gen_site_coos()

        return self.reindex(
            {
                self.site_ind(*coo): new_id.format(*coo) for coo in where
            },
            inplace=inplace
        )

    @site_ind_id.setter
    def site_ind_id(self, new_id):
        if self._site_ind_id != new_id:
            self.reindex_sites(new_id, inplace=True)
            self._site_ind_id = new_id

    @property
    def site_inds(self):
        """All of the site inds.
        """
        return tuple(starmap(self.site_ind, self.gen_site_coos()))

    def to_dense(self, *inds_seq, **contract_opts):
        """Return the dense ket version of this 3D vector, i.e. a ``qarray``
        with shape (-1, 1).
        """
        if not inds_seq:
            # just use list of site indices
            return do('reshape', TensorNetwork.to_dense(
                self, self.site_inds, **contract_opts
            ), (-1, 1))

        return TensorNetwork.to_dense(self, *inds_seq, **contract_opts)

    def phys_dim(self, i=None, j=None, k=None):
        """Get the size of the physical indices / a specific physical index.
        """
        if (i is not None) and (j is not None) and (k is not None):
            pix = self.site_ind(i, j, k)
        else:
            # allow for when some physical indices might have been contracted
            pix = next(iter(
                ix for ix in self.site_inds if ix in self.ind_map
            ))
        return self.ind_size(pix)

    def gate(
        self,
        G,
        where,
        contract=False,
        tags=None,
        info=None,
        inplace=False,
        **compress_opts,
    ):
        """Apply a gate ``G`` to sites ``where``, preserving the outer site
        inds.
        """
        if is_lone_coo(where):
            where = (where,)
        else:
            where = tuple(where)

        inds = tuple(starmap(self.site_ind, where))
        return super().gate_inds(
            G, inds, contract=contract, tags=tags, info=info, inplace=inplace,
            **compress_opts
        )

    gate_ = functools.partialmethod(gate, inplace=True)


class TensorNetwork3DFlat(TensorNetwork3D,
                          TensorNetwork):
    """Mixin class for a 3D square lattice tensor network with a single tensor
    per site, for example, both PEPS and PEPOs.
    """

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_x_tag_id',
        '_y_tag_id',
        '_z_tag_id',
        '_Lx',
        '_Ly',
        '_Lz',
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


class PEPS3D(TensorNetwork3DVector,
             TensorNetwork3DFlat,
             TensorNetwork3D,
             TensorNetwork):
    r"""Projected Entangled Pair States object (3D).

    Parameters
    ----------
    arrays : sequence of sequence of sequence of array
        The core tensor data arrays.
    shape : str, optional
        Which order the dimensions of the arrays are stored in, the default
        ``'urfdlbp'`` stands for ('up', 'right', 'front', down', 'left',
        'behind', 'physical') meaning (x+, y+, z+, x-, y-, z-, physical)
        respectively. Arrays on the edge of lattice are assumed to be missing
        the corresponding dimension.
    tags : set[str], optional
        Extra global tags to add to the tensor network.
    site_ind_id : str, optional
        String specifier for naming convention of site indices.
    site_tag_id : str, optional
        String specifier for naming convention of site tags.
    x_tag_id : str, optional
        String specifier for naming convention of x-slice tags.
    y_tag_id : str, optional
        String specifier for naming convention of y-slice tags.
    z_tag_id : str, optional
        String specifier for naming convention of z-slice tags.
    """

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_x_tag_id',
        '_y_tag_id',
        '_z_tag_id',
        '_Lx',
        '_Ly',
        '_Lz',
        '_site_ind_id',
    )

    def __init__(
        self,
        arrays,
        *,
        shape='urfdlbp',
        tags=None,
        site_ind_id='k{},{},{}',
        site_tag_id='I{},{},{}',
        x_tag_id='X{}',
        y_tag_id='Y{}',
        z_tag_id='Z{}',
        **tn_opts
    ):

        if isinstance(arrays, PEPS3D):
            super().__init__(arrays)
            return

        tags = tags_to_oset(tags)
        self._site_ind_id = site_ind_id
        self._site_tag_id = site_tag_id
        self._x_tag_id = x_tag_id
        self._y_tag_id = y_tag_id
        self._z_tag_id = z_tag_id

        arrays = tuple(tuple(tuple(z for z in y) for y in x) for x in arrays)
        self._Lx = len(arrays)
        self._Ly = len(arrays[0])
        self._Lz = len(arrays[0][0])
        tensors = []

        # cache for both creating and retrieving indices
        ix = defaultdict(rand_uuid)

        for i, j, k in self.gen_site_coos():
            array = arrays[i][j][k]

            # figure out if we need to transpose the arrays from some order
            #     other than up right front down left behind physical
            array_order = shape
            if i == self.Lx - 1:
                array_order = array_order.replace('u', '')
            if j == self.Ly - 1:
                array_order = array_order.replace('r', '')
            if k == self.Lz - 1:
                array_order = array_order.replace('f', '')
            if i == 0:
                array_order = array_order.replace('d', '')
            if j == 0:
                array_order = array_order.replace('l', '')
            if k == 0:
                array_order = array_order.replace('b', '')

            # allow convention of missing bonds to be singlet dimensions
            if len(array.shape) != len(array_order):
                array = do('squeeze', array)

            transpose_order = tuple(
                array_order.find(x) for x in 'urfdlbp' if x in array_order
            )
            if transpose_order != tuple(range(len(array_order))):
                array = do('transpose', array, transpose_order)

            # get the relevant indices corresponding to neighbours
            inds = []
            if 'u' in array_order:
                inds.append(ix[(i, j, k), (i + 1, j, k)])
            if 'r' in array_order:
                inds.append(ix[(i, j, k), (i, j + 1, k)])
            if 'f' in array_order:
                inds.append(ix[(i, j, k), (i, j, k + 1)])
            if 'd' in array_order:
                inds.append(ix[(i - 1, j, k), (i, j, k)])
            if 'l' in array_order:
                inds.append(ix[(i, j - 1, k), (i, j, k)])
            if 'b' in array_order:
                inds.append(ix[(i, j, k - 1), (i, j, k)])
            inds.append(self.site_ind(i, j, k))

            # mix site, slice and global tags
            ijk_tags = tags | oset((self.site_tag(i, j, k), self.x_tag(i),
                                   self.y_tag(j), self.z_tag(k)))

            # create the site tensor!
            tensors.append(Tensor(data=array, inds=inds, tags=ijk_tags))

        super().__init__(tensors, virtual=True, **tn_opts)

    @classmethod
    def from_fill_fn(
        cls, fill_fn, Lx, Ly, Lz, bond_dim, phys_dim=2, **peps3d_opts
    ):
        """Create a 3D PEPS from a filling function with signature
        ``fill_fn(shape)``.

        Parameters
        ----------
        Lx : int
            The number of x-slices.
        Ly : int
            The number of y-slices.
        Lz : int
            The number of z-slices.
        bond_dim : int
            The bond dimension.
        physical : int, optional
            The physical index dimension.
        peps_opts
            Supplied to :class:`~quimb.tensor.tensor_3d.PEPS3D`.

        Returns
        -------
        psi : PEPS3D
        """
        arrays = [[[None
                    for _ in range(Lz)]
                   for _ in range(Ly)]
                  for _ in range(Lx)]

        for i, j, k in product(range(Lx), range(Ly), range(Lz)):

            shape = []
            if i != Lx - 1:  # bond up
                shape.append(bond_dim)
            if j != Ly - 1:  # bond right
                shape.append(bond_dim)
            if k != Lz - 1:  # bond front
                shape.append(bond_dim)
            if i != 0:  # bond down
                shape.append(bond_dim)
            if j != 0:  # bond left
                shape.append(bond_dim)
            if k != 0:  # bond behind
                shape.append(bond_dim)
            shape.append(phys_dim)

            arrays[i][j][k] = fill_fn(shape)

        return cls(arrays, **peps3d_opts)

    @classmethod
    def empty(
        self, Lx, Ly, Lz, bond_dim, phys_dim=2, like='numpy', **peps3d_opts,
    ):
        """Create an empty 3D PEPS.

        Parameters
        ----------
        Lx : int
            The number of x-slices.
        Ly : int
            The number of y-slices.
        Lz : int
            The number of z-slices.
        bond_dim : int
            The bond dimension.
        physical : int, optional
            The physical index dimension.
        peps3d_opts
            Supplied to :class:`~quimb.tensor.tensor_3d.PEPS3D`.

        Returns
        -------
        psi : PEPS3D

        See Also
        --------
        PEPS3D.from_fill_fn
        """
        return self.from_fill_fn(
            lambda shape: do("zeros", shape, like=like),
            Lx, Ly, Lz, bond_dim, phys_dim, **peps3d_opts
        )

    @classmethod
    def ones(
        self, Lx, Ly, Lz, bond_dim, phys_dim=2, like='numpy', **peps3d_opts,
    ):
        """Create a 3D PEPS whose tensors are filled with ones.

        Parameters
        ----------
        Lx : int
            The number of x-slices.
        Ly : int
            The number of y-slices.
        Lz : int
            The number of z-slices.
        bond_dim : int
            The bond dimension.
        physical : int, optional
            The physical index dimension.
        peps3d_opts
            Supplied to :class:`~quimb.tensor.tensor_3d.PEPS3D`.

        Returns
        -------
        psi : PEPS3D

        See Also
        --------
        PEPS3D.from_fill_fn
        """
        return self.from_fill_fn(
            lambda shape: do("ones", shape, like=like),
            Lx, Ly, Lz, bond_dim, phys_dim, **peps3d_opts
        )

    @classmethod
    def rand(
        cls, Lx, Ly, Lz, bond_dim, phys_dim=2,
        dtype='float64', seed=None, **peps3d_opts
    ):
        """Create a random (un-normalized) 3D PEPS.

        Parameters
        ----------
        Lx : int
            The number of x-slices.
        Ly : int
            The number of y-slices.
        Lz : int
            The number of z-slices.
        bond_dim : int
            The bond dimension.
        physical : int, optional
            The physical index dimension.
        dtype : dtype, optional
            The dtype to create the arrays with, default is real double.
        seed : int, optional
            A random seed.
        peps_opts
            Supplied to :class:`~quimb.tensor.tensor_3d.PEPS3D`.

        Returns
        -------
        psi : PEPS3D

        See Also
        --------
        PEPS3D.from_fill_fn
        """
        if seed is not None:
            seed_rand(seed)

        def fill_fn(shape):
            return ops.sensibly_scale(ops.sensibly_scale(
                randn(shape, dtype=dtype)))

        return cls.from_fill_fn(
            fill_fn, Lx, Ly, Lz, bond_dim, phys_dim, **peps3d_opts
        )
