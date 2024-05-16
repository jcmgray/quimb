from itertools import product
from ...utils import pairwise
from ..tensor_2d_tebd import SimpleUpdate, conditioner, LocalHam2D
from ..tensor_2d import (
    gen_long_range_path,
    nearest_neighbors,
    gen_long_range_swap_path,
    swap_path_to_long_range_path,
    gen_2d_bonds,
)

from .block_interface import eye, Hubbard
from . import block_tools
from .fermion_core import _get_gauge_location
from .fermion_arbgeom_tebd import LocalHamGen


def Hubbard2D(t, u, Lx, Ly, mu=0.0, symmetry=None):
    """Create a LocalHam2D object for 2D Hubbard Model

    Parameters
    ----------
    t : scalar
        The hopping parameter
    u : scalar
        Onsite columb repulsion
    Lx: int
        Size in x direction
    Ly: int
        Size in y direction
    mu: scalar, optional
        Chemical potential
    symmetry: {"z2",'u1', 'z22', 'u11'}, optional
        Symmetry in the backend

    Returns
    -------
    a LocalHam2D object
    """
    ham = dict()

    def count_neighbour(i, j):
        return (i > 0) + (i < Lx - 1) + (j > 0) + (j < Ly - 1)

    for i, j in product(range(Lx), range(Ly)):
        count_ij = count_neighbour(i, j)
        if i + 1 != Lx:
            where = ((i, j), (i + 1, j))
            count_b = count_neighbour(i + 1, j)
            uop = Hubbard(
                t, u, mu, (1.0 / count_ij, 1.0 / count_b), symmetry=symmetry
            )
            ham[where] = uop
        if j + 1 != Ly:
            where = ((i, j), (i, j + 1))
            count_b = count_neighbour(i, j + 1)
            uop = Hubbard(
                t, u, mu, (1.0 / count_ij, 1.0 / count_b), symmetry=symmetry
            )
            ham[where] = uop
    return LocalHam2D(Lx, Ly, ham)


class LocalHam2D(LocalHamGen):
    """A 2D Fermion Hamiltonian represented as local terms. Different from
    class:`~quimb.tensor.tensor_2d_tebd.LocalHam2D`, this does not combine
    two sites and one site term into a single interaction per lattice pair.

    Parameters
    ----------
    Lx : int
        The number of rows.
    Ly : int
        The number of columns.
    H2 : pyblock3 tensors or dict[tuple[tuple[int]], pyblock3 tensors]
        The two site term(s). If a single array is given, assume to be the
        default interaction for all nearest neighbours. If a dict is supplied,
        the keys should represent specific pairs of coordinates like
        ``((ia, ja), (ib, jb))`` with the values the array representing the
        interaction for that pair. A default term for all remaining nearest
        neighbours interactions can still be supplied with the key ``None``.
    H1 : array_like or dict[tuple[int], array_like], optional
        The one site term(s). If a single array is given, assume to be the
        default onsite term for all terms. If a dict is supplied,
        the keys should represent specific coordinates like
        ``(i, j)`` with the values the array representing the local term for
        that site. A default term for all remaining sites can still be supplied
        with the key ``None``.

    Attributes
    ----------
    terms : dict[tuple[tuple[int]], pyblock3 tensors]
        The total effective local term for each interaction (with single site
        terms appropriately absorbed). Each key is a pair of coordinates
        ``ija, ijb`` with ``ija < ijb``.

    """

    def __init__(self, Lx, Ly, H2, H1=None):
        self.Lx = int(Lx)
        self.Ly = int(Ly)

        # parse two site terms
        if hasattr(H2, "shape"):
            # use as default nearest neighbour term
            H2 = {None: H2}
        else:
            H2 = dict(H2)

        # possibly fill in default gates
        default_H2 = H2.pop(None, None)
        if default_H2 is not None:
            for coo_a, coo_b in gen_2d_bonds(
                Lx,
                Ly,
                steppers=[
                    lambda i, j: (i, j + 1),
                    lambda i, j: (i + 1, j),
                ],
            ):
                if (coo_a, coo_b) not in H2 and (coo_b, coo_a) not in H2:
                    H2[coo_a, coo_b] = default_H2

        super().__init__(H2=H2, H1=H1)

    @property
    def nsites(self):
        """The number of sites in the system."""
        return self.Lx * self.Ly

    draw = LocalHam2D.draw
    __repr__ = LocalHam2D.__repr__


class SimpleUpdate(SimpleUpdate):
    def _initialize_gauges(self):
        """Create unit singular values, stored as tensors."""
        self._gauges = dict()
        string_inv = {"+": "-", "-": "+"}
        for ija, ijb in self._psi.gen_bond_coos():
            Tija = self._psi[ija]
            Tijb = self._psi[ijb]
            site_a = Tija.get_fermion_info()[1]
            site_b = Tijb.get_fermion_info()[1]
            if site_a > site_b:
                Tija, Tijb = Tijb, Tija
                ija, ijb = ijb, ija
            bnd = self._psi.bond(ija, ijb)
            sign_ija = Tija.data.pattern[Tija.inds.index(bnd)]
            bond_info = Tija.bond_info(bnd)
            ax = Tija.inds.index(bnd)
            if sign_ija == "-":
                new_bond_info = dict()
                for dq, dim in bond_info.items():
                    new_bond_info[-dq] = dim
                bond_info = new_bond_info
            Tsval = eye(bond_info)
            Tsval.pattern = sign_ija + string_inv[sign_ija]
            self._gauges[(ija, ijb)] = Tsval

    def _unpack_gauge(self, ija, ijb):
        Ta = self._psi[ija]
        Tb = self._psi[ijb]
        if (ija, ijb) in self.gauges:
            Tsval = self.gauges[(ija, ijb)]
            loca, locb, flip_pattern = _get_gauge_location(Ta, Tb)
        elif (ijb, ija) in self.gauges:
            Tsval = self.gauges[(ijb, ija)]
            locb, loca, flip_pattern = _get_gauge_location(Tb, Ta)
        else:
            raise KeyError("gauge not found")
        return Tsval, (loca, locb), flip_pattern

    def gate(self, U, where):
        """Like ``TEBD2D.gate`` but absorb and extract the relevant gauges
        before and after each gate application.
        """
        ija, ijb = where

        if callable(self.long_range_path_sequence):
            long_range_path_sequence = self.long_range_path_sequence(ija, ijb)
        else:
            long_range_path_sequence = self.long_range_path_sequence

        if self.long_range_use_swaps:
            path = tuple(
                gen_long_range_swap_path(
                    ija, ijb, sequence=long_range_path_sequence
                )
            )
            string = swap_path_to_long_range_path(path, ija)
        else:
            # get the string linking the two sites
            string = path = tuple(
                gen_long_range_path(
                    ija, ijb, sequence=long_range_path_sequence
                )
            )

        def env_neighbours(i, j):
            return tuple(
                filter(
                    lambda coo: self._psi.valid_coo((coo))
                    and coo not in string,
                    nearest_neighbors((i, j)),
                )
            )

        # get the relevant neighbours for string of sites
        neighbours = {site: env_neighbours(*site) for site in string}

        # absorb the 'outer' gauges from these neighbours
        for site in string:
            Tij = self._psi[site]
            for neighbour in neighbours[site]:
                Tsval, (loc_ij, _), flip_pattern = self._unpack_gauge(
                    site, neighbour
                )
                bnd = self._psi.bond(site, neighbour)
                Tij.multiply_index_diagonal_(
                    ind=bnd,
                    x=Tsval,
                    location=loc_ij,
                    flip_pattern=flip_pattern,
                    smudge=self.gauge_smudge,
                )

        # absorb the inner bond gauges equally into both sites along string
        for site_a, site_b in pairwise(string):
            Ta, Tb = self._psi[site_a], self._psi[site_b]
            Tsval, (loca, locb), flip_pattern = self._unpack_gauge(
                site_a, site_b
            )
            bnd = self._psi.bond(site_a, site_b)
            mult_val = block_tools.sqrt(Tsval)
            Ta.multiply_index_diagonal_(
                ind=bnd, x=mult_val, location=loca, flip_pattern=flip_pattern
            )
            Tb.multiply_index_diagonal_(
                ind=bnd, x=mult_val, location=locb, flip_pattern=flip_pattern
            )

        # perform the gate, retrieving new bond singular values
        info = dict()
        self._psi.gate_(
            U,
            where,
            absorb=None,
            info=info,
            long_range_path_sequence=path,
            **self.gate_opts,
        )

        # set the new singualar values all along the chain
        for site_a, site_b in pairwise(string):
            if ("singular_values", (site_a, site_b)) in info:
                bond_pair = (site_a, site_b)
            else:
                bond_pair = (site_b, site_a)
            s = info["singular_values", bond_pair]
            if self.gauge_renorm:
                s = s / s.norm()
            if bond_pair not in self.gauges:
                self.gauges.pop((bond_pair[1], bond_pair[0]), None)
            self.gauges[bond_pair] = s

        # absorb the 'outer' gauges from these neighbours
        for site in string:
            Tij = self._psi[site]
            for neighbour in neighbours[site]:
                Tsval, (loc_ij, _), flip_pattern = self._unpack_gauge(
                    site, neighbour
                )
                bnd = self._psi.bond(site, neighbour)
                Tij.multiply_index_diagonal_(
                    ind=bnd,
                    x=Tsval,
                    location=loc_ij,
                    flip_pattern=flip_pattern,
                    inverse=True,
                )

    def get_state(self, absorb_gauges=True):
        """Return the state, with the diagonal bond gauges either absorbed
        equally into the tensors on either side of them
        (``absorb_gauges=True``, the default), or left lazily represented in
        the tensor network with hyperedges (``absorb_gauges=False``).
        """
        psi = self._psi.copy()

        if not absorb_gauges:
            raise NotImplementedError
        else:
            for (ija, ijb), Tsval in self.gauges.items():
                Ta = psi[ija]
                Tb = psi[ijb]
                (bnd,) = Ta.bonds(Tb)
                _, (loca, locb), flip_pattern = self._unpack_gauge(ija, ijb)
                mult_val = block_tools.sqrt(Tsval)
                Ta.multiply_index_diagonal_(
                    bnd, mult_val, location=loca, flip_pattern=flip_pattern
                )
                Tb.multiply_index_diagonal_(
                    bnd, mult_val, location=locb, flip_pattern=flip_pattern
                )

        if self.condition_tensors:
            conditioner(psi, balance_bonds=self.condition_balance_bonds)

        return psi
