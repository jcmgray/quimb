import numpy as np
import random
import collections
from itertools import product
from pyblock3.algebra.core import SubTensor
from pyblock3.algebra.fermion import SparseFermionTensor, FlatFermionTensor
from quimb.tensor.fermion_2d import FPEPS,FermionTensorNetwork2DVector
from quimb.tensor.fermion_ops import to_exp, eye, hubbard
from pyblock3.algebra.symmetry import SZ, BondInfo
from quimb.tensor.tensor_2d_tebd import SimpleUpdate as _SimpleUpdate
from quimb.tensor.tensor_2d_tebd import conditioner
from quimb.utils import pairwise
from quimb.tensor.tensor_2d import (gen_long_range_path,
                                    nearest_neighbors)

SMALL_VAL = 1e-10

def Hubbard2D(t, u, Lx, Ly):
    ham = dict()
    count_neighbour = lambda i,j: (i>0) + (i<Lx-1) + (j>0) + (j<Ly-1)
    for i, j in product(range(Lx), range(Ly)):
        count_ij = count_neighbour(i,j)
        if i+1 != Lx:
            where = ((i,j), (i+1,j))
            count_b = count_neighbour(i+1,j)
            uop = hubbard(t,u, (1./count_ij, 1./count_b))
            ham[where] = uop
        if j+1 != Ly:
            where = ((i,j), (i,j+1))
            count_b = count_neighbour(i,j+1)
            uop = hubbard(t,u, (1./count_ij, 1./count_b))
            ham[where] = uop
    return LocalHam2D(Lx, Ly, ham)

class LocalHam2D:
    """A 2D Hamiltonian represented as local terms. This combines all two site
    and one site terms into a single interaction per lattice pair, and caches
    operations on the terms such as getting their exponential.

    Parameters
    ----------
    Lx : int
        The number of rows.
    Ly : int
        The number of columns.
    H2 : array_like or dict[tuple[tuple[int]], array_like]
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
    terms : dict[tuple[tuple[int]], array_like]
        The total effective local term for each interaction (with single site
        terms appropriately absorbed). Each key is a pair of coordinates
        ``ija, ijb`` with ``ija < ijb``.

    """

    def __init__(self, Lx, Ly, H2, H1=None):
        self.Lx = int(Lx)
        self.Ly = int(Ly)

        # caches for not repeating operations / duplicating tensors
        self._op_cache = collections.defaultdict(dict)

        # parse two site terms
        if hasattr(H2, 'shape'):
            # use as default nearest neighbour term
            self.terms = {None: H2}
        else:
            self.terms = dict(H2)

        # first combine terms to ensure coo1 < coo2
        for where in tuple(filter(bool, self.terms)):
            coo1, coo2 = where
            if coo1 < coo2:
                continue

            # pop and flip the term
            X12 = self._flip_cached(self.terms.pop(where))

            # add to, or create, term with flipped coos
            new_where = coo2, coo1
            if new_where in self.terms:
                raise KeyError("appearing twice", where, new_where)
            else:
                self.terms[new_where] = X12

        # possibly fill in default gates
        default_H2 = self.terms.pop(None, None)
        if default_H2 is not None:
            for i, j in product(range(self.Lx), range(self.Ly)):
                if i + 1 < self.Lx:
                    where = ((i, j), (i + 1, j))
                    self.terms.setdefault(where, default_H2)
                if j + 1 < self.Ly:
                    where = ((i, j), (i, j + 1))
                    self.terms.setdefault(where, default_H2)
        if H1 is None:
            return

        if hasattr(H1, 'shape'):
            # use as default nearest neighbour term
            self.terms.update({None: H1})
        else:
            self.terms.update(H1)

        default_H1 = self.terms.pop(None, None)
        if default_H1 is not None:
            for i, j in product(range(self.Lx), range(self.Ly)):
                where = (i, j)
                self.terms.setdefault(where, default_H1)

    def _flip_cached(self, x):
        cache = self._op_cache['flip']
        key = id(x)
        if key not in cache:
            xf = np.transpose(x, (1,0,3,2))
            cache[key] = xf
        return cache[key]

    def _expm_cached(self, x, y):
        cache = self._op_cache['expm']
        key = (id(x), y)
        if key not in cache:
            out = to_exp(x, y)
            cache[key] = out
        return cache[key]

    def get_gate(self, where):
        """Get the local term for pair ``where``, cached.
        """
        return self.terms[tuple(where)]

    def get_gate_expm(self, where, x):
        """Get the local term for pair ``where``, matrix exponentiated by
        ``x``, and cached.
        """
        return self._expm_cached(self.get_gate(where), x)

    def get_auto_ordering(self, order='sort', **kwargs):
        """Get an ordering of the terms to use with TEBD, for example. The
        default is to sort the coordinates then greedily group them into
        commuting sets.

        Parameters
        ----------
        order : {'sort', None, 'random', str}
            How to order the terms *before* greedily grouping them into
            commuting (non-coordinate overlapping) sets. ``'sort'`` will sort
            the coordinate pairs first. ``None`` will use the current order of
            terms which should match the order they were supplied to this
            ``LocalHam2D`` instance.  ``'random'`` will randomly shuffle the
            coordinate pairs before grouping them - *not* the same as returning
            a completely random order. Any other option will be passed as a
            strategy to ``networkx.coloring.greedy_color`` to generate the
            ordering.

        Returns
        -------
        list[tuple[tuple[int]]]
            Sequence of coordinate pairs.
        """
        if order is None:
            pairs = self.terms
        elif order == 'sort':
            pairs = sorted(self.terms)
        elif order == 'random':
            pairs = list(self.terms)
            random.shuffle(pairs)
        elif order == 'random-ungrouped':
            pairs = list(self.terms)
            random.shuffle(pairs)
            return pairs
        else:
            raise NotImplementedError

        pairs = {x: None for x in pairs}

        cover = set()
        ordering = list()
        while pairs:
            for pair in tuple(pairs):
                ij1, ij2 = pair
                if (ij1 not in cover) and (ij2 not in cover):
                    ordering.append(pair)
                    pairs.pop(pair)
                    cover.add(ij1)
                    cover.add(ij2)
            cover.clear()

        return ordering

    def apply_to_arrays(self, fn):
        """Apply the function ``fn`` to all the arrays representing terms.
        """
        for k, x in self.terms.items():
            self.terms[k] = fn(x)

    def __repr__(self):
        s = "<LocalHam2D(Lx={}, Ly={}, num_terms={})>"
        return s.format(self.Lx, self.Ly, len(self.terms))

class SimpleUpdate(_SimpleUpdate):

    def _initialize_gauges(self):
        """Create unit singular values, stored as tensors.
        """
        # create the gauges like whatever data array is in the first site.
        data00 = self._psi[0, 0].data

        self._gauges = dict()
        for ija, ijb in self._psi.gen_bond_coos():
            bnd = self._psi.bond(ija, ijb)
            bond_dimension = self._psi.ind_size(bnd)
            Tsval = eye(bond_dimension)
            self._gauges[tuple(sorted((ija, ijb)))] = Tsval

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
            raise NotImplementedError
        else:
            # get the string linking the two sites
            string = path = tuple(gen_long_range_path(
                ija, ijb, sequence=long_range_path_sequence))

        def env_neighbours(i, j):
            return tuple(filter(
                lambda coo: self._psi.valid_coo((coo)) and coo not in string,
                nearest_neighbors((i, j))
            ))

        # get the relevant neighbours for string of sites
        neighbours = {site: env_neighbours(*site) for site in string}

        # absorb the 'outer' gauges from these neighbours
        for site in string:
            Tij = self._psi[site]
            for neighbour in neighbours[site]:
                if (site, neighbour) in self.gauges:
                    Tsval = self.gauges[(site, neighbour)]
                    location = "back"
                elif (neighbour, site) in self.gauges:
                    Tsval = self.gauges[(neighbour, site)]
                    location = "front"
                else:
                    raise KeyError("gauge not found")
                bond_ind = self._psi.bond(site, neighbour)
                mult_val = Tsval.copy()
                mult_val.data[abs(mult_val.data)>SMALL_VAL] += self.gauge_smudge
                Tij.multiply_index_diagonal_(
                    ind=bond_ind, x=mult_val, location=location)

        # absorb the inner bond gauges equally into both sites along string
        for site_a, site_b in pairwise(string):
            Ta, Tb = self._psi[site_a], self._psi[site_b]
            if (site_a, site_b) in self.gauges:
                Tsval = self.gauges[(site_a, site_b)]
                loca, locb = ("back", "front")
            elif (site_b, site_a) in self.gauges:
                Tsval = self.gauges[(site_b, site_a)]
                loca, locb = ("front", "back")
            else:
                raise KeyError("gauge not found")

            mult_val = Tsval.copy()
            mult_val.data = Tsval.data ** .5
            bnd = self._psi.bond(site_a, site_b)
            Ta.multiply_index_diagonal_(ind=bnd, x=mult_val, location=loca)
            Tb.multiply_index_diagonal_(ind=bnd, x=mult_val, location=locb)

        # perform the gate, retrieving new bond singular values
        info = dict()
        self._psi.gate_(U, where, absorb=None, info=info,
                        long_range_path_sequence=path, **self.gate_opts)

        # set the new singualar values all along the chain
        for site_a, site_b in pairwise(string):
            if ('singular_values', (site_a, site_b)) in info:
                bond_pair = (site_a, site_b)
            else:
                bond_pair = (site_b, site_a)
            s = info['singular_values', bond_pair]
            if self.gauge_renorm:
                # keep the singular values from blowing up
                s = s / np.sum(s.data**2) ** 0.5

            if bond_pair not in self.gauges:
                del self.gauges[(bond_pair[1], bond_pair[0])]

            self.gauges[bond_pair] = s

        # absorb the 'outer' gauges from these neighbours
        for site in string:
            Tij = self._psi[site]
            for neighbour in neighbours[site]:
                if (site, neighbour) in self.gauges:
                    Tsval = self.gauges[(site, neighbour)]
                    location = "back"
                elif (neighbour, site) in self.gauges:
                    Tsval = self.gauges[(neighbour, site)]
                    location = "front"
                else:
                    raise KeyError("gauge not found")
                bnd = self._psi.bond(site, neighbour)
                mult_val = Tsval.copy()
                non_zero_ind = abs(mult_val.data)>SMALL_VAL
                mult_val.data[non_zero_ind] = (mult_val.data[non_zero_ind] + self.gauge_smudge) ** -1
                Tij.multiply_index_diagonal_(
                    ind=bnd, x=mult_val, location=location)

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
                bnd = psi.bond(ija, ijb)
                Ta = psi[ija]
                Tb = psi[ijb]
                mult_val = Tsval.copy()
                mult_val.data = Tsval.data ** .5
                Ta.multiply_index_diagonal_(bnd, mult_val, location='back')
                Tb.multiply_index_diagonal_(bnd, mult_val, location='front')

        if self.condition_tensors:
            conditioner(psi, balance_bonds=self.condition_balance_bonds)

        return psi
