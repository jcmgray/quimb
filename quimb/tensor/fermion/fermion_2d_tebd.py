import numpy as np
import random
import collections
from itertools import product
from ...utils import pairwise
from ..tensor_2d_tebd import SimpleUpdate as _SimpleUpdate
from ..tensor_2d_tebd import conditioner
from ..tensor_2d import gen_long_range_path, nearest_neighbors
from .block_interface import eye, to_exponential, Hubbard
from . import block_tools

INVERSE_CUTOFF = 1e-10

def Hubbard2D(t, u, Lx, Ly, mu=0., symmetry=None):
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
    count_neighbour = lambda i,j: (i>0) + (i<Lx-1) + (j>0) + (j<Ly-1)
    for i, j in product(range(Lx), range(Ly)):
        count_ij = count_neighbour(i,j)
        if i+1 != Lx:
            where = ((i,j), (i+1,j))
            count_b = count_neighbour(i+1,j)
            uop = Hubbard(t,u, mu, (1./count_ij, 1./count_b), symmetry=symmetry)
            ham[where] = uop
        if j+1 != Ly:
            where = ((i,j), (i,j+1))
            count_b = count_neighbour(i,j+1)
            uop = Hubbard(t,u, mu, (1./count_ij, 1./count_b), symmetry=symmetry)
            ham[where] = uop
    return LocalHam2D(Lx, Ly, ham)

class LocalHam2D:
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
            out = to_exponential(x, y)
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

def _get_location(Ti, Tj):
    if Ti.get_fermion_info()[1]<Tj.get_fermion_info()[1]:
        return "front", "back"
    else:
        return "back", "front"

class SimpleUpdate(_SimpleUpdate):
    """A subclass of ``quimb.tensor.tensor_2d_tebd.SimpleUpdate`` that overrides one key method in
    order to ensure the gauge is placed in the right order within the PEPS. The gauges
    are stored separately from the main PEPS in the ``gauges`` attribute.
    Before and after a gate is applied they are absorbed and then extracted.
    When absorbing the gauge back to the PEPS, one needs to retrive the right order
    based on the relative ordering of the two gauged tensors.

    Parameters
    ----------
    psi0 : FermionTensorNetwork2DVector
        The initial state.
    ham : LocalHam2D
        The Hamtiltonian consisting of local terms.
    tau : float, optional
        The default local exponent, if considered as time real values here
        imply imaginary time.
    max_bond : {'psi0', int, None}, optional
        The maximum bond dimension to keep when applying each gate.
    gate_opts : dict, optional
        Supplied to :meth:`quimb.tensor.fermion_2d.FermionTensorNetwork2DVector.gate`,
        in addition to ``max_bond``. By default ``contract`` is set to
        'reduce-split' and ``cutoff`` is set to ``0.0``.
    ordering : str, tuple[tuple[int]], callable, optional
        How to order the terms, if a string is given then use this as the
        strategy given to
        :meth:`~quimb.tensor.fermion_2d_tebd.LocalHam2D.get_auto_ordering`. An
        explicit list of coordinate pairs can also be given. The default is to
        greedily form an 'edge coloring' based on the sorted list of
        Hamiltonian pair coordinates. If a callable is supplied it will be used
        to generate the ordering before each sweep.
    compute_energy_every : None or int, optional
        How often to compute and record the energy. If a positive integer 'n',
        the energy is computed *before* every nth sweep (i.e. including before
        the zeroth).
    compute_energy_final : bool, optional
        Whether to compute and record the energy at the end of the sweeps
        regardless of the value of ``compute_energy_every``. If you start
        sweeping again then this final energy is the same as the zeroth of the
        next set of sweeps and won't be recomputed.
    compute_energy_opts : dict, optional
        Supplied to
        :meth:`~quimb.tensor.tensor_2d.PEPS.compute_local_expectation`. By
        default ``max_bond`` is set to ``max(8, D**2)`` where ``D`` is the
        maximum bond to use for applying the gate, ``cutoff`` is set to ``0.0``
        and ``normalized`` is set to ``True``.
    compute_energy_fn : callable, optional
        Supply your own function to compute the energy, it should take the
        ``TEBD2D`` object as its only argument.
    callback : callable, optional
        A custom callback to run after every sweep, it should take the
        ``TEBD2D`` object as its only argument. If it returns any value
        that boolean evaluates to ``True`` then terminal the evolution.
    progbar : boolean, optional
        Whether to show a live progress bar during the evolution.
    gauge_renorm : bool, optional
        Whether to actively renormalize the singular value gauges.
    gauge_smudge : float, optional
        A small offset to use when applying the guage and its inverse to avoid
        numerical problems.
    condition_tensors : bool, optional
        Whether to actively equalize tensor norms for numerical stability.
    condition_balance_bonds : bool, optional
        If and when equalizing tensor norms, whether to also balance bonds as
        an additional conditioning.
    long_range_use_swaps : bool, optional
        disenabled option
    long_range_path_sequence : str or callable, optional
        disenabled option

    Attributes
    ----------
    state : FermionTensorNetwork2DVector
        The current state.
    ham : LocalHam2D
        The Hamiltonian being used to evolve.
    energy : float
        The current of the current state, this will trigger a computation if
        the energy at this iteration hasn't been computed yet.
    energies : list[float]
        The energies that have been computed, if any.
    its : list[int]
        The corresponding sequence of iteration numbers that energies have been
        computed at.
    taus : list[float]
        The corresponding sequence of time steps that energies have been
        computed at.
    best : dict
        If ``keep_best`` was set then the best recorded energy and the
        corresponding state that was computed - keys ``'energy'`` and
        ``'state'`` respectively.
    """

    def _initialize_gauges(self):
        """Create unit singular values, stored as tensors.
        """
        # create the gauges like whatever data array is in the first site.
        data00 = self._psi[0, 0].data

        self._gauges = dict()
        inv_dict = {"+":"-", "-":"+"}
        for ija, ijb in self._psi.gen_bond_coos():
            Tija = self._psi[ija]
            Tijb = self._psi[ijb]
            bnd = self._psi.bond(ija, ijb)
            sign_ija = Tija.data.pattern[Tija.inds.index(bnd)]
            bond_info = Tija.bond_info(bnd)
            Tsval = eye(bond_info)
            Tsval.pattern = sign_ija + inv_dict[sign_ija]
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
                elif (neighbour, site) in self.gauges:
                    Tsval = self.gauges[(neighbour, site)]
                else:
                    raise KeyError("gauge not found")
                T2 = self._psi[neighbour]
                location = _get_location(Tij, T2)[0]
                bond_ind = self._psi.bond(site, neighbour)
                mult_val = block_tools.add_with_smudge(Tsval, INVERSE_CUTOFF, self.gauge_smudge)
                Tij.multiply_index_diagonal_(
                    ind=bond_ind, x=mult_val, location=location)

        # absorb the inner bond gauges equally into both sites along string
        for site_a, site_b in pairwise(string):
            Ta, Tb = self._psi[site_a], self._psi[site_b]
            if (site_a, site_b) in self.gauges:
                Tsval = self.gauges[(site_a, site_b)]
            elif (site_b, site_a) in self.gauges:
                Tsval = self.gauges[(site_b, site_a)]
            else:
                raise KeyError("gauge not found")
            loca, locb = _get_location(Ta, Tb)
            mult_val = block_tools.sqrt(Tsval)
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
                s = s / s.norm()

            if bond_pair not in self.gauges:
                del self.gauges[(bond_pair[1], bond_pair[0])]

            self.gauges[bond_pair] = s

        # absorb the 'outer' gauges from these neighbours
        for site in string:
            Tij = self._psi[site]
            for neighbour in neighbours[site]:
                if (site, neighbour) in self.gauges:
                    Tsval = self.gauges[(site, neighbour)]
                elif (neighbour, site) in self.gauges:
                    Tsval = self.gauges[(neighbour, site)]
                else:
                    raise KeyError("gauge not found")
                bnd = self._psi.bond(site, neighbour)
                mult_val = block_tools.inv_with_smudge(Tsval, INVERSE_CUTOFF, self.gauge_smudge)
                location = _get_location(Tij, self._psi[neighbour])[0]
                Tij.multiply_index_diagonal_(
                    ind=bnd, x=mult_val, location=location)

    def get_state(self, absorb_gauges=True):
        """Return the state, with the diagonal bond gauges either absorbed
        equally into the tensors (``absorb_gauges=True``, the default),
        lazy representation with hyperedges disenabled
        """
        psi = self._psi.copy()

        if not absorb_gauges:
            raise NotImplementedError
        else:
            for (ija, ijb), Tsval in self.gauges.items():
                bnd = psi.bond(ija, ijb)
                Ta = psi[ija]
                Tb = psi[ijb]
                loca, locb = _get_location(Ta, Tb)
                mult_val = block_tools.sqrt(Tsval)
                Ta.multiply_index_diagonal_(bnd, mult_val, location=loca)
                Tb.multiply_index_diagonal_(bnd, mult_val, location=locb)

        if self.condition_tensors:
            conditioner(psi, balance_bonds=self.condition_balance_bonds)

        return psi
