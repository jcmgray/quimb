"""Tools for defining and constructing sparse operators with:

    * arbitrary geometries,
    * numba acceleration,
    * support for reduced bases,
    * efficient parallelization,

and optionally producing:

    * sparse matrix form
    * matrix product operators,
    * dict of local gates form
    * VMC 'coupled configs' form

Currently only supports composing operators which are sums of products of
diagonal or anti-diagonal real dimension 2 operators.

TODO::

    - [ ] fix sparse matrix being built in opposite direction
    - [ ] product of operators generator (e.g. for PEPS DMRG)
    - [ ] complex and single precision support (lower priority)
    - [ ] support for non-diagonal and qudit operators (lower priority)

DONE::

    - [x] use compact bitbasis
    - [x] design interface for HilbertSpace / OperatorBuilder interaction
    - [x] automatic symbolic jordan wigner transformation
    - [x] numba accelerated coupled config
    - [x] general definition and automatic 'symbolic' jordan wigner
    - [x] multithreaded sparse matrix construction
    - [x] LocalHam generator (e.g. for simple update, normal PEPS algs)
    - [x] automatic MPO generator

"""

import operator
import functools

import numpy as np
from numba import njit


@njit
def get_local_size(n, rank, world_size):
    """Given global size n, and a rank in [0, world_size), return the size of
    the portion assigned to this rank.
    """
    cutoff_rank = n % world_size
    return n // world_size + int(rank < cutoff_rank)


@njit
def get_local_range(n, rank, world_size):
    """Given global size n, and a rank in [0, world_size), return the range of
    indices assigned to this rank.
    """
    ri = 0
    for rank_below in range(rank):
        ri += get_local_size(n, rank_below, world_size)
    rf = ri + get_local_size(n, rank, world_size)
    return ri, rf


def parse_edges_to_unique(edges):
    """Given a list of edges, return a sorted list of unique sites and edges.

    Parameters
    ----------
    edges : Iterable[tuple[hashable, hashable]]]
        The edges to parse.

    Returns
    -------
    sites : list of hashable
        The unique sites in the edges, sorted.
    edges : list of (hashable, hashable)
        The unique edges, sorted.
    """
    sites = set()
    uniq_edges = set()
    for i, j in edges:
        if j < i:
            i, j = j, i
        sites.add(i)
        sites.add(j)
        uniq_edges.add((i, j))
    return sorted(sites), sorted(uniq_edges)


class HilbertSpace:
    """Take a set of 'sites' (any sequence of sortable, hashable objects), and
    map this into a 'register' or linearly indexed range, optionally using a
    particular ordering. One can then calculate the size of the Hilbert space,
    including number conserving subspaces, and get a compact 'bitbasis' with
    which to construct sparse operators.

    Parameters
    ----------
    sites : int or sequence of hashable objects
        The sites to map into a linear register. If an integer, simply use
        ``range(sites)``.
    order : callable or sequence of hashable objects, optional
        If provided, use this to order the sites. If a callable, it should be a
        sorting key. If a sequence, it should be a permutation of the sites,
        and ``key=order.index`` will be used.
    """

    def __init__(self, sites, order=None):
        if isinstance(sites, int):
            sites = range(sites)

        if (order is not None) and (not callable(order)):
            order = order.index
        self._order = order
        self._sites = tuple(sorted(sites, key=self._order))
        self._mapping_inv = dict(enumerate(self._sites))
        self._mapping = {s: i for i, s in self._mapping_inv.items()}

    def set_ordering(self, order=None):
        if (order is not None) and (not callable(order)):
            order = order.index
        self._order = order
        self._sites = tuple(sorted(self._sites, key=self._order))
        self._mapping_inv = dict(enumerate(self._sites))
        self._mapping = {s: i for i, s in self._mapping_inv.items()}

    @classmethod
    def from_edges(cls, edges, order=None):
        """Construct a HilbertSpace from a set of edges, which are pairs of
        sites.
        """
        sites, _ = parse_edges_to_unique(edges)
        return cls(sites, order=order)

    def site_to_reg(self, site):
        """Convert a site to a linear register index."""
        return self._mapping[site]

    def reg_to_site(self, reg):
        """Convert a linear register index back to a site."""
        return self._mapping_inv[reg]

    def has_site(self, site):
        """Check if this HilbertSpace contains a given site."""
        return site in self._mapping

    def config_to_bit(self, config):
        """Encode a 'configuration' as a bit.

        Parameters
        ----------
        config : dict[hashable, int]
            A dictionary mapping sites to their occupation number / spin.

        Returns
        -------
        bit : int
            The bit corresponding to this configuration.
        """
        bit = 0
        for site, val in config.items():
            if val:
                bit |= 1 << self.site_to_reg(site)
        return bit

    def config_to_flatconfig(self, config):
        """Turn a configuration into a flat configuration, assuming the order
        given by this ``HilbertSpace``.

        Parameters
        ----------
        config : dict[hashable, int]
            A dictionary mapping sites to their occupation number / spin.

        Returns
        -------
        flatconfig : ndarray[uint8]
            A flat configuration, with the occupation number of each site in
            the order given by this ``HilbertSpace``.
        """
        flatconfig = np.empty(self.nsites, dtype=np.uint8)
        for site, val in config.items():
            flatconfig[self.site_to_reg(site)] = val
        return flatconfig

    def bit_to_config(self, bit):
        """Decode a bit to a configuration.

        Parameters
        ----------
        bit : int
            The bit to decode.

        Returns
        -------
        config : dict[hashable, int]
            A dictionary mapping sites to their occupation number / spin.
        """
        config = {}
        for site, reg in self._mapping.items():
            config[site] = (bit >> reg) & 1
        return config

    def flatconfig_to_config(self, flatconfig):
        """Turn a flat configuration into a configuration, assuming the order
        given by this ``HilbertSpace``.

        Parameters
        ----------
        flatconfig : ndarray[uint8]
            A flat configuration, with the occupation number of each site in
            the order given by this ``HilbertSpace``.

        Returns
        -------
        config : dict[hashable, int]
            A dictionary mapping sites to their occupation number / spin.
        """
        config = {}
        for i in range(self.nsites):
            config[self.reg_to_site(i)] = flatconfig[i]
        return config

    def rand_config(self, k=None):
        """Get a random configuration, optionally requiring it has ``k`` bits
        set.
        """
        if k is None:
            return {site: np.random.randint(2) for site in self.sites}
        r = np.random.randint(self.get_size(k))
        b = rank_to_bit(r, self.nsites, k)
        return self.bit_to_config(b)

    @property
    def sites(self):
        """The ordered tuple of all sites in the Hilbert space."""
        return self._sites

    @property
    def nsites(self):
        """The total number of sites in the Hilbert space."""
        return len(self._sites)

    def get_size(self, *k):
        """Compute the size of this Hilbert space, optionally taking into
        account number / z-spin conservation.

        Parameters
        ----------
        k : int or tuple of (int, int)
            If provided, compute the size of number conserving subspace(s)::

                - If a single int, compute the size of the subspace with
                  ``k`` particles / up states: ``comb(nsites, k)``.
                - If a tuple of (int, int), compute the size of the subspace
                  of the product of spaces where each pair (n, k) corresponds
                  to n sites with k particles / up states. The sum of every n
                  should equal ``nsites``.
        """
        if not k:
            return 2**self.nsites

        if (len(k) == 1) and isinstance(k[0], int):
            # single interger - take as k
            return comb(self.nsites, k[0])

        size = 1
        ncheck = 0
        for n, k in k:
            size *= comb(n, k)
            ncheck += n

        if ncheck != self.nsites:
            raise ValueError("`k` must sum to the number of sites")

        return size

    def get_bitbasis(self, *k, dtype=np.int64):
        """Get a basis for the Hilbert space, in terms of an integer bitarray,
        optionally taking into account number / z-spin conservation.

        Parameters
        ----------
        k : int or tuple of (int, int)
            If provided, get the basis for a number conserving subspace(s)::

                - If a single int, compute the size of the subspace with
                  ``k`` particles / up states: ``comb(nsites, k)``.
                - If a tuple of (int, int), compute the size of the subspace
                  of the product of spaces where each pair (n, k) corresponds
                  to n sites with k particles / up states. The sum of every n
                  should equal ``nsites``.

        dtype : np.dtype, optional
            The dtype of the bitarray, should be an integer type with at least
            ``nsites`` bits.

        Returns
        -------
        bits : numpy.ndarray
            The basis, each integer element being a binary representation of a
            configuration.
        """
        if not k:
            return np.arange(1 << self.nsites, dtype=dtype)

        if (len(k) == 1) and isinstance(k[0], int):
            k = ((self.nsites, k[0]),)

        return get_number_bitbasis(*k, dtype=dtype)

    def __repr__(self):
        return (
            f"HilbertSpace(nsites={self.nsites}, total_size={self.get_size()})"
        )


_OPMAP = {
    "I": {0: (0, 1.0), 1: (1, 1.0)},
    # pauli matrices
    "x": {0: (1, 1.0), 1: (0, 1.0)},
    "y": {0: (1, 1.0j), 1: (0, -1.0j)},
    "z": {0: (0, 1.0), 1: (1, -1.0)},
    # spin 1/2 matrices (scaled paulis)
    "sx": {0: (1, 0.5), 1: (0, 0.5)},
    "sy": {0: (1, 0.5j), 1: (0, -0.5j)},
    "sz": {0: (0, 0.5), 1: (1, -0.5)},
    # creation / annihilation operators
    "+": {0: (1, 1.0)},
    "-": {1: (0, 1.0)},
    # number, symmetric number, and hole operators
    "n": {1: (1, 1.0)},
    "sn": {0: (0, -0.5), 1: (1, 0.5)},
    "h": {0: (0, 1.0)},
}


@functools.lru_cache(maxsize=None)
def get_mat(op, dtype=None):
    if dtype is None:
        if any(
            isinstance(coeff, complex) for _, (_, coeff) in _OPMAP[op].items()
        ):
            dtype = np.complex128
        else:
            dtype = np.float64

    a = np.zeros((2, 2), dtype=dtype)
    for j, (i, xij) in _OPMAP[op].items():
        a[i, j] = xij
    # make immutable since caching
    a.flags.writeable = False
    return a


@functools.lru_cache(maxsize=2**14)
def simplify_single_site_ops(coeff, ops):
    """Simplify a sequence of operators acting on the same site.

    Parameters
    ----------
    coeff : float or complex
        The coefficient of the operator sequence.
    ops : tuple of str
        The operator sequence.

    Returns
    -------
    new_coeff : float or complex
        The simplified coefficient.
    new_op : str
        The single, simplified operator that the sequence maps to, up to
        scalar multiplication.

    Examples
    --------

        >>> simplify_single_site_ops(1.0, ('+', 'z', 'z', 'z', 'z', '-'))
        (1.0, 'n')

        >>> simplify_single_site_ops(1.0, ("x", "y", "z"))
        (-1j, 'I')

    """

    if len(ops) == 1:
        return coeff, ops[0]

    # product all the matrices
    combo_mat = functools.reduce(operator.matmul, map(get_mat, ops))
    combo_coeff = combo_mat.flat[np.argmax(np.abs(combo_mat))]

    if combo_coeff == 0.0:
        # null-term
        return 0, None

    # find the reference operator that maps to this matrix
    for op in _OPMAP:
        ref_mat = get_mat(op)
        ref_coeff = ref_mat.flat[np.argmax(np.abs(ref_mat))]
        if (
            (combo_mat / combo_coeff).round(12)
            == (ref_mat / ref_coeff).round(12)
        ).all():
            break
    else:
        raise ValueError(f"No match found for '{ops}'")

    coeff *= ref_coeff / combo_coeff
    if coeff.imag == 0.0:
        coeff = coeff.real

    return coeff, op


class SparseOperatorBuilder:
    """Object for building operators with sparse structure. Specifically,
    a sum of terms, where each term is a product of operators, where each of
    these local operators acts on a single site and has at most one entry per
    row.

    Parameters
    ----------
    terms : sequence, optional
        The terms to initialize the builder with. ``add_term`` is simply called
        on each of these.
    hilbert_space : HilbertSpace
        The Hilbert space to build the operator in. If this is not supplied
        then a minimal Hilbert space will be constructed from the sites used,
        when required.
    """

    def __init__(self, terms=(), hilbert_space=None):
        self._term_store = {}
        self._sites_used = set()
        self._hilbert_space = hilbert_space
        self._coupling_map = None
        for term in terms:
            self.add_term(*term)

    @property
    def sites_used(self):
        """A tuple of the sorted coordinates/sites seen so far."""
        return tuple(sorted(self._sites_used))

    @property
    def nsites(self):
        """The total number of coordinates/sites seen so far."""
        return self.hilbert_space.nsites

    @property
    def terms(self):
        """A tuple of the simplified terms seen so far."""
        return tuple((coeff, ops) for ops, coeff in self._term_store.items())

    @property
    def nterms(self):
        """The total number of terms seen so far."""
        return len(self._term_store)

    @property
    def locality(self):
        """The locality of the operator, the maximum support of any term."""
        return max(len(ops) for ops in self._term_store)

    @property
    def hilbert_space(self):
        """The Hilbert space of the operator. Created from the sites seen
        so far if not supplied at construction.
        """
        if self._hilbert_space is None:
            self._hilbert_space = HilbertSpace(self.sites_used)
        return self._hilbert_space

    @property
    def coupling_map(self):
        if self._coupling_map is None:
            self._coupling_map = build_coupling_numba(
                self._term_store, self.hilbert_space.site_to_reg
            )
        return self._coupling_map

    def site_to_reg(self, site):
        """Get the register / linear index of coordinate ``site``."""
        return self.hilbert_space.site_to_reg(site)

    def reg_to_site(self, reg):
        """Get the site of register / linear index ``reg``."""
        return self.hilbert_space.reg_to_site(reg)

    def add_term(self, *coeff_ops):
        """Add a term to the operator.

        Parameters
        ----------
        coeff : float, optional
            The overall coefficient of the term.
        ops : sequence of tuple[str, hashable]
            The operators of the term, together with the sites they act on.
            Each term should be a pair of ``(operator, site)``, where
            ``operator`` can be:

                - ``'x'``, ``'y'``, ``'z'``: Pauli matrices
                - ``'sx'``, ``'sy'``, ``'sz'``: spin operators (i.e. scaled
                  Pauli matrices)
                - ``'+'``, ``'-'``: creation/annihilation operators
                - ``'n'``, ``'sn'``, or ``'h'``: number, symmetric
                  number (n - 1/2) and hole (1 - n) operators

            And ``site`` is a hashable object that represents the site that
            the operator acts on.

        """
        if isinstance(coeff_ops[0], (tuple, list)):
            # assume coeff is 1.0
            coeff = 1
            ops = coeff_ops
        else:
            coeff, *ops = coeff_ops

        if coeff == 0.0:
            # null-term
            return

        # print(coeff, ops, '->')

        # collect operators acting on the same site
        collected = {}
        for op, site in ops:
            # check that the site is valid if the Hilbert space is known
            if (
                self._hilbert_space is not None
            ) and not self._hilbert_space.has_site(site):
                raise ValueError(f"Site {site} not in the Hilbert space.")
            self._sites_used.add(site)
            collected.setdefault(site, []).append(op)

        # simplify operators acting on the smae site & don't add null-terms
        simplified_ops = []
        for site, collected_ops in collected.items():
            coeff, op = simplify_single_site_ops(coeff, tuple(collected_ops))

            if op is None:
                # null-term ('e.g. '++' or '--')
                # print('null-term')
                # print()
                return

            if op != "I":
                # only need to record non-identity operators
                simplified_ops.append((op, site))

        key = tuple(simplified_ops)

        # print(coeff, key)
        # print()

        new_coeff = self._term_store.pop(key, 0.0) + coeff
        if new_coeff != 0.0:
            self._term_store[key] = new_coeff

    def __iadd__(self, term):
        self.add_term(*term)
        return self

    def jordan_wigner_transform(self):
        """Transform the terms in this operator by pre-prending pauli Z
        strings to all creation and annihilation operators, and then
        simplifying the resulting terms.
        """
        # TODO: check if transform has been applied already
        # TODO: store untransformed terms, so we can re-order at will

        old_term_store = self._term_store.copy()
        self._term_store.clear()

        for term, coeff in old_term_store.items():
            if not term:
                self.add_term(coeff, *term)
                continue

            ops, site = zip(*term)
            if {"+", "-"}.intersection(ops):
                new_term = []

                for op, site in term:
                    reg = self.site_to_reg(site)
                    if op in {"+", "-"}:
                        for r in range(reg):
                            site_below = self.reg_to_site(r)
                            new_term.append(("z", site_below))
                    new_term.append((op, site))

                self.add_term(coeff, *new_term)
            else:
                self.add_term(coeff, *term)

    def build_coo_data(self, *k, parallel=False):
        """Build the raw data for a sparse matrix in COO format. Optionally
        in a reduced k basis and in parallel.

        Parameters
        ----------
        k : int or tuple of (int, int)
            If provided, get the basis for a number conserving subspace(s)::

                - If a single int, compute the size of the subspace with
                  ``k`` particles / up states: ``comb(nsites, k)``.
                - If a tuple of (int, int), compute the size of the subspace
                  of the product of spaces where each pair (n, k) corresponds
                  to n sites with k particles / up states. The sum of every n
                  should equal ``nsites``.

        parallel : bool, optional
            Whether to build the matrix in parallel (multi-threaded).

        Returns
        -------
        coo : array
            The data entries for the sparse matrix in COO format.
        cis : array
            The row indices for the sparse matrix in COO format.
        cjs : array
            The column indices for the sparse matrix in COO format.
        N : int
            The total number of basis states.
        """
        hs = self.hilbert_space
        bits = hs.get_bitbasis(*k)
        coupling_map = self.coupling_map
        coo, cis, cjs = build_coo_numba(bits, coupling_map, parallel=parallel)
        return coo, cis, cjs, bits.size

    def build_sparse_matrix(self, *k, stype="csr", parallel=False):
        """Build a sparse matrix in the given format. Optionally in a reduced
        k basis and in parallel.

        Parameters
        ----------
        k : int or tuple of (int, int)
            If provided, get the basis for a number conserving subspace(s)::

                - If a single int, compute the size of the subspace with
                  ``k`` particles / up states: ``comb(nsites, k)``.
                - If a tuple of (int, int), compute the size of the subspace
                  of the product of spaces where each pair (n, k) corresponds
                  to n sites with k particles / up states. The sum of every n
                  should equal ``nsites``.

        parallel : bool, optional
            Whether to build the matrix in parallel (multi-threaded).

        Returns
        -------
        scipy.sparse matrix
        """
        import scipy.sparse as sp

        coo, cis, cjs, N = self.build_coo_data(*k, parallel=parallel)
        A = sp.coo_matrix((coo, (cis, cjs)), shape=(N, N))
        if stype != "coo":
            A = A.asformat(stype)
        return A

    def build_dense(self):
        """Get the dense (`numpy.ndarray`) matrix representation of this
        operator.
        """
        A = self.build_sparse_matrix(stype="coo")
        return A.A

    def build_local_terms(self):
        """Get a dictionary of local terms, where each key is a sorted tuple
        of sites, and each value is the local matrix representation of the
        operator on those sites. For use with e.g. tensor network algorithms.

        Note terms acting on the same sites are summed together and the size of
        each local matrix is exponential in the locality of that term.

        Returns
        -------
        Hk : dict[tuple[hashable], numpy.ndarray]
            The local terms.
        """
        Hk = {}

        for term, coeff in self._term_store.items():
            ops, sites = zip(*term)
            mats = tuple(get_mat(op) for op in ops)
            hk = coeff * functools.reduce(np.kron, mats)
            if sites not in Hk:
                Hk[sites] = hk.copy()
            else:
                Hk[sites] += hk
        return Hk

    def config_coupling(self, config):
        """Get a list of other configurations coupled to ``config`` by this
        operator, and the corresponding coupling coefficients. This is for
        use with VMC for example.

        Parameters
        ----------
        config : dict[site, int]
            The configuration to get the coupling for.

        Returns
        -------
        coupled_configs : list[dict[site, int]]
            Each distinct configuration coupled to ``config``.
        coeffs: list[float]
            The corresponding coupling coefficients.
        """
        bit_to_config = self.hilbert_space.bit_to_config
        config_to_bit = self.hilbert_space.config_to_bit
        b = config_to_bit(config)
        bjs, coeffs = coupled_bits_numba(b, self.coupling_map)
        return [bit_to_config(bj) for bj in bjs], coeffs

    def show(self, filler="."):
        """Print an ascii representation of the terms in this operator."""
        print(self)
        for t, (term, coeff) in enumerate(self._term_store.items()):
            s = [f"{filler} "] * self.nsites
            for op, site in term:
                s[self.site_to_reg(site)] = f"{op:<2}"
            print("".join(s), f"{coeff:+}")

    def build_state_machine_greedy(self):
        # XXX: optimal method : https://arxiv.org/abs/2006.02056

        import networkx as nx

        # - nodes of the state machine are a 2D grid of (register, 'rail'),
        #   with the maximum number of rails giving the eventual bond dimension
        # - there are N + 1 registers for N sites
        # - each edge from (r, i) to (r + 1, j) represents a term that will be
        #   placed in the rth MPO tensor at entry like W[i, j, :, :] = op
        # - each node has either a single inwards or outwards edge
        G = nx.DiGraph()
        G.add_node((0, 0))

        # count how many rails are at each register
        num_rails = [1] + [0] * self.nsites

        def new_edge(a, b):
            # need to track which terms pass through this edge so we can
            # place the coefficient somewhere unique at the end
            G.add_edge(a, b, op=op, weight=1, unique_term=t)

        def check_right():
            # check if can **right share**
            # - check all existing potential next nodes
            # - current op must match or not exist
            # - right strings must match
            # - must be single output node
            for rail in range(num_rails[reg + 1]):
                cand_node = (reg + 1, rail)
                if G.out_degree(cand_node) > 1:
                    continue

                if G.nodes[cand_node]["out_string"] != string[reg + 1 :]:
                    continue

                e = (current_node, cand_node)
                if e not in G.edges:
                    new_edge(current_node, cand_node)
                    return cand_node
                else:
                    if G.edges[e]["op"] != op:
                        continue
                    G.edges[e]["weight"] += 1
                    G.edges[e]["unique_term"] = None
                    return cand_node

                # XXX: if we can right share, don't need to do anything
                # more since whole remaining string is shared?

        def check_left():
            # check if can **left share**
            # - check all out edges
            # - current op must match AND
            # - must be single input node
            for e in G.edges(current_node):
                cand_node = e[1]
                if G.in_degree(cand_node) <= 1:
                    if G.edges[e]["op"] == op:
                        G.edges[e]["weight"] += 1
                        G.edges[e]["unique_term"] = None
                        return cand_node

        def create_new():
            # create a new rail at the next register
            next_node = (reg + 1, num_rails[reg + 1])
            num_rails[reg + 1] += 1
            new_edge(current_node, next_node)
            return next_node

        # each term guaranteed has a unique edge somewhere, which we can place
        # the coefficient on later
        coeffs_to_place = {}

        for t, (term, coeff) in enumerate(self._term_store.items()):
            # build full string for this term including identity ops
            rmap = {self.site_to_reg(site): op for op, site in term}
            string = tuple(rmap.get(r, "I") for r in range(self.nsites))

            current_node = (0, 0)
            for reg, op in enumerate(string):
                cand_node = check_right()
                if cand_node is not None:
                    # can share right part of string
                    current_node = cand_node
                else:
                    cand_node = check_left()
                    if cand_node is not None:
                        # can share left part of string
                        current_node = cand_node
                    else:
                        # have to create new node
                        current_node = create_new()

                if G.out_degree(current_node) <= 1:
                    # record what the right matching string is
                    G.nodes[current_node]["out_string"] = string[reg + 1 :]
                else:
                    G.nodes[current_node]["out_string"] = None

            if coeff != 1:
                # record that we still need to place coeff somewhere
                coeffs_to_place[t] = coeff

        for _, _, data in G.edges(data=True):
            data["coeff"] = coeffs_to_place.pop(data["unique_term"], None)

        if coeffs_to_place:
            raise ValueError("Failed to place all coefficients.")

        G.graph["nsites"] = self.nsites
        G.graph["num_rails"] = tuple(num_rails)
        G.graph["max_num_rails"] = max(num_rails)

        return G

    def draw_state_machine(
        self,
        method="greedy",
        figsize="auto",
    ):
        import math
        from matplotlib import pyplot as plt
        from quimb.tensor.drawing import auto_colors

        if method == "greedy":
            G = self.build_state_machine_greedy()
        else:
            raise ValueError(f"Unknown method {method}")

        def labelled_arrow(ax, p1, p2, label, color, width):
            angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            ax.annotate(
                "",
                xy=p2,
                xycoords="data",
                xytext=p1,
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    alpha=0.75,
                    linewidth=width,
                ),
            )
            p_middle = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            ax.text(
                *p_middle,
                label,
                color=color,
                ha="center",
                va="center",
                rotation=angle * 180 / math.pi,
                alpha=1.0,
                transform_rotates_text=True,
            )

        if figsize == "auto":
            width = G.graph["nsites"]
            # maximum number of nodes on any rail
            height = G.graph["max_num_rails"] / 2
            figsize = (width, height)

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_alpha(0.0)
        ax.set_axis_off()
        ax.set_xlim(-0.5, G.graph["nsites"] + 0.5)
        ax.set_ylim(-0.5, G.graph["max_num_rails"] + 0.5)

        # draw each edge as a colored, labelled node
        ops = sorted(
            {data["op"] for _, _, data in G.edges(data=True)} - {"I"}, key=str
        )
        all_colors = auto_colors(len(ops))
        colors = {op: c for op, c in zip(ops, all_colors)}
        colors["I"] = "grey"
        for n1, n2, data in G.edges(data=True):
            color = colors[data["op"]]
            width = math.log2(1 + data["weight"])
            label = data["op"]
            if data["coeff"] is not None:
                label += f" * {data['coeff']}"
            label += "\n"
            labelled_arrow(ax, n1, n2, label, color, width)

        # label which MPO site along the bottom
        for i in range(G.graph["nsites"]):
            ax.text(
                i + 0.5,
                -1.0,
                "$W_{" + str(i) + "}$",
                ha="center",
                va="center",
                color=(0.5, 0.5, 0.5),
                fontsize=12,
            )

        plt.show()
        plt.close(fig)

    def build_mpo(self, method="greedy", **mpo_opts):
        import numpy as np
        import quimb as qu
        import quimb.tensor as qtn

        if method == "greedy":
            G = self.build_state_machine_greedy()
        else:
            raise ValueError(f"Unknown method {method}.")

        Wts = [
            np.zeros((dl, dr, 2, 2), dtype=float)
            for dl, dr in qu.utils.pairwise(G.graph["num_rails"])
        ]

        for node_a, node_b, data in G.edges(data=True):
            op = data["op"]
            coeff = data["coeff"]
            if coeff is not None:
                mat = coeff * get_mat(op)
            else:
                mat = get_mat(op)

            rega, raila = node_a
            _, railb = node_b
            Wts[rega][raila, railb, :, :] = mat

        Wts[0] = Wts[0].sum(axis=0)
        Wts[-1] = Wts[-1].sum(axis=1)

        return qtn.MatrixProductOperator(Wts, **mpo_opts)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(nsites={self.nsites}, "
            f"nterms={self.nterms}, "
            f"locality={self.locality})"
            ")"
        )


@njit
def get_nth_bit(val, n):
    """Get the nth bit of val.

    Examples
    --------

        >>> get_nth_bit(0b101, 1)
        0
    """
    return (val >> n) & 1


@njit
def flip_nth_bit(val, n):
    """Flip the nth bit of val.

    Examples
    --------

        >>> bin(flip_nth_bit(0b101, 1))
        0b111
    """
    return val ^ (1 << n)


@njit
def comb(n, k):
    """Compute the binomial coefficient n choose k."""
    r = 1
    for dk, dn in zip(range(1, k + 1), range(n, n - k, -1)):
        r *= dn
        r //= dk
    return r


@njit
def get_all_equal_weight_bits(n, k, dtype=np.int64):
    """Get an array of all 'bits' (integers), with n bits, and k of them set."""
    if k == 0:
        return np.array([0], dtype=dtype)

    m = comb(n, k)
    b = np.empty(m, dtype=dtype)
    i = 0
    val = (1 << k) - 1
    while val < (1 << (n)):
        b[i] = val
        c = val & -val
        r = val + c
        val = (((r ^ val) >> 2) // c) | r
        i += 1
    return b


@njit
def bit_to_rank(b, n, k):
    """Given a bitstring b, return the rank of the bitstring in the
    basis of all bitstrings of length n with k bits set. Adapted from
    https://dlbeer.co.nz/articles/kwbs.html.
    """
    c = comb(n, k)
    r = 0
    while n:
        c0 = c * (n - k) // n
        if (b >> n - 1) & 1:
            r += c0
            k -= 1
            c -= c0
        else:
            c = c0
        n -= 1
    return r


@njit
def rank_to_bit(r, n, k):
    """Given a rank r, return the bitstring of length n with k bits set
    that has rank r in the basis of all bitstrings of length n with k
    bits set. Adapted from https://dlbeer.co.nz/articles/kwbs.html.
    """
    b = 0
    c = comb(n, k)
    while n:
        c0 = c * (n - k) // n
        if r >= c0:
            b |= 1 << (n - 1)
            r -= c0
            k -= 1
            c -= c0
        else:
            c = c0
        n -= 1
    return b


@njit
def _recursively_fill_flatconfigs(flatconfigs, n, k, c, r):
    c0 = c * (n - k) // n
    # set the entries of the left binary subtree
    flatconfigs[r : r + c0, -n] = 0
    # set the entries of the right binary subtree
    flatconfigs[r + c0 : r + c, -n] = 1
    if n > 1:
        # process each subtree recursively
        _recursively_fill_flatconfigs(flatconfigs, n - 1, k, c0, r)
        _recursively_fill_flatconfigs(
            flatconfigs, n - 1, k - 1, c - c0, r + c0
        )


@njit
def get_all_equal_weight_flatconfigs(n, k):
    """Get every flat configuration of length n with k bits set."""
    c = comb(n, k)
    flatconfigs = np.empty((c, n), dtype=np.uint8)
    _recursively_fill_flatconfigs(flatconfigs, n, k, c, 0)
    return flatconfigs


@njit
def flatconfig_to_bit(flatconfig):
    """Given a flat configuration, return the corresponding bitstring."""
    b = 0
    for i, x in enumerate(flatconfig):
        if x:
            b |= 1 << i
    return b


@njit
def flatconfig_to_rank(flatconfig, n, k):
    """Given a flat configuration ``flatconfig``, return the rank of the
    bitstring in the basis of all bitstrings of length ``n`` with ``k`` bits
    set. Adapted from https://dlbeer.co.nz/articles/kwbs.html.
    """
    c = comb(n, k)
    r = 0
    while n:
        c0 = c * (n - k) // n
        if flatconfig[-n]:
            r += c0
            k -= 1
            c -= c0
        else:
            c = c0
        n -= 1
    return r


@njit
def rank_to_flatconfig(r, n, k):
    """Given a rank ``r``, return the flat configuration of length ``n``
    with ``k`` bits set that has rank ``r`` in the basis of all bitstrings of
    length ``n`` with ``k`` bits set. Adapted from
    https://dlbeer.co.nz/articles/kwbs.html.
    """
    flatconfig = np.zeros(n, dtype=np.uint8)
    c = comb(n, k)
    while n:
        c0 = c * (n - k) // n
        if r >= c0:
            flatconfig[-n] = 1
            r -= c0
            k -= 1
            c -= c0
        else:
            c = c0
        n -= 1
    return flatconfig


@njit
def product_of_bits(b1, b2, n2, dtype=np.int64):
    """Get the outer product of two bit arrays.

    Parameters
    ----------
    b1, b2 : array
        The bit arrays to take the outer product of.
    n2 : int
        The number of bits in ``b2``
    """
    b = np.empty(len(b1) * len(b2), dtype=dtype)
    i = 0
    for x in b1:
        for y in b2:
            b[i] = (x << n2) | y
            i += 1
    return b


def get_number_bitbasis(*nk_pairs, dtype=np.int64):
    """Create a bit basis with number conservation.

    Parameters
    ----------
    nk_pairs : sequence of (int, int)
        Each element is a pair (n, k) where n is the number of bits, and k is
        the number of bits that are set. The basis will be the product of all
        supplied pairs.

    Returns
    -------
    basis : ndarray
        An array of integers, each representing a bit string. The size will be
        ``prod(comb(n, k) for n, k in nk_pairs)``.

    Examples
    --------

    A single number conserving basis:

        >>> for i, b in enumerate(get_number_bitbasis((4, 2))):
        >>>    print(f"{i}: {b:0>4b}")
        0: 0011
        1: 0101
        2: 0110
        3: 1001
        4: 1010
        5: 1100

    A product of two number conserving bases, e.g. n_up and n_down:

        >>> for b in get_number_bitbasis((3, 2), (3, 1)):
        >>>     print(f"{b:0>6b}")
        011001
        011010
        011100
        101001
        101010
        101100
        110001
        110010
        110100

    """
    configs = None
    for n, k in nk_pairs:
        next_configs = get_all_equal_weight_bits(n, k, dtype=dtype)
        if configs is None:
            configs = next_configs
        else:
            configs = product_of_bits(configs, next_configs, n, dtype=dtype)
    return configs


@njit
def build_bitmap(configs):
    """Build of map of bits to linear indices, suitable for use with numba."""
    return {b: i for i, b in enumerate(configs)}


def build_coupling_numba(term_store, site_to_reg):
    """Create a sparse nested dictionary of how each term couples each
    local site configuration to which other local site configuration, and
    with what coefficient, suitable for use with numba.

    Parameters
    ----------
    term_store : dict[term, coeff]
        The terms of the operator.
    site_to_reg : callable
        A function that maps a site to a linear register index.

    Returns
    -------
    coupling_map : numba.typed.Dict
        A nested numba dictionary of the form
        ``{term: {reg: {bit_in: (bit_out, coeff), ...}, ...}, ...}``.
    """
    from numba.core import types
    from numba.typed import Dict

    ty_xj = types.Tuple((types.int64, types.float64))
    ty_xi = types.DictType(types.int64, ty_xj)
    ty_site = types.DictType(types.int64, ty_xi)
    coupling_map = Dict.empty(types.int64, ty_site)

    # for term t ...
    for t, (term, coeff) in enumerate(term_store.items()):
        first = True
        # which couples sites with product of ops ...
        for op, site in term:
            reg = site_to_reg(site)
            # -> bit `xi` at `reg` is coupled to `xj` with coeff `cij`
            #          : reg
            #     ...10010...    xi=0  ->
            #     ...10110...    xj=1  with coeff cij
            for xi, (xj, cij) in _OPMAP[op].items():
                if first:
                    # absorb overall coefficient into first coupling
                    cij = coeff * cij

                # populate just the term/reg/bit maps we need
                coupling_map.setdefault(
                    t, Dict.empty(types.int64, ty_xi)
                ).setdefault(reg, Dict.empty(types.int64, ty_xj))[xi] = (
                    xj,
                    cij,
                )
            first = False

    return coupling_map


def build_coupling(term_store, site_to_reg):
    coupling_map = dict()
    # for term t ...
    for t, (term, coeff) in enumerate(term_store.items()):
        first = True
        # which couples sites with product of ops ...
        for op, site in term:
            reg = site_to_reg(site)
            # -> bit `xi` at `reg` is coupled to `xj` with coeff `cij`
            #          : reg
            #     ...10010...    xi=0  ->
            #     ...10110...    xj=1  with coeff cij
            for xi, (xj, cij) in _OPMAP[op].items():
                if first:
                    # absorb overall coefficient into first coupling
                    cij = coeff * cij

                # populate just the term/reg/bit maps we need
                coupling_map.setdefault(t, {}).setdefault(reg, {})[xi] = (
                    xj,
                    cij,
                )
            first = False

    return coupling_map


@njit(nogil=True)
def coupled_flatconfigs_numba(flatconfig, coupling_map):
    """Get the coupled flat configurations for a given flat configuration
    and coupling map.

    Parameters
    ----------
    flatconfig : ndarray[uint8]
        The flat configuration to get the coupled configurations for.
    coupling_map : numba.typed.Dict
        A nested numba dictionary of the form
        ``{term: {reg: {bit_in: (bit_out, coeff), ...}, ...}, ...}``.

    Returns
    -------
    coupled_flatconfigs : ndarray[uint8]
        A list of coupled flat configurations, each with the corresponding
        coefficient.
    coeffs : ndarray[float64]
        The coefficients for each coupled flat configuration.
    """
    buf_ptr = 0
    bjs = np.empty((len(coupling_map), flatconfig.size), dtype=np.uint8)
    cijs = np.empty(len(coupling_map), dtype=np.float64)
    for coupling_t in coupling_map.values():
        cij = 1.0
        bj = flatconfig.copy()
        # bjs[buf_ptr, :] = flatconfig
        for reg, coupling_t_reg in coupling_t.items():
            xi = flatconfig[reg]
            if xi not in coupling_t_reg:
                # zero coupling - whole branch dead
                break
            # update coeff and config
            xj, cij = coupling_t_reg[xi]
            cij *= cij
            if xi != xj:
                bj[reg] = xj
        else:
            # no break - all terms survived
            bjs[buf_ptr, :] = bj
            cijs[buf_ptr] = cij
            buf_ptr += 1
    return bjs[:buf_ptr], cijs[:buf_ptr]


@njit(nogil=True)
def coupled_bits_numba(bi, coupling_map):
    buf_ptr = 0
    bjs = np.empty(len(coupling_map), dtype=np.int64)
    cijs = np.empty(len(coupling_map), dtype=np.float64)
    bitmap = {}

    for coupling_t in coupling_map.values():
        cij = 1.0
        bj = bi
        for reg, coupling_t_reg in coupling_t.items():
            xi = get_nth_bit(bi, reg)
            if xi not in coupling_t_reg:
                # zero coupling - whole branch dead
                break
            # update coeff and config
            xj, cij = coupling_t_reg[xi]
            cij *= cij
            if xi != xj:
                bj = flip_nth_bit(bj, reg)
        else:
            # no break - all terms survived
            if bj in bitmap:
                # already seed this config - just add coeff
                loc = bitmap[bj]
                cijs[loc] += cij
            else:
                # TODO: check performance of numba exception catching
                bjs[buf_ptr] = bj
                cijs[buf_ptr] = cij
                bitmap[bj] = buf_ptr
                buf_ptr += 1

    return bjs[:buf_ptr], cijs[:buf_ptr]


@njit(nogil=True)
def _build_coo_numba_core(bits, coupling_map, bitmap=None):
    # the bit map is needed if we only have a partial set of `bits`, which
    # might couple to other bits that are not in `bits` -> we need to look up
    # the linear register of the coupled bit
    if bitmap is None:
        bitmap = {bi: ci for ci, bi in enumerate(bits)}

    buf_size = len(bits)
    data = np.empty(buf_size, dtype=np.float64)
    cis = np.empty(buf_size, dtype=np.int64)
    cjs = np.empty(buf_size, dtype=np.int64)
    buf_ptr = 0

    for bi in bits:
        ci = bitmap[bi]
        for coupling_t in coupling_map.values():
            hij = 1.0
            bj = bi
            for reg, coupling_t_reg in coupling_t.items():
                xi = get_nth_bit(bi, reg)
                if xi not in coupling_t_reg:
                    # zero coupling - whole branch dead
                    break
                # update coeff and config
                xj, cij = coupling_t_reg[xi]
                hij *= cij
                if xi != xj:
                    bj = flip_nth_bit(bj, reg)
            else:
                # didn't break out of loop
                if buf_ptr >= buf_size:
                    # need to double our storage
                    data = np.concatenate((data, np.empty_like(data)))
                    cis = np.concatenate((cis, np.empty_like(cis)))
                    cjs = np.concatenate((cjs, np.empty_like(cjs)))
                    buf_size *= 2
                data[buf_ptr] = hij
                cis[buf_ptr] = ci
                cjs[buf_ptr] = bitmap[bj]
                buf_ptr += 1

    return data[:buf_ptr], cis[:buf_ptr], cjs[:buf_ptr]


def build_coo_numba(bits, coupling_map, parallel=False):
    """Build an operator in COO form, using the basis ``bits`` and the
     ``coupling_map``, optionally multithreaded.

    Parameters
    ----------
    bits : array
        An array of integers, each representing a bit string.
    coupling_map : Dict[int, Dict[int, Dict[int, Tuple[int, float]]]]
        A nested numba dictionary of couplings. The outermost key is the term
        index, the next key is the register, and the innermost key is the bit
        index. The value is a tuple of (coupled bit index, coupling
        coefficient).
    parallel : bool or int, optional
        Whether to parallelize the computation. If an integer is given, it
        specifies the number of threads to use.

    Returns
    -------
    data : array
        The non-zero elements of the operator.
    cis : array
        The row indices of the non-zero elements.
    cjs : array
        The column indices of the non-zero elements.
    """
    if not parallel:
        return _build_coo_numba_core(bits, coupling_map)

    from quimb import get_thread_pool

    if isinstance(parallel, int):
        n_thread_workers = parallel
    else:
        n_thread_workers = None

    pool = get_thread_pool(n_thread_workers)
    n_thread_workers = pool._max_workers

    # need a global mapping of bits to linear indices
    kws = dict(coupling_map=coupling_map, bitmap=build_bitmap(bits))

    # launch the threads! note we distribtue in cyclic fashion as the sparsity
    # can be concentrated in certain ranges and we want each thread to have
    # roughly the same amount of work to do
    fs = [
        pool.submit(
            _build_coo_numba_core, bits=bits[i::n_thread_workers], **kws
        )
        for i in range(n_thread_workers)
    ]

    # gather and concatenate the results (probably some memory overhead here)
    data = []
    cis = []
    cjs = []
    for f in fs:
        d, ci, cj = f.result()
        data.append(d)
        cis.append(ci)
        cjs.append(cj)

    data = np.concatenate(data)
    cis = np.concatenate(cis)
    cjs = np.concatenate(cjs)

    return data, cis, cjs


# -------------------------- specific hamiltonians -------------------------- #


def fermi_hubbard_from_edges(edges, t=1.0, U=1.0, mu=0.0):
    """ """
    H = SparseOperatorBuilder()
    sites, edges = parse_edges_to_unique(edges)

    if t != 0.0:
        for cooa, coob in edges:
            # hopping
            for s in "↑↓":
                H += -t, ("+", (s, *cooa)), ("-", (s, *coob))
                H += -t, ("+", (s, *coob)), ("-", (s, *cooa))

    for coo in sites:
        # interaction
        H += U, ("n", ("↑", *coo)), ("n", ("↓", *coo))

        # chemical potential
        H += mu, ("n", ("↑", *coo))
        H += mu, ("n", ("↓", *coo))

    H.jordan_wigner_transform()
    return H


def fermi_hubbard_spinless_from_edges(edges, t=1.0, mu=0.0):
    """
    """
    H = SparseOperatorBuilder()
    sites, edges = parse_edges_to_unique(edges)

    for cooa, coob in edges:
        # hopping
        H += -t, ("+", cooa), ("-", coob)
        H += -t, ("+", coob), ("-", cooa)

    # chemical potential
    for coo in sites:
        H += mu, ("n", coo)

    H.jordan_wigner_transform()
    return H


def heisenberg_from_edges(edges, j=1.0, b=0.0, hilbert_space=None):
    """Create a Heisenberg Hamiltonian on the graph defined by ``edges``.

    Parameters
    ----------
    edges : Iterable[tuple[hashable, hashable]]
        The edges, as pairs of hashable 'sites', that define the graph.
        Multiple edges are allowed, and will be treated as a single edge.
    j : float or tuple[float, float, float], optional
        The Heisenberg exchange coupling constant(s). If a single float is
        given, it is used for all three terms. If a tuple of three floats is
        given, they are used for the xx, yy, and zz terms respectively. Note
        that positive values of ``j`` correspond to antiferromagnetic coupling.
    b : float or tuple[float, float, float], optional
        The magnetic field strength(s). If a single float is given, it is used
        taken as a z-field. If a tuple of three floats is given, they are used
        for the x, y, and z fields respectively.
    hilbert_space : HilbertSpace, optional
        The Hilbert space to use. If not given, one will be constructed
        automatically from the edges.

    Returns
    -------
    H : SparseOperatorBuilder
    """
    try:
        jx, jy, jz = j
    except TypeError:
        jx, jy, jz = j, j, j

    try:
        bx, by, bz = b
    except TypeError:
        bx, by, bz = 0, 0, b

    H = SparseOperatorBuilder(hilbert_space=hilbert_space)
    sites, edges = parse_edges_to_unique(edges)

    for cooa, coob in edges:
        if jx == jy:
            # keep things real
            H += jx / 2, ("+", cooa), ("-", coob)
            H += jx / 2, ("-", cooa), ("+", coob)
        else:
            H += jx, ("sx", cooa), ("sx", coob)
            H += jy, ("sy", cooa), ("sy", coob)

        H += (jz, ("sz", "sz"), cooa, coob)

    for site in sites:
        H += bx, ("sx", site)
        H += by, ("sy", site)
        H += bz, ("sz", site)

    return H
