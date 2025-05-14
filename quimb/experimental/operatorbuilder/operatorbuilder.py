import functools
import operator

import numpy as np
from numba import njit

from .hilbertspace import HilbertSpace
from . import flatconfig

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

_OPCOMPLEX = {
    op
    for op, mat in _OPMAP.items()
    if any(isinstance(coeff, complex) for _, (_, coeff) in mat.items())
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


def build_coupling_numba(term_store, site_to_reg, dtype=None):
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

    if (dtype is None) or np.issubdtype(dtype, np.float64):
        ty_coeff = types.float64
    elif np.issubdtype(dtype, np.complex128):
        ty_coeff = types.complex128
    elif np.issubdtype(dtype, np.complex64):
        ty_coeff = types.complex64
    elif np.issubdtype(dtype, np.float32):
        ty_coeff = types.float32
    else:
        raise ValueError(f"Unknown dtype {dtype}")

    ty_xj = types.Tuple((types.int64, ty_coeff))
    ty_xi = types.DictType(types.int64, ty_xj)
    ty_site = types.DictType(types.int64, ty_xi)
    coupling_map = Dict.empty(types.int64, ty_site)

    # for term t ...
    for t, (term, coeff) in enumerate(term_store.items()):
        if (len(term) == 0) and (coeff != 1.0):
            # special case: all identity term, if coeff != 1.0, need to add it
            term = [("I", 0)]
            # directly add to first *register*
            map_to_reg = False
        else:
            map_to_reg = True

        first = True
        # which couples sites with product of ops ...
        for op, site in term:
            if map_to_reg:
                reg = site_to_reg(site)
            else:
                # special all identity term -> always first register
                reg = site

            # -> bit `xi` at `reg` is coupled to `xj` with coeff `cij`
            #          : reg
            #     ...10010...    xi=0  ->
            #     ...10110...    xj=1  with coeff cij

            # populate just the term/reg/bit maps we need
            for xi, (xj, cij) in _OPMAP[op].items():
                if first:
                    # absorb overall coefficient into first coupling
                    cij = coeff * cij

                # what sites does this term act non-trivially on?
                term_to_reg = coupling_map.setdefault(
                    t, Dict.empty(types.int64, ty_xi)
                )

                # at each site, what bits are coupled to which other bits?
                # and with what coefficient?
                reg_to_bits = term_to_reg.setdefault(
                    reg, Dict.empty(types.int64, ty_xj)
                )
                reg_to_bits[xi] = (xj, cij)

            first = False

    return coupling_map


@functools.cache
def calc_dtype_cached(dtype, iscomplex):
    if dtype is None:
        # choose automatically
        if iscomplex:
            dtype = np.complex128
        else:
            dtype = np.float64
    else:
        if np.issubdtype(dtype, np.float64):
            dtype = np.float64
        elif np.issubdtype(dtype, np.complex128):
            dtype = np.complex128
        elif np.issubdtype(dtype, np.complex64):
            dtype = np.complex64
        elif np.issubdtype(dtype, np.float32):
            dtype = np.float32
        else:
            raise TypeError(f"Unsupported dtype {dtype}.")

    return dtype


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

    def __init__(
        self,
        terms=(),
        hilbert_space: HilbertSpace = None,
    ):
        self._term_store = {}
        self._sites_used = set()
        self._hilbert_space = hilbert_space
        self._coupling_maps = {}
        self._iscomplex = False
        for term in terms:
            self.add_term(*term)

    @property
    def sites_used(self):
        """A tuple of the sorted coordinates/sites seen so far."""
        return tuple(sorted(self._sites_used))

    @property
    def hilbert_space(self) -> HilbertSpace:
        """The Hilbert space of the operator. Created from the sites seen
        so far if not supplied at construction.
        """
        if self._hilbert_space is None:
            self._hilbert_space = HilbertSpace(self.sites_used)
        return self._hilbert_space

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
    def iscomplex(self):
        """Whether the operator has complex terms."""
        return self._iscomplex

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
        elif np.iscomplexobj(coeff):
            self._iscomplex = True

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
                return

            if op != "I":
                # only need to record non-identity operators
                simplified_ops.append((op, site))

                if op in _OPCOMPLEX:
                    # if we have complex operators, need to use complex dtype
                    self._iscomplex = True

        # if we have already seen this exact term, just add the coeff
        key = tuple(simplified_ops)
        new_coeff = self._term_store.pop(key, 0.0) + coeff
        if new_coeff != 0.0:
            # but only if it doesn't cancel to zero
            self._term_store[key] = new_coeff

    def __iadd__(self, term):
        self.add_term(*term)
        return self

    def __isub__(self, term):
        self.add_term(-term[0], *term[1:])
        return self

    def jordan_wigner_transform(self):
        """Transform the terms in this operator by pre-prending pauli Z
        strings to all creation and annihilation operators, and then
        simplifying the resulting terms.
        """
        # TODO: check if transform has been applied already?
        # TODO: store untransformed terms, so we can re-order at will?

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

    def show(self, filler="."):
        """Print an ascii representation of the terms in this operator."""
        print(self)
        for _, (term, coeff) in enumerate(self._term_store.items()):
            s = [f"{filler} "] * self.nsites
            for op, site in term:
                s[self.site_to_reg(site)] = f"{op:<2}"
            print("".join(s), f"{coeff:+}")

    def calc_dtype(self, dtype=None):
        """Calculate the numpy data type of the operator to use.

        Parameters
        ----------
        dtype : numpy.dtype or str, optional
            The data type of the coefficients. If not provided, will be
            automatically determined based on the terms in the operator.

        Returns
        -------
        dtype : numpy.dtype
            The data type of the coefficients.
        """
        return calc_dtype_cached(dtype, self._iscomplex)

    def build_coupling_map(self, dtype=None):
        """Build and cache the coupling map for the specified dtype.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            The data type of the coefficients. If not provided, will be
            automatically determined based on the terms in the operator.

        Returns
        -------
        coupling_map : numba.typed.Dict
            A nested numba dictionary of the form
            ``{term: {reg: {bit_in: (bit_out, coeff), ...}, ...}, ...}``.
        """
        dtype = self.calc_dtype(dtype)

        try:
            coupling_map = self._coupling_maps[dtype]
        except KeyError:
            coupling_map = build_coupling_numba(
                self._term_store,
                self.site_to_reg,
                dtype=dtype,
            )
            self._coupling_maps[dtype] = coupling_map

        return coupling_map

    def build_coo_data(self, dtype=None, parallel=False):
        """Build the raw data for a sparse matrix in COO format, optionally
        in parallel.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            The data type of the matrix. If not provided, will be
            automatically determined based on the terms in the operator.
        parallel : bool or int, optional
            Whether to build the matrix in parallel (multi-threaded). If True,
            will use number of threads equal to the number of available CPU
            cores. If False, will use a single thread. If an integer is
            provided, it will be used as the number of threads to use.

        Returns
        -------
        coo : array
            The data entries for the sparse matrix in COO format.
        cis : array
            The row indices for the sparse matrix in COO format.
        cjs : array
            The column indices for the sparse matrix in COO format.
        d : int
            The total number of basis states.
        """
        dtype = self.calc_dtype(dtype)
        coupling_map = self.build_coupling_map(dtype=dtype)
        kwargs = {
            "coupling_map": coupling_map,
            "sector": self.hilbert_space.sector_numba,
            "symmetry": self.hilbert_space.symmetry,
            "dtype": dtype,
        }

        if not parallel:
            coo, cis, cjs = flatconfig.build_coo_numba_core(**kwargs)
        else:
            # figure out how many threads to use etc.
            from quimb import get_thread_pool

            if parallel is True:
                n_thread_workers = None
            elif isinstance(parallel, int):
                n_thread_workers = parallel
            else:
                raise ValueError(f"Unknown parallel option {parallel}.")

            pool = get_thread_pool(n_thread_workers)
            n_thread_workers = pool._max_workers

            # launch the threads! note we distribtue in cyclic fashion as the
            # sparsity can be concentrated in certain ranges and we want each
            # thread to have roughly the same amount of work to do
            fs = [
                pool.submit(
                    flatconfig.build_coo_numba_core,
                    world_rank=i,
                    world_size=n_thread_workers,
                    **kwargs,
                )
                for i in range(n_thread_workers)
            ]

            # gather and concatenate the results (some memory overhead here)
            coo = []
            cis = []
            cjs = []
            for f in fs:
                d, ci, cj = f.result()
                coo.append(d)
                cis.append(ci)
                cjs.append(cj)

            coo = np.concatenate(coo)
            cis = np.concatenate(cis)
            cjs = np.concatenate(cjs)

        return coo, cis, cjs, self.hilbert_space.size

    def build_sparse_matrix(
        self,
        stype="csr",
        dtype=None,
        parallel=False,
    ):
        """Build a sparse matrix in the given format. Optionally in parallel.

        Parameters
        ----------
        stype : str, optional
            The sparse matrix format to use. Can be one of 'coo', 'csr', 'csc',
            'bsr', 'lil', 'dok', or 'dia'. Default is 'csr'.
        dtype : numpy.dtype, optional
            The data type of the matrix. If not provided, will be
            automatically determined based on the terms in the operator.
        parallel : bool, optional
            Whether to build the matrix in parallel (multi-threaded).

        Returns
        -------
        scipy.sparse matrix
        """
        import scipy.sparse as sp

        coo, cis, cjs, d = self.build_coo_data(
            dtype=dtype,
            parallel=parallel,
        )
        A = sp.coo_matrix((coo, (cis, cjs)), shape=(d, d))
        if stype != "coo":
            A = A.asformat(stype)
        return A

    def build_dense(self, dtype=None):
        """Get the dense (`numpy.ndarray`) matrix representation of this
        operator.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            The data type of the matrix. If not provided, will be
            automatically determined based on the terms in the operator.

        Returns
        -------
        A : numpy.ndarray
            The dense matrix representation of this operator.
        """
        A = self.build_sparse_matrix(stype="coo", dtype=dtype)
        return A.toarray()

    def build_local_terms(self, dtype=None):
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

        dtype = self.calc_dtype(dtype)

        for term, coeff in self._term_store.items():
            ops, sites = zip(*term)
            mats = (get_mat(op, dtype=dtype) for op in ops)
            hk = coeff * functools.reduce(np.kron, mats)
            if sites not in Hk:
                Hk[sites] = hk
            else:
                Hk[sites] = Hk[sites] + hk

        return Hk

    def flatconfig_coupling(self, flatconfig, dtype=None):
        """Get an array of other configurations coupled to the given
        ``flatconfig`` by this operator, and the corresponding coupling
        coefficients. This is for use with VMC for example.

        Parameters
        ----------
        flatconfig : array[np.uint8]
            The linear array of the configuration to get the coupling for.
        dtype : numpy.dtype, optional
            The data type of the matrix. If not provided, will be
            automatically determined based on the terms in the operator.

        Returns
        -------
        """
        dtype = calc_dtype_cached(dtype, self._iscomplex)
        coupling_map = self.build_coupling_map(dtype=dtype)
        return flatconfig.flatconfig_coupling(
            flatconfig,
            coupling_map=coupling_map,
            dtype=dtype,
        )

    def config_coupling(self, config, dtype=None):
        """Get a list of other configurations coupled to ``config`` by this
        operator, and the corresponding coupling coefficients. This is for
        use with VMC for example.

        Parameters
        ----------
        config : dict[site, int]
            The configuration to get the coupling for.

        Returns
        -------
        coupled_configs : list[dict[site, np.uint8]]
            Each distinct configuration coupled to ``config``.
        coeffs: list[dtype]
            The corresponding coupling coefficients.
        """
        flatconfig = self.hilbert_space.config_to_flatconfig(config)
        bjs, cijs = self.flatconfig_coupling(flatconfig, dtype=dtype)
        coupled_configs = [
            self.hilbert_space.flatconfig_to_config(bj) for bj in bjs
        ]
        return coupled_configs, cijs

    def build_state_machine_greedy(self):
        # XXX: upgrade to optimal method : https://arxiv.org/abs/2006.02056

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

        # track which terms pass through each edge and vice versa
        edges_to_terms = {}
        terms_to_edges = {}
        # so that we can place all the coefficients at the end
        coeffs_to_place = {}

        def new_edge(a, b):
            # need to track which terms pass through this edge so we can
            # place the coefficient somewhere unique at the end
            G.add_edge(a, b, op=op, weight=1, coeff=None)
            edges_to_terms.setdefault((a, b), set()).add(t)
            terms_to_edges.setdefault(t, set()).add((a, b))

            if G.out_degree(a) > 1:
                # no longer valid out_string, for all ancestor nodes
                G.nodes[a]["out_string"] = None
                for prev_node in nx.ancestors(G, a):
                    G.nodes[prev_node]["out_string"] = None

        def check_right():
            # check if can **right share**
            # - check all existing potential next nodes
            # - right strings must match
            # - must be single output node
            # - current op must match or not exist
            for rail in range(num_rails[reg + 1]):
                cand_node = (reg + 1, rail)
                if G.out_degree(cand_node) > 1:
                    # can't share if there's not a single output string
                    continue

                if G.nodes[cand_node]["out_string"] != string[reg + 1 :]:
                    # output string must match
                    continue

                e = (current_node, cand_node)
                if e not in G.edges:
                    new_edge(current_node, cand_node)
                    return cand_node
                else:
                    if G.edges[e]["op"] != op:
                        continue
                    G.edges[e]["weight"] += 1
                    edges_to_terms.setdefault(e, set()).add(t)
                    terms_to_edges.setdefault(t, set()).add(e)
                    return cand_node

                # XXX: if we can right share, don't need to do anything
                # more since whole remaining string is shared?
                # -> possibly, to track term congestion for coeff placement

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
                        edges_to_terms.setdefault(e, set()).add(t)
                        terms_to_edges.setdefault(t, set()).add(e)
                        return cand_node

        def create_new():
            # create a new rail at the next register
            next_node = (reg + 1, num_rails[reg + 1])
            num_rails[reg + 1] += 1
            new_edge(current_node, next_node)

            # if G.out_degree(current_node) > 1:
            #     # no longer valid out_string, for all ancestor nodes
            #     G.nodes[current_node]["out_string"] = None
            #     for prev_node in nx.ancestors(G, current_node):
            #         G.nodes[prev_node]["out_string"] = None

            # the new node is always a single output node so far
            G.nodes[next_node]["out_string"] = string[reg + 1 :]

            return next_node

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

            if coeff != 1.0:
                # record that we still need to place coeff somewhere
                coeffs_to_place[t] = coeff

        G.graph["nsites"] = self.nsites
        G.graph["num_rails"] = tuple(num_rails)
        G.graph["max_num_rails"] = max(num_rails)

        # how many terms pass through each edge
        edge_scores = {e: len(ts) for e, ts in edges_to_terms.items()}

        # the least congested edge a term passes through
        term_scores = {
            t: min(edge_scores[e] for e in es)
            for t, es in terms_to_edges.items()
        }

        def place_coeff(edge, coeff):
            G.edges[edge]["coeff"] = coeff

            # every term passing through this edge is multiplied by this coeff
            for t in edges_to_terms.pop(edge):
                new_coeff = coeffs_to_place.pop(t, 1.0) / coeff
                if new_coeff != 1.0:
                    # if another term doesn't have matching coeff, still need
                    # to place the updated coeff
                    coeffs_to_place[t] = new_coeff

                # # remove edge as candidate for placing other coefficients
                # terms_to_edges[t].remove(edge)
            edge_scores.pop(edge)

        while coeffs_to_place:
            # get the remaining term with the maximum congestion
            t = max(coeffs_to_place, key=term_scores.get)
            # get the least congested edge it passes through
            best = min(
                (e for e in terms_to_edges[t] if e in edge_scores),
                key=edge_scores.get,
            )
            # place it and update everything
            place_coeff(best, coeffs_to_place[t])

        return G

    def draw_state_machine(
        self,
        method="greedy",
        figsize="auto",
        G=None,
    ):
        """Draw the fintie state machine for this operator, as if buildling
        the MPO.
        """
        import math

        from matplotlib import pyplot as plt

        from quimb.schematic import auto_colors

        if G is None:
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
            if data.get("coeff", None) is not None:
                label += f" * {data['coeff']:.4g}"
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

    def build_mpo(self, method="greedy", dtype=None, **mpo_opts):
        """Build a matrix product operator (MPO) representation of this
        operator.

        Parameters
        ----------
        method : str, optional
            The method to use for building the MPO. Currently only "greedy"
            is supported.
        dtype : type, optional
            The data type of the MPO. If not supplied, will be chosen
            automatically based on the terms in the operator.
        mpo_opts : keyword arguments
            Additional options to pass to the MPO constructor.
            See `MatrixProductOperator` for details.

        Returns
        -------
        mpo : MatrixProductOperator
            The MPO representation of this operator.
        """
        import numpy as np

        import quimb as qu
        import quimb.tensor as qtn

        if method == "greedy":
            G = self.build_state_machine_greedy()
        else:
            raise ValueError(f"Unknown method {method}.")

        dtype = self.calc_dtype(dtype)
        Wts = [
            np.zeros((dl, dr, 2, 2), dtype=dtype)
            for dl, dr in qu.utils.pairwise(G.graph["num_rails"])
        ]

        for node_a, node_b, data in G.edges(data=True):
            op = data["op"]
            coeff = data.get("coeff", None)
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
