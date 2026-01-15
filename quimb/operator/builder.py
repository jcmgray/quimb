"""Given a single definition of hilbert space and set of terms, build out
various representations of the operator.

This includes:

- Sparse matrix in various formats (CSR, CSC, COO, etc.)
- Dense matrix
- Matrix product operator (MPO) for DMRG etc.
- Dict of k-local terms (for use with PEPS etc.)
- Coupling function, for use with VMC etc.
"""

import functools
import operator

import numpy as np

from . import configcore
from .hilbertspace import HilbertSpace

_OPMAP = {
    "I": {0: (0, 1.0), 1: (1, 1.0)},
    # pauli matrices
    "x": {0: (1, 1.0), 1: (0, 1.0)},
    "y": {0: (1, 1.0j), 1: (0, -1.0j)},
    "z": {0: (0, 1.0), 1: (1, -1.0)},
    # ⴵ = ZX = iY: 'real Y'
    "ⴵ": {0: (1, -1.0), 1: (0, 1.0)},
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


def _identity_fn(x):
    return x


def jordan_wigner_transform(terms, site_to_reg=None, reg_to_site=None):
    """Transform the terms in this operator by pre-prending pauli Z
    strings to all creation and annihilation operators. This is always
    applied directly to the raw terms, so that the any fermionic ordering
    is respected. Any further transformations (e.g. simplification or
    pauli decomposition) should thus be applied after this transformation.

    Note this doesn't decompose +, - into (X + iY) / 2 and (X - iY) / 2, it
    just prepends Z strings. Call `pauli_decompose` after this to get the
    full decomposition.

    The placement of the Z strings is defined by the ordering supplied by
    `site_to_reg` and `reg_to_site`, by default, it assumes the terms already
    are specified on a linear range of integers.

    Parameters
    ----------
    terms : dict[term, coeff]
        The terms of the operator. Each term is a tuple of tuples, where each
        inner tuple is a pair of ``(operator, site)``.
    site_to_reg : callable, optional
        A function that maps a site to a linear register index. If not
        provided, the sites are assumed to be linear integers already.
    reg_to_site : callable, optional
        A function that maps a linear register index to a site. If not
        provided, the sites are assumed to be linear integers already.

    Returns
    -------
    terms_jordan_wigner : dict[term, coeff]
        The transformed terms of the operator. Each term is a tuple of tuples,
        where each inner tuple is a pair of ``(operator, site)``.
    """
    if site_to_reg is None:
        site_to_reg = _identity_fn
    if reg_to_site is None:
        reg_to_site = _identity_fn

    terms_jordan_wigner = {}

    for term, coeff in terms.items():
        if not term:
            # all identity term, can't unzip
            terms_jordan_wigner[term] = coeff
            continue

        ops, _ = zip(*term)
        if {"+", "-"}.intersection(ops):
            # need to insert jordan-wigner strings
            new_term = []
            for op, site in term:
                reg = site_to_reg(site)
                if op in {"+", "-"}:
                    for r in range(reg):
                        site_below = reg_to_site(r)
                        new_term.append(("z", site_below))
                new_term.append((op, site))
            terms_jordan_wigner[tuple(new_term)] = coeff
        else:
            # no creation/annihilation operators, just add the term
            terms_jordan_wigner[term] = coeff

    return terms_jordan_wigner


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
    for op in _OPMAP.keys():
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


def simplify(terms, atol=1e-12, site_to_reg=None):
    """Simplify the given terms by combining operators acting on the same site,
    putting them in canonical order, removing null-terms, and combining
    equivalent operator strings.

    Parameters
    ----------
    terms : dict[term, coeff]
        The terms of the operator. Each term is a tuple of tuples, where each
        inner tuple is a pair of ``(operator, site)``.
    atol : float, optional
        The absolute tolerance for considering coefficients after
        simplification to be null.
    site_to_reg : callable, optional
        A function that maps a site to a linear register index. If not
        provided, the sites are assumed to be linear integers already.
        This is just used to sort the operators into canonical order.

    Returns
    -------
    terms_simplified : dict[term, coeff]
        The simplified terms of the operator. Each term is a tuple of tuples,
        where each inner tuple is a pair of ``(operator, site)``.
    """
    if site_to_reg is None:
        site_to_reg = _identity_fn

    terms_simplified = {}

    for term, coeff in terms.items():
        # collect operators acting on the same site
        collected = {}
        for op, site in term:
            collected.setdefault(site, []).append(op)

        # simplify operators acting on the same site & don't add null-terms
        simplified_ops = []
        for site, collected_ops in collected.items():
            coeff, op = simplify_single_site_ops(coeff, tuple(collected_ops))

            if op is None:
                # null-term ('e.g. '++' or '--')
                coeff = 0.0
                break

            if op != "I":
                # only need to record non-identity operators
                simplified_ops.append((op, site))

        if abs(coeff) < atol:
            # null-term
            continue

        # assume we can sort the operators into canonical order now
        simplified_ops.sort(key=lambda x: (site_to_reg(x[1]), x[0]))
        # combine coefficients of equivalent terms
        key = tuple(simplified_ops)
        coeff = terms_simplified.pop(key, 0.0) + coeff

        if abs(coeff) < atol:
            # null-term after combining coefficients
            continue

        if abs(coeff.imag) < atol:
            # if the coefficient is real, convert to real
            coeff = coeff.real

        terms_simplified[key] = coeff

    return terms_simplified


@functools.lru_cache(maxsize=None)
def get_pauli_decomp(op, atol=1e-12, use_zx=False):
    """Decompose the given operator (specified as a label) into a sum of
    Pauli components.

    Parameters
    ----------
    op : str
        The operator to decompose.
    atol : float, optional
        The absolute tolerance for considering coefficients to be null.
    use_zx : bool, optional
        Whether to decompose in terms of the real `ⴵ = ZX = iY` instead of `Y`.

    Returns
    -------
    list[tuple[float, str]]
        The decomposition of the operator into Pauli components. Each tuple
        contains the coefficient and the Pauli operator label.
    """
    bops = ("I", "x", "y", "z")

    if op in bops:
        terms = [(1.0, op)]
    else:
        terms = []
        mat = get_mat(op)
        for bop in bops:
            bmat = get_mat(bop)

            # Hilbert-Schmidt inner product
            cb = np.trace(bmat @ mat) / 2

            if abs(cb.imag) < atol:
                # realify if possible
                cb = cb.real

            # ignore zero components
            if abs(cb) >= atol:
                terms.append((cb, bop))

    if use_zx:
        # convert Y -> -iZX
        terms = [
            (-1j * coeff, "ⴵ") if op == "y" else (coeff, op)
            for coeff, op in terms
        ]

    return terms


def pauli_decompose(terms, atol=1e-12, use_zx=False, site_to_reg=None):
    """Decompose the given terms into a sum of Pauli strings.

    Parameters
    ----------
    terms : dict[term, coeff]
        The terms of the operator. Each term is a tuple of tuples, where each
        inner tuple is a pair of ``(operator, site)``.
    atol : float, optional
        The absolute tolerance for considering coefficients after
        decomposition to be null.
    use_xz : bool, optional
        Whether to decompose in terms of the real `ⴵ = ZX = iY` instead of `Y`.
    site_to_reg : callable, optional
        A function that maps a site to a linear register index. If not
        provided, the sites are assumed to be linear integers already.
        This is just used to sort the operators into canonical order.

    Returns
    -------
    terms_pauli_decomposed : dict[term, coeff]
        The decomposed terms of the operator. Each term is a tuple of tuples,
        where each inner tuple is a pair of ``(operator, site)``.
    """
    terms_pauli_decomposed = {}

    for ops, coeff in terms.items():
        # for each current term -> turn into potential sum
        new_ts = [(coeff, ())]

        for op, reg in ops:
            # for each operator in the string
            new_ts = [
                # extend with weighted pauli ...
                (coeff_t * dcoeff, (*ops_t, (dop, reg)))
                # ... for each weighted pauli in the decomposition
                for dcoeff, dop in get_pauli_decomp(op, atol, use_zx)
                # ... for each term in current sum
                for coeff_t, ops_t in new_ts
            ]

        for coeff, ops in new_ts:
            # we can sort paulis strings into canonical order, strip identities
            key = tuple(
                (op, site)
                for op, site in sorted(
                    ops, key=lambda x: (site_to_reg(x[1]), x[0])
                )
                if op != "I"
            )

            coeff = terms_pauli_decomposed.pop(key, 0.0) + coeff

            if abs(coeff) < atol:
                # null-term after combining coefficients
                continue

            if abs(coeff.imag) < atol:
                # if the coefficient is real, convert to real
                coeff = coeff.real

            terms_pauli_decomposed[key] = coeff

    return terms_pauli_decomposed


def get_pool_and_world_size(parallel):
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

    return pool, n_thread_workers


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
    coupling_map : tuple[ndarray]
        The operator defined as tuple of flat arrays.
    """
    if (dtype is None) or np.issubdtype(dtype, np.float64):
        dtype = np.float64
    elif np.issubdtype(dtype, np.complex128):
        dtype = np.complex128
    elif np.issubdtype(dtype, np.complex64):
        dtype = np.complex64
    elif np.issubdtype(dtype, np.float32):
        dtype = np.float32
    else:
        raise ValueError(f"Unknown dtype {dtype}")

    # number of operators per term e.g. 5 for '+zzz-'
    sizes_term = []
    # the registers each term acts on
    regs = []
    # number of elements per operator e.g. 2 for 'z', 1 for '+'
    sizes_op = []
    # input / output bits, e.g. 0 -> 1 for 'sx'
    xis = []
    xjs = []
    # transition coeffs, e.g. 0.5 for 'sx'
    cijs = []

    # for term t ...
    for term, coeff in term_store.items():
        if len(term) == 0:
            # special case: all identity term
            term = [("I", 0)]
            # directly add to first *register*
            map_to_reg = False
        else:
            map_to_reg = True

        # what sites does this term act non-trivially on?
        size = 0
        first_reg = True
        # which couples sites with product of ops ...
        for op, site in term:
            if map_to_reg:
                reg = site_to_reg(site)
            else:
                # special all identity term -> always first register
                reg = site

            regs.append(reg)
            # -> bit `xi` at `reg` is coupled to `xj` with coeff `cij`
            #          : reg
            #     ...10010...    xi=0  ->
            #     ...10110...    xj=1  with coeff cij

            # populate just the term/reg/bit maps we need
            size_t = 0
            for xi, (xj, cij) in _OPMAP[op].items():
                if first_reg:
                    # absorb overall coefficient into first coupling
                    cij = coeff * cij

                xis.append(xi)
                xjs.append(xj)
                cijs.append(cij)
                size_t += 1

            sizes_op.append(size_t)
            size += 1
            first_reg = False

        sizes_term.append(size)

    return (
        np.array(sizes_term, dtype=np.uint32),
        np.array(regs, dtype=np.uint32),
        np.array(sizes_op, dtype=np.uint8),
        np.array(xis, dtype=np.uint8),
        np.array(xjs, dtype=np.uint8),
        np.array(cijs, dtype=dtype),
    )


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
        when required. The default symmetry and sector to build operators in is
        inherited from this Hilbert space, but can be overridden.
    dtype : numpy.dtype or str, optional
        A default data type for the coefficients of the operator. If not
        provided, will be automatically determined at building time based on
        the terms in the operator. If the operator is complex, will be set to
        ``np.complex128``. If the operator is real, will be set to
        ``np.float64``. Individual building methods can override this.
    jordan_wigner : bool, optional
        Whether to apply the Jordan-Wigner transformation to the terms
        automatically when processing them. This prepends pauli Z strings to
        all creation and annihilation operators.
    pauli_decompose : bool or "zx", optional
        Whether to apply the Pauli decomposition to the terms automatically
        when processing them. This decomposes all local operators into sums of
        Pauli operators. If "zx" is supplied, the decomposition is done in
        terms of the real `ZX = iY` operator instead of `Y`.
    atol : float, optional
        The absolute tolerance for considering coefficients to be null when
        simplifying and decomposing terms.
    """

    def __init__(
        self,
        terms=(),
        hilbert_space: HilbertSpace = None,
        dtype=None,
        jordan_wigner=False,
        pauli_decompose=False,
        atol=1e-12,
    ):
        self._sites_used = set()
        self._hilbert_space = hilbert_space

        # terms as they are supplied by user
        self._terms_raw = {}
        # terms after processing (jordan-wigner, pauli decomp, simplification)
        self._terms_final = None

        # processing flags
        self._transform_jordan_wigner = jordan_wigner
        self._transform_pauli_decompose = pauli_decompose
        self._atol = atol

        self._dtype = dtype
        self._coupling_maps = {}
        self._cache = {}

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

    def site_to_reg(self, site):
        """Get the register / linear index of coordinate ``site``."""
        return self.hilbert_space.site_to_reg(site)

    def reg_to_site(self, reg):
        """Get the site of register / linear index ``reg``."""
        return self.hilbert_space.reg_to_site(reg)

    @property
    def terms_raw(self):
        """A tuple of the raw terms seen so far, as a mapping from
        operator strings to coefficients.
        """
        return tuple((coeff, ops) for ops, coeff in self._terms_raw.items())

    def _reset_caches(self):
        """Reset any cached representations of the operator, used whenever the
        terms are modified in any way, and thus require reprocessing.
        """
        self._cache.clear()
        self._coupling_maps.clear()
        self._terms_final = None

    def _get_terms_final(self):
        """Get the processed terms, applying any requested transformations, if
        not already done.
        """
        if self._terms_final is None:
            # need to (re)process raw terms to final terms
            terms = self._terms_raw

            # 1. jordan wigner transform before anything else
            if self._transform_jordan_wigner:
                terms = jordan_wigner_transform(
                    terms=terms,
                    site_to_reg=self.site_to_reg,
                    reg_to_site=self.reg_to_site,
                )

            terms = simplify(
                terms=terms,
                atol=self._atol,
                site_to_reg=self.site_to_reg,
            )

            # 2. pauli decomposition next
            if self._transform_pauli_decompose:
                terms = pauli_decompose(
                    terms=terms,
                    atol=self._atol,
                    use_zx=self._transform_pauli_decompose == "zx",
                    site_to_reg=self.site_to_reg,
                )

            # 3. finally simplify into strict form last
            self._terms_final = simplify(
                terms=terms,
                atol=self._atol,
                site_to_reg=self.site_to_reg,
            )

        return self._terms_final

    @property
    def terms(self):
        """A tuple of the, possibly transformed, terms seen so far."""
        return tuple(
            (coeff, ops) for ops, coeff in self._get_terms_final().items()
        )

    @property
    def nterms(self):
        """The total number of terms seen so far."""
        return len(self._get_terms_final())

    @property
    def locality(self):
        """The locality of the operator, the maximum support of any term."""
        terms = self._get_terms_final()
        if not terms:
            return 0
        return max(len(ops) for ops in terms)

    @property
    def iscomplex(self):
        """Whether the operator has complex terms."""
        try:
            iscomplex = self._cache["iscomplex"]
        except KeyError:
            iscomplex = self._cache["iscomplex"] = any(
                np.iscomplexobj(coeff)
                or any(op in _OPCOMPLEX for op, _ in ops)
                for ops, coeff in self._get_terms_final().items()
            )
        return iscomplex

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
            - ``'sx'``, ``'sy'``, ``'sz'``: spin operators (i.e. scaled Pauli
              matrices)
            - ``'+'``, ``'-'``: creation/annihilation operators
            - ``'n'``, ``'sn'``, or ``'h'``: number, symmetric number (n - 1/2)
              and hole (1 - n) operators.

            And ``site`` is a hashable object that represents the site that
            the operator acts on. If this builder has an associated Hilbert
            space already, the site must be present in that Hilbert space, else
            a minimal Hilbert space will be constructed from the sites used,
            when required.
        """
        if isinstance(coeff_ops[0], (tuple, list)):
            # assume coeff is 1.0
            coeff = 1
            ops = coeff_ops
        else:
            coeff, *ops = coeff_ops
            if abs(coeff) < self._atol:
                # null-term
                return

        # parse the operator specification
        ops = tuple((operator, site) for operator, site in ops)
        for op, site in ops:
            # check that the site is valid if the Hilbert space is known
            if (
                self._hilbert_space is not None
            ) and not self._hilbert_space.has_site(site):
                raise ValueError(f"Site {site} not in the Hilbert space.")

            # record used sites, for later construction of the Hilbert space
            self._sites_used.add(site)

            if op not in _OPMAP:
                raise ValueError(f"Unknown operator '{op}'.")

        # if we have already seen this exact term, just add the coeff but note,
        # we do not simplify equivalent terms here, in case it is fermionic
        coeff = self._terms_raw.pop(ops, 0.0) + coeff

        if abs(coeff) < self._atol:
            # null-term after combining coefficients
            return

        if abs(coeff.imag) < self._atol:
            # if the coefficient is real, convert to real
            coeff = coeff.real

        self._terms_raw[ops] = coeff
        self._reset_caches()

    def __iadd__(self, term):
        self.add_term(*term)
        return self

    def __isub__(self, term):
        self.add_term(-term[0], *term[1:])
        return self

    def jordan_wigner_transform(self, value=None):
        """Toggle transforming the terms in this operator by pre-prending
        pauli Z strings to all creation and annihilation operators. This is
        always applied directly as the first processing step to the raw terms,
        so that the fermionic ordering is respected.

        Note this doesn't decompose +, - into (X + iY) / 2 and (X - iY) / 2, it
        just prepends Z strings. Use `pauli_decompose` to get the full
        decomposition.

        The placement of the Z strings is defined by the ordering of the
        hilbert space, by default, the sorted order of the site labels.

        Parameters
        ----------
        value : bool, optional
            Whether to apply the Jordan-Wigner transformation. If `None` (the
            default) then this method acts as toggle. Whereas supplying `True`
            or `False` explicitly sets or unsets the transformation.
        """
        if value is None:
            value = not self._transform_jordan_wigner
        else:
            value = bool(value)
        self._transform_jordan_wigner = value
        self._reset_caches()

    def pauli_decompose(self, value=None, atol=None, use_zx=False):
        """Transform the terms in this operator by decomposing them into
        Pauli strings.

        Parameters
        ----------
        value : bool, optional
            Whether to apply the Pauli decomposition. If `None` (the default)
            then this method acts as toggle. Whereas supplying `True` or
            `False` explicitly sets or unsets the transformation.
        """
        if value is None:
            value = not self._transform_pauli_decompose
        else:
            value = bool(value)

        if value and use_zx:
            self._transform_pauli_decompose = "zx"
        else:
            self._transform_pauli_decompose = value

        if atol is not None:
            self._atol = atol

        self._reset_caches()

    def show(self, filler="."):
        """Print an ascii representation of the terms in this operator."""
        print(self)
        for _, (term, coeff) in enumerate(self._get_terms_final().items()):
            s = [f"{filler} "] * self.nsites
            for op, site in term:
                s[self.site_to_reg(site)] = f"{op:<2}"
            print("".join(s), f"{coeff:+}")

    def get_dtype(self, dtype=None):
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
        if dtype is None:
            dtype = self._dtype
        return calc_dtype_cached(dtype, self.iscomplex)

    def get_coupling_map(self, dtype=None):
        """Build and cache the coupling map for the specified dtype.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            The data type of the coefficients. If not provided, will be
            automatically determined based on the terms in the operator.

        Returns
        -------
        coupling_map : tuple[ndarray]
            The operator defined as tuple of flat arrays.
        """
        dtype = self.get_dtype(dtype)

        try:
            coupling_map = self._coupling_maps[dtype]
        except KeyError:
            coupling_map = build_coupling_numba(
                self._get_terms_final(),
                self.site_to_reg,
                dtype=dtype,
            )
            self._coupling_maps[dtype] = coupling_map

        return coupling_map

    def flatconfig_coupling(self, flatconfig, dtype=None):
        """Get an array of other configurations coupled to the given individual
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
        coupled_flatconfigs : ndarray[np.uint8]
            Each distinct flatconfig coupled to ``flatconfig``.
        coeffs : ndarray[dtype]
            The corresponding coupling coefficients.
        """
        dtype = self.get_dtype(dtype)
        coupling_map = self.get_coupling_map(dtype=dtype)
        return configcore.flatconfig_coupling_numba(
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
        fc = self.hilbert_space.config_to_flatconfig(config)
        bjs, cijs = self.flatconfig_coupling(fc, dtype=dtype)
        coupled_configs = [
            self.hilbert_space.flatconfig_to_config(bj) for bj in bjs
        ]
        return coupled_configs, cijs

    def evaluate_exact_flatconfigs(self, fn_amplitude, progbar=False):
        """Calculate the expectation value of this operator with respect to
        a wavefunction provided as a function with signature::

            fn_amplitude(flatconfig: ndarray[np.uint8]) -> z: float | complex

        """
        O = 0.0
        p = 0.0

        if progbar:
            import tqdm

            iterator = tqdm.tqdm(range(self.hilbert_space.size))
        else:
            iterator = range(self.hilbert_space.size)

        for r in iterator:
            flatconfig = self.hilbert_space.rank_to_flatconfig(r)

            xpsi = fn_amplitude(flatconfig)
            if not xpsi:
                continue

            pi = abs(xpsi) ** 2
            p += pi

            Oloc = 0.0
            for fy, hxy in zip(*self.flatconfig_coupling(flatconfig)):
                ypsi = fn_amplitude(fy)
                Oloc = Oloc + hxy * ypsi / xpsi

            O += Oloc * pi

        return O / p

    def evaluate_exact_configs(self, fn_amplitude, progbar=False):
        """Calculate the expectation value of this operator with respect to
        a wavefunction provided as a function with signature::

            fn_amplitude(config: dict[site, int]) -> z: float | complex

        """
        O = 0.0
        p = 0.0

        if progbar:
            import tqdm

            iterator = tqdm.tqdm(range(self.hilbert_space.size))
        else:
            iterator = range(self.hilbert_space.size)

        for r in iterator:
            config = self.hilbert_space.rank_to_config(r)

            xpsi = fn_amplitude(config)
            if not xpsi:
                continue

            pi = abs(xpsi) ** 2
            p += pi

            Oloc = 0.0
            for fy, hxy in zip(*self.config_coupling(config)):
                ypsi = fn_amplitude(fy)
                Oloc = Oloc + hxy * ypsi / xpsi

            O += Oloc * pi

        return O / p

    def build_coo_data(
        self, sector=None, symmetry=None, dtype=None, parallel=False
    ):
        """Build the raw data for a sparse matrix in COO format, optionally
        in parallel.

        Parameters
        ----------
        sector : {None, str, int, ((int, int), (int, int))}, optional
            The sector of the Hilbert space. If None, the default sector is
            used.
        symmetry : {None, "Z2", "U1", "U1U1"}, optional
            The symmetry of the Hilbert space. If None, the default symmetry is
            used, or inferred from the supplied sector if possible.
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
        data : array
            The data entries for the sparse matrix in COO format.
        rows : array
            The row indices for the sparse matrix in COO format.
        cols : array
            The column indices for the sparse matrix in COO format.
        d : int
            The total number of basis states.
        """
        dtype = self.get_dtype(dtype)
        coupling_map = self.get_coupling_map(dtype=dtype)
        d = self.hilbert_space.get_size(sector, symmetry)
        sector_nb, symmetry_nb = self.hilbert_space.get_sector_numba(
            sector=sector, symmetry=symmetry
        )
        kwargs = {
            "coupling_map": coupling_map,
            "sector": sector_nb,
            "symmetry": symmetry_nb,
            "dtype": dtype,
        }

        if not parallel:
            data, rows, cols = configcore.build_coo_numba_core(**kwargs)
        else:
            pool, world_size = get_pool_and_world_size(parallel)

            # launch the threads! note we distribtue in cyclic fashion as the
            # sparsity can be concentrated in certain ranges and we want each
            # thread to have roughly the same amount of work to do
            fs = [
                pool.submit(
                    configcore.build_coo_numba_core,
                    world_rank=i,
                    world_size=world_size,
                    **kwargs,
                )
                for i in range(world_size)
            ]

            # gather and concatenate the results (some memory overhead here)
            data = []
            rows = []
            cols = []
            for f in fs:
                df, ci, cj = f.result()
                data.append(df)
                rows.append(ci)
                cols.append(cj)

            data = np.concatenate(data)
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)

        return data, rows, cols, d

    def build_sparse_matrix(
        self,
        sector=None,
        symmetry=None,
        dtype=None,
        stype="csr",
        parallel=False,
    ):
        """Build a sparse matrix in the given format. Optionally in parallel.

        Parameters
        ----------
        sector : {None, str, int, ((int, int), (int, int))}, optional
            The sector of the Hilbert space. If None, the default sector is
            used.
        symmetry : {None, "Z2", "U1", "U1U1"}, optional
            The symmetry of the Hilbert space. If None, the default symmetry is
            used, or inferred from the supplied sector if possible.
        dtype : numpy.dtype, optional
            The data type of the matrix. If not provided, will be
            automatically determined based on the terms in the operator.
        stype : str, optional
            The sparse matrix format to use. Can be one of 'coo', 'csr', 'csc',
            'bsr', 'lil', 'dok', or 'dia'. Default is 'csr'.
        parallel : bool, optional
            Whether to build the matrix in parallel (multi-threaded).

        Returns
        -------
        scipy.sparse matrix
        """
        import scipy.sparse as sp

        data, rows, cols, d = self.build_coo_data(
            sector=sector,
            symmetry=symmetry,
            dtype=dtype,
            parallel=parallel,
        )
        A = sp.coo_matrix((data, (rows, cols)), shape=(d, d))
        if stype != "coo":
            A = A.asformat(stype)

        return A

    def build_dense(
        self,
        sector=None,
        symmetry=None,
        dtype=None,
        parallel=False,
    ):
        """Get the dense (`numpy.ndarray`) matrix representation of this
        operator.

        Parameters
        ----------
        sector : {None, str, int, ((int, int), (int, int))}, optional
            The sector of the Hilbert space. If None, the default sector is
            used.
        symmetry : {None, "Z2", "U1", "U1U1"}, optional
            The symmetry of the Hilbert space. If None, the default symmetry is
            used, or inferred from the supplied sector if possible.
        dtype : numpy.dtype, optional
            The data type of the matrix. If not provided, will be
            automatically determined based on the terms in the operator.
        parallel : bool or int, optional
            Whether to build the matrix in parallel (multi-threaded). If
            True, will use number of threads equal to the number of
            available CPU cores. If False, will use a single thread. If an
            integer is provided, it will be used as the number of threads to
            use.

        Returns
        -------
        A : numpy.ndarray
            The dense matrix representation of this operator.
        """
        A = self.build_sparse_matrix(
            sector=sector,
            symmetry=symmetry,
            dtype=dtype,
            stype="coo",
            parallel=parallel,
        )
        return A.toarray()

    def matvec(
        self,
        x,
        out=None,
        sector=None,
        symmetry=None,
        dtype=None,
        parallel=False,
    ):
        """Apply this operator lazily (i.e. without constructing a sparse
        matrix) to a vector. This uses less memory but is much slower.

        Parameters
        ----------
        x : array
            The vector to apply the operator to.
        out : array, optional
            An array to store the result in. If not provided, a new array
            will be created.
        sector : {None, str, int, ((int, int), (int, int))}, optional
            The sector of the Hilbert space. If None, the default sector is
            used. The implicit size should match that of `x`.
        symmetry : {None, "Z2", "U1", "U1U1"}, optional
            The symmetry of the Hilbert space. If None, the default symmetry is
            used, or inferred from the supplied sector if possible.
        dtype : numpy.dtype, optional
            The data type of the matrix. If not provided, will be
            automatically set as the same as the input vector.
        parallel : bool or int, optional
            Whether to apply the operator in parallel (multi-threaded). If
            True, will use number of threads equal to the number of
            available CPU cores. If False, will use a single thread. If an
            integer is provided, it will be used as the number of threads to
            use. Uses `num_threads` more memory but is faster.

        Returns
        -------
        out : array
            The result of applying the operator to the vector.
        """
        if dtype is None:
            dtype = x.dtype

        sector_nb, symmetry_nb = self.hilbert_space.get_sector_numba(
            sector=sector, symmetry=symmetry
        )

        kwargs = {
            "coupling_map": self.get_coupling_map(dtype=dtype),
            "sector": sector_nb,
            "symmetry": symmetry_nb,
        }

        if not parallel:
            if out is None:
                out = np.zeros_like(x, dtype=dtype)
            configcore.matvec_numba(x, out, **kwargs)
            return out

        pool, world_size = get_pool_and_world_size(parallel)
        out_i = np.zeros_like(x, dtype=dtype, shape=(world_size, x.size))

        fs = [
            pool.submit(
                configcore.matvec_numba,
                x,
                out_i[i],
                world_rank=i,
                world_size=world_size,
                **kwargs,
            )
            for i in range(world_size)
        ]
        for f in fs:
            f.result()

        # sum the results from each thread
        return np.sum(out_i, axis=0, out=out)

    def aslinearoperator(
        self,
        sector=None,
        symmetry=None,
        dtype=None,
        parallel=False,
    ):
        """Get a `scipy.sparse.linalg.LinearOperator` for this operator.
        This is a lazy representation of the operator, which uses `matvec` to
        apply the operator to a vector. Less memory is required than
        constructing the full sparse matrix, but it is significantly slower.

        Note currently the operator is assumed to be hermitian.

        Parameters
        ----------
        sector : {None, str, int, ((int, int), (int, int))}, optional
            The sector of the Hilbert space. If None, the default sector is
            used.
        symmetry : {None, "Z2", "U1", "U1U1"}, optional
            The symmetry of the Hilbert space. If None, the default symmetry is
            used, or inferred from the supplied sector if possible.
        dtype : numpy.dtype, optional
            The data type of the matrix. If not provided, will be
            automatically determined based on the terms in the operator.
        parallel : bool or int, optional
            Whether to apply the operator in parallel (multi-threaded). If
            True, will use number of threads equal to the number of
            available CPU cores. If False, will use a single thread. If an
            integer is provided, it will be used as the number of threads to
            use. Uses `num_threads` more memory but is faster.

        Returns
        -------
        Alo : scipy.sparse.linalg.LinearOperator
            The linear operator representation of this operator.
        """
        from scipy.sparse.linalg import LinearOperator

        if dtype is None:
            dtype = self.get_dtype()

        d = self.hilbert_space.get_size(sector, symmetry)
        shape = (d, d)

        matvec = functools.partial(
            self.matvec,
            sector=sector,
            symmetry=symmetry,
            dtype=dtype,
            parallel=parallel,
        )

        return LinearOperator(
            shape=shape,
            matvec=matvec,
            rmatvec=matvec,
            dtype=dtype,
        )

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

        dtype = self.get_dtype(dtype)

        for term, coeff in self._get_terms_final().items():
            ops, sites = zip(*term)
            mats = (get_mat(op, dtype=dtype) for op in ops)
            hk = coeff * functools.reduce(np.kron, mats)
            if sites not in Hk:
                Hk[sites] = hk
            else:
                Hk[sites] = Hk[sites] + hk

        return Hk

    def build_local_ham(self, dtype=None):
        """Get a `LocalHamGen` object for this operator.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            The data type of the matrix. If not provided, will be
            automatically determined based on the terms in the operator.

        Returns
        -------
        H : LocalHamGen
            The local Hamiltonian representation of this operator.
        """
        from quimb.tensor.tensor_arbgeom_tebd import LocalHamGen

        terms = self.build_local_terms(dtype=dtype)

        H2 = {}
        H1 = {}
        for sites, hk in terms.items():
            if len(sites) == 2:
                H2[sites] = hk
            elif len(sites) == 1:
                H1[sites[0]] = hk
            else:
                raise NotImplementedError(
                    "Only supports 1- and 2-site terms for now."
                )

        if not H1:
            H1 = None
        if not H2:
            H2 = None

        return LocalHamGen(H2, H1)

    def build_state_machine_greedy(self, atol=1e-12):
        # XXX: also implement optimal method : https://arxiv.org/abs/2006.02056

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

        for t, (term, coeff) in enumerate(self._get_terms_final().items()):
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
                if abs(new_coeff - 1.0) > atol:
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

        dtype = self.get_dtype(dtype)

        if self.nsites == 1:
            # single site operator, just sum them
            Wt0 = self.build_dense(dtype=dtype)
            return qtn.MatrixProductOperator([Wt0], **mpo_opts)

        if method == "greedy":
            G = self.build_state_machine_greedy()
        else:
            raise ValueError(f"Unknown method {method}.")

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
        s = [f"{self.__class__.__name__}("]
        s.append(f"nsites={self.nsites}")
        s.append(f", nterms={self.nterms}")
        s.append(f", locality={self.locality}")

        if self._transform_jordan_wigner:
            s.append(", jordan_wigner=True")
        if self._transform_pauli_decompose:
            s.append(f", pauli_decompose={self._transform_pauli_decompose}")

        s.append(")")
        return "".join(s)

    def build_matrix_ikron(self, **ikron_opts):
        """Build either the dense or sparse matrix of this operator via
        explicit calls to `ikron`. This is a slower but useful alternative
        testing method.
        """
        from quimb import ikron

        A = None

        dims = [2] * self.nsites
        for ops, coeff in self._get_terms_final().items():
            if not ops:
                ops = [("I", 0)]

            mats = []
            inds = []
            for op, reg in ops:
                mats.append(get_mat(op))
                inds.append(reg)

            Aterm = coeff * ikron(mats, dims, inds, **ikron_opts)

            if A is None:
                A = Aterm
            else:
                A = A + Aterm

        return A
