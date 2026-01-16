"""Predefined hamiltonians for various models:

- Heisenberg model
- Fermi-Hubbard model
- Spinless Fermi-Hubbard model
- Random operators

"""

from .builder import SparseOperatorBuilder
from .hilbertspace import HilbertSpace, parse_edges_to_unique


def make_edge_factory(coeff):
    """Ensure `coeff` is a function that takes two sites and returns an edge
    coeff."""
    if isinstance(coeff, dict):

        def edge_factory(cooa, coob):
            try:
                return coeff[(cooa, coob)]
            except KeyError:
                return coeff[(coob, cooa)]

    elif callable(coeff):
        edge_factory = coeff

    else:

        def edge_factory(cooa, coob):
            # constant
            return coeff

    return edge_factory


def make_node_factory(coeff):
    """Ensure `coeff` is a function that takes a site and returns a node
    coeff."""
    if isinstance(coeff, dict):

        def node_factory(coo):
            return coeff[coo]

    elif callable(coeff):
        node_factory = coeff

    else:

        def node_factory(coo):
            # constant
            return coeff

    return node_factory


def heisenberg_from_edges(
    edges,
    j=1.0,
    b=0.0,
    order=None,
    sector=None,
    symmetry=None,
    hilbert_space=None,
    dtype=None,
):
    r"""Create a Heisenberg Hamiltonian on the graph defined by ``edges``.

    .. math::

        H =
        \sum_{\{i,j\}}^{|E|}
        \left(
        J_x S^x_i S^x_j + J_y S^y_i S^y_j + J_z S^z_i S^z_j
        \right)
        -
        \sum_{i}^{|V|}
        \left(
        B_x S^x_i + B_y S^y_i + B_z S^z_i
        \right)

    where :math:`\{i,j\}` are the edges of the graph, and :math:`S^x_i` is the
    spin-1/2 operator acting on site :math:`i` in the x-direction, etc. Note
    positive values of :math:`J` correspond to antiferromagnetic coupling here,
    and the magnetic field is in the z-direction by default.

    Parameters
    ----------
    edges : Iterable[tuple[hashable, hashable]]
        The edges, as pairs of hashable 'sites', that define the graph.
        Multiple edges are allowed, and will be treated as a single edge.
    j : float or tuple[float, float, float] or dict or callable, optional
        The Heisenberg exchange coupling constant(s). If a single float is
        given, it is used for all three terms. If a tuple of three floats is
        given, they are used for the xx, yy, and zz terms respectively. Note
        that positive values of ``j`` correspond to antiferromagnetic coupling.
        A dict or callable can be supplied to have edge-dependent couplings.
    b : float or tuple[float, float, float] or dict or callable, optional
        The magnetic field strength(s). If a single float is given, it is used
        taken as a z-field. If a tuple of three floats is given, they are used
        for the x, y, and z fields respectively. A dict or callable can be
        supplied to have site-dependent fields.
    order : callable or sequence of hashable objects, optional
        If provided, use this to order the sites. If a callable, it should be a
        sorting key. If a sequence, it should be a permutation of the sites,
        and ``key=order.index`` will be used.
    sector : {None, str, int, ((int, int), (int, int))}, optional
        The sector of the Hilbert space. If None, no sector is assumed.
    symmetry : {None, "Z2", "U1", "U1U1"}, optional
        The symmetry of the Hilbert space if any. If `None` and a `sector` is
        provided, the symmetry will be inferred from the sector if possible.
    hilbert_space : HilbertSpace, optional
        The Hilbert space to use. If not given, one will be constructed
        automatically from the edges. This overrides the ``order``,
        ``symmetry``, and ``sector`` parameters.
    dtype : {None, str, type}, optional
        The data type of the Hamiltonian. If None, a default dtype will be
        used, np.float64 for real and np.complex128 for complex.

    Returns
    -------
    H : SparseOperatorBuilder
        The Hamiltonian as a SparseOperatorBuilder object.
    """
    sites, edges = parse_edges_to_unique(edges)

    if hilbert_space is None:
        hilbert_space = HilbertSpace(
            sites=sites,
            order=order,
            sector=sector,
            symmetry=symmetry,
        )

    H = SparseOperatorBuilder(hilbert_space=hilbert_space, dtype=dtype)

    j_factory = make_edge_factory(j)
    b_factory = make_node_factory(b)

    for cooa, coob in edges:
        # get possibly edge-dependent couplings
        jab = j_factory(cooa, coob)
        try:
            jx, jy, jz = jab
        except TypeError:
            jx, jy, jz = jab, jab, jab

        if jx == jy:
            # keep things real
            H += jx / 2, ("+", cooa), ("-", coob)
            H += jx / 2, ("-", cooa), ("+", coob)
        else:
            H += jx, ("sx", cooa), ("sx", coob)
            H += jy, ("sy", cooa), ("sy", coob)

        H += jz, ("sz", cooa), ("sz", coob)

    for site in sites:
        # get possibly site-dependent field
        b = b_factory(site)
        try:
            bx, by, bz = b
        except TypeError:
            bx, by, bz = 0, 0, b
        H -= bx, ("sx", site)
        H -= by, ("sy", site)
        H -= bz, ("sz", site)

    return H


def fermi_hubbard_from_edges(
    edges,
    t=1.0,
    U=1.0,
    mu=0.0,
    order=None,
    sector=None,
    symmetry=None,
    hilbert_space=None,
    dtype=None,
    pauli_decompose=False,
):
    r"""Create a Fermi-Hubbard Hamiltonian on the graph defined by ``edges``.
    The Hamiltonian is given by:

    .. math::

        H =
        -t
        \sum_{\{i,j\}}^{|E|}
        \sum_{\sigma \in \uparrow, \downarrow}
        \left(
        c_{\sigma,i}^\dagger c_{\sigma,j} +
        c_{\sigma,j}^\dagger c_{\sigma,i}
        \right)
        +
        U
        \sum_{i}^{|V|}
        n_{\uparrow,i} n_{\downarrow,i}
        -
        \mu
        \sum_{i}^{|V|}
        \left(
        n_{\uparrow,i} + n_{\downarrow,i}
        \right)

    where :math:`\{i,j\}` are the edges of the graph, and :math:`c_{\sigma,i}`
    is the fermionic annihilation operator acting on site :math:`i` with spin
    :math:`\sigma`. The Jordan-Wigner transformation is used to implement
    fermionic statistics.

    Parameters
    ----------
    edges : Iterable[tuple[hashable, hashable]]
        The edges, as pairs of hashable 'sites', that define the graph.
        Multiple edges are allowed, but will be treated as a single edge.
    t : float or tuple[float, float] or dict or callable, optional
        The hopping amplitude. Default is 1.0. If a tuple it specifies the up
        and down spin hoppings respectively. A dict or callable can be supplied
        to have edge-dependent hoppings.
    U : float or dict or callable, optional
        The on-site interaction strength. Default is 1.0. A dict or callable
        can be supplied to have site-dependent interactions.
    mu : float or tuple[float, float] or dict or callable, optional
        The chemical potential. Default is 0.0. If a tuple it specifies the up
        and down spin chemical potentials respectively. A dict or callable can
        be supplied to have site-dependent chemical potentials.
    order : callable or sequence of hashable objects, optional
        If provided, use this to order the sites. If a callable, it should be a
        sorting key. If a sequence, it should be a permutation of the sites,
        and ``key=order.index`` will be used.
    sector : {None, str, int, ((int, int), (int, int))}, optional
        The sector of the Hilbert space. If None, no sector is assumed.
    symmetry : {None, "Z2", "U1", "U1U1"}, optional
        The symmetry of the Hilbert space if any. If `None` and a `sector` is
        provided, the symmetry will be inferred from the sector if possible.
    hilbert_space : HilbertSpace, optional
        The Hilbert space to use. If not given, one will be constructed
        automatically from the edges. This overrides the ``order``,
        ``symmetry``, and ``sector`` parameters.
    dtype : {None, str, type}, optional
        The data type of the Hamiltonian. If None, a default dtype will be
        used, np.float64 for real and np.complex128 for complex.
    pauli_decompose : bool, optional
        Whether to decompose the Hamiltonian into Pauli strings after
        Jordan-Wigner transforming. Default is False.

    Returns
    -------
    H : SparseOperatorBuilder
        The Hamiltonian as a SparseOperatorBuilder object.
    """
    sites, edges = parse_edges_to_unique(edges)

    if hilbert_space is None:
        hilbert_space = HilbertSpace(
            sites=[(s, coo) for s in "↑↓" for coo in sites],
            order=order,
            sector=sector,
            symmetry=symmetry,
        )

    H = SparseOperatorBuilder(
        hilbert_space=hilbert_space,
        dtype=dtype,
        jordan_wigner=True,
        pauli_decompose=pauli_decompose,
    )

    t_factory = make_edge_factory(t)
    U_factory = make_node_factory(U)
    mu_factory = make_node_factory(mu)

    for cooa, coob in edges:
        # hopping, possibly edge and spin-dependent
        t = t_factory(cooa, coob)
        try:
            t_u, t_d = t
        except TypeError:
            t_u, t_d = t, t
        H -= t_u, ("+", ("↑", cooa)), ("-", ("↑", coob))
        H -= t_u, ("+", ("↑", coob)), ("-", ("↑", cooa))
        H -= t_d, ("+", ("↓", cooa)), ("-", ("↓", coob))
        H -= t_d, ("+", ("↓", coob)), ("-", ("↓", cooa))

    for coo in sites:
        # interaction, possibly site-dependent
        U = U_factory(coo)
        H += U, ("n", ("↑", coo)), ("n", ("↓", coo))

        # chemical potential, possibly site and spin-dependent
        mu = mu_factory(coo)
        try:
            mu_u, mu_d = mu
        except TypeError:
            mu_u, mu_d = mu, mu
        H -= mu_u, ("n", ("↑", coo))
        H -= mu_d, ("n", ("↓", coo))

    return H


def fermi_hubbard_spinless_from_edges(
    edges,
    t=1.0,
    V=0.0,
    mu=0.0,
    delta=0.0,
    order=None,
    sector=None,
    symmetry=None,
    hilbert_space=None,
    dtype=None,
    pauli_decompose=False,
):
    r"""Create a spinless Fermi-Hubbard Hamiltonian on the graph defined by
    ``edges``. The Hamiltonian is given by:

    .. math::

        H =
        -t \sum_{\{i,j\}}^{|E|}
        \left( c_i^\dagger c_j + c_j^\dagger c_i \right)
        +
        V \sum_{\{i,j\}}^{|E|} n_i n_j
        -
        \mu \sum_{i}^{|V|} n_i
        +
        \Delta \sum_{\{i,j\}}^{|E|}
        \left( c_i^\dagger c_j^\dagger + c_j c_i \right)

    where :math:`\{i,j\}` are the edges of the graph, and :math:`c_i` is the
    fermionic annihilation operator acting on site :math:`i`. The Jordan-Wigner
    transformation is used to implement fermionic statistics.

    Parameters
    ----------
    edges : Iterable[tuple[hashable, hashable]]
        The edges, as pairs of hashable 'sites', that define the graph.
        Multiple edges are allowed, but will be treated as a single edge.
    t : float or dict or callable, optional
        The hopping amplitude. Default is 1.0. A dict or callable can be
        supplied to have edge-dependent hoppings.
    V : float or dict or callable, optional
        The nearest neighbor interaction strength. Default is 0.0. A dict or
        callable can be supplied to have edge-dependent interactions.
    mu : float or dict or callable, optional
        The chemical potential. Default is 0.0. A dict or callable can be
        supplied to have site-dependent chemical potentials.
    delta : float or dict or callable, optional
        The superconducting pairing strength. Default is 0.0. A dict or
        callable can be supplied to have edge-dependent pairings.
    order : callable or sequence of hashable objects, optional
        If provided, use this to order the sites. If a callable, it should be a
        sorting key. If a sequence, it should be a permutation of the sites,
        and ``key=order.index`` will be used.
    sector : {None, str, int, ((int, int), (int, int))}, optional
        The sector of the Hilbert space. If None, no sector is assumed.
    symmetry : {None, "Z2", "U1", "U1U1"}, optional
        The symmetry of the Hilbert space if any. If `None` and a `sector` is
        provided, the symmetry will be inferred from the sector if possible.
    hilbert_space : HilbertSpace, optional
        The Hilbert space to use. If not given, one will be constructed
        automatically from the edges. This overrides the ``order``,
        ``symmetry``, and ``sector`` parameters.
    dtype : {None, str, type}, optional
        The data type of the Hamiltonian. If None, a default dtype will be
        used, np.float64 for real and np.complex128 for complex.
    pauli_decompose : bool, optional
        Whether to decompose the Hamiltonian into Pauli strings after
        Jordan-Wigner transforming. Default is False.

    Returns
    -------
    H : SparseOperatorBuilder
        The Hamiltonian as a SparseOperatorBuilder object.
    """
    sites, edges = parse_edges_to_unique(edges)

    if hilbert_space is None:
        hilbert_space = HilbertSpace(
            sites=sites,
            order=order,
            sector=sector,
            symmetry=symmetry,
        )

    H = SparseOperatorBuilder(
        hilbert_space=hilbert_space,
        dtype=dtype,
        jordan_wigner=True,
        pauli_decompose=pauli_decompose,
    )

    t_factory = make_edge_factory(t)
    V_factory = make_edge_factory(V)
    mu_factory = make_node_factory(mu)
    delta_factory = make_edge_factory(delta)

    for cooa, coob in edges:
        # hopping, possibly edge-dependent
        t = t_factory(cooa, coob)
        H -= t, ("+", cooa), ("-", coob)
        H -= t, ("+", coob), ("-", cooa)

        # nearest neighbor interaction, possibly edge-dependent
        V = V_factory(cooa, coob)
        H += V, ("n", cooa), ("n", coob)

        # pairing, possibly edge-dependent
        delta = delta_factory(cooa, coob)
        H += delta, ("+", cooa), ("+", coob)
        H += delta, ("-", coob), ("-", cooa)

    # chemical potential, possibly site-dependent
    for coo in sites:
        mu = mu_factory(coo)
        H -= mu, ("n", coo)

    return H


def rand_operator(n, m, k, kmin=None, seed=None, ops="XYZ"):
    """Generate a random operator with n qubits and m terms.
    Each term is a sum of k operators acting on different qubits.
    The operators are chosen randomly from the set {X, Y, Z, +, -, n}.
    The coefficients are drawn from a normal distribution.

    Parameters
    ----------
    n : int
        The number of qubits.
    m : int
        The number of terms in the operator.
    k : int
        The number of operators in each term.
    kmin : int, optional
        The minimum number of operators in each term. If not given, kmin = k.
    seed : int, optional
        The random seed for reproducibility.
    ops : str, optional
        The set of operators to choose from.

    Returns
    -------
    SparseOperatorBuilder
        The random operator as a SparseOperatorBuilder object.
    """
    import numpy as np

    terms = []

    rng = np.random.default_rng(seed)
    allowed_ops = np.array(list(ops))

    if kmin is None:
        kmin = k
    if not (0 <= kmin <= k <= n):
        raise ValueError(
            "kmin must be positive and k must be between kmin and n"
        )

    for _ in range(m):
        coeff = rng.normal()
        ops = []

        if kmin == k:
            ki = k
        else:
            ki = rng.integers(kmin, k + 1)

        regs = rng.choice(np.arange(n), size=ki, replace=False)
        for reg in regs:
            op = rng.choice(allowed_ops)
            ops.append((str(op), int(reg)))
        terms.append((coeff, *ops))

    hilbert_space = HilbertSpace(sites=range(n))

    return SparseOperatorBuilder(terms=terms, hilbert_space=hilbert_space)
