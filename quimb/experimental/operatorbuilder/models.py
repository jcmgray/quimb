from .hilbertspace import parse_edges_to_unique
from .operatorbuilder import SparseOperatorBuilder


def heisenberg_from_edges(
    edges,
    j=1.0,
    b=0.0,
    hilbert_space=None,
):
    r"""Create a Heisenberg Hamiltonian on the graph defined by ``edges``.

    .. math::

        H =
        \sum_{\{i,j\}}^{|E|}
        \left(
        J_x S^x_i S^x_j + J_y S^y_i S^y_j + J_z S^z_i S^z_j
        \right)
        +
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
        The Hamiltonian as a SparseOperatorBuilder object.
    """
    try:
        jx, jy, jz = j
    except TypeError:
        jx, jy, jz = j, j, j

    try:
        bx, by, bz = b
    except TypeError:
        bx, by, bz = 0, 0, b

    sites, edges = parse_edges_to_unique(edges)

    H = SparseOperatorBuilder(hilbert_space=hilbert_space)

    for cooa, coob in edges:
        if jx == jy:
            # keep things real
            H += jx / 2, ("+", cooa), ("-", coob)
            H += jx / 2, ("-", cooa), ("+", coob)
        else:
            H += jx, ("sx", cooa), ("sx", coob)
            H += jy, ("sy", cooa), ("sy", coob)

        H += jz, ("sz", cooa), ("sz", coob)

    for site in sites:
        H += bx, ("sx", site)
        H += by, ("sy", site)
        H += bz, ("sz", site)

    return H


def fermi_hubbard_from_edges(edges, t=1.0, U=1.0, mu=0.0):
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
    t : float, optional
        The hopping amplitude. Default is 1.0.
    U : float, optional
        The on-site interaction strength. Default is 1.0.
    mu : float, optional
        The chemical potential. Default is 0.0.

    Returns
    -------
    H : SparseOperatorBuilder
        The Hamiltonian as a SparseOperatorBuilder object.
    """
    H = SparseOperatorBuilder()
    sites, edges = parse_edges_to_unique(edges)

    for cooa, coob in edges:
        # hopping
        for s in "↑↓":
            H += -t, ("+", (s, cooa)), ("-", (s, coob))
            H += -t, ("+", (s, coob)), ("-", (s, cooa))

    for coo in sites:
        # interaction
        H += U, ("n", ("↑", coo)), ("n", ("↓", coo))

        # chemical potential
        H += -mu, ("n", ("↑", coo))
        H += -mu, ("n", ("↓", coo))

    H.jordan_wigner_transform()
    return H


def fermi_hubbard_spinless_from_edges(edges, t=1.0, V=0.0, mu=0.0):
    r"""Create a spinless Fermi-Hubbard Hamiltonian on the graph defined by
    ``edges``. The Hamiltonian is given by:

    .. math::

        H =
        -t
        \sum_{\{i,j\}}^{|E|}
        \left(
        c_i^\dagger c_j + c_j^\dagger c_i
        \right)
        +
        V
        \sum_{\{i,j\}}^{|E|}
        n_i n_j
        -
        \mu
        \sum_{i}^{|V|}
        n_i

    where :math:`\{i,j\}` are the edges of the graph, and :math:`c_i` is the
    fermionic annihilation operator acting on site :math:`i`. The Jordan-Wigner
    transformation is used to implement fermionic statistics.

    Parameters
    ----------
    edges : Iterable[tuple[hashable, hashable]]
        The edges, as pairs of hashable 'sites', that define the graph.
        Multiple edges are allowed, but will be treated as a single edge.
    t : float, optional
        The hopping amplitude. Default is 1.0.
    V : float, optional
        The nearest neighbor interaction strength. Default is 0.0.
    mu : float, optional
        The chemical potential. Default is 0.0.

    Returns
    -------
    H : SparseOperatorBuilder
        The Hamiltonian as a SparseOperatorBuilder object.
    """
    H = SparseOperatorBuilder()
    sites, edges = parse_edges_to_unique(edges)

    for cooa, coob in edges:
        # hopping
        H += -t, ("+", cooa), ("-", coob)
        H += -t, ("+", coob), ("-", cooa)

        # nearest neighbor interaction
        H += V, ("n", cooa), ("n", coob)

    # chemical potential
    for coo in sites:
        H += -mu, ("n", coo)

    H.jordan_wigner_transform()
    return H
