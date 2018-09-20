"""Functions for generating quantum operators.
"""
from operator import add
import math
import functools
import itertools
import operator

from cytoolz import isiterable, concat, unique
import numpy as np
import scipy.sparse as sp
from scipy.special import comb

from ..core import (qarray, make_immutable, get_thread_pool,
                    par_reduce, isreal, qu, eye, kron, ikron)


# --------------------------------------------------------------------------- #
#                      gates and other simple operators                       #
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=16)
def spin_operator(label, S=1 / 2, **kwargs):
    """Generate a general spin-operator.

    Parameters
    ----------
    label : str
        The type of operator, can be one of six options:
            - ``{'x', 'X'}``, x-spin operator.
            - ``{'y', 'Y'}``, y-spin operator.
            - ``{'z', 'Z'}``, z-spin operator.
            - ``{'+', 'p'}``, Raising operator.
            - ``{'-', 'm'}``, Lowering operator.
            - ``{'i', 'I'}``, identity operator.
    S : float, optional
        The spin of particle to act on, default to spin-1/2.
    kwargs
        Passed to :func:`quimbify`.

    Returns
    -------
    S : immutable operator
        The spin operator.

    See Also
    --------
    pauli

    Examples
    --------
    >>> spin_operator('x')
    qarray([[0. +0.j, 0.5+0.j],
            [0.5+0.j, 0. +0.j]])

    >>> qu.spin_operator('+', S=1)
    qarray([[0.        +0.j, 1.41421356+0.j, 0.        +0.j],
            [0.        +0.j, 0.        +0.j, 1.41421356+0.j],
            [0.        +0.j, 0.        +0.j, 0.        +0.j]])

    >>> qu.spin_operator('Y', sparse=True)
    <2x2 sparse matrix of type '<class 'numpy.complex128'>'
        with 2 stored elements in Compressed Sparse Row format>

    """

    D = int(2 * S + 1)

    op = np.zeros((D, D), dtype=complex)
    ms = np.linspace(S, -S, D)

    label = label.lower()

    if label in {'x', 'y'}:
        for i in range(D - 1):
            c = 0.5 * (S * (S + 1) - (ms[i] * ms[i + 1]))**0.5
            op[i, i + 1] = -1.0j * c if (label == 'y') else c
            op[i + 1, i] = 1.0j * c if (label == 'y') else c

    elif label == 'z':
        for i in range(D):
            op[i, i] = ms[i]

    elif label in {'+', 'p', '-', 'm'}:
        for i in range(D - 1):
            c = (S * (S + 1) - (ms[i] * ms[i + 1]))**0.5
            if label in {'+', 'p'}:
                op[i, i + 1] = c
            else:
                op[i + 1, i] = c

    elif label in {'i', 'I'}:
        np.fill_diagonal(op, 1.0)

    else:
        raise ValueError("Label '{}'' not understood, should be one of ``['X',"
                         " 'Y', 'Z', '+', '-', 'I']``.".format(label))

    op = qu(np.real_if_close(op), **kwargs)

    make_immutable(op)
    return op


@functools.lru_cache(maxsize=8)
def pauli(xyz, dim=2, **kwargs):
    """Generates the pauli operators for dimension 2 or 3.

    Parameters
    ----------
    xyz : str
        Which spatial direction, upper or lower case from ``{'I', 'X', 'Y',
        'Z'}``.
    dim : int, optional
        Dimension of spin operator (e.g. 3 for spin-1), defaults to 2 for
        spin half.
    kwargs
        Passed to ``quimbify``.

    Returns
    -------
    P : immutable operator
        The pauli operator.

    See Also
    --------
    spin_operator
    """
    xyzmap = {0: 'i', 'i': 'i', 'I': 'i',
              1: 'x', 'x': 'x', 'X': 'x',
              2: 'y', 'y': 'y', 'Y': 'y',
              3: 'z', 'z': 'z', 'Z': 'z'}
    opmap = {('i', 2): lambda: eye(2, **kwargs),
             ('x', 2): lambda: qu([[0, 1],
                                   [1, 0]], **kwargs),
             ('y', 2): lambda: qu([[0, -1j],
                                   [1j, 0]], **kwargs),
             ('z', 2): lambda: qu([[1, 0],
                                   [0, -1]], **kwargs),
             ('i', 3): lambda: eye(3, **kwargs),
             ('x', 3): lambda: qu([[0, 1, 0],
                                   [1, 0, 1],
                                   [0, 1, 0]], **kwargs) / 2**.5,
             ('y', 3): lambda: qu([[0, -1j, 0],
                                   [1j, 0, -1j],
                                   [0, 1j, 0]], **kwargs) / 2**.5,
             ('z', 3): lambda: qu([[1, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, -1]], **kwargs)}
    op = opmap[(xyzmap[xyz], dim)]()

    # Operator is cached, so make sure it cannot be modified
    make_immutable(op)

    return op


@functools.lru_cache(8)
def hadamard(dtype=complex, sparse=False):
    """The Hadamard gate.
    """
    H = qu([[1., 1.],
            [1., -1.]], dtype=dtype, sparse=sparse) / 2**0.5
    make_immutable(H)
    return H


@functools.lru_cache(128)
def phase_gate(phi, dtype=complex, sparse=False):
    """The generalized qubit phase-gate, which adds phase ``phi`` to the
    ``|1>`` state.
    """
    Rp = qu([[1., 0.],
             [0., np.exp(1.0j * phi)]], dtype=dtype, sparse=sparse)
    make_immutable(Rp)
    return Rp


@functools.lru_cache(8)
def T_gate(dtype=complex, sparse=False):
    """The T-gate (pi/8 gate).
    """
    return phase_gate(math.pi / 4, dtype=dtype, sparse=sparse)


@functools.lru_cache(8)
def S_gate(dtype=complex, sparse=False):
    """The S-gate (phase gate).
    """
    return phase_gate(math.pi / 2, dtype=dtype, sparse=sparse)


@functools.lru_cache(128)
def rotation(phi, xyz='Z', dtype=complex, sparse=False):
    """The single qubit rotation gate.
    """
    R = math.cos(phi / 2) * pauli('I') - 1.0j * math.sin(phi / 2) * pauli(xyz)
    R = qu(R, dtype=dtype, sparse=sparse)
    make_immutable(R)
    return R


Rx = functools.partial(rotation, xyz='x')
Ry = functools.partial(rotation, xyz='y')
Rz = functools.partial(rotation, xyz='z')


@functools.lru_cache(maxsize=8)
def swap(dim=2, dtype=complex, **kwargs):
    """The SWAP operator acting on subsystems of dimension `dim`.
    """
    S = np.identity(dim**2, dtype=dtype)
    S = (S.reshape([dim, dim, dim, dim])
          .transpose([0, 3, 1, 2])
          .reshape([dim**2, dim**2]))
    S = qu(S, dtype=dtype, **kwargs)
    make_immutable(S)
    return S


@functools.lru_cache(maxsize=16)
def controlled(s, dtype=complex, sparse=False):
    """Construct a controlled pauli gate for two qubits.

    Parameters
    ----------
    s : str
        Which pauli to use, including 'not' aliased to 'x'.
    sparse : bool, optional
        Whether to construct a sparse operator.

    Returns
    -------
    C : immutable operator
        The controlled two-qubit gate operator.
    """
    # alias not and NOT to x
    s = {'NOT': 'x', 'not': 'x'}.get(s, s)
    kws = {'dtype': dtype, 'sparse': sparse}
    op = ((qu([1, 0], qtype='dop', **kws) & eye(2, **kws)) +
          (qu([0, 1], qtype='dop', **kws) & pauli(s, **kws)))
    make_immutable(op)
    return op


@functools.lru_cache(8)
def CNOT(dtype=complex, sparse=False):
    """The controlled-not gate.
    """
    return controlled('not', dtype=dtype, sparse=sparse)


@functools.lru_cache(8)
def cX(dtype=complex, sparse=False):
    """The controlled-X gate.
    """
    return controlled('not', dtype=dtype, sparse=sparse)


@functools.lru_cache(8)
def cY(dtype=complex, sparse=False):
    """The controlled-Y gate.
    """
    return controlled('Y', dtype=dtype, sparse=sparse)


@functools.lru_cache(8)
def cZ(dtype=complex, sparse=False):
    """The controlled-Z gate.
    """
    return controlled('Z', dtype=dtype, sparse=sparse)


# --------------------------------------------------------------------------- #
#                                Hamiltonians                                 #
# --------------------------------------------------------------------------- #


def hamiltonian_builder(fn):
    """Wrap a function to perform some generic postprocessing and take the
    kwargs ``stype`` and ``sparse``. This assumes the core function always
    builds the hamiltonian in sparse form. The wrapper then:

    1. Checks if the operator is real
    2. Converts the operator to dense or the correct sparse form
    3. Makes the operator immutable so it can be safely cached
    """

    @functools.wraps(fn)
    def ham_fn(*args, stype='csr', sparse=False, **kwargs):
        H = fn(*args, **kwargs)

        if isreal(H):
            H = H.real

        if not sparse:
            H = qarray(H.A)
        elif H.format != stype:
            H = H.asformat(stype)

        make_immutable(H)

        return H

    return ham_fn


@functools.lru_cache(maxsize=8)
@hamiltonian_builder
def ham_heis(n, j=1.0, b=0.0, cyclic=True,
             parallel=None, nthreads=None, ownership=None):
    """Constructs the nearest neighbour 1d heisenberg spin-1/2 hamiltonian.

    Parameters
    ----------
    n : int
        Number of spins.
    j : float or tuple(float, float, float), optional
        Coupling constant(s), with convention that positive =
        antiferromagnetic. Can supply scalar for isotropic coupling or
        vector ``(jx, jy, jz)``.
    b : float or tuple(float, float, float), optional
        Magnetic field, defaults to z-direction only if tuple not given.
    cyclic : bool, optional
        Whether to couple the first and last spins.
    sparse : bool, optional
        Whether to return the hamiltonian in sparse form.
    stype : str, optional
        What format of sparse operator to return if ``sparse``.
    parallel : bool, optional
        Whether to build the operator in parallel. By default will do this
        for n > 16.
    nthreads : int optional
        How mny threads to use in parallel to build the operator.
    ownership : (int, int), optional
        If given, which range of rows to generate.
    kwargs
        Supplied to :func:`~quimb.core.quimbify`.

    Returns
    -------
    H : immutable operator
        The Hamiltonian.
    """
    dims = (2,) * n
    try:
        jx, jy, jz = j
    except TypeError:
        jx = jy = jz = j

    try:
        bx, by, bz = b
    except TypeError:
        bz = b
        bx = by = 0.0

    parallel = (n > 16) if parallel is None else parallel

    op_kws = {'sparse': True, 'stype': 'coo'}
    ikron_kws = {'sparse': True, 'stype': 'coo',
                 'coo_build': True, 'ownership': ownership}

    # The basic operator (interaction and single b-field) that can be repeated.
    two_site_term = sum(
        j * kron(spin_operator(s, **op_kws), spin_operator(s, **op_kws))
        for j, s in zip((jx, jy, jz), 'xyz')
    ) - sum(
        b * kron(spin_operator(s, **op_kws), eye(2, **op_kws))
        for b, s in zip((bx, by, bz), 'xyz') if b != 0.0
    )

    single_site_b = sum(-b * spin_operator(s, **op_kws)
                        for b, s in zip((bx, by, bz), 'xyz') if b != 0.0)

    def gen_term(i):
        # special case: the last b term needs to be added manually
        if i == -1:
            return ikron(single_site_b, dims, n - 1, **ikron_kws)

        # special case: the interaction between first and last spins if cyclic
        if i == n - 1:
            return sum(
                j * ikron(spin_operator(s, **op_kws),
                          dims, [0, n - 1], **ikron_kws)
                for j, s in zip((jx, jy, jz), 'xyz') if j != 0.0)

        # General term, on-site b-field plus interaction with next site
        return ikron(two_site_term, dims, [i, i + 1], **ikron_kws)

    terms_needed = range(0 if single_site_b is 0 else -1,
                         n if cyclic else n - 1)

    if parallel:
        pool = get_thread_pool(nthreads)
        ham = par_reduce(add, pool.map(gen_term, terms_needed))
    else:
        ham = sum(map(gen_term, terms_needed))

    return ham


def ham_ising(n, jz=1.0, bx=1.0, **ham_opts):
    """Generate the quantum transverse field ising model hamiltonian. This is a
    simple alias for :func:`~quimb.gen.operators.ham_heis` with Z-interactions
    and an X-field.
    """
    return ham_heis(n, j=(0, 0, jz), b=(bx, 0, 0), **ham_opts)


def ham_XY(n, jxy, bz, **ham_opts):
    """Generate the quantum transverse field XY model hamiltonian. This is a
    simple alias for :func:`~quimb.gen.operators.ham_heis` with
    X- and Y-interactions and a Z-field.
    """
    return ham_heis(n, j=(jxy, jxy, 0), b=(0, 0, bz), **ham_opts)


def ham_XXZ(n, delta, jxy=1.0, **ham_opts):
    """Generate the XXZ-model hamiltonian. This is a
    simple alias for :func:`~quimb.gen.operators.ham_heis` with matched
    X- and Y-interactions and ``delta`` Z coupling.
    """
    return ham_heis(n, j=(jxy, jxy, delta), b=0, **ham_opts)


@functools.lru_cache(maxsize=8)
@hamiltonian_builder
def ham_j1j2(n, j1=1.0, j2=0.5, bz=0.0, cyclic=True, ownership=None):
    """Generate the j1-j2 hamiltonian, i.e. next nearest neighbour
    interactions.

    Parameters
    ----------
    n : int
        Number of spins.
    j1 : float, optional
        Nearest neighbour coupling strength.
    j2 : float, optional
        Next nearest neighbour coupling strength.
    bz : float, optional
        B-field strength in z-direction.
    cyclic : bool, optional
        Cyclic boundary conditions.
    sparse : bool, optional
        Return hamiltonian as sparse-csr operator.
    ownership : (int, int), optional
        If given, which range of rows to generate.
    kwargs
        Supplied to :func:`~quimb.core.quimbify`.

    Returns
    -------
    H : immutable operator
        The Hamiltonian.
    """
    dims = (2,) * n

    op_kws = {'sparse': True, 'stype': 'coo'}
    ikron_kws = {'sparse': True, 'stype': 'coo',
                 'coo_build': True, 'ownership': ownership}

    sxyz = [spin_operator(i, **op_kws) for i in 'xyz']

    coosj1 = np.array([(i, i + 1) for i in range(n)])
    coosj2 = np.array([(i, i + 2) for i in range(n)])
    if cyclic:
        coosj1, coosj2 = coosj1 % n, coosj2 % n
    else:
        coosj1 = coosj1[np.all(coosj1 < n, axis=1)]
        coosj2 = coosj2[np.all(coosj2 < n, axis=1)]

    def j1_terms():
        for coo in coosj1:
            if abs(coo[1] - coo[0]) == 1:  # can sum then tensor (faster)
                yield ikron(sum(op & op for op in sxyz),
                            dims, coo, **ikron_kws)
            else:  # tensor then sum (slower)
                yield sum(ikron(op, dims, coo, **ikron_kws) for op in sxyz)

    def j2_terms():
        for coo in coosj2:
            if abs(coo[1] - coo[0]) == 2:  # can add then tensor (faster)
                yield ikron(sum(op & eye(2, **op_kws) & op for op in sxyz),
                            dims, coo, **ikron_kws)
            else:
                yield sum(ikron(op, dims, coo, **ikron_kws) for op in sxyz)

    ham = j1 * sum(j1_terms()) + j2 * sum(j2_terms())

    if bz != 0:
        gen_bz = (ikron([sxyz[2]], dims, i, **ikron_kws) for i in range(n))
        ham += bz * sum(gen_bz)

    return ham


def _gen_mbl_random_factors(n, dh, dh_dim, dh_dist, seed=None, beta=None):
    # sort out a vector of noise strengths -> e.g. (0, 0, 1) for z-noise only
    if isinstance(dh, (tuple, list)):
        dhds = dh
    else:
        dh_dim = {0: '', 1: 'z', 2: 'xy', 3: 'xyz'}.get(dh_dim, dh_dim)
        dhds = tuple((dh if d in dh_dim else 0) for d in 'xyz')

    if seed is not None:
        np.random.seed(seed)

    # sort out the noise distribution
    if dh_dist in {'g', 'gauss', 'gaussian', 'normal'}:
        rs = np.random.randn(3, n)

    elif dh_dist in {'s', 'flat', 'square', 'uniform', 'box'}:
        rs = 2.0 * np.random.rand(3, n) - 1.0

    elif dh_dist in {'qp', 'quasiperiodic', 'qr', 'quasirandom'}:
        if dh_dim is not 'z':
            raise ValueError("dh_dim should be 1 or 'z' for dh_dist='qp'.")

        if beta is None:
            beta = (5**0.5 - 1) / 2

        # the random phase
        delta = 2 * np.pi * np.random.rand()

        # make sure get 3 by n different strengths
        inds = np.broadcast_to(range(n), (3, n))

        rs = np.cos(2 * np.pi * beta * inds + delta)

    return dhds, rs


@hamiltonian_builder
def ham_mbl(n, dh, j=1.0, bz=0.0, cyclic=True,
            seed=None, dh_dist="s", dh_dim=1, beta=None, ownership=None):
    """ Constructs a heisenberg hamiltonian with isotropic coupling and
    random fields acting on each spin - the many-body localized (MBL)
    spin hamiltonian.

    Parameters
    ----------
    n : int
        Number of spins.
    dh : float or (float, float, float)
        Strength of random fields (stdev of gaussian distribution), can be
        scalar (isotropic noise) or 3-vector for (x, y, z) directions.
    j : float or (float, float, float), optional
        Coupling strength, can be scalar (isotropic) or 3-vector.
    bz : float, optional
        Global magnetic field (in z-direction).
    cyclic : bool, optional
        Whether to use periodic boundary conditions.
    seed : int, optional
        Number to seed random number generator with.
    dh_dist : {'g', 's', 'qr'}, optional
        Type of random distribution for the noise:

        - "s": square, with bounds ``(-dh, dh)``
        - "g": gaussian, with standard deviation ``dh``
        - "qp": quasi periodic, with amplitude ``dh`` and
          'wavenumber' ``beta`` so that the field at site ``i`` is
          ``dh * cos(2 * pi * beta * i + delta)`` with ``delta`` a random
          offset between ``(0, 2 * pi)``, possibly seeded by ``seed``.

    dh_dim : {1, 2, 3} or str, optional
        The number of dimensions the noise acts in, or string
        specifier like ``'yz'``.
    beta : float, optional
        The wave number if ``dh_dist='qr'``, defaults to the golden
        ratio``(5**0.5 - 1) / 2``.
    sparse : bool, optional
        Whether to construct the hamiltonian in sparse form.
    stype : {'csr', 'csc', 'coo'}, optional
        The sparse format.
    ownership : (int, int), optional
        If given, which range of rows to generate.
    kwargs
        Supplied to :func:`~quimb.core.quimbify`.

    Returns
    -------
    H : operator
        The MBL hamiltonian for spin-1/2.

    See Also
    --------
    MPO_ham_mbl
    """
    dhds, rs = _gen_mbl_random_factors(n, dh, dh_dim, dh_dist, seed, beta)

    # the base hamiltonian ('csr' is most efficient format to add with)
    ham = ham_heis(n=n, j=j, b=bz, cyclic=cyclic,
                   sparse=True, stype='csr', ownership=ownership)

    op_kws = {'sparse': True, 'stype': 'coo'}
    ikron_kws = {'sparse': True, 'stype': 'coo',
                 'coo_build': True, 'ownership': ownership}

    def dh_terms():
        for i in range(n):
            # dhd - the total strength in direction x, y, or z
            # r - the random strength in direction x, y, or z for site i
            hdh = sum(dhd * r * spin_operator(s, **op_kws)
                      for dhd, r, s in zip(dhds, rs[:, i], 'xyz'))
            yield ikron(hdh, (2,) * n, i, **ikron_kws)

    ham = ham + sum(dh_terms())

    return ham


@hamiltonian_builder
def ham_heis_2D(n, m, j=1.0, bz=0.0, cyclic=False,
                parallel=False, ownership=None):
    """Construct the 2D spin-1/2 heisenberg model hamiltonian.

    Parameters
    ----------
    n : int
        The number of rows.
    m : int
        The number of columns.
    j : float or (float, float, float), optional
        The coupling strength(s). Isotropic if scalar else if
        vector ``(Jx, Jy, Jz) = j``.
    bz : float, optional
        The z direction magnetic field.
    cyclic : bool, optional
        Whether to use periodic boundary conditions.
    sparse : bool, optional
        Whether to construct the hamiltonian in sparse form.
    stype : {'csr', 'csc', 'coo'}, optional
        The sparse format.
    parallel : bool, optional
        Construct the hamiltonian in parallel. Faster but might use more
        memory.
    ownership : (int, int), optional
        If given, which range of rows to generate.
    kwargs
        Supplied to :func:`~quimb.core.quimbify`.

    Returns
    -------
    H : operator
        The hamiltonian.
    """

    # parse interaction strengths
    try:
        jx, jy, jz = j
    except (TypeError, ValueError):
        jx = jy = jz = j

    dims = [[2] * m] * n  # shape (n, m)

    sites = tuple(itertools.product(range(n), range(m)))

    # generate neighbouring pair coordinates
    def gen_pairs():
        for i, j in sites:
            above, right = (i + 1) % n, (j + 1) % m
            # ignore wraparound coordinates if not cyclic
            if cyclic or above != 0:
                yield ((i, j), (above, j))
            if cyclic or right != 0:
                yield ((i, j), (i, right))

    # build the hamiltonian in sparse 'coo' format always for efficiency
    op_kws = {'sparse': True, 'stype': 'coo'}
    ikron_kws = {'sparse': True, 'stype': 'coo',
                 'coo_build': True, 'ownership': ownership}

    # generate all pairs of coordinates and directions
    pairs_ss = tuple(itertools.product(gen_pairs(), 'xyz'))

    # generate XX, YY and ZZ interaction from
    #     e.g. arg ([(3, 4), (3, 5)], 'z')
    def interactions(pair_s):
        pair, s = pair_s
        Sxyz = spin_operator(s, **op_kws)
        J = {'x': jx, 'y': jy, 'z': jz}[s]
        return ikron(J * Sxyz, dims, inds=pair, **ikron_kws)

    # generate Z field
    def fields(site):
        Sz = spin_operator('z', **op_kws)
        return ikron(bz * Sz, dims, inds=[site], **ikron_kws)

    if not parallel:
        # combine all terms
        all_terms = itertools.chain(
            map(interactions, pairs_ss),
            map(fields, sites) if bz != 0.0 else ())
        H = sum(all_terms)
    else:
        pool = get_thread_pool()
        all_terms = itertools.chain(
            pool.map(interactions, pairs_ss),
            pool.map(fields, sites) if bz != 0.0 else ())
        H = par_reduce(add, all_terms)

    return H


def uniq_perms(xs):
    """Generate all the unique permutations of sequence ``xs``.

    Examples
    --------
    >>> list(uniq_perms('0011'))
    [('0', '0', '1', '1'),
     ('0', '1', '0', '1'),
     ('0', '1', '1', '0'),
     ('1', '0', '0', '1'),
     ('1', '0', '1', '0'),
     ('1', '1', '0', '0')]
    """
    if len(xs) == 1:
        yield (xs[0],)
    else:
        uniq_xs = unique(xs)
        for first_x in uniq_xs:
            rem_xs = list(xs)
            rem_xs.remove(first_x)
            for sub_perm in uniq_perms(rem_xs):
                yield (first_x,) + sub_perm


@functools.lru_cache(maxsize=8)
def zspin_projector(n, sz=0, stype="csr", dtype=float):
    """Construct the projector onto spin-z subpspaces.

    Parameters
    ----------
    n : int
        Total size of spin system.
    sz : float or sequence of floats
        Spin-z value(s) subspace(s) to find projector for.
    stype : str
        Sparse format of the output operator.
    dtype : {float, complex}, optional
        The data type of the operator to generate.

    Returns
    -------
    prj : immutable sparse operator, shape (2**n, D)
        The (non-square) projector onto the specified subspace(s). The subspace
        size ``D`` is given by ``n choose (n / 2 + s)`` for each ``s``
        specified in ``sz``.

    Examples
    --------
    >>> zspin_projector(n=2, sz=0).A
    array([[0., 0.],
           [1., 0.],
           [0., 1.],
           [0., 0.]]

    Project a 9-spin Heisenberg-Hamiltonian into its spin-1/2 subspace:

    >>> H = ham_heis(9, sparse=True)
    >>> H.shape
    (512, 512)

    >>> P = zspin_projector(n=9, sz=1 / 2)
    >>> H0 = P.T @ H @ P
    >>> H0.shape
    (126, 126)
    """
    if not isiterable(sz):
        sz = (sz,)

    p = 0
    all_perms = []

    for s in sz:
        # Number of 'up' spins
        k = n / 2 + s
        if not k.is_integer():
            raise ValueError("{} is not a valid spin half subspace for "
                             "{} spins.".format(s, n))
        k = int(round(k))
        # Size of subspace
        p += comb(n, k, exact=True)
        # Find all computational basis states with correct number of 0s and 1s
        base_perm = '0' * (n - k) + '1' * k
        all_perms += [uniq_perms(base_perm)]

    # Coordinates
    cis = tuple(range(p))  # arbitrary basis
    cjs = tuple(int("".join(perm), 2) for perm in concat(all_perms))

    # Construct matrix which projects only on to these basis states
    prj = sp.coo_matrix((np.ones(p, dtype=dtype), (cjs, cis)),
                        shape=(2**n, p), dtype=dtype)
    prj = qu(prj, stype=stype, dtype=dtype)
    make_immutable(prj)
    return prj


@functools.lru_cache(8)
def create(n=2, **qu_opts):
    """The creation operator acting on an n-level system.
    """
    data = np.zeros((n, n))

    for i in range(n):
        data[i, i - 1] = i ** 0.5

    ap = qu(data, **qu_opts)
    make_immutable(ap)
    return ap


@functools.lru_cache(8)
def destroy(n=2, **qu_opts):
    """The annihilation operator acting on an n-level system.
    """
    am = create(n, **qu_opts).T.copy()
    make_immutable(am)
    return am


@functools.lru_cache(8)
def num(n, **qu_opts):
    """The number operator acting on an n-level system.
    """
    ap, am = create(n, **qu_opts), destroy(n, **qu_opts)
    an = qu(ap @ am, **qu_opts)
    make_immutable(an)
    return an


@functools.lru_cache(maxsize=8)
@hamiltonian_builder
def ham_hubbard_hardcore(n, t=0.5, V=1., mu=1., cyclic=True,
                         parallel=None, ownership=None):
    """Generate the spinless fermion hopping hamiltonian.

    Parameters
    ----------
    n : int
        The number of sites.
    t : float, optional
        The hopping energy.
    V : float, optional
        The interaction energy.
    mu : float, optional
        The chemical potential - defaults to half-filling.
    cyclic : bool, optional
        Whether to use periodic boundary conditions.
    parallel : bool, optional
        Construct the hamiltonian in parallel. Faster but might use more
        memory.
    ownership : (int, int), optional
        If given, which range of rows to generate.
    kwargs
        Supplied to :func:`~quimb.core.quimbify`.

    Returns
    -------
    H : operator
        The hamiltonian.
    """

    op_kws = {'sparse': True, 'stype': 'coo'}
    ikron_kws = {'sparse': True, 'stype': 'csr',
                 'coo_build': True, 'ownership': ownership}

    cdag, c, cnum = (f(2, **op_kws) for f in (create, destroy, num))
    neighbor_term = t * ((cdag & c) + (c & cdag)) + V * (cnum & cnum)

    dims = [2] * n

    def terms():
        # interacting terms
        for i, j in [(i, i + 1) for i in range(n - 1)]:
            yield ikron(neighbor_term, dims, (i, j), **ikron_kws)

        if cyclic:
            # can't sum terms and kron later since identity in middle
            yield ikron([t * cdag, c], dims, (0, n - 1), **ikron_kws)
            yield ikron([t * c, cdag], dims, (0, n - 1), **ikron_kws)
            yield ikron([V * cnum, cnum], dims, (0, n - 1), **ikron_kws)

        # single site terms
        for i in range(n):
            yield ikron(-mu * cnum, dims, i, **ikron_kws)

    if parallel is None:
        parallel = (n >= 14)

    if parallel:
        return par_reduce(operator.add, terms())

    return functools.reduce(operator.add, terms())
