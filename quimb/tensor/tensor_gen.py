"""Generate specific tensor network states and operators.
"""
import random
import numpy as np

from ..accel import make_immutable
from ..linalg.base_linalg import norm_fro_dense
from ..gen.operators import spin_operator, eye
from .tensor_core import Tensor
from .tensor_1d import MatrixProductState, MatrixProductOperator


def randn(shape, dtype=float):
    """Generate normally distributed random array of certain shape and dtype.
    """
    # real datatypes
    if np.issubdtype(dtype, np.floating):
        x = np.random.randn(*shape)

        # convert type if not the default
        if dtype not in (float, np.float_):
            x = x.astype(dtype)

    # complex datatypes
    elif np.issubdtype(dtype, np.complexfloating):
        x = np.random.randn(*shape) + 1.0j * np.random.randn(*shape)

        # convert type if not the default
        if dtype not in (complex, np.complex_):
            x = x.astype(dtype)

    else:
        raise TypeError("dtype {} not understood - should be float or complex."
                        "".format(dtype))

    return x


def rand_tensor(shape, inds, tags=None, dtype=float):
    """Generate a random (complex) tensor with specified shape and inds.
    """
    data = randn(shape, dtype=dtype)
    return Tensor(data=data, inds=inds, tags=tags)


# --------------------------------------------------------------------------- #
#                                    MPSs                                     #
# --------------------------------------------------------------------------- #

def MPS_rand_state(n, bond_dim, phys_dim=2, normalize=True, cyclic=False,
                   dtype=float, **mps_opts):
    """Generate a random matrix product state.

    Parameters
    ----------
    n : int
        The number of sites.
    bond_dim : int
        The bond dimension.
    phys_dim : int, optional
        The physical (site) dimensions, defaults to 2.
    normalize : bool, optional
        Whether to normalize the state.
    cyclic : bool, optional
        Generate a MPS with periodic boundary conditions or not, default is
        open boundary conditions.
    dtype : {float, complex} or numpy dtype, optional
        Data type of the tensor network.
    mps_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductState`.
    """
    cyc_dim = (bond_dim,) if cyclic else ()

    def gen_shapes():
        yield (*cyc_dim, bond_dim, phys_dim)
        for _ in range(n - 2):
            yield (bond_dim, bond_dim, phys_dim)
        yield (bond_dim, *cyc_dim, phys_dim)

    def gen_data(shape):
        return randn(shape, dtype=dtype)

    def scale(x):
        return x / norm_fro_dense(x)**(1 / (x.ndim - 1))

    arrays = map(scale, map(gen_data, gen_shapes()))

    rmps = MatrixProductState(arrays, **mps_opts)

    if normalize:
        rmps /= (rmps.H @ rmps)**0.5

    return rmps


def MPS_product_state(arrays, **mps_opts):
    """Generate a product state in MatrixProductState form, i,e,
    with bond dimension 1, from single site vectors described by ``arrays``.
    """
    def gen_array_shapes():
        yield (1, -1)
        for _ in range(len(arrays) - 2):
            yield (1, 1, -1)
        yield (1, -1)

    mps_arrays = (np.asarray(array).reshape(*shape)
                  for array, shape in zip(arrays, gen_array_shapes()))

    return MatrixProductState(mps_arrays, shape='lrp', **mps_opts)


def MPS_computational_state(binary_str, dtype=float, **mps_opts):
    """A computational basis state in Matrix Product State form.

    Parameters
    ----------
    binary_str : str
        String specifying the state, e.g. '00101010111'
    mps_opts
        Supplied to MatrixProductState constructor.
    """
    array_map = {
        '0': np.array([1., 0.], dtype=dtype),
        '1': np.array([0., 1.], dtype=dtype),
    }

    def gen_arrays():
        for s in binary_str:
            yield array_map[s]

    return MPS_product_state(tuple(gen_arrays()), **mps_opts)


def MPS_neel_state(n, down_first=False, dtype=float, **mps_opts):
    """Generate the neel state in Matrix Product State form.

    Parameters
    ----------
    n : int
        The number of spins.
    down_first : bool, optional
        Whether to start with '1' (down) or '0' (up) first.
    mps_opts
        Supplied to MatrixProductState constructor.
    """
    binary_str = "01" * (n // 2) + (n % 2 == 1) * "0"
    if down_first:
        binary_str = "1" + binary_str[:-1]
    return MPS_computational_state(binary_str, dtype=dtype, **mps_opts)


def MPS_rand_computational_state(n, seed=None, dtype=float, **mps_opts):
    """Generate a random computation basis state, like '01101001010'.

    Parameters
    ----------
    n : int
        The number of qubits.
    seed : int, optional
        The seed to use.
    dtype : {float, complex} or numpy dtype, optional
        Data type of the tensor network.
    mps_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductState`.
    """
    if seed is not None:
        random.seed(seed)

    cstr = "".join(random.choice(('0', '1')) for _ in range(n))
    return MPS_computational_state(cstr, dtype=dtype, **mps_opts)


def MPS_zero_state(n, bond_dim=1, phys_dim=2, cyclic=False,
                   dtype=float, **mps_opts):
    """The all-zeros MPS state, of given bond-dimension.

    Parameters
    ----------
    n : int
        The number of sites.
    bond_dim : int, optional
        The bond dimension, defaults to 1.
    phys_dim : int, optional
        The physical (site) dimensions, defaults to 2.
    cyclic : bool, optional
        Generate a MPS with periodic boundary conditions or not, default is
        open boundary conditions.
    dtype : {float, complex} or numpy dtype, optional
        Data type of the tensor network.
    mps_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductState`.
    """
    cyc_dim = (bond_dim,) if cyclic else ()

    def gen_arrays():
        yield np.zeros((*cyc_dim, bond_dim, phys_dim), dtype=dtype)
        for _ in range(n - 2):
            yield np.zeros((bond_dim, bond_dim, phys_dim), dtype=dtype)
        yield np.zeros((bond_dim, *cyc_dim, phys_dim), dtype=dtype)

    return MatrixProductState(gen_arrays(), **mps_opts)


# --------------------------------------------------------------------------- #
#                                    MPOs                                     #
# --------------------------------------------------------------------------- #

def MPO_identity(n, phys_dim=2, dtype=float, cyclic=False, **mpo_opts):
    """Generate an identity MPO of size ``n``.

    Parameters
    ----------
    n : int
        The number of sites.
    phys_dim : int, optional
        The physical (site) dimensions, defaults to 2.
    dtype : {float, complex} or numpy dtype, optional
        Data type of the tensor network.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    mpo_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.
    """
    II = np.identity(phys_dim, dtype=dtype)
    cyc_dim = (1,) if cyclic else ()

    def gen_arrays():
        yield II.reshape(*cyc_dim, 1, phys_dim, phys_dim)
        for _ in range(n - 2):
            yield II.reshape(1, 1, phys_dim, phys_dim)
        yield II.reshape(1, *cyc_dim, phys_dim, phys_dim)

    return MatrixProductOperator(gen_arrays(), **mpo_opts)


def MPO_identity_like(mpo, **mpo_opts):
    """Return an identity matrix operator with the same physical index and
    inds/tags as ``mpo``.
    """
    return MPO_identity(n=mpo.nsites, phys_dim=mpo.phys_dim(), dtype=mpo.dtype,
                        site_tag_id=mpo.site_tag_id, cyclic=mpo.cyclic,
                        upper_ind_id=mpo.upper_ind_id,
                        lower_ind_id=mpo.lower_ind_id, **mpo_opts)


def MPO_zeros(n, phys_dim=2, dtype=float, cyclic=False, **mpo_opts):
    """Generate a zeros MPO of size ``n``.

    Parameters
    ----------
    n : int
        The number of sites.
    phys_dim : int, optional
        The physical (site) dimensions, defaults to 2.
    dtype : {float, complex} or numpy dtype, optional
        Data type of the tensor network.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    mpo_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.
    """
    cyc_dim = (1,) if cyclic else ()

    def gen_arrays():
        yield np.zeros((*cyc_dim, 1, phys_dim, phys_dim), dtype=dtype)
        for _ in range(n - 2):
            yield np.zeros((1, 1, phys_dim, phys_dim), dtype=dtype)
        yield np.zeros((1, *cyc_dim, phys_dim, phys_dim), dtype=dtype)

    return MatrixProductOperator(gen_arrays(), **mpo_opts)


def MPO_zeros_like(mpo, **mpo_opts):
    """Return a zeros matrix operator with the same physical index and
    inds/tags as ``mpo``.
    """
    return MPO_zeros(n=mpo.nsites, phys_dim=mpo.phys_dim(),
                     dtype=mpo.dtype, site_tag_id=mpo.site_tag_id,
                     upper_ind_id=mpo.upper_ind_id, cyclic=mpo.cyclic,
                     lower_ind_id=mpo.lower_ind_id, **mpo_opts)


def MPO_rand(n, bond_dim, phys_dim=2, normalize=True, cyclic=False,
             herm=False, dtype=float, **mpo_opts):
    """Generate a random matrix product state.

    Parameters
    ----------
    n : int
        The number of sites.
    bond_dim : int
        The bond dimension.
    phys_dim : int, optional
        The physical (site) dimensions, defaults to 2.
    normalize : bool, optional
        Whether to normalize the operator such that ``trace(A.H @ A) == 1``.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    dtype : {float, complex} or numpy dtype, optional
        Data type of the tensor network.
    herm : bool, optional
        Whether to make the matrix hermitian (or symmetric if real) or not.
    mpo_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.
    """
    cyc_shp = (bond_dim,) if cyclic else ()

    shapes = [(*cyc_shp, bond_dim, phys_dim, phys_dim),
              *((bond_dim, bond_dim, phys_dim, phys_dim),) * (n - 2),
              (bond_dim, *cyc_shp, phys_dim, phys_dim)]

    def gen_data(shape):
        data = randn(shape, dtype=dtype)
        if not herm:
            return data

        trans = (0, 2, 1) if len(shape) == 3 else (0, 1, 3, 2)
        return data + data.transpose(*trans).conj()

    arrays = map(lambda x: x / norm_fro_dense(x)**(1 / (x.ndim - 1)),
                 map(gen_data, shapes))

    rmpo = MatrixProductOperator(arrays, **mpo_opts)

    if normalize:
        rmpo /= (rmpo.H @ rmpo)**0.5

    return rmpo


def MPO_rand_herm(n, bond_dim, phys_dim=2, normalize=True,
                  dtype=float, **mpo_opts):
    """Generate a random hermitian matrix product operator.
    See :class:`~quimb.tensor.tensor_gen.MPO_rand`.
    """
    return MPO_rand(n, bond_dim, phys_dim=phys_dim, normalize=normalize,
                    dtype=dtype, herm=True, **mpo_opts)


# ---------------------------- MPO hamiltonians ----------------------------- #

def spin_ham_mpo_tensor(one_site_terms, two_site_terms, S=1 / 2,
                        which=None, cyclic=False):
    """Generate tensor(s) for a spin hamiltonian MPO.

    Parameters
    ----------
    one_site_terms : sequence of (scalar, operator)
        The terms that act on a single site, each ``operator`` can be a string
        suitable to be sent to :func:`spin_operator` or an actual 2d-array.
    two_site_terms : sequence of (scalar, operator operator)
        The terms that act on two neighbouring sites, each ``operator`` can be
        a string suitable to be sent to :func:`spin_operator` or an actual
        2d-array.
    S : fraction, optional
        What size spin to use, defaults to spin-1/2.
    which : {None, 'L', 'R', 'A'}, optional
        If ``None``, generate the middle tensor, if 'L' a left-end tensor, if
        'R' a right-end tensor and if 'A' all three.
    cyclic : bool, optional
        Whether to use periodic boundary conditions - default False.

    Returns
    -------
    numpy.ndarray{, numpy.ndarray, numpy.ndarray}
    """
    # local dimension
    D = int(2 * S + 1)
    # bond dimension
    B = len(two_site_terms) + 2

    H = np.zeros((B, B, D, D), dtype=complex)

    # add one-body terms
    for factor, s in one_site_terms:
        if isinstance(s, str):
            s = spin_operator(s, S=S)
        H[B - 1, 0, :, :] += factor * s

    # add two-body terms
    for i, (factor, s1, s2) in enumerate(two_site_terms):
        if isinstance(s1, str):
            s1 = spin_operator(s1, S=S)
        if isinstance(s2, str):
            s2 = spin_operator(s2, S=S)
        H[1 + i, 0, :, :] = s1
        H[-1, 1 + i, :, :] = factor * s2

    H[0, 0, :, :] = eye(D)
    H[B - 1, B - 1, :, :] = eye(D)

    if np.allclose(H.imag, np.zeros_like(H)):
        H = H.real

    make_immutable(H)

    if which in {None, 'M'}:
        return H

    if cyclic:
        # need special conditions for first MPO matrix
        HL = np.zeros_like(H)
        HL[0, :, :, :] = H[-1, :, :, :]
        HL[1:-1, -1, :, :] = H[1:-1, 0, :, :]
        HR = H
    else:
        HL = H[-1, :, :, :]
        HR = H[:, 0, :, :]

    if which == 'L':
        return HL
    elif which == 'R':
        return HR
    elif which == 'A':
        return HL, H, HR


class MPOSpinHam:
    """Class for easily building translationally invariant spin hamiltonians in
    MPO form. Currently limited to nearest neighbour interactions (and single
    site terms).

    Parameters
    ----------
    S : float, optional
        The type of spin, defaults to 1/2.
    cyclic : bool, optional
        Whether to use periodic boundary conditions - default is False.

    Example
    -------
    >>> builder = MPOSpinHam(S=3 / 2)
    >>> builder.add_term(-0.3, 'Z')
    >>> builder.add_term(0.5, '+', '-')
    >>> builder.add_term(0.5, '-', '+')
    >>> builder.add_term(1.0, 'Z', 'Z')
    >>> mpo_ham = builder.build(100)
    >>> mpo_ham
    <MatrixProductOperator(tensors=100, structure='I{}', nsites=100)>
    """

    def __init__(self, S=1 / 2, cyclic=False):
        self.S = S
        self.one_site_terms = []
        self.two_site_terms = []
        self.cyclic = cyclic

    def add_term(self, factor, *operators):
        """Add another term to the expression to be built.

        Parameters
        ----------
        factor : scalar
            Scalar factor to multiply this term by.
        *operators : str or array
            The operators to use. Can specify one or two for single or two site
            terms respectively. Can use strings, which are supplied to
            ``spin_operator``, or actual arrays as long as they have the
            correct dimension.
        """
        if len(operators) == 1:
            self.one_site_terms.append((factor, *operators))
        elif len(operators) == 2:
            self.two_site_terms.append((factor, *operators))
        else:
            raise NotImplementedError("3-body+ terms are not supported yet.")

    def build(self, n, upper_ind_id='k{}', lower_ind_id='b{}',
              site_tag_id='I{}', tags=None, bond_name=""):
        """Build an instance of this MPO of size ``n``. See also
        ``MatrixProductOperator``.
        """
        HL, H, HR = spin_ham_mpo_tensor(
            self.one_site_terms, self.two_site_terms,
            S=self.S, which='A', cyclic=self.cyclic)

        arrays = (HL, *[H] * (n - 2), HR)

        return MatrixProductOperator(arrays=arrays, bond_name=bond_name,
                                     upper_ind_id=upper_ind_id,
                                     lower_ind_id=lower_ind_id,
                                     site_tag_id=site_tag_id, tags=tags)


def MPO_ham_ising(n, j=1.0, bx=0.0, *, S=1 / 2, cyclic=False, **mpo_opts):
    """Ising Hamiltonian in matrix product operator form.

    Parameters
    ----------
    n : int
        The number of sites.
    j : float, optional
        The ZZ interaction strength.
    bx : float, optional
        The X-magnetic field strength.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    mpo_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.

    Returns
    -------
    MatrixProductOperator
    """
    H = MPOSpinHam(S=1 / 2, cyclic=cyclic)
    H.add_term(j, 'Z', 'Z')
    H.add_term(-bx, 'X')

    return H.build(n, **mpo_opts)


def MPO_ham_XY(n, j=1.0, bz=0.0, *, S=1 / 2, cyclic=False, **mpo_opts):
    """XY-Hamiltonian in matrix product operator form.

    Parameters
    ----------
    n : int
        The number of sites.
    j : float or (float, float), optional
        The XX and YY interaction strength.
    bz : float, optional
        The Z-magnetic field strength.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    mpo_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.

    Returns
    -------
    MatrixProductOperator
    """
    try:
        jx, jy = j
    except (TypeError, ValueError):
        jx = jy = j

    H = MPOSpinHam(S=S, cyclic=cyclic)
    if jx == jy:
        # easy way to enforce realness
        H.add_term(jx / 2, '+', '-')
        H.add_term(jx / 2, '-', '+')
    else:
        H.add_term(jx, 'X', 'X')
        H.add_term(jy, 'Y', 'Y')
    H.add_term(-bz, 'Z')

    return H.build(n, **mpo_opts)


def MPO_ham_heis(n, j=1.0, bz=0.0, *, S=1 / 2, cyclic=False, **mpo_opts):
    """Heisenberg Hamiltonian in matrix product operator form.

    Parameters
    ----------
    n : int
        The number of sites.
    j : float or (float, float, float), optional
        The XX, YY and ZZ interaction strength.
    bz : float, optional
        The Z-magnetic field strength.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Generate a MPO with periodic boundary conditions or not, default is
        open boundary conditions.
    mpo_opts
        Supplied to :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`.

    Returns
    -------
    MatrixProductOperator
    """
    try:
        jx, jy, jz = j
    except (TypeError, ValueError):
        jx = jy = jz = j

    H = MPOSpinHam(S=S, cyclic=cyclic)
    if jx == jy:
        # easy way to enforce realness
        H.add_term(jx / 2, '+', '-')
        H.add_term(jx / 2, '-', '+')
    else:
        H.add_term(jx, 'X', 'X')
        H.add_term(jy, 'Y', 'Y')
    H.add_term(jz, 'Z', 'Z')
    H.add_term(-bz, 'Z')

    return H.build(n, **mpo_opts)


def MPO_ham_mbl(n, dh, j=1.0, run=None, S=1 / 2, *, cyclic=False,
                dh_dist='s', dh_dim=1, beta=None, **mpo_opts):
    """The many-body-localized spin hamiltonian.

    Parameters
    ----------
    n : int
        Number of spins.
    dh : float
        Random noise strength.
    j : float, or (float, float, float), optional
        Interaction strength(s) e.g. 1 or (1., 1., 0.5).
    run : int, optional
        Random number to seed the noise with.
    S : {1/2, 1, 3/2, ...}, optional
        The underlying spin of the system, defaults to 1/2.
    cyclic : bool, optional
        Whether to use periodic boundary conditions - default is False.
    dh_dist : {'s', 'g', 'qp'}, optional
        Whether to use sqaure, guassian or quasiperiodic noise.
    beta : float, optional
        Frequency of the quasirandom noise, only if ``dh_dist='qr'``.
    mpo_opts
        Supplied to :class:`MatrixProductOperator`.

    Returns
    -------
    MatrixProductOperator
    """
    # Parse the interaction term and strengths
    try:
        jx, jy, jz = j
    except (TypeError, ValueError):
        jx = jy = jz = j

    if jy == jx:
        # can specify specifically real MPO-terms
        interaction = [(jx / 2, '+', '+'), (jx / 2, '-', '-'), (jz, 'Z', 'Z')]
    else:
        interaction = [(jx, 'X', 'X'), (jy, 'Y', 'Y'), (jz, 'Z', 'Z')]

    # sort out a vector of noise strengths -> e.g. (0, 0, 1) for z-noise only
    if isinstance(dh, (tuple, list)):
        dhds = dh
    else:
        dh_dim = {0: '', 1: 'z', 2: 'xy', 3: 'xyz'}.get(dh_dim, dh_dim)
        dhds = tuple((dh if d in dh_dim else 0) for d in 'xyz')

    if run is not None:
        np.random.seed(run)

    # sort out the noise distribution
    if dh_dist in {'g', 'gauss', 'gaussian', 'normal'}:
        rs = np.random.randn(3, n)

    elif dh_dist in {'s', 'flat', 'square', 'uniform', 'box'}:
        rs = 2.0 * np.random.rand(3, n) - 1.0

    elif dh_dist in {'qp', 'quasiperiodic'}:
        if dh_dim is not 'z':
            raise ValueError("dh_dim should be 1 or 'z' for dh_dist='qp'.")

        if beta is None:
            beta = (5**0.5 - 1) / 2

        # the random phase
        delta = 2 * np.pi * np.random.rand()

        # make sure get 3 by n different strengths
        inds = np.broadcast_to(range(n), (3, 10))

        rs = np.cos(2 * np.pi * beta * inds + delta)

    # generate noise, potentially in all directions, each with own strength
    def single_site_terms():
        for i in range(n):
            yield [(dh * r, s)
                   for dh, r, s in zip(dhds, rs[:, i], 'XYZ')
                   if dh != 0]

    dh_terms = iter(single_site_terms())

    def gen_arrays():
        yield spin_ham_mpo_tensor(next(dh_terms), interaction, S=S,
                                  which='L' if not cyclic else None)
        for _ in range(n - 2):
            yield spin_ham_mpo_tensor(next(dh_terms), interaction, S=S)
        yield spin_ham_mpo_tensor(next(dh_terms), interaction, S=S,
                                  which='R' if not cyclic else None)

    return MatrixProductOperator(gen_arrays(), **mpo_opts)
