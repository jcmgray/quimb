"""Functions for generating random quantum objects and states.
"""
import math
import random
from functools import wraps
from numbers import Integral
from itertools import count, chain
from concurrent.futures import wait

import numpy as np
import numpy.random
import scipy.sparse as sp

from ..core import (qarray, dag, dot, rdmul, complex_array, get_thread_pool,
                    _NUM_THREAD_WORKERS, qu, ptr, kron, nmlz, prod,
                    vectorize, pvectorize)


class _RGenHandler:
    """Private object that handles pool of random number generators for
    parallel number generation - seeding them & changing the underlying bit
    generators.
    """

    def __init__(self, initial_seed=None, initial_bitgen=None):
        self.rgs = []
        self.set_seed(initial_seed)
        self.set_bitgen(initial_bitgen)

    def set_bitgen(self, bitgen):
        """Set the core underlying bit-generator.

        Parameters
        ----------
        bitgen : {None, str}
            Which bit generator to use, either from numpy or `randomgen` -
            https://bashtage.github.io/randomgen/bit_generators/index.html.
        """

        if bitgen is None:
            self.gen_fn = numpy.random.default_rng

        else:
            try:
                bg = getattr(numpy.random, bitgen)
            except AttributeError:
                import randomgen
                bg = getattr(randomgen, bitgen)

            def gen(s):
                return numpy.random.Generator(bg(s))

            self.gen_fn = gen

        # delete any old rgens
        self.rgs = []

    def set_seed(self, seed=None):
        """Set the seed for the bit generators.

        Parameters
        ----------
        seed : {None, int}, optional
            Seed supplied to `numpy.random.SeedSequence`. None will randomly
            the generators (default).
        """
        seq = numpy.random.SeedSequence(seed)

        # compute seeds in batches of 4 for perf
        self.seeds = iter(chain.from_iterable(seq.spawn(4) for _ in count()))

        # delete any old rgens
        self.rgs = []

    def get_rgens(self, num_threads):
        """Get a list of the :class:`numpy.random.Generator` instances, having
        made sure there are at least ``num_threads``.

        Parameters
        ----------
        num_threads : int
            The number of generators to return.

        Returns
        -------
        list[numpy.random.Generator]
        """
        num_gens = len(self.rgs)

        if num_gens < num_threads:
            self.rgs.extend(self.gen_fn(next(self.seeds))
                            for _ in range(num_gens, num_threads))

        return self.rgs[:num_threads]


_RG_HANDLER = _RGenHandler()


def seed_rand(seed):
    """See the random number generators, by instantiating a new set of bit
    generators with a 'seed sequence'.
    """
    global _RG_HANDLER
    return _RG_HANDLER.set_seed(seed)


def set_rand_bitgen(bitgen):
    """Set the core bit generator type to use, from either ``numpy`` or
    ``randomgen``.

    Parameters
    ----------
    bitgen : {'PCG64', 'SFC64', 'MT19937', 'Philox', str}
        Which bit generator to use.
    """
    global _RG_HANDLER
    return _RG_HANDLER.set_bitgen(bitgen)


def _get_rgens(num_threads):
    global _RG_HANDLER
    return _RG_HANDLER.get_rgens(num_threads)


def randn(shape=(), dtype=float, scale=1.0, loc=0.0,
          num_threads=None, seed=None, dist='normal'):
    """Fast multithreaded generation of random normally distributed data
    using ``randomgen``.

    Parameters
    ----------
    shape : tuple[int]
        The shape of the output random array.
    dtype : {'complex128', 'float64', 'complex64' 'float32'}, optional
        The data-type of the output array.
    scale : float, optional
        The width of the distribution (standard deviation if
        ``dist='normal'``).
    loc : float, optional
        The location of the distribution (lower limit if
        ``dist='uniform'``).
    num_threads : int, optional
        How many threads to use. If ``None``, decide automatically.
    dist : {'normal', 'uniform', 'exp'}, optional
        Type of random number to generate.
    """
    if seed is not None:
        seed_rand(seed)

    if isinstance(shape, Integral):
        d = shape
        shape = (shape,)
    else:
        d = prod(shape)

    if num_threads is None:
        # only multi-thread for big ``d``
        if d <= 32768:
            num_threads = 1
        else:
            num_threads = _NUM_THREAD_WORKERS

    rgs = _get_rgens(num_threads)

    gen_method = {
        'uniform': 'random',
        'normal': 'standard_normal',
        'exp': 'standard_exponential',
    }.get(dist, dist)

    # sequential generation
    if num_threads <= 1:

        def create(d, dtype):
            out = np.empty(d, dtype)
            getattr(rgs[0], gen_method)(out=out, dtype=dtype)
            return out

    # threaded generation
    else:
        pool = get_thread_pool()
        S = math.ceil(d / num_threads)

        def _fill(gen, out, dtype, first, last):
            getattr(gen, gen_method)(out=out[first:last], dtype=dtype)

        def create(d, dtype):
            out = np.empty(d, dtype)
            # submit thread work
            fs = [
                pool.submit(_fill, gen, out, dtype, i * S, (i + 1) * S)
                for i, gen in enumerate(rgs)
            ]
            wait(fs)
            return out

    if np.issubdtype(dtype, np.floating):
        out = create(d, dtype)

    elif np.issubdtype(dtype, np.complexfloating):
        # need to sum two real arrays if generating complex numbers
        if np.issubdtype(dtype, np.complex64):
            sub_dtype = np.float32
        else:
            sub_dtype = np.float64

        out = complex_array(create(d, sub_dtype), create(d, sub_dtype))

    else:
        raise ValueError(f"dtype {dtype} not understood.")

    if out.dtype != dtype:
        out = out.astype(dtype)

    if scale != 1.0:
        out *= scale
    if loc != 0.0:
        out += loc

    return out.reshape(shape)


@wraps(randn)
def rand(*args, **kwargs):
    kwargs.setdefault('dist', 'uniform')
    return randn(*args, **kwargs)


def random_seed_fn(fn):
    """Modify ``fn`` to take a ``seed`` argument (so as to seed the random
    generators once-only at beginning of function not every ``randn`` call).
    """

    @wraps(fn)
    def wrapped_fn(*args, seed=None, **kwargs):
        if seed is not None:
            seed_rand(seed)
        return fn(*args, **kwargs)

    return wrapped_fn


def _randint(*args, **kwargs):
    return _get_rgens(1)[0].integers(*args, **kwargs)


def _choice(*args, **kwargs):
    return _get_rgens(1)[0].choice(*args, **kwargs)


choice = random_seed_fn(_choice)


@random_seed_fn
def rand_rademacher(shape, scale=1, dtype=float):
    """
    """
    if np.issubdtype(dtype, np.floating):
        entries = np.array([1.0, -1.0]) * scale
        need2convert = dtype not in (float, np.float_)

    elif np.issubdtype(dtype, np.complexfloating):
        entries = np.array([1.0, -1.0, 1.0j, -1.0j]) * scale
        need2convert = dtype not in (complex, np.complex_)

    else:
        raise TypeError(f"dtype {dtype} not understood - should be float or "
                        "complex.")

    x = _choice(entries, shape)
    if need2convert:
        x = x.astype(dtype)

    return x


def _phase_to_complex_base(x):
    return 1j * math.sin(x) + math.cos(x)


_phase_sigs = ['complex64(float32)', 'complex128(float64)']
_phase_to_complex_seq = vectorize(_phase_sigs)(_phase_to_complex_base)
"""Turn array of phases into unit circle complex numbers - sequential.
"""
_phase_to_complex_par = pvectorize(_phase_sigs)(_phase_to_complex_base)
"""Turn array of phases into unit circle complex numbers - parallel.
"""


def phase_to_complex(x):
    if x.size >= 512:
        return _phase_to_complex_par(x)
    # XXX: this is not as fast as numexpr - investigate?
    return _phase_to_complex_seq(x)


@random_seed_fn
def rand_phase(shape, scale=1, dtype=complex):
    """Generate random complex numbers distributed on the unit sphere.
    """
    if not np.issubdtype(dtype, np.complexfloating):
        raise ValueError(f"dtype must be complex, got '{dtype}'.")

    if np.issubdtype(dtype, np.complex64):
        sub_dtype = np.float32
    else:
        sub_dtype = np.float64

    phi = randn(shape, dtype=sub_dtype, scale=2 * math.pi, dist='uniform')
    z = phase_to_complex(phi)
    if scale != 1:
        z *= scale

    return z


def rand_matrix(d, scaled=True, sparse=False, stype='csr',
                density=None, dtype=complex, seed=None):
    """Generate a random matrix of order `d` with normally distributed
    entries. If `scaled` is `True`, then in the limit of large `d` the
    eigenvalues will be distributed on the unit complex disk.

    Parameters
    ----------
    d : int
        Matrix dimension.
    scaled : bool, optional
        Whether to scale the matrices values such that its spectrum
        approximately lies on the unit disk (for dense matrices).
    sparse : bool, optional
        Whether to produce a sparse matrix.
    stype : {'csr', 'csc', 'coo', ...}, optional
        The type of sparse matrix if ``sparse=True``.
    density : float, optional
        Target density of non-zero elements for the sparse matrix. By default
        aims for about 10 entries per row.
    dtype : {complex, float}, optional
        The data type of the matrix elements.

    Returns
    -------
    mat : qarray or sparse matrix
        Random matrix.
    """
    if np.issubdtype(dtype, np.floating):
        iscomplex = False
    elif np.issubdtype(dtype, np.complexfloating):
        iscomplex = True
    else:
        raise TypeError(f"dtype {dtype} not understood - should be "
                        "float or complex.")

    # handle seed manually since standard python random.seed might be called
    if seed is not None:
        seed_rand(seed)

    if sparse:
        # Aim for 10 non-zero values per row, but betwen 1 and d/2
        density = min(10, d / 2) / d if density is None else density
        density = min(max(d**-2, density, ), 1.0)
        nnz = round(density * d * d)

        if density > 0.1:
            # take special care to avoid duplicates
            if seed is not None:
                random.seed(seed)
            ijs = random.sample(range(0, d**2), k=nnz)
        else:
            ijs = _randint(0, d * d, size=nnz)

        # want to sample nnz unique (d, d) pairs without building list
        i, j = np.divmod(ijs, d)

        data = randn(nnz, dtype=dtype)
        mat = sp.coo_matrix((data, (i, j)), shape=(d, d)).asformat(stype)
    else:
        density = 1.0
        mat = qarray(randn((d, d), dtype=dtype))

    if scaled:
        mat /= ((2 if iscomplex else 1) * d * density)**0.5

    return mat


@random_seed_fn
def rand_herm(d, sparse=False, density=None, dtype=complex):
    """Generate a random hermitian operator of order `d` with normally
    distributed entries. In the limit of large `d` the spectrum will be a
    semi-circular distribution between [-1, 1].

    See Also
    --------
    rand_matrix, rand_pos, rand_rho, rand_uni
    """
    if sparse:
        density = 10 / d if density is None else density
        density = min(max(density, d**-2), 1 - d**-2)
        density /= 2  # to account of herm construction

    herm = rand_matrix(d, scaled=True, sparse=sparse,
                       density=density, dtype=dtype)

    if sparse:
        herm.data /= (2**1.5)
    else:
        herm /= (2**1.5)

    herm += dag(herm)

    return herm


@random_seed_fn
def rand_pos(d, sparse=False, density=None, dtype=complex):
    """Generate a random positive operator of size `d`, with normally
    distributed entries. In the limit of large `d` the spectrum will lie
    between [0, 1].

    See Also
    --------
    rand_matrix, rand_herm, rand_rho, rand_uni
    """
    if sparse:
        density = 10 / d if density is None else density
        density = min(max(density, d**-2), 1 - d**-2)
        density = 0.5 * (density / d)**0.5  # to account for pos construction

    pos = rand_matrix(d, scaled=True, sparse=sparse,
                      density=density, dtype=dtype)

    return dot(pos, dag(pos))


@random_seed_fn
def rand_rho(d, sparse=False, density=None, dtype=complex):
    """Generate a random positive operator of size `d` with normally
    distributed entries and unit trace.

    See Also
    --------
    rand_matrix, rand_herm, rand_pos, rand_uni
    """
    return nmlz(rand_pos(d, sparse=sparse, density=density, dtype=dtype))


@random_seed_fn
def rand_uni(d, dtype=complex):
    """Generate a random unitary operator of size `d`, distributed according to
    the Haar measure.

    See Also
    --------
    rand_matrix, rand_herm, rand_pos, rand_rho
    """
    q, r = np.linalg.qr(rand_matrix(d, dtype=dtype))
    r = np.diagonal(r)
    r = r / np.abs(r)  # read-only so not inplace
    return rdmul(q, r)


@random_seed_fn
def rand_ket(d, sparse=False, stype='csr', density=0.01, dtype=complex):
    """Generates a ket of length `d` with normally distributed entries.
    """
    if sparse:
        ket = sp.random(d, 1, format=stype, density=density)
        ket.data = randn((ket.nnz,), dtype=dtype)
    else:
        ket = qarray(randn((d, 1), dtype=dtype))
    return nmlz(ket)


@random_seed_fn
def rand_haar_state(d, dtype=complex):
    """Generate a random state of dimension `d` according to the Haar
    distribution.
    """
    u = rand_uni(d, dtype=dtype)
    return u[:, [0]]


@random_seed_fn
def gen_rand_haar_states(d, reps, dtype=complex):
    """Generate many random Haar states, recycling a random unitary operator
    by using all of its columns (not a good idea?).
    """
    for rep in range(reps):
        cyc = rep % d
        if cyc == 0:
            u = rand_uni(d, dtype=dtype)
        yield u[:, [cyc]]


@random_seed_fn
def rand_mix(d, tr_d_min=None, tr_d_max=None, mode='rand', dtype=complex):
    """Constructs a random mixed state by tracing out a random ket
    where the composite system varies in size between 2 and d. This produces
    a spread of states including more purity but has no real meaning.
    """
    if tr_d_min is None:
        tr_d_min = 2
    if tr_d_max is None:
        tr_d_max = d

    m = _randint(tr_d_min, tr_d_max)
    if mode == 'rand':
        psi = rand_ket(d * m, dtype=dtype)
    elif mode == 'haar':
        psi = rand_haar_state(d * m, dtype=dtype)

    return ptr(psi, [d, m], 0)


@random_seed_fn
def rand_product_state(n, qtype=None, dtype=complex):
    """Generates a ket of `n` many random pure qubits.
    """
    def gen_rand_pure_qubits(n):
        for _ in range(n):
            u, = rand(1)
            v, = rand(1)
            phi = 2 * np.pi * u
            theta = np.arccos(2 * v - 1)
            yield qu([[np.cos(theta / 2.0)],
                      [np.sin(theta / 2.0) * np.exp(1.0j * phi)]],
                     qtype=qtype, dtype=dtype)
    return kron(*gen_rand_pure_qubits(n))


@random_seed_fn
def rand_matrix_product_state(n, bond_dim, phys_dim=2, dtype=complex,
                              cyclic=False, trans_invar=False):
    """Generate a random matrix product state (in dense form, see
    :func:`~quimb.tensor.MPS_rand_state` for tensor network form).

    Parameters
    ----------
    n : int
        Number of sites.
    bond_dim : int
        Dimension of the bond (virtual) indices.
    phys_dim : int, optional
        Physical dimension of each local site, defaults to 2 (qubits).
    cyclic : bool (optional)
        Whether to impose cyclic boundary conditions on the entanglement
        structure.
    trans_invar : bool (optional)
        Whether to generate a translationally invariant state,
        requires cyclic=True.

    Returns
    -------
    ket : qarray
        The random state, with shape (phys_dim**n, 1)

    """
    from quimb.tensor import MPS_rand_state

    mps = MPS_rand_state(n, bond_dim, phys_dim=phys_dim, dtype=dtype,
                         cyclic=cyclic, trans_invar=trans_invar)
    return mps.to_dense()


rand_mps = rand_matrix_product_state


@random_seed_fn
def rand_seperable(dims, num_mix=10, dtype=complex):
    """Generate a random, mixed, seperable state. E.g rand_seperable([2, 2])
    for a mixed two qubit state with no entanglement.

    Parameters
    ----------
        dims : tuple of int
            The local dimensions across which to be seperable.
        num_mix : int, optional
            How many individual product states to sum together, each with
            random weight.

    Returns
    -------
        qarray
            Mixed seperable state.
    """

    def gen_single_sites():
        for dim in dims:
            yield rand_rho(dim, dtype=dtype)

    weights = rand(num_mix)

    def gen_single_states():
        for w in weights:
            yield w * kron(*gen_single_sites())

    return sum(gen_single_states()) / np.sum(weights)


@random_seed_fn
def rand_iso(n, m, dtype=complex):
    """Generate a random isometry of shape ``(n, m)``.
    """
    data = randn((n, m), dtype=dtype)

    q, _ = np.linalg.qr(data if n >= m else data.T)
    q = q.astype(dtype)

    return q if (n >= m) else q.T


@random_seed_fn
def rand_mera(n, invariant=False, dtype=complex):
    """Generate a random mera state of ``n`` qubits, which must be a power
    of 2. This uses ``quimb.tensor``.
    """
    import quimb.tensor as qt

    if invariant:
        constructor = qt.MERA.rand_invar
    else:
        constructor = qt.MERA.rand

    return constructor(n, dtype=dtype).to_dense()
