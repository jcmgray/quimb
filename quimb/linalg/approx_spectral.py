"""Use lanczos tri-diagonalization to approximate the spectrum of any operator
which has an efficient representation of its linear action on a vector.
"""
import functools
from math import sqrt, log2, exp, inf, nan
import random
import warnings

import numpy as np
import scipy.linalg as scla
from scipy.optimize import curve_fit

from ..core import ptr
from ..accel import prod, vdot
from ..utils import int2tup
from ..linalg.mpi_launcher import get_mpi_pool
from ..tensor.tensor_core import Tensor
from ..tensor.tensor_1d import MatrixProductOperator
from ..tensor.tensor_approx_spectral import (
    construct_lanczos_tridiag_MPO,
    PTPTLazyMPS,
    construct_lanczos_tridiag_PTPTLazyMPS,
)


# --------------------------------------------------------------------------- #
#                  'Lazy' representation tensor contractions                  #
# --------------------------------------------------------------------------- #


def lazy_ptr_linop(psi_ab, dims, sysa):
    r"""A linear operator representing action of partially tracing a bipartite
    state, then multiplying another 'unipartite' state::

          ( | )
        +-------+
        | psi_a |   ______
        +_______+  /      \
           a|      |b     |
        +-------------+   |
        |  psi_ab.H   |   |
        +_____________+   |
                          |
        +-------------+   |
        |   psi_ab    |   |
        +_____________+   |
           a|      |b     |
            |      \______/

    Parameters
    ----------
    psi_ab : ket
        State to partially trace and dot with another ket, with
        size ``prod(dims)``.
    dims : sequence of int, optional
        The sub dimensions of ``psi_ab``.
    sysa : int or sequence of int, optional
        Index(es) of the 'a' subsystem(s) to keep.
    """
    sysa = int2tup(sysa)

    Kab = Tensor(np.asarray(psi_ab).reshape(dims),
                 inds=[('kA{}' if i in sysa else 'xB{}').format(i)
                       for i in range(len(dims))])

    Bab = Tensor(Kab.data.conjugate(),
                 inds=[('bA{}' if i in sysa else 'xB{}').format(i)
                       for i in range(len(dims))])

    return (Kab & Bab).aslinearoperator(
        ['kA{}'.format(i) for i in sysa],
        ['bA{}'.format(i) for i in sysa],
    )


def lazy_ptr_ppt_linop(psi_abc, dims, sysa, sysb):
    r"""A linear operator representing action of partially tracing a tripartite
    state, partially transposing the remaining bipartite state, then
    multiplying another bipartite state::

             ( | )
        +--------------+
        |   psi_ab     |
        +______________+  _____
         a|  ____   b|   /     \
          | /   a\   |   |c    |
          | | +-------------+  |
          | | |  psi_abc.H  |  |
          \ / +-------------+  |
           X                   |
          / \ +-------------+  |
          | | |   psi_abc   |  |
          | | +-------------+  |
          | \____/a  |b  |c    |
         a|          |   \_____/

    Parameters
    ----------
    psi_abc : ket
        State to partially trace, partially tranpose, then dot with another
        ket, with size ``prod(dims)``.
        ``prod(dims[sysa] + dims[sysb])``.
    dims : sequence of int
        The sub dimensions of ``psi_abc``.
    sysa : int or sequence of int, optional
        Index(es) of the 'a' subsystem(s) to keep, with respect to all
        the dimensions, ``dims``, (i.e. pre-partial trace).
    sysa : int or sequence of int, optional
        Index(es) of the 'b' subsystem(s) to keep, with respect to all
        the dimensions, ``dims``, (i.e. pre-partial trace).
    """
    sysa, sysb = int2tup(sysa), int2tup(sysb)
    sys_ab = sorted(sysa + sysb)

    Kabc = Tensor(np.asarray(psi_abc).reshape(dims),
                  inds=[('kA{}' if i in sysa else 'kB{}' if i in sysb else
                         'xC{}').format(i) for i in range(len(dims))])

    Babc = Tensor(Kabc.data.conjugate(),
                  inds=[('bA{}' if i in sysa else 'bB{}' if i in sysb else
                         'xC{}').format(i) for i in range(len(dims))])

    return (Kabc & Babc).aslinearoperator(
        [('bA{}' if i in sysa else 'kB{}').format(i) for i in sys_ab],
        [('kA{}' if i in sysa else 'bB{}').format(i) for i in sys_ab],
    )


# --------------------------------------------------------------------------- #
#                         Lanczos tri-diag technique                          #
# --------------------------------------------------------------------------- #

def inner(a, b):
    """Inner product between two vectors
    """
    return vdot(a, b).real


def norm_fro(a):
    """'Frobenius' norm of a vector.
    """
    return sqrt(inner(a, a))


def random_rect(shape, dist='rademacher', orthog=False, norm=True, seed=False):
    """Generate a random matrix optionally orthogonal.

    Parameters
    ----------
    shape : tuple of int
        The shape of matrix.
    dist : {'guassian', 'rademacher'}
        Distribution of the random variables.
    orthog : bool or operator.
        Orthogonalize the columns if more than one.
    norm : bool
        Explicitly normalize the frobenius norm to 1.
    """
    if seed:
        # needs to be truly random so MPI processes don't overlap
        np.random.seed(random.SystemRandom().randint(0, 2**32 - 1))

    if dist == 'rademacher':
        # already normalized
        entries = np.array([1.0, -1.0, 1.0j, -1.0j]) / sqrt(prod(shape))
        V = np.random.choice(entries, shape)
    elif dist == 'gaussian':
        scale = 1 / (prod(shape)**0.5 * 2**0.5)
        V = (np.random.normal(scale=scale, size=shape) +
             1.0j * np.random.normal(scale=scale, size=shape))
        if norm:
            V /= norm_fro(V)
    else:
        raise ValueError("`dist={}` not understood.".format(dist))

    if orthog and min(shape) > 1:
        V = scla.orth(V)
        V /= sqrt(min(V.shape))

    return V


def construct_lanczos_tridiag(A, K, v0=None, bsz=1, k_min=10, orthog=False,
                              beta_tol=1e-6, seed=False, v0_opts=None):
    """Construct the tridiagonal lanczos matrix using only matvec operators.
    This is a generator that iteratively yields the alpha and beta digaonals
    at each step.

    Parameters
    ----------
    A : matrix-like or linear operator
        The operator to approximate, must implement ``.dot`` method to compute
        its action on a vector.
    K : int, optional
        The maximum number of iterations and thus rank of the matrix to find.
    v0 : vector, optional
        The starting vector to iterate with, default to random.
    bsz : int, optional
        The block size (number of columns) of random vectors to iterate with.
    beta_tol : float, optional
        The 'breakdown' tolerance. If the next beta ceofficient in the lanczos
        matrix is less that this, implying that the full non-null space has
        been found, terminate early.
    k_min : int, optional
        The minimum size of the krylov subspace for form.

    Yields
    ------
    alpha : sequence of float of length k
        The diagonal entries of the lanczos matrix.
    beta : sequence of float of length k
        The off-diagonal entries of the lanczos matrix, with the last entry
        the 'look' forward value.
    scaling : float
        How to scale the overall weights.
    """
    if isinstance(A, np.matrix):
        A = np.asarray(A)
    d = A.shape[0]

    if bsz == 1:
        v_shp = (d,)
    else:
        orthog = False
        v_shp = (d, bsz)

    alpha = np.zeros(K + 1)
    beta = np.zeros(K + 2)
    beta[1] = sqrt(prod(v_shp))  # by construction

    if v0 is None:
        if v0_opts is None:
            v0_opts = {}
        q = random_rect(v_shp, **v0_opts)
    else:
        q = v0.astype(np.complex128)
        q /= norm_fro(q)  # normalize (make sure has unit variance)
    v = np.zeros_like(q)

    if orthog:
        Q = np.copy(q).reshape(-1, 1)

    for j in range(1, K + 1):

        r = A.dot(q)
        r -= beta[j] * v
        alpha[j] = inner(q, r)
        r -= alpha[j] * q

        # perform full orthogonalization
        if orthog:
            r -= Q.dot(Q.conj().T.dot(r))

        beta[j + 1] = norm_fro(r)

        # check for convergence
        if abs(beta[j + 1]) < beta_tol:
            yield (np.copy(alpha[1:j + 1]),
                   np.copy(beta[2:j + 2]),
                   np.copy(beta[1])**2 / bsz)
            break

        np.copyto(v, q)
        np.divide(r, beta[j + 1], out=q)

        # keep all vectors
        if orthog:
            Q = np.concatenate((Q, q.reshape(-1, 1)), axis=1)

        if j >= k_min:
            yield (np.copy(alpha[1:j + 1]),
                   np.copy(beta[2:j + 2]),
                   np.copy(beta[1])**2 / bsz)


def lanczos_tridiag_eig(alpha, beta, check_finite=True):
    """Find the eigen-values and -vectors of the Lanczos triadiagonal matrix.

    Parameters
    ----------
    alpha : array of float
        The diagonal.
    beta : array of float
        The k={-1, 1} off-diagonal. Only first ``len(alpha) - 1`` entries used.
    """
    Tk_banded = np.empty((2, alpha.size))
    Tk_banded[1, -1] = 0.0  # sometimes can get nan here? -> breaks eig_banded
    Tk_banded[0, :] = alpha
    Tk_banded[1, :beta.size] = beta

    try:
        tl, tv = scla.eig_banded(
            Tk_banded, lower=True, check_finite=check_finite)

    # sometimes get no convergence -> use dense hermitian method
    except scla.LinAlgError:  # pragma: no cover
        tl, tv = np.linalg.eigh(
            np.diag(alpha) + np.diag(beta[:alpha.size - 1], -1), UPLO='L')

    return tl, tv


def calc_trace_fn_tridiag(tl, tv, f, pos=True):
    """Spectral ritz function sum, weighted by ritz vectors.
    """
    return sum(
        tv[0, i]**2 * f(max(tl[i], 0.0) if pos else tl[i])
        for i in range(tl.size)
    )


def ext_per_trim(x, p=0.6, s=1.0):
    r"""Extended percentile trimmed-mean. Makes the mean robust to asymmetric
    outliers, while using all data when it is nicely clustered. This can be
    visualized roughly as::

            |--------|=========|--------|
        x     x   xx xx xxxxx xxx   xx     x      x           x

    Where the inner range contains the central ``p`` proportion of the data,
    and the outer ranges entends this by a factor of ``s`` either side.

    Parameters
    ----------
    x : array
        Data to trim.
    p : Proportion of data used to define the 'central' percentile.
        For example, p=0.5 gives the inter-quartile range.
    s : Include data up to this factor times the central 'percentile' range
        away from the central percentile itself.

    Returns
    xt : array
        Trimmed data.
    """
    lb = np.percentile(x, 100 * (1 - p) / 2)
    ub = np.percentile(x, 100 * (1 + p) / 2)
    ib = ub - lb

    x = np.array(x)
    trimmed_x = x[(lb - s * ib < x) & (x < ub + s * ib)]

    return trimmed_x


def std(xs):
    """Simple standard deviation - don't invoke numpy for small lists.
    """
    N = len(xs)
    xm = sum(xs) / N
    var = sum((x - xm)**2 for x in xs) / N
    return var**0.5


def exp_approach(x, a, b, c):
    return a * np.exp(-np.array(x) / b) + c


def calc_est_fit(estimates, conv_n):
    """Make estimate by fitting exponential convergence to estimates.
    """
    n = len(estimates)

    if n < conv_n:
        return nan, inf

    try:
        # initial guess for offset, decay length and equilibrium
        p0 = (0, n, estimates[-1])
        ks = np.arange(len(estimates))

        # weight later estimates with less error as well
        sigma = 1 / (1 + ks)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(exp_approach, ks, estimates,
                                   p0=p0, sigma=sigma)

        est = popt[-1]
        err = abs(pcov[-1, -1])**0.5

    except (ValueError, RuntimeError):
        est = nan
        err = inf

    return est, err


def calc_est_window(estimates, mean_ests, conv_n):
    """Make estimate from mean of last ``m`` samples, following:

    1. Take between ``conv_n`` and 12 estimates.
    2. Pair the estimates as they are alternate upper/lower bounds
    3. Compute the standard error on the paired estimates.
    """
    m_est = min(max(conv_n, len(estimates) // 8), 12)

    est = sum(estimates[-m_est:]) / m_est
    mean_ests.append(est)

    if len(estimates) > conv_n:
        # check for convergence using variance of paired last m estimates
        #   -> paired because estimates alternate between upper and lower bound
        paired_ests = [
            (a + b) / 2 for a, b in
            zip(estimates[-m_est::2], estimates[-m_est + 1::2])
        ]
        err = std(paired_ests) / (m_est / 2) ** 0.5
    else:
        err = inf

    return est, err


def single_random_estimate(A, K, bsz, beta_tol, v0, f, pos, tau, tol_scale,
                           k_min=10, verbosity=0, *, seed=None,
                           v0_opts=None, **lanczos_opts):
    # choose normal (any LinearOperator) or MPO lanczos tridiag construction
    if isinstance(A, MatrixProductOperator):
        lanc_fn = construct_lanczos_tridiag_MPO
    elif isinstance(A, PTPTLazyMPS):
        lanc_fn = construct_lanczos_tridiag_PTPTLazyMPS
    else:
        lanc_fn = construct_lanczos_tridiag
        lanczos_opts['bsz'] = bsz

    estimates = []
    mean_ests = []

    # the number of samples to check standard deviation convergence with
    conv_n = 6  # 3 pairs

    # iteratively build the lanczos matrix, checking for convergence
    for alpha, beta, scaling in lanc_fn(
            A, K=K, beta_tol=beta_tol, seed=seed, k_min=k_min - 2 * conv_n,
            v0=v0() if callable(v0) else v0, v0_opts=v0_opts, **lanczos_opts):

        try:
            Tl, Tv = lanczos_tridiag_eig(alpha, beta, check_finite=False)
            Gf = scaling * calc_trace_fn_tridiag(Tl, Tv, f=f, pos=pos)
        except scla.LinAlgError:
            import warnings
            warnings.warn("Approx Spectral Gf tri-eig didn't converge.")
            estimates.append(np.nan)
            continue

        k = alpha.size
        estimates.append(Gf)

        # check for break-down convergence (e.g. found entire subspace)
        #     in which case latest estimate should be accurate
        if abs(beta[-1]) < beta_tol:
            if verbosity >= 2:
                print("k={}: Beta breadown, returning {}.".format(k, Gf))
            return Gf

        # compute an estimate and error using a window of the last few results
        win_est, win_err = calc_est_window(estimates, mean_ests, conv_n)

        # try and compute an estimate and error using exponential fit
        fit_est, fit_err = calc_est_fit(mean_ests, conv_n)

        # take whichever has lowest error
        est, err = min((win_est, win_err), (fit_est, fit_err),
                       key=lambda est_err: est_err[1])

        converged = err < tau * (abs(est) + tol_scale)

        if verbosity >= 2:

            if verbosity >= 3:
                print("est_win={}, err_win={}".format(win_est, win_err))
                print("est_fit={}, err_fit={}".format(fit_est, fit_err))

            print("k={}: Gf={}, Est={}, Err={}".format(k, Gf, est, err))
            if converged:
                print("k={}: Converged to tau {}.".format(k, tau))

        if converged:
            break

    if verbosity >= 1:
        print("k={}: Returning estimate {}.".format(k, est))

    return est


def calc_stats(samples, mean_p, mean_s, tol, tol_scale):
    """Get an estimate from samples.
    """
    xtrim = ext_per_trim(samples, p=mean_p, s=mean_s)

    # sometimes everything is an outlier...
    if xtrim.size == 0:  # pragma: no cover
        estimate, sdev = np.mean(samples), np.std(samples)
    else:
        estimate, sdev = np.mean(xtrim), np.std(xtrim)

    err = sdev / len(samples) ** 0.5

    converged = err < tol * (abs(estimate) + tol_scale)

    return estimate, err, converged


def approx_spectral_function(A, f, tol=1e-2, *, bsz=1, R=1024, tol_scale=1,
                             tau=None, k_min=10, k_max=256, beta_tol=1e-6,
                             mpi=False, mean_p=0.7, mean_s=1.0, pos=False,
                             v0=None, verbosity=0, **lanczos_opts):
    """Approximate a spectral function, that is, the quantity ``Tr(f(A))``.

    Parameters
    ----------
    A : matrix-like or LinearOperator
        Operator to approximate spectral function for. Should implement
        ``A.dot(vec)``.
    f : callable
        Scalar function with which to act on approximate eigenvalues.
    tol : float, optional
        Convergence tolerance threshold for error on mean of repeats. This can
        pretty much be relied on as the overall accuracy. See also
        ``tol_scale`` and ``tau``. Default: 1%.
    bsz : int, optional
        Number of simultenous vector columns to use at once, 1 equating to the
        standard lanczos method. If ``bsz > 1`` then ``A`` must implement
        matrix-matrix multiplication. This is a more performant way of
        essentially increasing ``R``, at the cost of more memory. Default: 1.
    R : int, optional
        The number of repeats with different initial random vectors to perform.
        Increasing this should increase accuracy as ``sqrt(R)``. Cost of
        algorithm thus scales linearly with ``R``. If ``tol`` is non-zero, this
        is the maximum number of repeats.
    tau : float, optional
        The relative tolerance required for a single lanczos run to converge.
        This needs to be small enough that each estimate with a single random
        vector produces an unbiased sample of the operators spectrum.
        Defaults to ``tol``.
    k_min : int, optional
        The minimum size of the krylov subspace to form for each sample.
    k_max : int, optional
        The maximum size of the kyrlov space to form. Cost of algorithm scales
        linearly with ``K``. If ``tau`` is non-zero, this is the maximum size
        matrix to form.
    tol_scale : float, optional
        This sets the overall expected scale of each estimate, so that an
        absolute tolerance can be used for values near zero. Default: 1.
    beta_tol : float, optional
        The 'breakdown' tolerance. If the next beta ceofficient in the lanczos
        matrix is less that this, implying that the full non-null space has
        been found, terminate early. Default: 1e-6.
    mpi : bool, optional
        Whether to parallelize repeat runs over MPI processes.
    mean_p : float, optional
        Factor for robustly finding mean and err of repeat estimates,
        see :func:`ext_per_trim`.
    mean_s : float, optional
        Factor for robustly finding mean and err of repeat estimates,
        see :func:`ext_per_trim`.
    v0 : vector, or callable
        Initial vector to iterate with, sets ``R=1`` if given. If callable, the
        function to produce a random intial vector (sequence).
    pos : bool, optional
        If True, make sure any approximate eigenvalues are positive by
        clipping below 0.
    verbosity : {0, 1, 2}, optional
        How much information to print while computing.
    lanczos_opts
        Supplied to
        :func:`~quimb.linalg.approx_spectral.single_random_estimate` or
        :func:`~quimb.linalg.approx_spectral.construct_lanczos_tridiag`.


    Returns
    -------
    scalar
        The approximate value ``Tr(f(a))``.

    See Also
    --------
    construct_lanczos_tridiag
    """
    if (v0 is not None) and not callable(v0):
        R = 1
    else:
        R = max(1, int(R / bsz))

    if tau is None:
        tau = tol / 2

    if verbosity:
        print("LANCZOS f(A) CALC: tol={}, tau={}, R={}, bsz={}"
              "".format(tol, tau, R, bsz))

    # generate repeat estimates
    kwargs = {'A': A, 'K': k_max, 'bsz': bsz, 'beta_tol': beta_tol,
              'v0': v0, 'f': f, 'pos': pos, 'tau': tau, 'k_min': k_min,
              'tol_scale': tol_scale, 'verbosity': verbosity, **lanczos_opts}

    if not mpi:
        def gen_results():
            for _ in range(R):
                yield single_random_estimate(**kwargs)
    else:
        pool = get_mpi_pool()
        kwargs['seed'] = True
        fs = [pool.submit(single_random_estimate, **kwargs) for _ in range(R)]

        def gen_results():
            for f in fs:
                yield f.result()

    # iterate through estimates, waiting for convergence
    results = gen_results()
    estimate = None
    samples = []
    for _ in range(R):
        samples.append(next(results))

        if verbosity >= 1:
            print("Repeat {}: estimate is {}"
                  "".format(len(samples), samples[-1]))

        # wait a few iterations before checking error on mean breakout
        if len(samples) >= 3:
            estimate, err, converged = calc_stats(
                samples, mean_p, mean_s, tol, tol_scale)

            if verbosity >= 1:
                print("Total estimate = {} ± {}".format(estimate, err))

            if converged:
                if verbosity >= 1:
                    print("Repeat {}: converged to tol {}"
                          "".format(len(samples), tol))
                break

    if mpi:
        # deal with remaining futures
        extra_futures = []
        for f in fs:
            if f.done() or f.running():
                extra_futures.append(f)
            else:
                f.cancel()

        if extra_futures:
            samples.extend(f.result() for f in extra_futures)
            estimate, err, converged = calc_stats(
                samples, mean_p, mean_s, tol, tol_scale)

    if estimate is None:
        estimate, err, _ = calc_stats(
            samples, mean_p, mean_s, tol, tol_scale)

    if verbosity >= 1:
        print("ESTIMATE is {} ± {}".format(estimate, err))

    return estimate


@functools.wraps(approx_spectral_function)
def tr_abs_approx(*args, **kwargs):
    return approx_spectral_function(*args, f=abs, **kwargs)


@functools.wraps(approx_spectral_function)
def tr_exp_approx(*args, **kwargs):
    return approx_spectral_function(*args, f=exp, **kwargs)


@functools.wraps(approx_spectral_function)
def tr_sqrt_approx(*args, **kwargs):
    return approx_spectral_function(*args, f=sqrt, pos=True, **kwargs)


def xlogx(x):
    return x * log2(x) if x > 0 else 0.0


@functools.wraps(approx_spectral_function)
def tr_xlogx_approx(*args, **kwargs):
    return approx_spectral_function(*args, f=xlogx, **kwargs)


# --------------------------------------------------------------------------- #
#                             Specific quantities                             #
# --------------------------------------------------------------------------- #


def choose_bsz_from_dims(dims, subsys):
    """Try to guess a good blocksize to ensure convergence quickly. Based
    on some benchmarks, plus the fact that if bsz is > rank, then the lanczos
    vectors will start to take up the most memory.
    """
    rank = prod(d for i, d in enumerate(dims) if i not in subsys)
    return min(max(1, int(rank / 8)), 128)


def entropy_subsys_approx(psi_ab, dims, sysa, bsz=None, **kwargs):
    """Approximate the (Von Neumann) entropy of a pure state's subsystem.

    Parameters
    ----------
    psi_ab : ket
        Bipartite state to partially trace and find entopy of.
    dims : sequence of int, optional
        The sub dimensions of ``psi_ab``.
    sysa : int or sequence of int, optional
        Index(es) of the 'a' subsystem(s) to keep.
    bsz : int, optional
        The size of the lanczos vector blocks to use. If None, guess a good
        value based on the effective rank of the subsystem.
    kwargs
        Supplied to :func:`approx_spectral_function`.
    """
    lo = lazy_ptr_linop(psi_ab, dims=dims, sysa=sysa)

    if bsz is None:
        bsz = choose_bsz_from_dims(dims, sysa)

    return - tr_xlogx_approx(lo, bsz=bsz, **kwargs)


def tr_sqrt_subsys_approx(psi_ab, dims, sysa, bsz=None, **kwargs):
    """Approximate the trace sqrt of a pure state's subsystem.

    Parameters
    ----------
    psi_ab : ket
        Bipartite state to partially trace and find trace sqrt of.
    dims : sequence of int, optional
        The sub dimensions of ``psi_ab``.
    sysa : int or sequence of int, optional
        Index(es) of the 'a' subsystem(s) to keep.
    bsz : int, optional
        The size of the lanczos vector blocks to use. If None, guess a good
        value based on the effective rank of the subsystem.
    kwargs
        Supplied to :func:`approx_spectral_function`.
    """
    lo = lazy_ptr_linop(psi_ab, dims=dims, sysa=sysa)

    if bsz is None:
        bsz = choose_bsz_from_dims(dims, sysa)

    return tr_sqrt_approx(lo, bsz=bsz, **kwargs)


def norm_ppt_subsys_approx(psi_abc, dims, sysa, sysb, bsz=None, **kwargs):
    """Estimate the norm of the partial tranpose of a pure state's subsystem.
    """
    lo = lazy_ptr_ppt_linop(psi_abc, dims=dims, sysa=sysa, sysb=sysb)

    if bsz is None:
        bsz = choose_bsz_from_dims(dims, sysa + sysb)

    return tr_abs_approx(lo, bsz=bsz, **kwargs)


def logneg_subsys_approx(psi_abc, dims, sysa, sysb, bsz=None, **kwargs):
    """Estimate the logarithmic negativity of a pure state's subsystem.

    Parameters
    ----------
    psi_abc : ket
        Pure tripartite state, for which estimate the entanglement between
        'a' and 'b'.
    dims : sequence of int
        The sub dimensions of ``psi_abc``.
    sysa : int or sequence of int, optional
        Index(es) of the 'a' subsystem(s) to keep, with respect to all
        the dimensions, ``dims``, (i.e. pre-partial trace).
    sysa : int or sequence of int, optional
        Index(es) of the 'b' subsystem(s) to keep, with respect to all
        the dimensions, ``dims``, (i.e. pre-partial trace).
    bsz : int, optional
        The size of the lanczos vector blocks to use. If None, guess a good
        value based on the effective rank of the subsystem.
    kwargs
        Supplied to :func:`approx_spectral_function`.
    """
    return max(log2(norm_ppt_subsys_approx(psi_abc, dims, sysa, sysb,
                                           bsz=bsz, **kwargs)), 0.0)


def negativity_subsys_approx(psi_abc, dims, sysa, sysb, bsz=None, **kwargs):
    """Estimate the negativity of a pure state's subsystem.

    Parameters
    ----------
    psi_abc : ket
        Pure tripartite state, for which estimate the entanglement between
        'a' and 'b'.
    dims : sequence of int
        The sub dimensions of ``psi_abc``.
    sysa : int or sequence of int, optional
        Index(es) of the 'a' subsystem(s) to keep, with respect to all
        the dimensions, ``dims``, (i.e. pre-partial trace).
    sysa : int or sequence of int, optional
        Index(es) of the 'b' subsystem(s) to keep, with respect to all
        the dimensions, ``dims``, (i.e. pre-partial trace).
    bsz : int, optional
        The size of the lanczos vector blocks to use. If None, guess a good
        value based on the effective rank of the subsystem.
    kwargs
        Supplied to :func:`approx_spectral_function`.
    """
    return max((norm_ppt_subsys_approx(psi_abc, dims, sysa, sysb,
                                       bsz=bsz, **kwargs) - 1) / 2, 0.0)


def gen_bipartite_spectral_fn(exact_fn, approx_fn, pure_default):
    """Generate a function that computes a spectral quantity of the subsystem
    of a pure state. Automatically computes for the smaller subsystem, or
    switches to the approximate method for large subsystems.

    Parameters
    ----------
    exact_fn : callable
        The function that computes the quantity on a density matrix, with
        signature: ``exact_fn(rho_a, rank=...)``.
    approx_fn : callable
        The function that approximately computes the quantity using a lazy
        representation of the whole system. With signature
        ``approx_fn(psi_ab, dims, sysa, **approx_opts)``.
    pure_default : float
        The default value when the whole state is the subsystem.

    Returns
    -------
    bipartite_spectral_fn : callable
        The function, with signature:
        ``(psi_ab, dims, sysa, approx_thresh=2**13, **approx_opts)``
    """
    def bipartite_spectral_fn(psi_ab, dims, sysa, approx_thresh=2**13,
                              **approx_opts):
        sysa = int2tup(sysa)
        sz_a = prod(d for i, d in enumerate(dims) if i in sysa)
        sz_b = prod(dims) // sz_a

        # pure state
        if sz_b == 1:
            return pure_default

        # also check if system b is smaller, since spectrum is same for both
        if sz_b < sz_a:
            # if so swap things around
            sz_a, sz_b = sz_b, sz_a
            sysb = [i for i in range(len(dims)) if i not in sysa]
            sysa = sysb

        # check whether to use approx lanczos method
        if (approx_thresh is not None) and (sz_a >= approx_thresh):
            return approx_fn(psi_ab, dims, sysa, **approx_opts)

        rho_a = ptr(psi_ab, dims, sysa)
        return exact_fn(rho_a)

    return bipartite_spectral_fn
