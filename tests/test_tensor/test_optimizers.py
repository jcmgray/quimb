import importlib

import pytest
from numpy.testing import assert_allclose
from autoray import real

import quimb as qu
import quimb.tensor as qtn


found_torch = importlib.util.find_spec('torch') is not None
found_autograd = importlib.util.find_spec('autograd') is not None
found_jax = importlib.util.find_spec('jax') is not None

try:
    import tensorflow as tf
    # needed so tensorflow doesn't allocate all gpu mem
    _TF_CONFIG = tf.ConfigProto()
    _TF_CONFIG.gpu_options.allow_growth = True
    found_tensorflow = True
except ImportError:
    found_tensorflow = False


@pytest.fixture
def heis_pbc():
    L = 10
    chi = 8
    dtype = 'float32'
    psi0 = qtn.MPS_rand_state(L, chi, cyclic=True, seed=42).astype(dtype)
    H = qtn.MPO_ham_heis(L, cyclic=True).astype(dtype)

    def norm_fn(psi):
        factor = (psi & psi).contract(all, optimize='random-greedy')
        return psi / factor**0.5

    def loss_fn(psi, H):
        k, H, b = qtn.align_TN_1D(psi, H, psi)
        energy = (k & H & b).contract(all, optimize='random-greedy')
        return energy

    en_ex = qu.groundenergy(qu.ham_heis(L, cyclic=True, sparse=True))

    return psi0, H, norm_fn, loss_fn, en_ex


@pytest.fixture
def ham_mbl_pbc_complex():
    L = 10
    chi = 8
    dtype = 'complex64'
    psi0 = qtn.MPS_rand_state(L, chi, cyclic=True, seed=42).astype(dtype)

    ham_opts = {'cyclic': True, 'dh': 0.7, 'dh_dim': 3, 'seed': 42}
    H = qtn.MPO_ham_mbl(L, **ham_opts).astype(dtype)

    def norm_fn(psi):
        factor = (psi.H & psi).contract(all, optimize='random-greedy')
        return psi * factor**-0.5

    def loss_fn(psi, H):
        k, H, b = qtn.align_TN_1D(psi, H, psi.H)
        energy = (k & H & b).contract(all, optimize='random-greedy')
        return real(energy)

    en_ex = qu.groundenergy(qu.ham_mbl(L, sparse=True, **ham_opts))

    return psi0, H, norm_fn, loss_fn, en_ex


@pytest.mark.skipif(not found_torch, reason="Torch not installed.")
def test_optimize_pbc_heis_torch(heis_pbc):
    from quimb.tensor.optimize_pytorch import TNOptimizer
    psi0, H, norm_fn, loss_fn, en_ex = heis_pbc
    tnopt = TNOptimizer(
        psi0,
        loss_fn,
        norm_fn,
        loss_constants={'H': H},
    )
    psi_opt = tnopt.optimize(100)
    assert loss_fn(psi_opt, H) == pytest.approx(en_ex, rel=1e-2)


@pytest.mark.skipif(not found_tensorflow, reason="Tensorflow not installed.")
@pytest.mark.parametrize('optimizer', ['Adam', 'scipy'])
def test_optimize_pbc_heis_tensorflow(heis_pbc, optimizer):
    from quimb.tensor.optimize_tensorflow import TNOptimizer
    psi0, H, norm_fn, loss_fn, en_ex = heis_pbc

    with tf.Session(config=_TF_CONFIG).as_default():
        tnopt = TNOptimizer(
            psi0,
            loss_fn,
            norm_fn,
            loss_constants={'H': H},
            optimizer=optimizer,
        )
        psi_opt = tnopt.optimize(100)
    assert loss_fn(psi_opt, H) == pytest.approx(en_ex, rel=1e-2)


@pytest.mark.skipif(not found_tensorflow, reason="Tensorflow not installed.")
@pytest.mark.parametrize('optimizer', ['Adam', 'scipy'])
def test_optimize_ham_mbl_complex_tensorflow(ham_mbl_pbc_complex, optimizer):
    from quimb.tensor.optimize_tensorflow import TNOptimizer
    psi0, H, norm_fn, loss_fn, en_ex = ham_mbl_pbc_complex
    with tf.Session(config=_TF_CONFIG).as_default():
        tnopt = TNOptimizer(
            psi0,
            loss_fn,
            norm_fn,
            loss_constants={'H': H},
            optimizer=optimizer,
        )
        psi_opt = tnopt.optimize(100)
    assert loss_fn(psi_opt, H) == pytest.approx(en_ex, rel=1e-2)


def test_vectorizer():
    from quimb.tensor.optimize_autograd import Vectorizer

    shapes = [(2, 3), (4, 5), (6, 7, 8)]
    dtypes = ['complex64', 'float32', 'complex64']
    arrays = [qu.randn(s, dtype=dtype) for s, dtype in zip(shapes, dtypes)]

    v = Vectorizer(arrays)

    grads = [qu.randn(s, dtype=dtype) for s, dtype in zip(shapes, dtypes)]
    v.pack(grads, 'grad')

    new_arrays = v.unpack(v.vector)
    for x, y in zip(arrays, new_arrays):
        assert_allclose(x, y)

    new_arrays = v.unpack(v.grad)
    for x, y in zip(grads, new_arrays):
        assert_allclose(x, y)


jax_case = pytest.param(
    'jax', marks=pytest.mark.skipif(not found_jax, reason='jax not installed'))
autograd_case = pytest.param(
    'autograd', marks=pytest.mark.skipif(not found_autograd,
                                         reason='autograd not installed'))


@pytest.mark.parametrize('backend', [jax_case, autograd_case])
@pytest.mark.parametrize('method', ['simple', 'basin'])
def test_optimize_pbc_heis_jax(heis_pbc, backend, method):
    from quimb.tensor.optimize_autograd import TNOptimizer
    psi0, H, norm_fn, loss_fn, en_ex = heis_pbc
    tnopt = TNOptimizer(
        psi0,
        loss_fn,
        norm_fn,
        loss_constants={'H': H},
        autograd_backend=backend,
    )
    if method == 'simple':
        psi_opt = tnopt.optimize(100)
    elif method == 'basin':
        psi_opt = tnopt.optimize_basinhopping(25, 4)
    assert loss_fn(psi_opt, H) == pytest.approx(en_ex, rel=1e-2)


@pytest.mark.parametrize('backend', [jax_case, autograd_case])
@pytest.mark.parametrize('method', ['simple', 'basin'])
def test_optimize_ham_mbl_complex_jax(ham_mbl_pbc_complex, backend, method):
    from quimb.tensor.optimize_autograd import TNOptimizer
    psi0, H, norm_fn, loss_fn, en_ex = ham_mbl_pbc_complex
    tnopt = TNOptimizer(
        psi0,
        loss_fn,
        norm_fn,
        loss_constants={'H': H},
        autograd_backend=backend,
    )
    if method == 'simple':
        psi_opt = tnopt.optimize(100)
    elif method == 'basin':
        psi_opt = tnopt.optimize_basinhopping(25, 4)
    assert loss_fn(psi_opt, H) == pytest.approx(en_ex, rel=1e-2)
