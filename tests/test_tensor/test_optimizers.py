import functools
import importlib

import pytest
import numpy as np
from numpy.testing import assert_allclose
from autoray import real
import opt_einsum as oe

import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.optimize import Vectorizer, parse_network_to_backend


found_torch = importlib.util.find_spec('torch') is not None
found_autograd = importlib.util.find_spec('autograd') is not None
found_jax = importlib.util.find_spec('jax') is not None
found_tensorflow = importlib.util.find_spec('tensorflow') is not None

if found_tensorflow:
    # XXX: tensorflow einsum gradient wrong for complex backends
    #      https://github.com/tensorflow/tensorflow/issues/37307
    oe.backends.dispatch._has_einsum['tensorflow'] = False

jax_case = pytest.param(
    'jax', marks=pytest.mark.skipif(
        not found_jax, reason='jax not installed'))
autograd_case = pytest.param(
    'autograd', marks=pytest.mark.skipif(
        not found_autograd, reason='autograd not installed'))
tensorflow_case = pytest.param(
    'tensorflow', marks=pytest.mark.skipif(
        not found_tensorflow, reason='tensorflow not installed'))
pytorch_case = pytest.param(
    'torch', marks=pytest.mark.skipif(
        not found_torch, reason='pytorch not installed'))


@pytest.fixture
def tagged_qaoa_tn():
    """
    make qaoa tensor network, with RZZ and RX tagged on a per-round basis
    so that these tags can be used as shared_tags to TNOptimizer
    """

    n = 8
    depth = 4
    terms = [(i, (i+1) % n) for i in range(n)]
    gammas = qu.randn(depth)
    betas = qu.randn(depth)

    # make circuit
    circuit_opts = {'gate_opts': {'contract': False}}
    circ = qtn.Circuit(n, **circuit_opts)

    # layer of hadamards to get into plus state
    for i in range(n):
        circ.apply_gate('H', i, gate_round=0)

    for d in range(depth):
        for (i, j) in terms:
            circ.apply_gate('RZZ', -gammas[d], i, j, gate_round=d,
                            parametrize=True)

        for i in range(n):
            circ.apply_gate('RX', betas[d] * 2, i, gate_round=d,
                            parametrize=True)

    # tag circuit for shared_tags
    tn_tagged = circ.psi.copy()
    for i in range(depth):
        tn_tagged.select(['RZZ', f'ROUND_{i}']).add_tag(f'p{2 * i}')
        tn_tagged.select(['RX', f'ROUND_{i}']).add_tag(f'p{2 * i + 1}')

    return n, depth, tn_tagged


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
        k, H, b = qtn.tensor_network_align(psi, H, psi)
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
        k, H, b = qtn.tensor_network_align(psi, H, psi.H)
        energy = (k & H & b).contract(all, optimize='random-greedy')
        return real(energy)

    en_ex = qu.groundenergy(qu.ham_mbl(L, sparse=True, **ham_opts))

    return psi0, H, norm_fn, loss_fn, en_ex


def test_vectorizer():
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


def rand_array(rng):
    ndim = rng.integers(1, 6)
    shape = rng.integers(2, 6, size=ndim)
    dtype = rng.choice(['float32', 'float64', 'complex64', 'complex128'])
    x = rng.normal(shape).astype(dtype)
    if 'complex' in dtype:
        x += 1j * rng.normal(shape).astype(dtype)
    return x


def random_array_pytree(rng, max_depth=3):

    def _recurse(d=0):
        if d >= max_depth:
            return rand_array(rng)
        t = rng.choice(['array', 'list', 'tuple', 'dict'])
        if t == 'array':
            return rand_array(rng)
        elif t == 'list':
            return [_recurse(d + 1) for _ in range(rng.integers(2, 6))]
        elif t == 'tuple':
            return tuple([_recurse(d + 1) for _ in range(rng.integers(2, 6))])
        elif t == 'dict':
            cs = (chr(i) for i in range(ord('a'), ord('z') + 1))
            return {
                next(cs): _recurse(d + 1) for _ in range(rng.integers(2, 6))
            }

    return _recurse()


def test_vectorizer_pytree():
    tree = random_array_pytree(np.random.default_rng(666))
    v = Vectorizer(tree)
    new_tree = v.unpack()
    assert tree is not new_tree
    assert str(tree) == str(new_tree)


@pytest.mark.parametrize('backend', [jax_case, autograd_case,
                                     tensorflow_case, pytorch_case])
@pytest.mark.parametrize('method', ['simple', 'basin'])
def test_optimize_pbc_heis(heis_pbc, backend, method):
    psi0, H, norm_fn, loss_fn, en_ex = heis_pbc
    tnopt = qtn.TNOptimizer(
        psi0,
        loss_fn,
        norm_fn,
        loss_constants={'H': H},
        autodiff_backend=backend,
    )
    if method == 'simple':
        psi_opt = tnopt.optimize(100)
    elif method == 'basin':
        psi_opt = tnopt.optimize_basinhopping(25, 4)
    assert loss_fn(psi_opt, H) == pytest.approx(en_ex, rel=1e-2)


@pytest.mark.parametrize('backend', [jax_case, autograd_case,
                                     tensorflow_case])
@pytest.mark.parametrize('method', ['simple', 'basin'])
def test_optimize_ham_mbl_complex(ham_mbl_pbc_complex, backend, method):
    psi0, H, norm_fn, loss_fn, en_ex = ham_mbl_pbc_complex
    tnopt = qtn.TNOptimizer(
        psi0,
        loss_fn,
        norm_fn,
        loss_constants={'H': H},
        autodiff_backend=backend,
    )
    if method == 'simple':
        psi_opt = tnopt.optimize(100)
    elif method == 'basin':
        psi_opt = tnopt.optimize_basinhopping(25, 4)
    assert loss_fn(psi_opt, H) == pytest.approx(en_ex, rel=1e-2)


@pytest.mark.parametrize('backend', [jax_case, autograd_case,
                                     tensorflow_case])
def test_parametrized_circuit(backend):
    H = qu.ham_mbl(4, dh=3.0, dh_dim=3)
    gs = qu.groundstate(H)
    T_gs = qtn.Dense1D(gs)

    def loss(psi, target):
        f = psi.H & target
        f.rank_simplify_()
        return -abs(f ^ all)

    circ = qtn.circ_ansatz_1D_brickwork(4, depth=4)
    psi0 = circ.psi
    tnopt = qtn.TNOptimizer(
        psi0,
        loss,
        tags='U3',
        loss_constants=dict(target=T_gs),
        autodiff_backend=backend,
        loss_target=-0.99,
    )
    psi_opt = tnopt.optimize(20)
    assert sum(loss < -0.99 for loss in tnopt.losses) == 1
    assert qu.fidelity(psi_opt.to_dense(), gs) > 0.99


def mera_norm_fn(mera):
    return mera.isometrize(method='cayley')


def mera_local_expectation(mera, terms, where):
    tags = [mera.site_tag(coo) for coo in where]
    mera_ij = mera.select(tags, 'any')
    mera_ij_G = mera_ij.gate(terms[where], where)
    mera_ij_ex = (mera_ij_G & mera_ij.H)
    return mera_ij_ex.contract(all, optimize='auto-hq')


@pytest.mark.parametrize('backend', [autograd_case, jax_case,
                                     tensorflow_case, pytorch_case])
@pytest.mark.parametrize('executor', [None, 'threads'])
def test_multiloss(backend, executor):
    if executor == 'threads':
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(2)

    L = 8
    D = 3
    dtype = 'float32'

    mera = qtn.MERA.rand(L, max_bond=D, dtype=dtype)

    H2 = qu.ham_heis(2).real.astype(dtype)
    terms = {(i, (i + 1) % L): H2 for i in range(L)}

    loss_fns = [functools.partial(mera_local_expectation, where=where)
                for where in terms]

    tnopt = qtn.TNOptimizer(
        mera,
        loss_fn=loss_fns,
        norm_fn=mera_norm_fn,
        loss_constants={'terms': terms},
        autodiff_backend=backend,
        executor=executor,
        device='cpu',
    )

    tnopt.optimize(10)
    # ex = -3.6510934089371734
    assert tnopt.loss < -2.5

    if executor is not None:
        executor.shutdown()


def test_parse_network_to_backend_shared_tags(tagged_qaoa_tn):
    n, depth, psi0 = tagged_qaoa_tn

    def to_constant(x):
        return np.asarray(x)

    tags = [f'p{i}' for i in range(2 * depth)]
    tn_tagged, variabes = parse_network_to_backend(psi0,
                                                   tags=tags,
                                                   shared_tags=tags,
                                                   to_constant=to_constant,
                                                   )
    # test number of variables identified
    assert len(variabes) == 2 * depth
    # each variable tag should be in n tensors
    for i in range(len(tags)):
        var_tag = f"__VARIABLE{i}__"
        assert len(tn_tagged.select(var_tag).tensors) == n


def test_parse_network_to_backend_individual_tags(tagged_qaoa_tn):
    n, depth, psi0 = tagged_qaoa_tn

    def to_constant(x):
        return np.asarray(x)

    tags = [f'p{i}' for i in range(2*depth)]
    tn_tagged, variabes = parse_network_to_backend(
        psi0, tags=tags, to_constant=to_constant)
    # test number of variables identified
    assert len(variabes) == 2 * depth * n
    # each variable tag should only be in 1 tensors
    for i in range(len(tags)):
        var_tag = f"__VARIABLE{i}__"
        assert len(tn_tagged.select_tensors(var_tag)) == 1


def test_parse_network_to_backend_constant_tags(tagged_qaoa_tn):
    n, depth, psi0 = tagged_qaoa_tn

    def to_constant(x):
        return np.asarray(x)

    # constant tags, include shared variable tags for first QAOA layer
    constant_tags = ['PSI0', 'H', 'p0', 'p1']
    tn_tagged, variabes = parse_network_to_backend(
        psi0, constant_tags=constant_tags, to_constant=to_constant)

    # test number of variables identified
    assert len(variabes) == 2 * (depth - 1) * n
    # each variable tag should only be in 1 tensors
    for i in range(len(variabes)):
        var_tag = f"__VARIABLE{i}__"
        assert len(tn_tagged.select(var_tag).tensors) == 1


@pytest.mark.parametrize('backend', [jax_case, autograd_case,
                                     tensorflow_case])
def test_shared_tags(tagged_qaoa_tn, backend):
    n, depth, psi0 = tagged_qaoa_tn

    H = qu.ham_heis(n, j=(0., 0., -1.), b=(1., 0., 0.), cyclic=True,)
    gs = qu.groundstate(H)
    T_gs = qtn.Dense1D(gs).astype(complex)  # tensorflow needs all same dtype

    def loss(psi, target):
        f = psi.H & target
        f.rank_simplify_()
        return -abs(f ^ all)

    tags = [f'p{i}' for i in range(2 * depth)]
    tnopt = qtn.TNOptimizer(
        psi0,
        loss_fn=loss,
        tags=tags,
        shared_tags=tags,
        loss_constants={'target': T_gs},
        autodiff_backend=backend,
        # loss_target=-0.99,
    )

    # run optimisation and test output
    psi_opt = tnopt.optimize_basinhopping(n=10, nhop=5)
    # assert sum(loss < -0.99 for loss in tnopt.losses) == 1
    assert qu.fidelity(psi_opt.to_dense(), gs) > 0.99

    # test dimension of optimisation space
    assert tnopt.res.x.size == 2*depth

    # examine tensors inside optimised TN and check sharing was done
    for tag in tags:
        test_data = None
        for t in psi_opt.select_tensors(tag):
            if test_data is None:
                test_data = t.get_params()
            else:
                assert_allclose(test_data, t.get_params())
