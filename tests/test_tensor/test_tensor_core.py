import pytest
import operator

import numpy as np
from numpy.testing import assert_allclose
import scipy.sparse.linalg as spla

import quimb as qu
import quimb.tensor as qtn
from quimb.tensor import (
    oset,
    bonds,
    tensor_contract,
    tensor_direct_product,
    Tensor,
    TensorNetwork,
    rand_tensor,
    MPS_rand_state,
    TNLinearOperator1D,
)
from quimb.tensor.decomp import _trim_singular_vals
from quimb.tensor.tensor_core import _CONTRACT_BACKEND, _TENSOR_LINOP_BACKEND


def test__trim_singular_vals():
    s = np.array([3., 2., 1., 0.1])
    assert _trim_singular_vals(s, 0.5, 1) == 3
    assert _trim_singular_vals(s, 0.5, 2) == 2
    assert _trim_singular_vals(s, 2, 3) == 2
    assert _trim_singular_vals(s, 5.02, 3) == 1


class TestContractOpts:

    def test_contract_strategy(self):
        assert qtn.get_contract_strategy() == 'greedy'
        with qtn.contract_strategy('auto'):
            assert qtn.get_contract_strategy() == 'auto'
        assert qtn.get_contract_strategy() == 'greedy'

    def test_contract_backend(self):
        assert qtn.get_contract_backend() == _CONTRACT_BACKEND
        with qtn.contract_backend('cupy'):
            assert qtn.get_contract_backend() == 'cupy'
        assert qtn.get_contract_backend() == _CONTRACT_BACKEND

    def test_tensor_linop_backend(self):
        assert qtn.get_tensor_linop_backend() == _TENSOR_LINOP_BACKEND
        with qtn.tensor_linop_backend('cupy'):
            assert qtn.get_tensor_linop_backend() == 'cupy'
        assert qtn.get_tensor_linop_backend() == _TENSOR_LINOP_BACKEND

    def test_contract_cache(self):
        import tempfile
        import os
        from opt_einsum.paths import register_path_fn

        info = {'num_calls': 0}

        def my_custom_opt(inputs, output, size_dict, memory_limit=None):
            info['num_calls'] += 1
            return [(0, 1)] * (len(inputs) - 1)

        register_path_fn('quimb_test_opt', my_custom_opt)

        tn = qtn.MPS_rand_state(4, 3) & qtn.MPS_rand_state(4, 3)
        assert (
            tn.contract(all, optimize='quimb_test_opt', get='expression')
            is
            tn.contract(all, optimize='quimb_test_opt', get='expression'))
        assert info['num_calls'] == 1

        # contraction pathinfo objects are now cached together
        assert (
            tn.contract(all, optimize='quimb_test_opt', get='path-info')
            is
            tn.contract(all, optimize='quimb_test_opt', get='path-info'))
        assert info['num_calls'] == 1

        # set a directory cache - functions will be run fresh again
        with tempfile.TemporaryDirectory() as tdir:
            assert len(os.listdir(tdir)) == 0
            qtn.set_contract_path_cache(tdir)
            assert (
                tn.contract(all, optimize='quimb_test_opt', get='expression')
                is
                tn.contract(all, optimize='quimb_test_opt', get='expression'))
            assert info['num_calls'] == 2
            assert (
                tn.contract(all, optimize='quimb_test_opt', get='path-info')
                is
                tn.contract(all, optimize='quimb_test_opt', get='path-info'))
            assert info['num_calls'] == 2
            assert len(os.listdir(tdir)) != 0

            # need to release close the cache so the directory can be deleted
            qtn.set_contract_path_cache(None)


class TestBasicTensorOperations:

    def test_tensor_construct(self):
        x = np.random.randn(2, 3, 4)
        a = Tensor(x, inds=[0, 1, 2], tags='blue')
        assert_allclose(a.H.data, x.conj())
        assert a.size == 24

        with pytest.raises(ValueError):
            Tensor(x, inds=[0, 2], tags='blue')

        assert repr(a) == ("Tensor(shape=(2, 3, 4), "
                           "inds=(0, 1, 2), tags=oset(['blue']))")
        assert str(a) == ("Tensor(shape=(2, 3, 4), inds=(0, 1, 2), "
                          "tags=oset(['blue']), backend='numpy', "
                          "dtype='float64')")

    def test_tensor_copy(self):
        a = Tensor(np.random.randn(2, 3, 4), inds=[0, 1, 2], tags='blue')
        b = a.copy()
        b.add_tag('foo')
        assert 'foo' not in a.tags
        b.data[:] = b.data / 2
        # still reference the same underlying array
        assert_allclose(a.data, b.data)

    def test_tensor_deep_copy(self):
        a = Tensor(np.random.randn(2, 3, 4), inds=[0, 1, 2], tags='blue')
        b = a.copy(deep=True)
        b.add_tag('foo')
        assert 'foo' not in a.tags
        b.data[:] = b.data / 2
        # still reference the same underlying array
        assert_allclose(a.data / 2, b.data)

    def test_with_alpha_construct(self):
        x = np.random.randn(2, 3, 4)
        a = Tensor(x, inds='ijk', tags='blue')
        assert_allclose(a.H.data, x.conj())
        assert a.size == 24

        with pytest.raises(ValueError):
            Tensor(x, inds='ij', tags='blue')

        x = np.random.randn(2, 3, 4)
        a = Tensor(x, inds=['a1', 'b2', 'c3'], tags='blue')
        assert_allclose(a.H.data, x.conj())
        assert a.size == 24

        with pytest.raises(ValueError):
            Tensor(x, inds=['ijk'], tags='blue')

    def test_arithmetic_scalar(self):
        x = np.random.randn(2, 3, 4)
        a = Tensor(x, inds=[0, 1, 2], tags='blue')
        assert_allclose((a + 2).data, x + 2)
        assert_allclose((a - 3).data, x - 3)
        assert_allclose((a * 4).data, x * 4)
        assert_allclose((a / 5).data, x / 5)
        assert_allclose((a ** 2).data, x ** 2)
        assert_allclose((2 + a).data, 2 + x)
        assert_allclose((3 - a).data, 3 - x)
        assert_allclose((4 * a).data, 4 * x)
        assert_allclose((5 / a).data, 5 / x)
        assert_allclose((5 ** a).data, 5 ** x)

    @pytest.mark.parametrize("op", [operator.__add__,
                                    operator.__sub__,
                                    operator.__mul__,
                                    operator.__pow__,
                                    operator.__truediv__])
    @pytest.mark.parametrize("mismatch", (True, False))
    def test_tensor_tensor_arithmetic(self, op, mismatch):
        a = Tensor(np.random.rand(2, 3, 4), inds=[0, 1, 2], tags='blue')
        b = Tensor(np.random.rand(2, 3, 4), inds=[0, 1, 2], tags='red')
        if mismatch:
            b.modify(inds=(0, 1, 3))
            with pytest.raises(ValueError):
                op(a, b)
        else:
            c = op(a, b)
            assert_allclose(c.data, op(a.data, b.data))

    def test_tensor_conj_inplace(self):
        data = np.random.rand(2, 3, 4) + 1.0j * np.random.rand(2, 3, 4)
        a = Tensor(data, inds=[0, 1, 2], tags='blue')
        a.conj_()
        assert_allclose(data.conj(), a.data)

    def test_contract_some(self):
        a = Tensor(np.random.randn(2, 3, 4), inds=[0, 1, 2])
        b = Tensor(np.random.randn(3, 4, 5), inds=[1, 2, 3])

        assert a.shared_bond_size(b) == 12

        c = a @ b

        assert isinstance(c, Tensor)
        assert c.shape == (2, 5)
        assert c.inds == (0, 3)

    def test_contract_all(self):
        a = Tensor(np.random.randn(2, 3, 4), inds=[0, 1, 2])
        b = Tensor(np.random.randn(3, 4, 2), inds=[1, 2, 0])
        c = a @ b
        assert isinstance(c, float)
        assert not isinstance(c, Tensor)

    def test_contract_None(self):
        a = Tensor(np.random.randn(2, 3, 4), inds=[0, 1, 2])
        b = Tensor(np.random.randn(3, 4, 5), inds=[3, 4, 5])
        c = a @ b
        assert c.shape == (2, 3, 4, 3, 4, 5)
        assert c.inds == (0, 1, 2, 3, 4, 5)

        a = Tensor(np.random.randn(2, 3, 4), inds=[0, 1, 2])
        b = Tensor(np.random.randn(3, 4, 5), inds=[5, 4, 3])
        c = a @ b

        assert c.shape == (2, 3, 4, 3, 4, 5)
        assert c.inds == (0, 1, 2, 5, 4, 3)

    def test_raise_on_triple_inds(self):
        a = Tensor(np.random.randn(2, 3, 4), inds=[0, 1, 2])
        b = Tensor(np.random.randn(3, 4, 5), inds=[1, 1, 2])
        with pytest.raises(ValueError):
            a @ b

    def test_multi_contract(self):
        a = Tensor(np.random.randn(2, 3, 4), inds=[0, 1, 2],
                   tags='red')
        b = Tensor(np.random.randn(3, 4, 5), inds=[1, 2, 3],
                   tags='blue')
        c = Tensor(np.random.randn(5, 2, 6), inds=[3, 0, 4],
                   tags='blue')
        d = tensor_contract(a, b, c)
        assert isinstance(d, Tensor)
        assert d.shape == (6,)
        assert d.inds == (4,)
        assert d.tags == oset(('red', 'blue'))

    def test_contract_with_legal_characters(self):
        a = Tensor(np.random.randn(2, 3, 4), inds='abc',
                   tags='red')
        b = Tensor(np.random.randn(3, 4, 5), inds='bcd',
                   tags='blue')
        c = a @ b
        assert c.shape == (2, 5)
        assert c.inds == ('a', 'd')

    def test_contract_with_out_of_range_inds(self):
        a = Tensor(np.random.randn(2, 3, 4), inds=[-1, 100, 2200],
                   tags='red')
        b = Tensor(np.random.randn(3, 4, 5), inds=[100, 2200, -3],
                   tags='blue')
        c = a @ b
        assert c.shape == (2, 5)
        assert c.inds == (-1, -3)

    def test_contract_with_wild_mix(self):
        a = Tensor(np.random.randn(2, 3, 4), inds=['-1', 'a', 'foo'],
                   tags='red')
        b = Tensor(np.random.randn(3, 4, 5), inds=['a', 'foo', '42.42'],
                   tags='blue')
        c = a @ b
        assert c.shape == (2, 5)
        assert c.inds == ('-1', '42.42')

    def test_fuse(self):
        a = Tensor(np.random.rand(2, 3, 4, 5), 'abcd', tags={'blue'})
        b = a.fuse({'bra': ['a', 'c'], 'ket': 'bd'})
        assert set(b.shape) == {8, 15}
        assert set(b.inds) == {'bra', 'ket'}
        assert b.tags == oset(('blue',))

        b = a.fuse({'ket': 'bd', 'bra': 'ac'})
        assert set(b.shape) == {15, 8}
        assert set(b.inds) == {'ket', 'bra'}
        assert b.tags == oset(('blue',))

    def test_fuse_leftover(self):
        a = Tensor(np.random.rand(2, 3, 4, 5, 2, 2), 'abcdef', tags={'blue'})
        b = a.fuse({'bra': 'ac', 'ket': 'bd'})
        assert b.shape == (8, 15, 2, 2)
        assert b.inds == ('bra', 'ket', 'e', 'f')
        assert b.tags == oset(('blue',))

    def test_tensor_transpose(self):
        a = Tensor(np.random.rand(2, 3, 4, 5, 2, 2), 'abcdef', tags={'blue'})
        at = a.transpose(*'cdfeba')
        assert at.shape == (4, 5, 2, 2, 3, 2)
        assert at.inds == ('c', 'd', 'f', 'e', 'b', 'a')

        with pytest.raises(ValueError):
            a.transpose(*'cdfebz')

    def test_ownership(self):
        a = rand_tensor((2, 2), ('a', 'b'), tags={'X', 'Y'})
        b = rand_tensor((2, 2), ('b', 'c'), tags={'X', 'Z'})
        assert not a.check_owners()
        assert not b.check_owners()
        tn = TensorNetwork((a, b), virtual=True)
        assert a.check_owners()
        assert b.check_owners()
        assert a.owners[hash(tn)][0]() is tn
        assert b.owners[hash(tn)][0]() is tn
        assert all(map(tn.ind_map.__contains__, ('a', 'b', 'c')))
        assert all(map(tn.tag_map.__contains__, ('X', 'Y', 'Z')))
        a.reindex_({'a': 'd'})
        assert 'a' not in tn.ind_map
        assert 'd' in tn.ind_map
        assert len(tn.tag_map['X']) == 2
        b.retag_({'X': 'W'})
        assert len(tn.tag_map['X']) == 1
        assert 'W' in tn.tag_map
        del tn
        assert not a.check_owners()
        assert not b.check_owners()

    def test_isel(self):
        T = rand_tensor((2, 3, 4, 5, 6), inds=['a', 'b', 'c', 'd', 'e'])
        Tis = T.isel({'d': 2, 'b': 0})
        assert Tis.shape == (2, 4, 6)
        assert Tis.inds == ('a', 'c', 'e')
        assert_allclose(Tis.data, T.data[:, 0, :, 2, :])

    def test_cut_iter(self):
        psi = MPS_rand_state(10, 7, cyclic=True)
        pp = psi.H & psi
        bnds = bonds(pp[0], pp[-1])
        assert sum(tn ^ all for tn in pp.cut_iter(*bnds)) == pytest.approx(1.0)
        assert pp ^ all == pytest.approx(1.0)

    @pytest.mark.parametrize("method", ['qr', 'exp', 'mgs', 'svd'])
    def test_unitize(self, method):
        t = rand_tensor((2, 3, 4), 'abc')
        assert t.H @ t != pytest.approx(3.0)
        t.unitize('b', inplace=True, method=method)
        assert t.H @ t == pytest.approx(3.0)
        assert t.inds == ('b', 'a', 'c')

    def test_connect(self):
        x = rand_tensor((2, 3), 'ab')
        y = rand_tensor((3, 2), 'cd')

        with pytest.raises(ValueError):
            qtn.connect(x, y, 0, 0)

        tn = x | y
        assert len(tn.outer_inds()) == 4
        qtn.connect(x, y, 0, 1)
        assert len(tn.outer_inds()) == 2
        qtn.connect(x, y, 1, 0)
        assert len(tn.outer_inds()) == 0
        assert (tn ^ all).shape == ()
        # make sure bond is newly labelled
        assert set('abcd') & set(tn.all_inds()) == set()


class TestTensorFunctions:
    @pytest.mark.parametrize('method', ['svd', 'eig', 'isvd', 'svds'])
    @pytest.mark.parametrize('linds', [('a', 'b', 'd'), ('c', 'e')])
    @pytest.mark.parametrize('cutoff', [-1.0, 1e-13, 1e-10])
    @pytest.mark.parametrize('cutoff_mode', ['abs', 'rel', 'sum2'])
    @pytest.mark.parametrize('absorb', ['left', 'both', 'right'])
    def test_split_tensor_rank_revealing(self, method, linds, cutoff,
                                         cutoff_mode, absorb):
        a = rand_tensor((2, 3, 4, 5, 6), inds='abcde', tags='red')
        a_split = a.split(linds, method=method, cutoff=cutoff,
                          cutoff_mode=cutoff_mode, absorb=absorb)
        assert len(a_split.tensors) == 2
        if linds == 'abd':
            assert ((a_split.shape == (2, 3, 5, 4, 6)) or
                    (a_split.shape == (4, 6, 2, 3, 5)))
        elif linds == 'edc':
            assert ((a_split.shape == (6, 5, 4, 2, 3)) or
                    (a_split.shape == (2, 3, 6, 5, 4)))
        assert (a_split ^ ...).almost_equals(a)

    @pytest.mark.parametrize('method', ['qr', 'lq'])
    @pytest.mark.parametrize('linds', [('a', 'b', 'd'), ('c', 'e')])
    def test_split_tensor_rank_hidden(self, method, linds):
        a = rand_tensor((2, 3, 4, 5, 6), inds='abcde', tags='red')
        a_split = a.split(linds, method=method)
        assert len(a_split.tensors) == 2
        if linds == 'abd':
            assert ((a_split.shape == (2, 3, 5, 4, 6)) or
                    (a_split.shape == (4, 6, 2, 3, 5)))
        elif linds == 'edc':
            assert ((a_split.shape == (6, 5, 4, 2, 3)) or
                    (a_split.shape == (2, 3, 6, 5, 4)))
        assert (a_split ^ ...).almost_equals(a)

    @pytest.mark.parametrize('method', ['svd', 'eig'])
    def test_singular_values(self, method):
        psim = Tensor(np.eye(2) * 2**-0.5, inds='ab')
        assert_allclose(psim.H @ psim, 1.0)
        assert_allclose(psim.singular_values('a', method=method)**2,
                        [0.5, 0.5])

    def test_absorb_none(self):
        x = qtn.rand_tensor((4, 5, 6, 7), inds='abcd', tags='X', seed=42)
        e = x.H @ x

        with pytest.raises(ValueError):
            x.split(['a', 'c'], absorb=None, method='qr')

        xs_tn = x.split(['a', 'c'], absorb=None, stags='S')
        assert isinstance(xs_tn, TensorNetwork)
        assert xs_tn.num_tensors == 3
        e1 = (xs_tn.H & xs_tn).contract(all, output_inds=())
        assert e1 == pytest.approx(e)
        assert 'S' in xs_tn.tags

        Tl, Ts, Tr = x.split(['a', 'c'], absorb=None, get='tensors')
        assert isinstance(Ts, Tensor)
        assert len(Ts.inds) == 1
        assert 'X' in Ts.tags
        Tl.multiply_index_diagonal_(Ts.inds[0], Ts.data)
        xs_tn = Tl & Tr
        e2 = (xs_tn.H & xs_tn).contract(all)
        assert e2 == pytest.approx(e)

        l, s, r = x.split(['a', 'c'], absorb=None, get='arrays')
        assert s.size == 24
        y_data = np.einsum('acx,x,xbd->abcd', l, s, r)
        assert_allclose(y_data, x.data)

        l, s, r = x.split(['a', 'c'], absorb=None, get='arrays', max_bond=20)
        assert s.size == 20
        y_data = np.einsum('acx,x,xbd->abcd', l, s, r)
        assert (
            qu.norm(y_data, 'fro') ==
            pytest.approx(qu.norm(x.data, 'fro'), rel=0.1)
        )

    @pytest.mark.parametrize('method', ['svd', 'eig'])
    def test_renorm(self, method):
        U = qu.rand_uni(10)
        s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        x = np.einsum('ab,b,bc->ac', U, s, qu.dag(U))

        t = qtn.Tensor(x, inds='ab')
        fn2 = t.norm()**2
        trc = np.einsum('aa', t.data).real

        assert fn2 == pytest.approx(385.)
        assert trc == pytest.approx(55.)

        tn2 = t.split('a', method='svd', cutoff=0.1, cutoff_mode='rsum2')
        a_fn2 = tn2.H @ tn2
        assert qtn.bonds_size(*tn2) == 6
        assert a_fn2 == pytest.approx(fn2)

        tn2 = t.split('a', method='svd', cutoff=40, cutoff_mode='sum2')
        a_fn2 = tn2.H @ tn2
        assert qtn.bonds_size(*tn2) == 6
        assert a_fn2 == pytest.approx(fn2)

        tn1 = t.split('a', method='svd', cutoff=0.2, cutoff_mode='rsum1')
        a_trc = tn1.trace('a', 'b').real
        assert qtn.bonds_size(*tn1) == 6
        assert a_trc == pytest.approx(trc)

        tn1 = t.split('a', method='svd', cutoff=11, cutoff_mode='sum1')
        a_trc = tn1.trace('a', 'b').real
        assert qtn.bonds_size(*tn1) == 6
        assert a_trc == pytest.approx(trc)

    @pytest.mark.parametrize('method', ['svd', 'eig'])
    def test_entropy(self, method):
        psim = Tensor(np.eye(2) * 2**-0.5, inds='ab')
        assert_allclose(psim.H @ psim, 1.0)
        assert_allclose(psim.entropy('a', method=method)**2, 1)

    @pytest.mark.parametrize('method', ['svd', 'eig'])
    def test_entropy_matches_dense(self, method):
        p = MPS_rand_state(5, 32)
        p_dense = p.to_dense()
        real_svn = qu.entropy(p_dense.ptr([2] * 5, [0, 1, 2]))

        svn = (p ^ ...).entropy(('k0', 'k1', 'k2'))
        assert_allclose(real_svn, svn)

        # use tensor to left of bipartition
        p.canonize(2)
        t1 = p['I2']
        left_inds = set(t1.inds) - set(p['I3'].inds)
        svn = (t1).entropy(left_inds, method=method)
        assert_allclose(real_svn, svn)

        # use tensor to right of bipartition
        p.canonize(3)
        t2 = p['I3']
        left_inds = set(t2.inds) & set(p['I2'].inds)
        svn = (t2).entropy(left_inds, method=method)
        assert_allclose(real_svn, svn)

    def test_direct_product(self):
        a1 = rand_tensor((2, 3, 4), inds='abc')
        b1 = rand_tensor((3, 4, 5), inds='bcd')
        a2 = rand_tensor((2, 3, 4), inds='abc')
        b2 = rand_tensor((3, 4, 5), inds='bcd')

        c1 = (a1 @ b1) + (a2 @ b2)
        c2 = (tensor_direct_product(a1, a2, sum_inds=('a')) @
              tensor_direct_product(b1, b2, sum_inds=('d')))
        assert c1.almost_equals(c2)

    def test_direct_product_triple(self):
        a1 = rand_tensor((2, 3, 4), inds='abc')
        b1 = rand_tensor((3, 4, 5, 6), inds='bcde')
        c1 = rand_tensor((6, 7), inds='ef')

        a2 = rand_tensor((2, 3, 4), inds='abc')
        b2 = rand_tensor((3, 4, 5, 6), inds='bcde').transpose(*'decb')
        c2 = rand_tensor((6, 7), inds='ef')

        d1 = (a1 @ b1 @ c1) + (a2 @ b2 @ c2)
        d2 = (tensor_direct_product(a1, a2, sum_inds=('a')) @
              tensor_direct_product(b1, b2, sum_inds=('d')) @
              tensor_direct_product(c1, c2, sum_inds=('f')))
        assert d1.almost_equals(d2)

    @pytest.mark.parametrize("dtype", [float, complex, np.complex128,
                                       np.float_, np.float32, 'raise'])
    def test_rand_tensor(self, dtype):
        if dtype == 'raise':
            with pytest.raises(TypeError):
                rand_tensor((2, 3, 4), 'abc', dtype=dtype)
        else:
            t = rand_tensor((2, 3, 4), 'abc', dtype=dtype)
            assert t.dtype == np.dtype(dtype)

            tn = t & t
            assert tn.dtype == np.dtype(dtype)

    def test_squeeze(self):
        a = rand_tensor((1, 2, 3, 1, 4), inds='abcde', tags=['hello'])
        b = a.squeeze()
        assert b.shape == (2, 3, 4)
        assert b.inds == ('b', 'c', 'e')
        assert 'hello' in b.tags
        assert a.shape == (1, 2, 3, 1, 4)

    @pytest.mark.parametrize('dtype', [None, 'complex128', 'float32'])
    def test_randomize(self, dtype):
        a = rand_tensor((2, 3, 4), ['a', 'b', 'c'], dtype='float64')
        if dtype is not None:
            assert a.dtype != dtype
        x1 = a.norm()
        a.randomize_(dtype=dtype)
        x2 = a.norm()
        assert x1 != pytest.approx(x2)
        assert a.shape == (2, 3, 4)
        if dtype is not None:
            assert a.dtype == dtype
        else:
            assert a.dtype == 'float64'

    def test_multiply_index_diagonal(self):
        x = rand_tensor((3, 4), 'ab')
        y = rand_tensor((4, 5), 'bc')
        z1 = x @ y
        # insert a diagonal gauge
        s = qu.randn(4)
        z2 = (
            x.multiply_index_diagonal('b', s) @
            y.multiply_index_diagonal('b', 1 / s)
        )
        assert z1.almost_equals(z2)

    @pytest.mark.parametrize("smudge", [1e-6, 1e-12])
    def test_balance_bonds(self, smudge):
        t1 = rand_tensor((3, 4), 'ab')
        t2 = rand_tensor((4, 5), 'bc')
        col_nrm_x1 = tensor_contract(t1.H, t1, output_inds='b').data
        col_nrm_y1 = tensor_contract(t2.H, t2, output_inds='b').data
        assert not np.allclose(col_nrm_x1, col_nrm_y1, rtol=1e-6)
        z1 = (t1 @ t2).data
        qtn.tensor_balance_bond(t1, t2, smudge=smudge)
        col_nrm_x2 = tensor_contract(t1.H, t1, output_inds='b').data
        col_nrm_y2 = tensor_contract(t2.H, t2, output_inds='b').data
        assert_allclose(col_nrm_x2, col_nrm_y2, rtol=10 * smudge)
        z2 = (t1 @ t2).data
        assert_allclose(z1, z2)


class TestTensorNetwork:
    def test_combining_tensors(self):
        a = rand_tensor((2, 3, 4), inds=[0, 1, 2], tags='red')
        b = rand_tensor((3, 4, 5), inds=[1, 2, 3], tags='blue')
        c = rand_tensor((5, 2, 6), inds=[3, 0, 4], tags='blue')

        with pytest.raises(TypeError):
            a & np.array([0, 0])

        abc1 = (a & b & c).H.contract()
        abc2 = (a & (b & c)).H.contract()
        abc3 = (TensorNetwork([a, b, c])).H.contract()
        abc4 = (TensorNetwork([a, TensorNetwork([b, c])])).H.contract()
        abc5 = (TensorNetwork([a]) & TensorNetwork([b, c])).H.contract()

        assert_allclose(abc1.data, abc2.data)
        assert_allclose(abc1.data, abc3.data)
        assert_allclose(abc1.data, abc4.data)
        assert_allclose(abc1.data, abc5.data)

    def test_copy(self):
        a = rand_tensor((2, 3, 4), inds='abc', tags='t0')
        b = rand_tensor((2, 3, 4), inds='abd', tags='t1')
        tn1 = TensorNetwork((a, b))
        tn2 = tn1.copy()
        # check can modify tensor structure
        tn2['t1'].modify(inds=('a', 'b', 'X'))
        assert tn1['t1'] is not tn2['t1']
        assert tn2['t1'].inds == ('a', 'b', 'X')
        assert tn1['t1'].inds == ('a', 'b', 'd')
        # but that data remains the same
        assert tn1['t1'].data is tn2['t1'].data
        tn2['t1'].data[:] /= 2
        assert_allclose(tn1['t1'].data, tn2['t1'].data)

    def test_copy_deep(self):
        a = rand_tensor((2, 3, 4), inds='abc', tags='t0')
        b = rand_tensor((2, 3, 4), inds='abd', tags='t1')
        tn1 = TensorNetwork((a, b))
        tn2 = tn1.copy(deep=True)
        # check can modify tensor structure
        tn2['t1'].modify(inds=('a', 'b', 'X'))
        assert tn1['t1'] is not tn2['t1']
        assert tn2['t1'].inds == ('a', 'b', 'X')
        assert tn1['t1'].inds == ('a', 'b', 'd')
        # and that data is not the same
        assert tn1['t1'].data is not tn2['t1'].data
        tn2['t1'].data[:] /= 2
        assert_allclose(tn1['t1'].data / 2, tn2['t1'].data)

    def test_TensorNetwork_init_checks(self):
        a = rand_tensor((2, 3, 4), inds=[0, 1, 2], tags={'red'})
        b = rand_tensor((3, 4, 5), inds=[1, 2, 3], tags={'blue'})
        c = rand_tensor((3, 4, 5), inds=[1, 2, 3], tags={'blue', 'c'})

        with pytest.raises(TypeError):
            TensorNetwork(a, b)  # missing brackets around ``a, b``.

        tn = a & b
        with pytest.raises(TypeError):
            tn['red'] = 1

        tn.add_tag('foo')
        assert len(tn['foo']) == 2
        with pytest.raises(KeyError):
            tn['foo'] = c

        tn[('foo', 'blue')] = c
        assert 'c' in tn.tags
        assert tn[('blue', 'c')] is c

        assert 'red' in tn.tags
        del tn['red']
        assert 'red' not in tn.tags

        assert set(tn.tag_map.keys()) == {'blue', 'c'}

        tn.drop_tags('c')
        assert set(tn.tag_map.keys()) == {'blue'}
        tn.drop_tags(['blue'])
        assert set(tn.tag_map.keys()) == set()

    def test_conj(self):
        a_data = np.random.randn(2, 3, 4) + 1.0j * np.random.randn(2, 3, 4)
        b_data = np.random.randn(3, 4, 5) + 1.0j * np.random.randn(3, 4, 5)
        c_data = np.random.randn(5, 2, 6) + 1.0j * np.random.randn(5, 2, 6)

        a = Tensor(a_data, inds=[0, 1, 2], tags={'red', '0'})
        b = Tensor(b_data, inds=[1, 2, 3], tags={'blue', '1'})
        c = Tensor(c_data, inds=[3, 0, 4], tags={'blue', '2'})

        tn = a & b & c
        new_tn = tn.conj()

        for i, arr in enumerate((a_data, b_data, c_data)):
            assert_allclose(new_tn[str(i)].data, arr.conj())

        # make sure original network unchanged
        for i, arr in enumerate((a_data, b_data, c_data)):
            assert_allclose(tn[str(i)].data, arr)

    def test_conj_inplace(self):
        a_data = np.random.randn(2, 3, 4) + 1.0j * np.random.randn(2, 3, 4)
        b_data = np.random.randn(3, 4, 5) + 1.0j * np.random.randn(3, 4, 5)
        c_data = np.random.randn(5, 2, 6) + 1.0j * np.random.randn(5, 2, 6)

        a = Tensor(a_data, inds=[0, 1, 2], tags={'red', 'I0'})
        b = Tensor(b_data, inds=[1, 2, 3], tags={'blue', 'I1'})
        c = Tensor(c_data, inds=[3, 0, 4], tags={'blue', 'I2'})

        tn = a & b & c
        tn.conj_()

        for i, arr in enumerate((a_data, b_data, c_data)):
            assert_allclose(tn[f"I{i}"].data, arr.conj())

    def test_multiply(self):
        a = rand_tensor((2, 3, 4), inds=['0', '1', '2'], tags='red')
        b = rand_tensor((3, 4, 5), inds=['1', '2', '3'], tags='blue')
        c = rand_tensor((5, 2, 6), inds=['3', '0', '4'], tags='blue')
        tn = a & b & c
        x1 = (tn & tn.H) ^ ...
        x2 = ((2 * tn) & tn.H) ^ ...
        assert_allclose(2 * x1, x2)

    def test_multiply_inplace(self):
        a = rand_tensor((2, 3, 4), inds=['0', '1', '2'], tags='red')
        b = rand_tensor((3, 4, 5), inds=['1', '2', '3'], tags='blue')
        c = rand_tensor((5, 2, 6), inds=['3', '0', '4'], tags='blue')
        tn = a & b & c
        x1 = (tn & tn.H) ^ ...
        tn *= 2
        x2 = (tn & tn.H) ^ ...
        assert_allclose(4 * x1, x2)

    def test_multiply_each(self):
        a = rand_tensor((2, 3, 4), inds=['0', '1', '2'], tags='red')
        b = rand_tensor((3, 4, 5), inds=['1', '2', '3'], tags='blue')
        c = rand_tensor((5, 2, 6), inds=['3', '0', '4'], tags='blue')
        tn = a & b & c
        x1 = (tn & tn.H) ^ ...
        x2 = (tn.multiply_each(2) & tn.H) ^ ...
        assert_allclose(2**3 * x1, x2)

    def test_divide(self):
        a = rand_tensor((2, 3, 4), inds=['0', '1', '2'], tags='red')
        b = rand_tensor((3, 4, 5), inds=['1', '2', '3'], tags='blue')
        c = rand_tensor((5, 2, 6), inds=['3', '0', '4'], tags='blue')
        tn = a & b & c
        x1 = (tn & tn.H) ^ ...
        x2 = ((tn / 2) & tn.H) ^ ...
        assert_allclose(x1 / 2, x2)

    def test_divide_inplace(self):
        a = rand_tensor((2, 3, 4), inds=['0', '1', '2'], tags='red')
        b = rand_tensor((3, 4, 5), inds=['1', '2', '3'], tags='blue')
        c = rand_tensor((5, 2, 6), inds=['3', '0', '4'], tags='blue')
        tn = a & b & c
        x1 = (tn & tn.H) ^ ...
        tn /= 2
        x2 = (tn & tn.H) ^ ...
        assert_allclose(x1 / 4, x2)

    def test_multiply_spread(self):
        a = rand_tensor([2, 2], inds=['a', 'b'], tags='A')
        b = Tensor(a.data, ['b', 'c'], tags='B')
        c = Tensor(a.data, ['c', 'd'], tags='C')
        tn = (a | b | c)
        tn.multiply_(-8j + 1 / 3, spread_over=3)
        assert_allclose(tn['A'].data, tn['B'].data)
        assert_allclose(tn['B'].data, tn['C'].data)

    def test_multiply_spread_neg_stays_real(self):
        a = rand_tensor([2, 2], inds=['a', 'b'], tags='A', dtype='float32')
        b = Tensor(a.data, ['b', 'c'], tags='B')
        c = Tensor(a.data, ['c', 'd'], tags='C')
        tn = (a | b | c)
        tn.multiply_(-1000)
        assert a.dtype == b.dtype == c.dtype == 'float32'
        assert_allclose(abs(tn['A'].data), abs(tn['B'].data))
        assert_allclose(abs(tn['B'].data), abs(tn['C'].data))

    def test_contracting_tensors(self):
        a = rand_tensor((2, 3, 4), inds=[0, 1, 2], tags='red')
        b = rand_tensor((3, 4, 5), inds=[1, 2, 3], tags='blue')
        c = rand_tensor((5, 2, 6), inds=[3, 0, 4], tags='blue')

        a_b_c = a & b & c
        print(a_b_c)
        repr(a_b_c)

        assert isinstance(a_b_c, TensorNetwork)
        a_bc = a_b_c ^ 'blue'
        assert isinstance(a_bc, TensorNetwork)
        assert len(a_bc.tensors) == 2
        abc = a_bc ^ ['red', 'blue']
        assert isinstance(abc, Tensor)
        assert_allclose(abc.data, a_b_c.contract().data)

        assert len(a_b_c.tensors) == 3
        a_b_c ^= 'blue'
        assert len(a_b_c.tensors) == 2

    def test_cumulative_contract(self):
        a = rand_tensor((2, 3, 4), inds=[0, 1, 2], tags='red')
        b = rand_tensor((3, 4, 5), inds=[1, 2, 3], tags='blue')
        c = rand_tensor((5, 2, 6), inds=[3, 0, 4], tags='green')

        d = (a & b & c)
        d2 = d.copy()

        cd = d >> ['red', 'green', 'blue']
        assert cd.shape == (6,)
        assert cd.inds == (4,)

        # make sure inplace operations didn't effect original tensor
        for tag, names in d2.tag_map.items():
            assert d.tag_map[tag] == names

        # test inplace
        d >>= ['red', 'green', 'blue']
        assert isinstance(d, Tensor)

    def test_contract_with_slices(self):
        a = rand_tensor((2, 3, 4), inds=[0, 1, 2], tags='I0')
        b = rand_tensor((3, 4, 5), inds=[1, 2, 3], tags='I1')
        c = rand_tensor((5, 2, 6), inds=[3, 0, 4], tags='I2')
        d = rand_tensor((5, 2, 6), inds=[5, 6, 4], tags='I3')
        tn = TensorNetwork((a, b, c, d), structure="I{}")

        assert len((tn ^ slice(2)).tensors) == 3
        assert len((tn ^ slice(..., 1, -1)).tensors) == 3
        assert len((tn ^ slice(-1, 1)).tensors) == 3
        assert len((tn ^ slice(None, -2, -1)).tensors) == 3
        assert len((tn ^ slice(-2, 0)).tensors) == 3

    def test_reindex(self):
        a = Tensor(np.random.randn(2, 3, 4), inds=[0, 1, 2], tags='red')
        b = Tensor(np.random.randn(3, 4, 5), inds=[1, 2, 3], tags='blue')
        c = Tensor(np.random.randn(5, 2, 6), inds=[3, 0, 4], tags='green')

        a_b_c = (a & b & c)

        d = a_b_c.reindex({4: 'foo', 2: 'bar'})

        assert a_b_c.outer_inds() == (4,)
        assert d.outer_inds() == ('foo',)
        assert set(a_b_c.inner_inds()) == {0, 1, 2, 3}
        assert set(d.inner_inds()) == {0, 1, 'bar', 3}
        assert d.tensors[0].inds == (0, 1, 'bar')

        d = a_b_c.reindex_({4: 'foo', 2: 'bar'})

        assert a_b_c.outer_inds() == ('foo',)
        assert set(d.inner_inds()) == {0, 1, 'bar', 3}
        assert d.tensors[0].inds == (0, 1, 'bar')

    def test_add_tag(self):
        a = rand_tensor((2, 3, 4), inds='abc', tags={'red'})
        b = rand_tensor((2, 3, 4), inds='abc', tags={'blue'})
        tn = a & b
        tn.add_tag('green')
        assert 'green' in tn.tag_map
        assert 'green' in tn['red'].tags
        assert 'green' in tn['blue'].tags
        tn.add_tag('blue')
        for t in tn:
            assert 'blue' in t.tags

    def test_index_by_site(self):
        a_data = np.random.randn(2, 3, 4)
        b_data = np.random.randn(2, 3, 4)
        a = Tensor(a_data, inds='abc', tags={'I0'})
        b = Tensor(b_data, inds='abc', tags={'I1'})
        tn = TensorNetwork((a, b), structure="I{}")
        assert_allclose(tn[0].data, a_data)
        new_data = np.random.randn(2, 3, 4)
        tn[1] = Tensor(new_data, inds='abc', tags={'I1', 'red'})
        assert_allclose(tn['I1'].data, new_data)
        assert 'red' in tn['I1'].tags

    def test_set_data_in_tensor(self):
        a_data = np.random.randn(2, 3, 4)
        b_data = np.random.randn(2, 3, 4)
        a = Tensor(a_data, inds='abc', tags={'I0'})
        b = Tensor(b_data, inds='abc', tags={'I1'})
        tn = TensorNetwork((a, b), structure="I{}")
        assert_allclose(tn[0].data, a_data)
        new_data = np.random.randn(2, 3, 4)
        tn[1].modify(data=new_data)
        assert_allclose(tn['I1'].data, new_data)

    def test_combining_with_no_check_collisions(self):
        p1 = MPS_rand_state(5, 3, phys_dim=3)
        p2 = MPS_rand_state(5, 3, phys_dim=3)
        # shouldn't need to check any collisions
        tn = TensorNetwork((p1, p2), check_collisions=False)
        # test can contract
        assert 0 < abs(tn ^ ...) < 1

    def test_retagging(self):
        x = rand_tensor((2, 4), inds='ab', tags={'X', 'I0'})
        y = rand_tensor((4, 2, 5), inds='bcd', tags={'Y', 'I1'})
        z = rand_tensor((5, 3), inds='de', tags={'Z', 'I2'})
        tn = TensorNetwork((x, y, z))
        tn.retag_({"I0": "I1", "I1": "I2", "I2": "I3", "Z": "A"})
        assert set(tn.tag_map.keys()) == {'X', 'I1', 'I2', 'I3', 'Y', 'A'}

    def test_squeeze(self):
        A, B, C = (rand_tensor((1, 2, 3), 'abc', tags=['I0']),
                   rand_tensor((2, 3, 4), 'bcd', tags=['I1']),
                   rand_tensor((4, 1, 1), 'dae', tags=['I2']))
        tn = A & B & C

        x1 = tn ^ ...
        stn = tn.squeeze()

        assert tn['I0'].shape == (1, 2, 3)
        assert tn['I1'].shape == (2, 3, 4)
        assert tn['I2'].shape == (4, 1, 1)

        assert stn['I0'].shape == (2, 3)
        assert stn['I1'].shape == (2, 3, 4)
        assert stn['I2'].shape == (4,)

        x2 = stn ^ ...
        assert_allclose(x1.data, x2)  # x2 should be scalar already

    def test_tensors_sorted(self):
        tn1, tn2 = TensorNetwork([]), TensorNetwork([])
        A, B, C = (rand_tensor((1, 2, 3), 'abc', tags=['I0']),
                   rand_tensor((2, 3, 4), 'bcd', tags=['I1']),
                   rand_tensor((4, 1, 1), 'dae', tags=['I2']))

        tn1 &= A
        tn1 &= B
        tn1 &= C

        tn2 &= C
        tn2 &= A
        tn2 &= B

        for t1, t2 in zip(tn1.tensors_sorted(), tn2.tensors_sorted()):
            assert t1.tags == t2.tags
            assert t1.almost_equals(t2)

    def test_select_tensors_mode(self):
        A, B, C = (rand_tensor((2, 2), 'ab', tags={'0', 'X'}),
                   rand_tensor((2, 2), 'bc', tags={'1', 'X', 'Y'}),
                   rand_tensor((2, 3), 'cd', tags={'2', 'Y'}))
        tn = A & B & C

        ts = tn.select_tensors(('X', 'Y'), which='all')
        assert len(ts) == 1
        assert not any(map(A.almost_equals, ts))
        assert any(map(B.almost_equals, ts))
        assert not any(map(C.almost_equals, ts))

        ts = tn.select_tensors(('X', 'Y'), which='any')
        assert len(ts) == 3
        assert any(map(A.almost_equals, ts))
        assert any(map(B.almost_equals, ts))
        assert any(map(C.almost_equals, ts))

    def test_replace_with_identity(self):
        A, B, C, D = (rand_tensor((2, 3, 4), 'abc', tags=['I0']),
                      rand_tensor((4, 5, 6), 'cde', tags=['I1']),
                      rand_tensor((5, 6, 7), 'def', tags=['I2']),
                      rand_tensor((7,), 'f', tags=['I3']))

        tn = (A & B & C & D)

        with pytest.raises(ValueError):
            tn.replace_with_identity(('I1', 'I2'), inplace=True)

        tn['I2'] = rand_tensor((5, 6, 4), 'def', tags=['I2'])
        tn['I3'] = rand_tensor((4,), 'f', tags=['I3'])

        tn1 = tn.replace_with_identity(('I1', 'I2'))
        assert len(tn1.tensors) == 2
        x = tn1 ^ ...
        assert set(x.inds) == {'a', 'b'}

        A, B, C = (rand_tensor((2, 2), 'ab', tags={'0'}),
                   rand_tensor((2, 2), 'bc', tags={'1'}),
                   rand_tensor((2, 3), 'cd', tags={'2'}))

        tn = A & B & C

        tn2 = tn.replace_with_identity('1')
        assert len(tn2.tensors) == 2
        x = tn2 ^ ...
        assert set(x.inds) == {'a', 'd'}

    def test_partition(self):
        k = MPS_rand_state(10, 7, site_tag_id='Q{}', structure_bsz=4)
        where = [f'Q{i}' for i in range(10) if i % 2 == 1]
        k.add_tag('odd', where=where, which='any')

        tn_even, tn_odd = k.partition('odd')

        assert len(tn_even.tensors) == len(tn_odd.tensors) == 5

        assert tn_even.structure == 'Q{}'
        assert tn_even.structure_bsz == 4
        assert tn_odd.structure == 'Q{}'
        assert tn_odd.structure_bsz == 4

        assert (tn_even & tn_odd).sites == range(10)

    def test_compress_multibond(self):
        A = rand_tensor((7, 2, 2), 'abc', tags='A')
        A.expand_ind('c', 3)
        B = rand_tensor((3, 2, 7), 'cbd', tags='B')
        x0 = (A & B).trace('a', 'd')
        qtn.tensor_compress_bond(A, B, absorb='left')
        A.transpose_('a', 'b')
        assert A.shape == (7, 4)
        B.transpose_('b', 'd')
        assert B.shape == (4, 7)
        assert B.H @ B == pytest.approx(4)
        x1 = (A & B).trace('a', 'd')
        assert x1 == pytest.approx(x0)

    def test_canonize_multibond(self):
        A = rand_tensor((3, 4, 5), 'abc', tags='A')
        assert A.H @ A != pytest.approx(3)
        B = rand_tensor((5, 4, 3), 'cbd', tags='B')
        x0 = (A & B).trace('a', 'd')
        qtn.tensor_canonize_bond(A, B)
        assert A.shape == (3, 3)
        assert B.shape == (3, 3)
        assert A.H @ A == pytest.approx(3)
        x1 = (A & B).trace('a', 'd')
        assert x1 == pytest.approx(x0)

    @pytest.mark.parametrize("method", ['svd', 'eig', 'isvd', 'svds', 'rsvd'])
    def test_compress_between(self, method):
        A = rand_tensor((3, 4, 5), 'abd', tags={'T1'})
        A.expand_ind('d', 10)
        B = rand_tensor((5, 6), 'dc', tags={'T2'})
        B.expand_ind('d', 10)
        tn = A | B
        assert A.shared_bond_size(B) == 10
        tn.compress_between('T1', 'T2', method=method)
        assert A.shared_bond_size(B) == 5

    @pytest.mark.parametrize("method", ['svd', 'eig', 'isvd', 'svds', 'rsvd'])
    def test_compress_all(self, method):
        k = MPS_rand_state(10, 7)
        k += k
        k /= 2
        k.compress_all_(max_bond=7, method=method)
        assert k.max_bond() == 7
        assert_allclose(k.H @ k, 1.0)

    def test_canonize_between(self):
        k = MPS_rand_state(4, 3)
        k.canonize_between('I1', 'I2')
        assert k.H @ k == pytest.approx(1)
        t = k[1]
        assert t.H @ t == pytest.approx(3)
        t = k[2]
        assert t.H @ t != pytest.approx(3)
        k.canonize_between('I2', 'I1')
        assert k.H @ k == pytest.approx(1)
        t = k[1]
        assert t.H @ t != pytest.approx(3)
        t = k[2]
        assert t.H @ t == pytest.approx(3)

    def test_canonize_around(self):
        # make a small tree tensor network
        #
        #             U2--                         v--
        #             |                            |
        #             U1--         ==>             v--        etc
        #             |                            |
        #   L2---L1---C---R1---R2        >---->---->----O---<
        #  /     |    |    |    \       /     |    |    |    \
        #
        C = qtn.rand_tensor([2], inds=['kC'], tags='C', dtype=complex)

        # left arm
        L1 = qtn.rand_tensor([2], inds=['kL1'], tags='L1', dtype=complex)
        qtn.new_bond(C, L1, size=7)
        L2 = qtn.rand_tensor([2], inds=['kL2'], tags='L2', dtype=complex)
        qtn.new_bond(L1, L2, size=7)

        # right arm
        R1 = qtn.rand_tensor([2], inds=['kR1'], tags='R1', dtype=complex)
        qtn.new_bond(C, R1, size=7)
        R2 = qtn.rand_tensor([2], inds=['kR2'], tags='R2', dtype=complex)
        qtn.new_bond(R1, R2, size=7)

        # upper arm
        U1 = qtn.rand_tensor([2], inds=['kU1'], tags='U1', dtype=complex)
        qtn.new_bond(C, U1, size=7)
        U2 = qtn.rand_tensor([2], inds=['kU2'], tags='U2', dtype=complex)
        qtn.new_bond(U1, U2, size=7)

        # make the TN and randomize the data then normalize
        ttn = qtn.TensorNetwork([C, L1, L2, R1, R2, U1, U2])
        ttn.randomize_()
        ttn /= (ttn.H @ ttn)**0.5
        assert ttn.H @ ttn == pytest.approx(1.0)

        # test max distance
        ttn.canonize_around_('C', max_distance=1)
        assert ttn.H @ ttn == pytest.approx(1.0)
        assert ttn['C'].H @ ttn['C'] != pytest.approx(1.0)

        # tensors one-away from center should be isometries
        for tg in ['L1', 'R1', 'U1']:
            assert ttn[tg].H @ ttn[tg] == pytest.approx(7)
        # tensors two-away from center should be random
        for tg in ['L2', 'R2', 'U2']:
            assert ttn[tg].H @ ttn[tg] != pytest.approx(2)

        ttn.canonize_around_('C', max_distance=2)
        for tg in ['L2', 'R2', 'U2']:
            assert ttn[tg].H @ ttn[tg] == pytest.approx(2)

        # test can set the orthogonality center anywhere
        for tg in ['C', 'L1', 'L2', 'R1', 'R2', 'U1', 'U2']:
            ttn.canonize_around_(tg)
            assert ttn.H @ ttn == pytest.approx(1.0)
            assert ttn[tg].H @ ttn[tg] == pytest.approx(1.0)

            # tensors two-away from center should now be isometries
            for far_tg in ['L2', 'R2', 'U2']:
                if far_tg != tg:
                    ttn[far_tg].H @ ttn[far_tg] == pytest.approx(2)

    def test_insert_operator(self):
        p = MPS_rand_state(3, 7, tags='KET')
        q = p.H.retag({'KET': 'BRA'})
        qp = q & p
        sz = qu.spin_operator('z').real
        qp.insert_operator(sz, ('KET', 'I1'), ('BRA', 'I1'),
                           tags='SZ', inplace=True)
        assert 'SZ' in qp.tags
        assert len(qp.tensors) == 7
        x1 = qp ^ all
        x2 = qu.expec(p.to_dense(), qu.ikron(sz, [2, 2, 2], inds=1))
        assert x1 == pytest.approx(x2)

    @pytest.mark.parametrize("dtype", (float, complex))
    def test_insert_gauge(self, dtype):
        k = MPS_rand_state(10, 7, dtype=dtype, normalize=False)
        kU = k.copy()

        U = rand_tensor((7, 7), dtype=dtype, inds='ab').data
        kU.insert_gauge(U, 4, 5)

        assert k[3].almost_equals(kU[3])
        assert not k[4].almost_equals(kU[4])
        assert not k[5].almost_equals(kU[5])
        assert k[6].almost_equals(kU[6])

        assert k[4].inds == kU[4].inds
        assert k[5].inds == kU[5].inds

        assert_allclose(k.H @ k, kU.H @ kU)

    def test_fuse_multibonds(self):
        x = rand_tensor((2, 2, 2), ['a', 'b', 'c'])
        y = rand_tensor((2, 2, 2, 2), ['b', 'c', 'd', 'e'])
        z = rand_tensor((2, 2, 2), ['a', 'e', 'd'])
        tn = (x & y & z)
        assert len(tn.inner_inds()) == 5
        tn.fuse_multibonds(inplace=True)
        assert len(tn.inner_inds()) == 3

    def test_graph(self):
        import matplotlib
        from matplotlib import pyplot as plt
        matplotlib.use('Template')
        k = MPS_rand_state(10, 7, normalize=False)
        fig = k.graph(color=['I0', 'I2'], return_fig=True)
        plt.close(fig)

    def test_graph_with_fixed_pos(self):
        import matplotlib
        from matplotlib import pyplot as plt
        matplotlib.use('Template')
        n = 7
        p = MPS_rand_state(n, 7, tags='KET')
        q = MPS_rand_state(n, 7, tags='BRA')
        fix = {**{('KET', f'I{i}'): (i, 0) for i in range(n)},
               **{('BRA', f'I{i}'): (i, 1) for i in range(n)}}
        fig = (q | p).graph(colors=['KET', 'BRA'], fix=fix, return_fig=True)
        plt.close(fig)

    def test_pickle(self):
        import tempfile
        import os

        pytest.importorskip("joblib")

        tn = MPS_rand_state(10, 7, tags='KET')

        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "tn.dmp")
            qu.save_to_disk(tn, fname)
            tn2 = qu.load_from_disk(fname)

        assert tn.H @ tn2 == pytest.approx(1.0)

        assert all(hash(tn) not in t.owners for t in tn2)
        assert all(hash(tn2) in t.owners for t in tn2)

    @pytest.mark.parametrize('dtype', [None, 'float32', 'complex128'])
    def test_randomize(self, dtype):
        psi = MPS_rand_state(5, 3, dtype='float64')
        x1 = psi.H @ psi
        psi.randomize_(seed=42, dtype=dtype)
        x2 = psi.H @ psi
        assert x1 != pytest.approx(x2)
        if dtype is None:
            assert psi.dtype == 'float64'
        else:
            assert psi.dtype == dtype
        psi.randomize_(seed=42, dtype=dtype)
        x3 = psi.H @ psi
        assert x2 == pytest.approx(x3)

    @pytest.mark.parametrize('dtype', ['float32', 'complex128'])
    @pytest.mark.parametrize('value', [None, 42])
    def test_equalize_norms(self, dtype, value):
        psi = MPS_rand_state(5, 3, dtype=dtype)
        psi.randomize_(seed=42)
        x_exp = psi.H @ psi
        norms = [t.norm() for t in psi]
        psi.equalize_norms_(value)
        enorms = [t.norm() for t in psi]
        if value is None:
            assert all(n1 != n2 for n1, n2 in zip(norms, enorms))
            assert psi.H @ psi == pytest.approx(x_exp)
        else:
            assert all(n1 == pytest.approx(value) for n1 in enorms)

    @pytest.mark.parametrize("append", [None, '*'])
    def test_mangle_inner(self, append):
        a = MPS_rand_state(6, 3)
        b = a.copy()
        assert tuple(a.ind_map) == tuple(b.ind_map)
        b.mangle_inner_(append)
        assert tuple(a.ind_map) != tuple(b.ind_map)
        ab = a & b
        assert all(ix in ab.ind_map for ix in a.ind_map)
        assert all(ix in ab.ind_map for ix in b.ind_map)


class TestTensorNetworkSimplifications:

    def test_rank_simplify(self):
        A = rand_tensor([2, 2, 3], 'abc', tags='A')
        B = rand_tensor([3, 2], 'cd', tags='B')
        C = rand_tensor([2, 2, 2], 'def', tags='C')
        tn = A & B & C
        tn_s = tn.rank_simplify()
        assert tn.num_tensors == 3
        assert tn_s.num_tensors == 2
        assert (tn ^ all).almost_equals(tn_s ^ all)
        # checl that 'B' was absorbed into 'A' not 'C'
        assert set(tn_s['B'].tags) == {'A', 'B'}

    def test_diagonal_reduce(self):
        A = rand_tensor([2, 2], 'ab', dtype=complex)
        B = Tensor([[3j, 0.], [0., 4j]], 'bc')
        C = rand_tensor([2, 2], 'ca', dtype=complex)
        tn = A & B & C
        tn_s = tn.diagonal_reduce()
        assert tn.num_indices == 3
        assert tn_s.num_indices == 2
        assert tn ^ all == pytest.approx(tn_s.contract(all, output_inds=[]))

    def test_antidiag_gauge(self):
        A = rand_tensor([2, 2], 'ab', dtype=complex)
        B = Tensor([[0., 3j], [4j, 0.]], 'bc')
        C = rand_tensor([2, 2], 'ca', dtype=complex)
        tn = A & B & C
        assert tn.num_indices == 3
        # can't use diagonal reduction yet
        assert tn.diagonal_reduce().num_indices == 3
        # initial gauge doesn't change indices
        tn_a = tn.antidiag_gauge()
        assert tn_a.num_indices == 3
        # but allows the diagonal reduction
        tn_ad = tn_a.diagonal_reduce()
        assert tn_ad.num_indices == 2
        assert tn ^ all == pytest.approx(tn_ad.contract(all, output_inds=[]))

    def test_column_reduce(self):
        A = rand_tensor([2, 3], 'ab')
        A.new_ind('c', size=4)
        B = rand_tensor([4, 5, 6], 'cde')
        tn = A & B
        assert tn.num_indices == 5
        tn_s = tn.column_reduce()
        assert tn_s.num_indices == 4
        assert (tn ^ all).almost_equals(tn_s ^ all)


class TestTensorNetworkAsLinearOperator:

    def test_against_dense(self):
        A, B, C, D = (
            rand_tensor([3, 5, 5], 'aef'),
            rand_tensor([3, 5, 5], 'beg'),
            rand_tensor([3, 5, 5], 'cfh'),
            rand_tensor([3, 5, 5], 'dhg'),
        )

        tn = A & B & C & D
        tn_lo = tn.aslinearoperator(('a', 'b'), ('c', 'd'))
        tn_d = tn.to_dense(['a', 'b'], ['c', 'd'])

        u, s, v = qu.svds(tn_lo, k=5, backend='scipy')
        ud, sd, vd = qu.svds(tn_d, k=5, backend='scipy')

        assert_allclose(s, sd)

        # test matmat
        X = np.random.randn(9, 8) + 1.0j * np.random.randn(9, 8)
        assert_allclose(tn_lo.dot(X), tn_d.dot(X))

    @pytest.mark.parametrize("dtype", (float, complex))
    @pytest.mark.parametrize("method", ('isvd', 'rsvd'))
    def test_replace_with_svd_using_linear_operator(self, dtype, method):
        k = MPS_rand_state(100, 10, dtype=dtype, cyclic=True)
        b = k.H
        b.expand_bond_dimension(11)
        k.add_tag('_KET')
        b.add_tag('_BRA')
        tn = b & k

        x1 = tn ^ ...

        ul, = tn['_KET', 'I1'].bonds(tn['_KET', 'I2'])
        ll, = tn['_BRA', 'I1'].bonds(tn['_BRA', 'I2'])

        where = [f'I{i}' for i in range(2, 40)]

        tn.replace_with_svd(where, left_inds=(ul, ll), eps=1e-3, method=method,
                            inplace=True, ltags='_U', rtags='_V')
        tn.structure = None
        x2 = tn ^ ...

        # check ltags and rtags have gone in
        assert isinstance(tn['_U'], Tensor)
        assert isinstance(tn['_V'], Tensor)

        assert_allclose(x1, x2, rtol=1e-4)

    def test_TNLinearOperator1D(self):
        p = MPS_rand_state(40, 10, dtype=complex)
        pp = p.H & p
        start, stop = 10, 30
        lix = bonds(pp[start - 1], pp[start])
        rix = bonds(pp[stop - 1], pp[stop])

        sec = pp[start:stop]

        A = TNLinearOperator1D(sec, lix, rix, start, stop)
        B = sec.aslinearoperator(lix, rix)

        s1 = spla.svds(A)[1]
        s2 = spla.svds(B)[1]

        assert_allclose(s1, s2)
