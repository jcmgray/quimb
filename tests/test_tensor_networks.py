import pytest
import operator

import numpy as np
from numpy.testing import assert_allclose

from quimb import (
    ham_heis,
    expec,
    seigsys,
    entropy,
    entropy_subsys,
    schmidt_gap,
    plus,
    neel_state,
    is_eigenvector,
    eigsys,
)

from quimb.tensor import (
    tensor_contract,
    tensor_direct_product,
    Tensor,
    TensorNetwork,
    rand_tensor,
    MatrixProductState,
    MatrixProductOperator,
    align_TN_1D,
    MPS_rand_state,
    MPS_product_state,
    MPS_computational_state,
    MPS_neel_state,
    MPS_zero_state,
    MPO_identity,
    MPO_identity_like,
    MPO_rand_herm,
    MPO_ham_ising,
    MPO_ham_XY,
    MPO_ham_heis,
    MPO_ham_mbl,
    MovingEnvironment,
    DMRG1,
    DMRG2,
    DMRGX,
)
from quimb.tensor.tensor_core import _trim_singular_vals


def test__trim_singular_vals():
    s = np.array([3., 2., 1., 0.1])
    assert _trim_singular_vals(s, 0.5, 1) == 3
    assert _trim_singular_vals(s, 0.5, 2) == 2
    assert _trim_singular_vals(s, 2, 3) == 2
    assert _trim_singular_vals(s, 5.02, 3) == 1


class TestBasicTensorOperations:

    def test_tensor_construct(self):
        x = np.random.randn(2, 3, 4)
        a = Tensor(x, inds=[0, 1, 2], tags='blue')
        assert_allclose(a.H.data, x.conj())
        assert a.size == 24

        with pytest.raises(ValueError):
            Tensor(x, inds=[0, 2], tags='blue')

    def test_tensor_copy(self):
        a = Tensor(np.random.randn(2, 3, 4), inds=[0, 1, 2], tags='blue')
        b = a.copy()
        b.tags.add('foo')
        assert 'foo' not in a.tags
        b.data /= 2
        # still reference the same underlying array
        assert_allclose(a.data, b.data)

    def test_tensor_deep_copy(self):
        a = Tensor(np.random.randn(2, 3, 4), inds=[0, 1, 2], tags='blue')
        b = a.copy(deep=True)
        b.tags.add('foo')
        assert 'foo' not in a.tags
        b.data /= 2
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
            b.inds = (0, 1, 3)
            with pytest.raises(ValueError):
                op(a, b)
        else:
            c = op(a, b)
            assert_allclose(c.data, op(a.data, b.data))

    def test_tensor_conj_inplace(self):
        data = np.random.rand(2, 3, 4) + 1.0j * np.random.rand(2, 3, 4)
        a = Tensor(data, inds=[0, 1, 2], tags='blue')
        a.conj(inplace=True)
        assert_allclose(data.conj(), a.data)

    def test_contract_some(self):
        a = Tensor(np.random.randn(2, 3, 4), inds=[0, 1, 2])
        b = Tensor(np.random.randn(3, 4, 5), inds=[1, 2, 3])
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
        assert d.tags == {'red', 'blue'}

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
        assert b.tags == {'blue'}

        b = a.fuse({'ket': 'bd', 'bra': 'ac'})
        assert set(b.shape) == {15, 8}
        assert set(b.inds) == {'ket', 'bra'}
        assert b.tags == {'blue'}

    def test_fuse_leftover(self):
        a = Tensor(np.random.rand(2, 3, 4, 5, 2, 2), 'abcdef', tags={'blue'})
        b = a.fuse({'bra': 'ac', 'ket': 'bd'})
        assert b.shape == (8, 15, 2, 2)
        assert b.inds == ('bra', 'ket', 'e', 'f')
        assert b.tags == {'blue'}

    def test_tensor_transpose(self):
        a = Tensor(np.random.rand(2, 3, 4, 5, 2, 2), 'abcdef', tags={'blue'})
        at = a.transpose(*'cdfeba')
        assert at.shape == (4, 5, 2, 2, 3, 2)
        assert at.inds == ('c', 'd', 'f', 'e', 'b', 'a')

        with pytest.raises(ValueError):
            a.transpose(*'cdfebz')


class TestTensorFunctions:
    @pytest.mark.parametrize('method', ['svd', 'eig'])
    @pytest.mark.parametrize('linds', ['abd', 'ce'])
    @pytest.mark.parametrize('cutoff', [-1.0, 1e-13, 1e-10])
    @pytest.mark.parametrize('cutoff_mode', ['abs', 'rel', 'sum2'])
    @pytest.mark.parametrize('absorb', ['left', 'both', 'right'])
    def test_split_tensor_with_vals(self, method, linds, cutoff,
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
    @pytest.mark.parametrize('linds', ['abd', 'ce'])
    def test_split_tensor_no_vals(self, method, linds):
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

    @pytest.mark.parametrize('method', ['svd', 'eig'])
    def test_entropy(self, method):
        psim = Tensor(np.eye(2) * 2**-0.5, inds='ab')
        assert_allclose(psim.H @ psim, 1.0)
        assert_allclose(psim.entropy('a', method=method)**2, 1)

    @pytest.mark.parametrize('method', ['svd', 'eig'])
    def test_entropy_matches_dense(self, method):
        p = MPS_rand_state(5, 32)
        p_dense = p.to_dense()
        real_svn = entropy(p_dense.ptr([2] * 5, [0, 1, 2]))

        svn = (p ^ ...).entropy(('k0', 'k1', 'k2'))
        assert_allclose(real_svn, svn)

        # use tensor to left of bipartition
        p.canonize(2)
        t1 = p['i2']
        left_inds = set(t1.inds) - set(p['i3'].inds)
        svn = (t1).entropy(left_inds, method=method)
        assert_allclose(real_svn, svn)

        # use tensor to right of bipartition
        p.canonize(3)
        t2 = p['i3']
        left_inds = set(t2.inds) & set(p['i2'].inds)
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
                                       np.float_, 'raise'])
    def test_rand_tensor(self, dtype):
        if dtype == 'raise':
            with pytest.raises(TypeError):
                rand_tensor((2, 3, 4), 'abc', dtype=dtype)
        else:
            t = rand_tensor((2, 3, 4), 'abc', dtype=dtype)
            assert t.dtype == dtype


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
        tn2['t1'].inds = ('a', 'b', 'X')
        assert tn1['t1'] is not tn2['t1']
        assert tn2['t1'].inds == ('a', 'b', 'X')
        assert tn1['t1'].inds == ('a', 'b', 'd')
        # but that data remains the same
        assert tn1['t1'].data is tn2['t1'].data
        tn2['t1'].data /= 2
        assert_allclose(tn1['t1'].data, tn2['t1'].data)

    def test_copy_deep(self):
        a = rand_tensor((2, 3, 4), inds='abc', tags='t0')
        b = rand_tensor((2, 3, 4), inds='abd', tags='t1')
        tn1 = TensorNetwork((a, b))
        tn2 = tn1.copy(deep=True)
        # check can modify tensor structure
        tn2['t1'].inds = ('a', 'b', 'X')
        assert tn1['t1'] is not tn2['t1']
        assert tn2['t1'].inds == ('a', 'b', 'X')
        assert tn1['t1'].inds == ('a', 'b', 'd')
        # and that data is not the same
        assert tn1['t1'].data is not tn2['t1'].data
        tn2['t1'].data /= 2
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
        # assert len(tn['foo']) == 2
        with pytest.raises(KeyError):
            tn['foo'] = c

        tn[('foo', 'blue')] = c
        assert 'c' in tn.tags
        assert tn[('blue', 'c')] is c

        del tn['red']
        assert 'red' not in tn.tags

        assert set(tn.tag_index.keys()) == {'blue', 'c'}

        tn.drop_tags('c')
        assert set(tn.tag_index.keys()) == {'blue'}
        tn.drop_tags(['blue'])
        assert set(tn.tag_index.keys()) == set()

    def test_conj(self):
        a_data = np.random.randn(2, 3, 4) + 1.0j * np.random.randn(2, 3, 4)
        b_data = np.random.randn(3, 4, 5) + 1.0j * np.random.randn(3, 4, 5)
        c_data = np.random.randn(5, 2, 6) + 1.0j * np.random.randn(5, 2, 6)

        a = Tensor(a_data, inds=[0, 1, 2], tags={'red', 0})
        b = Tensor(b_data, inds=[1, 2, 3], tags={'blue', 1})
        c = Tensor(c_data, inds=[3, 0, 4], tags={'blue', 2})

        tn = a & b & c
        new_tn = tn.conj()

        for i, arr in enumerate((a_data, b_data, c_data)):
            assert_allclose(new_tn[i].data, arr.conj())

        # make sure original network unchanged
        for i, arr in enumerate((a_data, b_data, c_data)):
            assert_allclose(tn[i].data, arr)

    def test_conj_inplace(self):
        a_data = np.random.randn(2, 3, 4) + 1.0j * np.random.randn(2, 3, 4)
        b_data = np.random.randn(3, 4, 5) + 1.0j * np.random.randn(3, 4, 5)
        c_data = np.random.randn(5, 2, 6) + 1.0j * np.random.randn(5, 2, 6)

        a = Tensor(a_data, inds=[0, 1, 2], tags={'red', 'i0'})
        b = Tensor(b_data, inds=[1, 2, 3], tags={'blue', 'i1'})
        c = Tensor(c_data, inds=[3, 0, 4], tags={'blue', 'i2'})

        tn = a & b & c
        tn.conj(inplace=True)

        for i, arr in enumerate((a_data, b_data, c_data)):
            assert_allclose(tn["i{}".format(i)].data, arr.conj())

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
        for tag, names in d2.tag_index.items():
            assert d.tag_index[tag] == names

        # test inplace
        d >>= ['red', 'green', 'blue']
        assert isinstance(d, Tensor)

    def test_contract_with_slices(self):
        a = rand_tensor((2, 3, 4), inds=[0, 1, 2], tags='i0')
        b = rand_tensor((3, 4, 5), inds=[1, 2, 3], tags='i1')
        c = rand_tensor((5, 2, 6), inds=[3, 0, 4], tags='i2')
        d = rand_tensor((5, 2, 6), inds=[5, 6, 4], tags='i3')
        tn = TensorNetwork((a, b, c, d), structure="i{}")

        assert len((tn ^ slice(2)).tensors) == 3
        assert len((tn ^ slice(..., 1)).tensors) == 3
        assert len((tn ^ slice(-1, 0)).tensors) == 2
        assert len((tn ^ slice(None, -2)).tensors) == 3
        assert len((tn ^ slice(-2, ...)).tensors) == 3

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

        d = a_b_c.reindex({4: 'foo', 2: 'bar'}, inplace=True)

        assert a_b_c.outer_inds() == ('foo',)
        assert set(d.inner_inds()) == {0, 1, 'bar', 3}
        assert d.tensors[0].inds == (0, 1, 'bar')

    def test_add_tag(self):
        a = rand_tensor((2, 3, 4), inds='abc', tags={'red'})
        b = rand_tensor((2, 3, 4), inds='abc', tags={'blue'})
        tn = a & b
        tn.add_tag('green')
        assert 'green' in tn.tag_index
        assert 'green' in tn['red'].tags
        assert 'green' in tn['blue'].tags
        tn.add_tag('blue')
        for t in tn.tensors:
            assert 'blue' in t.tags

    def test_index_by_site(self):
        a_data = np.random.randn(2, 3, 4)
        b_data = np.random.randn(2, 3, 4)
        a = Tensor(a_data, inds='abc', tags={'i0'})
        b = Tensor(b_data, inds='abc', tags={'i1'})
        tn = TensorNetwork((a, b), structure="i{}")
        assert_allclose(tn.site[0].data, a_data)
        new_data = np.random.randn(2, 3, 4)
        tn.site[1] = Tensor(new_data, inds='abc', tags={'i1', 'red'})
        assert_allclose(tn['i1'].data, new_data)
        assert 'red' in tn['i1'].tags

    def test_set_data_in_tensor(self):
        a_data = np.random.randn(2, 3, 4)
        b_data = np.random.randn(2, 3, 4)
        a = Tensor(a_data, inds='abc', tags={'i0'})
        b = Tensor(b_data, inds='abc', tags={'i1'})
        tn = TensorNetwork((a, b), structure="i{}")
        assert_allclose(tn.site[0].data, a_data)
        new_data = np.random.randn(24)
        tn.site[1].data = new_data
        assert_allclose(tn['i1'].data, new_data.reshape(2, 3, 4))

    def test_combining_with_no_check_collisions(self):
        p1 = MPS_rand_state(5, 3, phys_dim=3)
        p2 = MPS_rand_state(5, 3, phys_dim=3)
        # shouldn't need to check any collisions
        tn = TensorNetwork((p1, p2), check_collisions=False)
        # test can contract
        assert 0 < abs(tn ^ ...) < 1


class TestMatrixProductState:

    def test_matrix_product_state(self):
        tensors = ([np.random.rand(5, 2)] +
                   [np.random.rand(5, 5, 2) for _ in range(3)] +
                   [np.random.rand(5, 2)])
        mps = MatrixProductState(tensors)
        assert len(mps.tensors) == 5
        nmps = mps.reindex_sites('foo{}', inplace=False, where=slice(0, 3))
        assert nmps.site_ind_id == "k{}"
        assert isinstance(nmps, MatrixProductState)
        assert set(nmps.outer_inds()) == {'foo0', 'foo1',
                                          'foo2', 'k3', 'k4'}
        assert set(mps.outer_inds()) == {'k0', 'k1',
                                         'k2', 'k3', 'k4'}
        mps.site_ind_id = 'foo{}'
        assert set(mps.outer_inds()) == {'foo0', 'foo1',
                                         'foo2', 'foo3', 'foo4'}
        assert mps.site_inds == ('foo0', 'foo1', 'foo2', 'foo3', 'foo4')
        assert mps.site_ind_id == 'foo{}'
        mps.plot()

    @pytest.mark.parametrize("dtype", [float, complex, np.complex128,
                                       np.float64, 'raise'])
    def test_rand_mps_dtype(self, dtype):
        if dtype == 'raise':
            with pytest.raises(TypeError):
                MPS_rand_state(10, 7, dtype=dtype)
        else:
            p = MPS_rand_state(10, 7, dtype=dtype)
            assert p.site[0].dtype == dtype
            assert p.site[7].dtype == dtype

    def test_left_canonize_site(self):
        a = np.random.randn(7, 2) + 1.0j * np.random.randn(7, 2)
        b = np.random.randn(7, 7, 2) + 1.0j * np.random.randn(7, 7, 2)
        c = np.random.randn(7, 2) + 1.0j * np.random.randn(7, 2)
        mps = MatrixProductState([a, b, c], site_tag_id="i{}")

        mps.left_canonize_site(0)
        assert mps['i0'].shape == (2, 2)
        assert mps['i0'].tags == {'i0'}
        assert mps['i1'].tags == {'i1'}

        U = (mps['i0'].data)
        assert_allclose(U.conj().T @ U, np.eye(2), atol=1e-13)
        assert_allclose(U @ U.conj().T, np.eye(2), atol=1e-13)

        # combined two site contraction is identity also
        mps.left_canonize_site(1)
        ptn = (mps.H & mps) ^ ['i0', 'i1']
        assert_allclose(ptn['i1'].data, np.eye(4), atol=1e-13)

        # try normalizing the state
        mps['i2'] /= mps['i2'].norm()

        assert_allclose(abs(mps.H @ mps), 1.0)

    def test_right_canonize_site(self):
        a = np.random.randn(7, 2) + 1.0j * np.random.randn(7, 2)
        b = np.random.randn(7, 7, 2) + 1.0j * np.random.randn(7, 7, 2)
        c = np.random.randn(7, 2) + 1.0j * np.random.randn(7, 2)
        mps = MatrixProductState([a, b, c], site_tag_id="i{}")

        mps.right_canonize_site(2)
        assert mps['i2'].shape == (2, 2)
        assert mps['i2'].tags == {'i2'}
        assert mps['i1'].tags == {'i1'}

        U = (mps['i2'].data)
        assert_allclose(U.conj().T @ U, np.eye(2), atol=1e-13)
        assert_allclose(U @ U.conj().T, np.eye(2), atol=1e-13)

        # combined two site contraction is identity also
        mps.right_canonize_site(1)
        ptn = (mps.H & mps) ^ ['i1', 'i2']
        assert_allclose(ptn['i1'].data, np.eye(4), atol=1e-13)

        # try normalizing the state
        mps['i0'] /= mps['i0'].norm()

        assert_allclose(mps.H @ mps, 1)

    def test_rand_mps_left_canonize(self):
        n = 10
        k = MPS_rand_state(n, 10, site_tag_id="foo{}",
                           tags='bar', normalize=False)
        k.left_canonize(normalize=True)

        assert k.count_canonized() == (9, 0)

        assert_allclose(k.H @ k, 1)
        p_tn = (k.H & k) ^ slice(0, 9)
        assert_allclose(p_tn['foo8'].data, np.eye(10), atol=1e-13)

    def test_rand_mps_left_canonize_with_bra(self):
        n = 10
        k = MPS_rand_state(n, 10, site_tag_id="foo{}",
                           tags='bar', normalize=False)
        b = k.H
        k.left_canonize(normalize=True, bra=b)
        assert_allclose(b @ k, 1)
        p_tn = (b & k) ^ slice(0, 9)
        assert_allclose(p_tn['foo8'].data, np.eye(10), atol=1e-13)

    def test_rand_mps_right_canonize(self):
        n = 10
        k = MPS_rand_state(n, 10, site_tag_id="foo{}",
                           tags='bar', normalize=False)
        k.right_canonize(normalize=True)
        assert_allclose(k.H @ k, 1)
        p_tn = (k.H & k) ^ slice(..., 0)
        assert_allclose(p_tn['foo1'].data, np.eye(10), atol=1e-13)

    def test_rand_mps_right_canonize_with_bra(self):
        n = 10
        k = MPS_rand_state(n, 10, site_tag_id="foo{}",
                           tags='bar', normalize=False)
        b = k.H
        k.right_canonize(normalize=True, bra=b)
        assert_allclose(b @ k, 1)
        p_tn = (b & k) ^ slice(..., 0)
        assert_allclose(p_tn['foo1'].data, np.eye(10), atol=1e-13)

    def test_rand_mps_mixed_canonize(self):
        n = 10
        rmps = MPS_rand_state(n, 10, site_tag_id="foo{}",
                              tags='bar', normalize=True)

        # move to the center
        rmps.canonize(orthogonality_center=4)
        assert rmps.count_canonized() == (4, 5)
        assert_allclose(rmps.H @ rmps, 1)
        p_tn = (rmps.H & rmps) ^ slice(0, 4) ^ slice(..., 4)
        assert_allclose(p_tn['foo3'].data, np.eye(10), atol=1e-13)
        assert_allclose(p_tn['foo5'].data, np.eye(10), atol=1e-13)

        # try shifting to the right
        rmps.shift_orthogonality_center(current=4, new=8)
        assert_allclose(rmps.H @ rmps, 1)
        p_tn = (rmps.H & rmps) ^ slice(0, 8) ^ slice(..., 8)
        assert_allclose(p_tn['foo7'].data, np.eye(4), atol=1e-13)
        assert_allclose(p_tn['foo9'].data, np.eye(2), atol=1e-13)

        # try shifting to the left
        rmps.shift_orthogonality_center(current=8, new=6)
        assert_allclose(rmps.H @ rmps, 1)
        p_tn = (rmps.H & rmps) ^ slice(0, 6) ^ slice(..., 6)
        assert_allclose(p_tn['foo5'].data, np.eye(10), atol=1e-13)
        assert_allclose(p_tn['foo7'].data, np.eye(8), atol=1e-13)

    def test_can_change_data(self):
        p = MPS_rand_state(3, 10)
        assert_allclose(p.H @ p, 1)
        p.site[1].data = np.random.randn(200)
        assert abs(p.H @ p - 1) > 1e-13

    def test_can_change_data_using_subnetwork(self):
        p = MPS_rand_state(3, 10)
        pH = p.H
        p.add_tag('__ket__')
        pH.add_tag('__bra__')
        tn = p | pH
        assert_allclose((tn ^ ...), 1)
        assert_allclose(tn[('__ket__', 'i1')].data,
                        tn[('__bra__', 'i1')].data.conj())
        p.site[1].data = np.random.randn(200)
        assert abs((tn ^ ...) - 1) > 1e-13
        assert not np.allclose(tn[('__ket__', 'i1')].data,
                               tn[('__bra__', 'i1')].data.conj())

    def test_adding_mps(self):
        p = MPS_rand_state(10, 7)
        assert max(p['i4'].shape) == 7
        p2 = p + p
        assert max(p2['i4'].shape) == 14
        assert_allclose(p2.H @ p, 2)
        p += p
        assert max(p['i4'].shape) == 14
        assert_allclose(p.H @ p, 4)

    @pytest.mark.parametrize("method", ['svd', 'eig'])
    @pytest.mark.parametrize('cutoff_mode', ['abs', 'rel', 'sum2'])
    def test_compress_mps(self, method, cutoff_mode):
        n = 10
        chi = 7
        p = MPS_rand_state(n, chi)
        assert max(p['i4'].shape) == chi
        p2 = p + p
        assert max(p2['i4'].shape) == chi * 2
        assert_allclose(p2.H @ p, 2)
        p2.left_compress(method=method, cutoff=1e-6, cutoff_mode=cutoff_mode)
        assert max(p2['i4'].shape) == chi
        assert_allclose(p2.H @ p, 2)
        assert p2.count_canonized() == (n - 1, 0)

    def test_compress_mps_right(self):
        p = MPS_rand_state(10, 7)
        assert max(p['i4'].shape) == 7
        p2 = p + p
        assert max(p2['i4'].shape) == 14
        assert_allclose(p2.H @ p, 2)
        p2.right_compress()
        assert max(p2['i4'].shape) == 7
        assert_allclose(p2.H @ p, 2)

    @pytest.mark.parametrize("method", ['svd', 'eig'])
    def test_compress_trim_max_bond(self, method):
        p = MPS_rand_state(20, 20)
        p.compress(method=method)
        assert max(p['i4'].shape) == 20
        p.compress(max_bond=13, method=method)
        assert max(p['i4'].shape) == 13
        assert p.H @ p < 1.0

    def test_compress_form(self):
        p = MPS_rand_state(20, 20)
        p.compress('left')
        assert p.count_canonized() == (19, 0)
        p.compress('right')
        assert p.count_canonized() == (0, 19)
        p.compress(7)
        assert p.count_canonized() == (7, 12)
        p = MPS_rand_state(20, 20)
        p.compress('flat', absorb='left')
        assert p.count_canonized() == (0, 0)

    @pytest.mark.parametrize("method", ['svd', 'eig'])
    @pytest.mark.parametrize("form", ['left', 'right', 'raise'])
    def test_add_and_compress_mps(self, method, form):
        p = MPS_rand_state(10, 7)
        assert max(p['i4'].shape) == 7

        if form == 'raise':
            with pytest.raises(ValueError):
                p.add_MPS(p, compress=True, method=method,
                          form=form, cutoff=1e-6)
            return

        p2 = p.add_MPS(p, compress=True, method=method, form=form, cutoff=1e-6)
        assert max(p2['i4'].shape) == 7
        assert_allclose(p2.H @ p, 2)

    def test_adding_mpo(self):
        h = MPO_ham_heis(6)
        hd = h.to_dense()
        assert_allclose(h @ h.H, (hd @ hd.H).tr())
        h2 = h + h
        assert_allclose(h2 @ h2.H, (hd @ hd.H).tr() * 4)
        h2.right_compress()
        assert_allclose(h2 @ h2.H, (hd @ hd.H).tr() * 4)
        assert max(h2['i3'].shape) == 5

    def test_schmidt_values_entropy_gap_simple(self):
        n = 12
        p = MPS_rand_state(n, 16)
        p.right_canonize()
        svns = []
        sgs = []
        for i in range(1, n):
            sgs.append(p.schmidt_gap(i, current_orthog_centre=i - 1))
            svns.append(p.entropy(i, current_orthog_centre=i))

        pd = p.to_dense()
        ex_svns = [entropy_subsys(pd, [2] * n, range(i)) for i in range(1, n)]
        ex_sgs = [schmidt_gap(pd, [2] * n, range(i)) for i in range(1, n)]
        assert_allclose(ex_svns, svns)
        assert_allclose(ex_sgs, sgs)


class TestMatrixProductOperator:

    def test_matrix_product_operator(self):
        tensors = ([np.random.rand(5, 2, 2)] +
                   [np.random.rand(5, 5, 2, 2) for _ in range(3)] +
                   [np.random.rand(5, 2, 2)])
        mpo = MatrixProductOperator(tensors)
        mpo.plot()
        assert len(mpo.tensors) == 5
        assert mpo.upper_inds == ('k0', 'k1', 'k2', 'k3', 'k4')
        assert mpo.lower_inds == ('b0', 'b1', 'b2', 'b3', 'b4')
        op = mpo ^ ...
        # this would rely on left to right contraction if not in set form
        assert set(op.inds) == {'k0', 'b0', 'k1', 'b1', 'k2', 'b2',
                                'k3', 'b3', 'k4', 'b4'}

    def test_add_mpo(self):
        h = MPO_ham_heis(12)
        h2 = h + h
        assert max(h2.site[6].shape) == 10
        h.lower_ind_id = h.upper_ind_id
        t = h ^ ...
        h2.upper_ind_id = h2.lower_ind_id
        t2 = h2 ^ ...
        assert_allclose(2 * t, t2)

    def test_expand_mpo(self):
        h = MPO_ham_heis(12)
        assert h.site[0].dtype == float
        he = h.expand_bond_dimension(13)
        assert h.site[0].dtype == float
        assert max(he.site[6].shape) == 13
        h.lower_ind_id = h.upper_ind_id
        t = h ^ ...
        he.upper_ind_id = he.lower_ind_id
        te = he ^ ...
        assert_allclose(t, te)

    def test_expand_mpo_limited(self):
        h = MPO_ham_heis(12)
        he = h.expand_bond_dimension(3)  # should do nothing
        assert max(he.site[6].shape) == 5

    def test_mpo_identity(self):
        k = MPS_rand_state(13, 7)
        b = MPS_rand_state(13, 7)
        o1 = k @ b
        i = MPO_identity(13)
        k, i, b = align_TN_1D(k, i, b)
        o2 = (k & i & b) ^ ...
        assert_allclose(o1, o2)

    @pytest.mark.parametrize("dtype", (complex, float))
    def test_mpo_rand_herm_and_trace(self, dtype):
        op = MPO_rand_herm(20, bond_dim=5, phys_dim=3, dtype=dtype)
        assert_allclose(op.H @ op, 1.0)
        tr_val = op.trace()
        assert tr_val != 0.0
        assert_allclose(tr_val.imag, 0.0, atol=1e-14)

    def test_mpo_rand_herm_trace_and_identity_like(self):
        op = MPO_rand_herm(20, bond_dim=5, phys_dim=3, upper_ind_id='foo{}')
        t = op.trace()
        assert t != 0.0
        Id = MPO_identity_like(op)
        assert_allclose(Id.trace(), 3**20)
        Id.site[0] *= 3 / 3**20
        op += Id
        assert_allclose(op.trace(), t + 3)


class TestSpecificStatesOperators:

    def test_rand_ket_mps(self):
        n = 10
        rmps = MPS_rand_state(n, 10, site_tag_id="foo{}", tags='bar')
        assert rmps.site[0].tags == {'foo0', 'bar'}
        assert rmps.site[3].tags == {'foo3', 'bar'}
        assert rmps.site[-1].tags == {'foo9', 'bar'}

        rmpsH_rmps = rmps.H & rmps
        assert len(rmpsH_rmps.tag_index['foo0']) == 2
        assert len(rmpsH_rmps.tag_index['bar']) == n * 2

        assert_allclose(rmps.H @ rmps, 1)
        c = (rmps.H & rmps) ^ slice(0, 5) ^ slice(9, 4, -1) ^ slice(4, 6)
        assert_allclose(c, 1)

    def test_mps_computation_state(self):
        p = MPS_neel_state(10)
        pd = neel_state(10)
        assert_allclose(p.to_dense(), pd)

    def test_zero_state(self):
        z = MPS_zero_state(21, 7)
        p = MPS_rand_state(21, 13)
        assert_allclose(p.H @ z, 0.0)
        assert_allclose(p.H @ p, 1.0)
        zp = z + p
        assert max(zp.site[13].shape) == 20
        assert_allclose(zp.H @ p, 1.0)

    def test_mpo_site_ham_heis(self):
        hh_mpo = MPO_ham_heis(5, tags=['foo'])
        assert hh_mpo.site[0].tags == {'i0', 'foo'}
        assert hh_mpo.site[3].tags == {'i3', 'foo'}
        assert hh_mpo.site[-1].tags == {'i4', 'foo'}
        assert hh_mpo.shape == (2,) * 10
        hh_ = (hh_mpo ^ ...).fuse({'k': ['k0', 'k1', 'k2', 'k3', 'k4'],
                                   'b': ['b0', 'b1', 'b2', 'b3', 'b4']})
        hh = ham_heis(5, cyclic=False) / 4  # /4 :ham_heis uses paulis not spin
        assert_allclose(hh, hh_.data)


class TestMovingEnvironment:
    def test_bsz1_start_left(self):
        tn = MPS_rand_state(6, bond_dim=7)
        env = MovingEnvironment(tn, n=6, start='left', bsz=1)
        assert env.pos == 0
        assert len(env().tensors) == 2
        env.move_right()
        assert env.pos == 1
        assert len(env().tensors) == 3
        env.move_right()
        assert env.pos == 2
        assert len(env().tensors) == 3
        env.move_to(5)
        assert env.pos == 5
        assert len(env().tensors) == 2

    def test_bsz1_start_right(self):
        tn = MPS_rand_state(6, bond_dim=7)
        env = MovingEnvironment(tn, n=6, start='right', bsz=1)
        assert env.pos == 5
        assert len(env().tensors) == 2
        env.move_left()
        assert env.pos == 4
        assert len(env().tensors) == 3
        env.move_left()
        assert env.pos == 3
        assert len(env().tensors) == 3
        env.move_to(0)
        assert env.pos == 0
        assert len(env().tensors) == 2

    def test_bsz2_start_left(self):
        tn = MPS_rand_state(6, bond_dim=7)
        env = MovingEnvironment(tn, n=6, start='left', bsz=2)
        assert len(env().tensors) == 3
        env.move_right()
        assert len(env().tensors) == 4
        env.move_right()
        assert len(env().tensors) == 4
        with pytest.raises(ValueError):
            env.move_to(5)
        env.move_to(4)
        assert env.pos == 4
        assert len(env().tensors) == 3

    def test_bsz2_start_right(self):
        tn = MPS_rand_state(6, bond_dim=7)
        env = MovingEnvironment(tn, n=6, start='right', bsz=2)
        assert env.pos == 4
        assert len(env().tensors) == 3
        env.move_left()
        assert env.pos == 3
        assert len(env().tensors) == 4
        env.move_left()
        assert env.pos == 2
        assert len(env().tensors) == 4
        with pytest.raises(ValueError):
            env.move_to(-1)
        env.move_to(0)
        assert env.pos == 0
        assert len(env().tensors) == 3


class TestDMRG1:

    def test_single_explicit_sweep(self):
        h = MPO_ham_heis(5)
        dmrg = DMRG1(h, bond_dims=3)
        assert dmrg._k.site[0].dtype == float

        energy_tn = (dmrg._b | dmrg.ham | dmrg._k)

        e0 = energy_tn ^ ...
        assert abs(e0.imag) < 1e-13

        de1 = dmrg.sweep_right()
        e1 = energy_tn ^ ...
        assert_allclose(de1, e1)
        assert abs(e1.imag) < 1e-13

        de2 = dmrg.sweep_right()
        e2 = energy_tn ^ ...
        assert_allclose(de2, e2)
        assert abs(e2.imag) < 1e-13

        # state is already left canonized after right sweep
        de3 = dmrg.sweep_left(canonize=False)
        e3 = energy_tn ^ ...
        assert_allclose(de3, e3)
        assert abs(e2.imag) < 1e-13

        de4 = dmrg.sweep_left()
        e4 = energy_tn ^ ...
        assert_allclose(de4, e4)
        assert abs(e2.imag) < 1e-13

        # test still normalized
        assert dmrg._k.site[0].dtype == float
        align_TN_1D(dmrg._k, dmrg._b, inplace=True)
        assert_allclose(abs(dmrg._b @ dmrg._k), 1)

        assert e1.real < e0.real
        assert e2.real < e1.real
        assert e3.real < e2.real
        assert e4.real < e3.real

    @pytest.mark.parametrize("dense", [False, True])
    @pytest.mark.parametrize("MPO_ham", [MPO_ham_XY, MPO_ham_heis])
    def test_ground_state_matches(self, dense, MPO_ham):
        h = MPO_ham(6)
        dmrg = DMRG1(h, bond_dims=8)
        dmrg.opts['eff_eig_dense'] = dense
        assert dmrg.solve()
        eff_e, mps_gs = dmrg.energy, dmrg.state
        mps_gs_dense = mps_gs.to_dense()

        assert_allclose(mps_gs_dense.H @ mps_gs_dense, 1.0)

        h_dense = h.to_dense()

        # check against dense form
        actual_e, gs = seigsys(h_dense, k=1)
        assert_allclose(actual_e, eff_e)
        assert_allclose(abs(expec(mps_gs_dense, gs)), 1.0)

        # check against actual MPO_ham
        if MPO_ham is MPO_ham_XY:
            ham_dense = ham_heis(6, cyclic=False, j=(1.0, 1.0, 0.0)) / 4
        elif MPO_ham is MPO_ham_heis:
            ham_dense = ham_heis(6, cyclic=False) / 4

        actual_e, gs = seigsys(ham_dense, k=1)
        assert_allclose(actual_e, eff_e)
        assert_allclose(abs(expec(mps_gs_dense, gs)), 1.0)

    def test_ising_and_MPS_product_state(self):
        h = MPO_ham_ising(6, bx=2.0, j=0.1)
        dmrg = DMRG1(h, bond_dims=8)
        assert dmrg.solve()
        eff_e, mps_gs = dmrg.energy, dmrg.state
        mps_gs_dense = mps_gs.to_dense()
        assert_allclose(mps_gs_dense.H @ mps_gs_dense, 1.0)

        # check against dense
        h_dense = h.to_dense()
        actual_e, gs = seigsys(h_dense, k=1)
        assert_allclose(actual_e, eff_e)
        assert_allclose(abs(expec(mps_gs_dense, gs)), 1.0)

        exp_gs = MPS_product_state([plus()] * 6)
        assert_allclose(abs(exp_gs.H @ mps_gs), 1.0, rtol=1e-3)


class TestDMRG2:
    @pytest.mark.parametrize("dense", [False, True])
    @pytest.mark.parametrize("MPO_ham", [MPO_ham_XY, MPO_ham_heis])
    def test_matches_exact(self, dense, MPO_ham):
        h = MPO_ham(6)
        dmrg = DMRG2(h, bond_dims=8)
        assert dmrg._k.site[0].dtype == float
        dmrg.opts['eff_eig_dense'] = dense
        assert dmrg.solve()

        # XXX: need to dispatch SLEPc seigsys on real input
        # assert dmrg._k.site[0].dtype == float

        eff_e, mps_gs = dmrg.energy, dmrg.state
        mps_gs_dense = mps_gs.to_dense()

        assert_allclose(mps_gs_dense.H @ mps_gs_dense, 1.0)

        h_dense = h.to_dense()

        # check against dense form
        actual_e, gs = seigsys(h_dense, k=1)
        assert_allclose(actual_e, eff_e)
        assert_allclose(abs(expec(mps_gs_dense, gs)), 1.0)

        # check against actual MPO_ham
        if MPO_ham is MPO_ham_XY:
            ham_dense = ham_heis(6, cyclic=False, j=(1.0, 1.0, 0.0)) / 4
        elif MPO_ham is MPO_ham_heis:
            ham_dense = ham_heis(6, cyclic=False) / 4

        actual_e, gs = seigsys(ham_dense, k=1)
        assert_allclose(actual_e, eff_e)
        assert_allclose(abs(expec(mps_gs_dense, gs)), 1.0)


class TestDMRGX:

    def test_explicit_sweeps(self):
        # import pdb; pdb.set_trace()
        n = 8
        chi = 16
        ham = MPO_ham_mbl(n, dh=5, run=42)
        p0 = MPS_neel_state(n).expand_bond_dimension(chi)

        b0 = p0.H
        align_TN_1D(p0, ham, b0, inplace=True)
        en0 = np.asscalar(p0 & ham & b0 ^ ...)

        dmrgx = DMRGX(ham, p0, chi)
        dmrgx.sweep_right()
        en1 = dmrgx.sweep_left(canonize=False)
        assert en0 != en1

        dmrgx.sweep_right(canonize=False)
        en = dmrgx.sweep_right(canonize=True)

        # check normalized
        assert_allclose(dmrgx._k.H @ dmrgx._k, 1.0)

        k = dmrgx._k.to_dense()
        h = ham.to_dense()
        el, ev = eigsys(h)

        # check variance very low
        assert np.abs((k.H @ h @ h @ k) - (k.H @ h @ k)**2) < 1e-12

        # check exactly one eigenvalue matched well
        assert np.sum(np.abs(el - en) < 1e-12) == 1

        # check exactly one eigenvector is matched with high fidelity
        ovlps = (ev.H @ k).A**2
        big_ovlps = ovlps[ovlps > 1e-12]
        assert_allclose(big_ovlps, [1])

        # check fully
        assert is_eigenvector(k, h)

    def test_solve_bigger(self):
        n = 14
        chi = 16
        ham = MPO_ham_mbl(n, dh=8, run=42)
        p0 = MPS_computational_state('00110111000101')
        dmrgx = DMRGX(ham, p0, chi)
        assert dmrgx.solve(tol=1e-5, sweep_sequence='R')
        assert dmrgx.state.site[0].dtype == float
