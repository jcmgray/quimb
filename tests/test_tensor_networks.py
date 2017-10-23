import pytest
import operator

import numpy as np
from numpy.testing import assert_allclose

from quimb import ham_heis, qu, expec, seigsys

from quimb.tensor_networks import (
    Tensor,
    tensor_contract,
    TensorNetwork,
    MatrixProductState,
    MatrixProductOperator,
    MPS_rand,
    MPO_ham_heis,
    rand_tensor,
    dmrg1_sweep,
    dmrg1,
)


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
        array = np.random.rand(2, 3, 4) + 1.0j * np.random.rand(2, 3, 4)
        a = Tensor(array, inds=[0, 1, 2], tags='blue')
        a.conj(inplace=True)
        assert_allclose(array.conj(), a.data)

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
        assert b.shape == (15, 8)
        assert b.inds == ('ket', 'bra')
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
    @pytest.mark.parametrize('method', ['svd', 'eig', 'qr', 'lq'])
    @pytest.mark.parametrize('linds', ['abd', 'ce'])
    @pytest.mark.parametrize('tol', [-1.0, 1e-13])
    def test_split_tensor_full_svd(self, method, linds, tol):
        a = rand_tensor((2, 3, 4, 5, 6), inds='abcde', tags='red')
        a_split = a.split(linds, method=method, tol=tol)
        assert len(a_split.tensors) == 2
        if linds == 'abd':
            assert ((a_split.shape == (2, 3, 5, 4, 6)) or
                    (a_split.shape == (4, 6, 2, 3, 5)))
        elif linds == 'edc':
            assert ((a_split.shape == (6, 5, 4, 2, 3)) or
                    (a_split.shape == (2, 3, 6, 5, 4)))
        assert (a_split ^ ...).almost_equals(a)


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

    def test_TensorNetwork_init_checks(self):
        a = Tensor(np.random.randn(2, 3, 4), inds=[0, 1, 2], tags='red')
        b = Tensor(np.random.randn(3, 4, 5), inds=[1, 2, 3], tags='blue')

        with pytest.raises(TypeError):
            TensorNetwork(a, b)  # missing brackets around ``a, b``.

    def test_conj(self):
        a_array = np.random.randn(2, 3, 4) + 1.0j * np.random.randn(2, 3, 4)
        b_array = np.random.randn(3, 4, 5) + 1.0j * np.random.randn(3, 4, 5)
        c_array = np.random.randn(5, 2, 6) + 1.0j * np.random.randn(5, 2, 6)

        a = Tensor(a_array, inds=[0, 1, 2], tags={'red', 0})
        b = Tensor(b_array, inds=[1, 2, 3], tags={'blue', 1})
        c = Tensor(c_array, inds=[3, 0, 4], tags={'blue', 2})

        tn = a & b & c
        new_tn = tn.conj()

        for i, arr in enumerate((a_array, b_array, c_array)):
            assert_allclose(new_tn[i].data, arr.conj())

        # make sure original network unchanged
        for i, arr in enumerate((a_array, b_array, c_array)):
            assert_allclose(tn[i].data, arr)

    def test_conj_inplace(self):
        a_array = np.random.randn(2, 3, 4) + 1.0j * np.random.randn(2, 3, 4)
        b_array = np.random.randn(3, 4, 5) + 1.0j * np.random.randn(3, 4, 5)
        c_array = np.random.randn(5, 2, 6) + 1.0j * np.random.randn(5, 2, 6)

        a = Tensor(a_array, inds=[0, 1, 2], tags={'red', 'i0'})
        b = Tensor(b_array, inds=[1, 2, 3], tags={'blue', 'i1'})
        c = Tensor(c_array, inds=[3, 0, 4], tags={'blue', 'i2'})

        tn = a & b & c
        tn.conj(inplace=True)

        for i, arr in enumerate((a_array, b_array, c_array)):
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
        a_array = np.random.randn(2, 3, 4)
        b_array = np.random.randn(2, 3, 4)
        a = Tensor(a_array, inds='abc', tags={'i0'})
        b = Tensor(b_array, inds='abc', tags={'i1'})
        tn = TensorNetwork((a, b), contract_strategy="i{}")
        assert_allclose(tn.site[0].data, a_array)
        new_array = np.random.randn(2, 3, 4)
        tn.site[1] = Tensor(new_array, inds='abc', tags={'i1', 'red'})
        assert_allclose(tn['i1'].data, new_array)
        assert 'red' in tn['i1'].tags

    def test_set_data_in_tensor(self):
        a_array = np.random.randn(2, 3, 4)
        b_array = np.random.randn(2, 3, 4)
        a = Tensor(a_array, inds='abc', tags={'i0'})
        b = Tensor(b_array, inds='abc', tags={'i1'})
        tn = TensorNetwork((a, b), contract_strategy="i{}")
        assert_allclose(tn.site[0].data, a_array)
        new_array = np.random.randn(24)
        tn.site[1].data = new_array
        assert_allclose(tn['i1'].data, new_array.reshape(2, 3, 4))


class TestMatrixProductState:

    def test_matrix_product_state(self):
        tensors = ([np.random.rand(5, 2)] +
                   [np.random.rand(5, 5, 2) for _ in range(3)] +
                   [np.random.rand(5, 2)])
        mps = MatrixProductState(tensors)
        assert len(mps.tensors) == 5
        nmps = mps.reindex_sites('foo{}', inplace=False, where=slice(0, 3))
        assert nmps.site_inds == "k{}"
        assert isinstance(nmps, MatrixProductState)
        assert set(nmps.outer_inds()) == {'foo0', 'foo1',
                                          'foo2', 'k3', 'k4'}
        assert set(mps.outer_inds()) == {'k0', 'k1',
                                         'k2', 'k3', 'k4'}
        mps.set_site_inds('foo{}')
        assert set(mps.outer_inds()) == {'foo0', 'foo1',
                                         'foo2', 'foo3', 'foo4'}
        assert mps.site_inds == 'foo{}'

    def test_left_canonize_site(self):
        a = np.random.randn(7, 2) + 1.0j * np.random.randn(7, 2)
        b = np.random.randn(7, 7, 2) + 1.0j * np.random.randn(7, 7, 2)
        c = np.random.randn(7, 2) + 1.0j * np.random.randn(7, 2)
        mps = MatrixProductState([a, b, c], site_tags="i{}")

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

        assert abs(mps.H @ mps - 1) < 1e-13

    def test_right_canonize_site(self):
        a = np.random.randn(7, 2) + 1.0j * np.random.randn(7, 2)
        b = np.random.randn(7, 7, 2) + 1.0j * np.random.randn(7, 7, 2)
        c = np.random.randn(7, 2) + 1.0j * np.random.randn(7, 2)
        mps = MatrixProductState([a, b, c], site_tags="i{}")

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

        assert abs(mps.H @ mps - 1) < 1e-13

    def test_rand_mps_left_canonize(self):
        n = 10
        rmps = MPS_rand(n, 10, site_tags="foo{}", tags='bar', normalize=False)
        rmps.left_canonize(normalize=True)
        assert abs(rmps.H @ rmps - 1) < 1e-13
        p_tn = (rmps.H & rmps) ^ slice(0, 9)
        assert_allclose(p_tn['foo8'].data, np.eye(10), atol=1e-13)

    def test_rand_mps_left_canonize_with_bra(self):
        n = 10
        k = MPS_rand(n, 10, site_tags="foo{}", tags='bar', normalize=False)
        b = k.H
        k.left_canonize(normalize=True, bra=b)
        assert abs(b @ k - 1) < 1e-13
        p_tn = (b & k) ^ slice(0, 9)
        assert_allclose(p_tn['foo8'].data, np.eye(10), atol=1e-13)

    def test_rand_mps_right_canonize(self):
        n = 10
        rmps = MPS_rand(n, 10, site_tags="foo{}", tags='bar', normalize=False)
        rmps.right_canonize(normalize=True)
        assert abs(rmps.H @ rmps - 1) < 1e-13
        p_tn = (rmps.H & rmps) ^ slice(..., 0)
        assert_allclose(p_tn['foo1'].data, np.eye(10), atol=1e-13)

    def test_rand_mps_right_canonize_with_bra(self):
        n = 10
        k = MPS_rand(n, 10, site_tags="foo{}", tags='bar', normalize=False)
        b = k.H
        k.right_canonize(normalize=True, bra=b)
        assert abs(b @ k - 1) < 1e-13
        p_tn = (b & k) ^ slice(..., 0)
        assert_allclose(p_tn['foo1'].data, np.eye(10), atol=1e-13)

    def test_rand_mps_mixed_canonize(self):
        n = 10
        rmps = MPS_rand(n, 10, site_tags="foo{}", tags='bar', normalize=True)

        # move to the center
        rmps.canonize(orthogonality_center=4)
        assert abs(rmps.H @ rmps - 1) < 1e-13
        p_tn = (rmps.H & rmps) ^ slice(0, 4) ^ slice(..., 4)
        assert_allclose(p_tn['foo3'].data, np.eye(10), atol=1e-13)
        assert_allclose(p_tn['foo5'].data, np.eye(10), atol=1e-13)

        # try shifting to the right
        rmps.shift_orthogonality_center(current=4, new=8)
        assert abs(rmps.H @ rmps - 1) < 1e-13
        p_tn = (rmps.H & rmps) ^ slice(0, 8) ^ slice(..., 8)
        assert_allclose(p_tn['foo7'].data, np.eye(4), atol=1e-13)
        assert_allclose(p_tn['foo9'].data, np.eye(2), atol=1e-13)

        # try shifting to the left
        rmps.shift_orthogonality_center(current=8, new=6)
        assert abs(rmps.H @ rmps - 1) < 1e-13
        p_tn = (rmps.H & rmps) ^ slice(0, 6) ^ slice(..., 6)
        assert_allclose(p_tn['foo5'].data, np.eye(10), atol=1e-13)
        assert_allclose(p_tn['foo7'].data, np.eye(8), atol=1e-13)

    def test_can_change_data(self):
        p = MPS_rand(3, 10)
        assert abs(p.H @ p - 1) < 1e-13
        p.site[1].data = np.random.randn(200)
        assert abs(p.H @ p - 1) > 1e-13

    def test_can_change_data_using_subnetwork(self):
        p = MPS_rand(3, 10)
        pH = p.H
        p.add_tag('__ket__')
        pH.add_tag('__bra__')
        tn = p & pH
        assert abs((tn ^ ...) - 1) < 1e-13
        assert_allclose(tn[('__ket__', 'i1')].data,
                        tn[('__bra__', 'i1')].data.conj())
        p.site[1].data = np.random.randn(200)
        assert abs((tn ^ ...) - 1) > 1e-13
        assert not np.allclose(tn[('__ket__', 'i1')].data,
                               tn[('__bra__', 'i1')].data.conj())


class TestMatrixProductOperator:

    def test_matrix_product_operator(self):
        tensors = ([np.random.rand(5, 2, 2)] +
                   [np.random.rand(5, 5, 2, 2) for _ in range(3)] +
                   [np.random.rand(5, 2, 2)])
        mpo = MatrixProductOperator(tensors)
        assert len(mpo.tensors) == 5
        op = mpo ^ ...
        # this would rely on left to right contraction if not in set form
        assert set(op.inds) == {'k0', 'b0', 'k1', 'b1', 'k2', 'b2',
                                'k3', 'b3', 'k4', 'b4'}


class TestSpecificStatesOperators:

    def test_rand_ket_mps(self):
        n = 10
        rmps = MPS_rand(n, 10, site_tags="foo{}", tags='bar')
        assert rmps.site[0].tags == {'foo0', 'bar'}
        assert rmps.site[3].tags == {'foo3', 'bar'}
        assert rmps.site[-1].tags == {'foo9', 'bar'}

        rmpsH_rmps = rmps.H & rmps
        assert len(rmpsH_rmps.tag_index['foo0']) == 2
        assert len(rmpsH_rmps.tag_index['bar']) == n * 2

        assert abs(rmps.H @ rmps - 1) < 1e-13
        c = (rmps.H & rmps) ^ slice(0, 5) ^ slice(9, 4, -1) ^ slice(4, 6)
        assert abs(c - 1) < 1e-13

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


class TestDMRG1:

    def test_single_explicit_sweep(self):
        h = MPO_ham_heis(5)

        k = MPS_rand(5, 3)
        b = k.H
        b.set_site_inds(h.bra_site_inds)

        k.add_tag("__ket__")
        b.add_tag("__bra__")
        h.add_tag("__ham__")

        energy_tn = (b & h & k)

        e0 = energy_tn ^ ...
        assert abs(e0.imag) < 1e-13

        de1 = dmrg1_sweep(energy_tn, k, b, direction='right')
        e1 = energy_tn ^ ...
        assert_allclose(de1, e1)
        assert abs(e1.imag) < 1e-13

        de2 = dmrg1_sweep(energy_tn, k, b, direction='right')
        e2 = energy_tn ^ ...
        assert_allclose(de2, e2)
        assert abs(e2.imag) < 1e-13

        de3 = dmrg1_sweep(energy_tn, k, b, direction='left', canonize=False)
        e3 = energy_tn ^ ...
        assert_allclose(de3, e3)
        assert abs(e2.imag) < 1e-13

        de4 = dmrg1_sweep(energy_tn, k, b, direction='left')
        e4 = energy_tn ^ ...
        assert_allclose(de4, e4)
        assert abs(e2.imag) < 1e-13

        # test still normalized
        b.set_site_inds(k.site_inds)
        assert abs(b @ k) - 1 < 1e-13

        assert e1.real < e0.real
        assert e2.real < e1.real
        assert e3.real < e2.real
        assert e4.real < e3.real

    def test_ground_state_matches(self):
        h = MPO_ham_heis(5)
        eff_e, mps_gs = dmrg1(h, 5)
        mps_inds = [mps_gs.site_inds.format(i) for i in range(mps_gs.nsites)]
        mps_gs_dense = qu((mps_gs ^ ...).fuse({'all': mps_inds}).data)

        h_lower_inds = [h.bra_site_inds.format(i) for i in range(h.nsites)]
        h_upper_inds = [h.ket_site_inds.format(i) for i in range(h.nsites)]

        h_dense = (h ^ ...).fuse((('lower', h_lower_inds),
                                  ('upper', h_upper_inds))).data

        actual_e, gs = seigsys(h_dense, k=1)
        assert abs(expec(mps_gs_dense, gs)) - 1 < 1e-12
        assert_allclose(actual_e, eff_e)

        actual_e, gs = seigsys(ham_heis(5, cyclic=False) / 4, k=1)
        assert abs(expec(mps_gs_dense, gs)) - 1 < 1e-12
        assert_allclose(actual_e, eff_e)
