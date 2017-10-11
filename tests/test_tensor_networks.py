import pytest
import operator

import numpy as np
from numpy.testing import assert_allclose

from quimb import ham_heis

from quimb.tensor_networks import (
    Tensor,
    tensor_contract,
    tensor_split,
    TensorNetwork,
    matrix_product_state,
    matrix_product_operator,
    rand_ket_mps,
    ham_heis_mpo,
)


class TestBasicTensorOperations:

    def test_tensor_construct(self):
        x = np.random.randn(2, 3, 4)
        a = Tensor(x, inds=[0, 1, 2], tags='blue')
        assert_allclose(a.H.array, x.conj())
        assert a.size == 24

        with pytest.raises(ValueError):
            Tensor(x, inds=[0, 2], tags='blue')

    def test_with_alpha_construct(self):
        x = np.random.randn(2, 3, 4)
        a = Tensor(x, inds='ijk', tags='blue')
        assert_allclose(a.H.array, x.conj())
        assert a.size == 24

        with pytest.raises(ValueError):
            Tensor(x, inds='ij', tags='blue')

        x = np.random.randn(2, 3, 4)
        a = Tensor(x, inds=['a1', 'b2', 'c3'], tags='blue')
        assert_allclose(a.H.array, x.conj())
        assert a.size == 24

        with pytest.raises(ValueError):
            Tensor(x, inds=['ijk'], tags='blue')

    def test_arithmetic_scalar(self):
        x = np.random.randn(2, 3, 4)
        a = Tensor(x, inds=[0, 1, 2], tags='blue')
        assert_allclose((a + 2).array, x + 2)
        assert_allclose((a - 3).array, x - 3)
        assert_allclose((a * 4).array, x * 4)
        assert_allclose((a / 5).array, x / 5)
        assert_allclose((a ** 2).array, x ** 2)
        assert_allclose((2 + a).array, 2 + x)
        assert_allclose((3 - a).array, 3 - x)
        assert_allclose((4 * a).array, 4 * x)
        assert_allclose((5 / a).array, 5 / x)
        assert_allclose((5 ** a).array, 5 ** x)

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
            assert_allclose(c.array, op(a.array, b.array))

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
        assert b.shape == (8, 15)
        assert b.inds == ('bra', 'ket')
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


class TestTensorFunctions:
    @pytest.mark.parametrize('method', ['svd', 'eig', 'qr', 'lq'])
    @pytest.mark.parametrize('linds', ['abd', 'ce'])
    @pytest.mark.parametrize('tol', [-1.0, 1e-13])
    def test_split_tensor_full_svd(self, method, linds, tol):
        a = Tensor(np.random.randn(2, 3, 4, 5, 6), inds='abcde', tags='red')

        a_split = tensor_split(a, linds, method=method, tol=tol)
        assert len(a_split.tensors) == 2
        if linds == 'abd':
            assert a_split.shape == (2, 3, 5, 4, 6)
        elif linds == 'ce':
            assert a_split.shape == (4, 6, 2, 3, 5)
        assert (a_split ^ ...).almost_equals(a)


class TestTensorNetworkBasic:
    def test_combining_tensors(self):
        a = Tensor(np.random.randn(2, 3, 4), inds=[0, 1, 2],
                   tags='red')
        b = Tensor(np.random.randn(3, 4, 5), inds=[1, 2, 3],
                   tags='blue')
        c = Tensor(np.random.randn(5, 2, 6), inds=[3, 0, 4],
                   tags='blue')

        with pytest.raises(TypeError):
            a & np.array([0, 0])

        abc1 = (a & b & c).H.contract()
        abc2 = (a & (b & c)).H.contract()
        abc3 = (TensorNetwork(a, b, c)).H.contract()
        abc4 = (TensorNetwork(a, TensorNetwork(b, c))).H.contract()
        abc5 = (TensorNetwork(a) & TensorNetwork(b, c)).H.contract()

        assert_allclose(abc1.array, abc2.array)
        assert_allclose(abc1.array, abc3.array)
        assert_allclose(abc1.array, abc4.array)
        assert_allclose(abc1.array, abc5.array)

    def test_TensorNetwork_init_checks(self):
        a = Tensor(np.random.randn(2, 3, 4), inds=[0, 1, 2],
                   tags='red')
        b = Tensor(np.random.randn(3, 4, 5), inds=[1, 2, 3],
                   tags='blue')

        with pytest.raises(TypeError):
            TensorNetwork((a, b))  # note extra bracket

    def test_contracting_tensors(self):
        a = Tensor(np.random.randn(2, 3, 4), inds=[0, 1, 2],
                   tags='red')
        b = Tensor(np.random.randn(3, 4, 5), inds=[1, 2, 3],
                   tags='blue')
        c = Tensor(np.random.randn(5, 2, 6), inds=[3, 0, 4],
                   tags='blue')

        a_b_c = a & b & c
        print(a_b_c)
        repr(a_b_c)

        assert isinstance(a_b_c, TensorNetwork)
        a_bc = a_b_c ^ 'blue'
        assert isinstance(a_bc, TensorNetwork)
        assert len(a_bc.tensors) == 2
        abc = a_bc ^ ['red', 'blue']
        assert isinstance(abc, Tensor)
        assert_allclose(abc.array, a_b_c.contract().array)

    def test_cumulative_contract(self):
        a = Tensor(np.random.randn(2, 3, 4), inds=[0, 1, 2],
                   tags='red')
        b = Tensor(np.random.randn(3, 4, 5), inds=[1, 2, 3],
                   tags='blue')
        c = Tensor(np.random.randn(5, 2, 6), inds=[3, 0, 4],
                   tags='green')

        d = (a & b & c) >> ['red', 'green', 'blue']
        assert d.shape == (6,)
        assert d.inds == (4,)

    def test_entanglement_of_mps_state(self):

        mps = TensorNetwork(
            Tensor(np.random.randn(2, 8), [0, 1]),
            *[Tensor(np.random.randn(8, 2, 8), [2 * i - 1, 2 * i, 2 * i + 1])
              for i in range(1, 9)],
            Tensor(np.random.randn(8, 2), [17, 18])
        )

        psi = mps.contract()
        assert psi.shape == (2,) * 10

    def test_reindex(self):
        a = Tensor(np.random.randn(2, 3, 4), inds=[0, 1, 2],
                   tags='red')
        b = Tensor(np.random.randn(3, 4, 5), inds=[1, 2, 3],
                   tags='blue')
        c = Tensor(np.random.randn(5, 2, 6), inds=[3, 0, 4],
                   tags='green')

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


class TestSpecificNetworks:

    def test_matrix_product_state(self):
        tensors = ([np.random.rand(5, 2)] +
                   [np.random.rand(5, 5, 2) for _ in range(3)] +
                   [np.random.rand(5, 2)])
        mps = matrix_product_state(*tensors)
        assert len(mps.tensors) == 5

    def test_matrix_product_operator(self):
        tensors = ([np.random.rand(5, 2, 2)] +
                   [np.random.rand(5, 5, 2, 2) for _ in range(3)] +
                   [np.random.rand(5, 2, 2)])
        mpo = matrix_product_operator(*tensors)
        assert len(mpo.tensors) == 5
        op = mpo ^ ...
        # this relies on left to right contraction
        assert op.inds == ('k0', 'b0', 'k1', 'b1', 'k2', 'b2',
                           'k3', 'b3', 'k4', 'b4')


class TestSpecificStatesOperators:

    def test_rand_ket_mps(self):
        rmps = rand_ket_mps(10, 10, site_tags="foo{}", tags='bar')
        assert rmps.tensors[0].tags == {'foo0', 'bar'}
        assert rmps.tensors[3].tags == {'foo3', 'bar'}
        assert rmps.tensors[-1].tags == {'foo9', 'bar'}

        rmpsH_rmps = rmps.H & rmps
        assert rmpsH_rmps.tag_index['foo0'] == [0, 10]
        assert rmpsH_rmps.tag_index['bar'] == list(range(20))

        assert abs(rmps.H @ rmps - 1) < 1e-13
        # import pdb;pdb.set_trace()
        c = (rmps.H & rmps) ^ slice(0, 5) ^ slice(9, 4, -1) ^ slice(4, 6)
        assert abs(c - 1) < 1e-13

    def test_mpo_site_ham_heis(self):
        hh_mpo = ham_heis_mpo(5, tags=['foo'])

        assert hh_mpo.tensors[0].tags == {'i0', 'foo'}
        assert hh_mpo.tensors[3].tags == {'i3', 'foo'}
        assert hh_mpo.tensors[-1].tags == {'i4', 'foo'}

        assert hh_mpo.shape == (2,) * 10

        hh_ = (hh_mpo ^ ...).fuse({'k': ['k0', 'k1', 'k2', 'k3', 'k4'],
                                   'b': ['b0', 'b1', 'b2', 'b3', 'b4']})

        hh = ham_heis(5, cyclic=False) / 4

        assert_allclose(hh, hh_.array)
