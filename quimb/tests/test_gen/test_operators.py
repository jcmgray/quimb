from pytest import raises, mark
import numpy as np
from numpy.testing import assert_allclose
from ... import issparse, eigvals, groundstate, overlap, singlet, seigvals
from ...gen.operators import sig, controlled, ham_heis, ham_j1j2


class TestSig:
    def test_sig_dim2(self):
        for dir in (1, 'x', 'X',
                    2, 'y', 'Y',
                    3, 'z', 'Z'):
            x = sig(dir)
            assert_allclose(eigvals(x), [-1, 1])

    def test_sig_dim3(self):
        for dir in (1, 'x', 'X',
                    2, 'y', 'Y',
                    3, 'z', 'Z'):
            x = sig(dir, dim=3)
            assert_allclose(eigvals(x), [-1, 0, 1],
                            atol=1e-15)

    def test_sig_bad_dim(self):
        with raises(KeyError):
            sig('x', 4)

    def test_sig_bad_dir(self):
        with raises(KeyError):
            sig('w', 2)


class TestControlledZ:
    def test_controlled_z_dense(self):
        cz = controlled('z')
        assert_allclose(cz, np.diag([1, 1, 1, -1]))

    def test_controlled_z_sparse(self):
        cz = controlled('z', sparse=True)
        assert(issparse(cz))
        assert_allclose(cz.A, np.diag([1, 1, 1, -1]))


class TestHamHeis:
    def test_ham_heis_2(self):
        h = ham_heis(2, cyclic=False)
        l = eigvals(h)
        assert_allclose(l, [-3, 1, 1, 1])
        gs = groundstate(h)
        assert_allclose(overlap(gs, singlet()), 1.)

    def test_ham_heis_sparse_cyclic_4(self):
        h = ham_heis(4, sparse=True, cyclic=True)
        lk = seigvals(h, 4)
        assert_allclose(lk, [-8, -4, -4, -4])

    # @mark.parametrize("sformat", ["csr", "csc", "bsr", "coo"])
    # TODO:
    # def test_sformat_construct(self, sformat):
    #     h = ham_heis(4, sparse=True, sformat=sformat)
    #     assert h.format == sformat


class TestHamJ1J2:
    def test_ham_j1j2_3_dense(self):
        h = ham_j1j2(3, j2=1.0, cyclic=False)
        h2 = ham_heis(3, cyclic=True)
        assert_allclose(h, h2)

    def test_ham_j1j2_6_sparse_cyc(self):
        h = ham_j1j2(6, j2=0.5, sparse=True, cyclic=True)
        lk = seigvals(h, 5)
        assert_allclose(lk, [-9, -9, -7, -7, -7])

    def test_ham_j1j2_4_bz(self):
        h = ham_j1j2(4, cyclic=True, bz=0)
        lk = seigvals(h, 11)
        assert_allclose(lk, [-6, -6, -2, -2, -2, -2, -2, -2, -2, -2, -2])
        h = ham_j1j2(4, cyclic=True, bz=0.1)
        lk = seigvals(h, 11)
        assert_allclose(lk, [-6, -6,
                             -2.2, -2.2, -2.2,
                             -2.0, -2.0, -2.0,
                             -1.8, -1.8, -1.8])
