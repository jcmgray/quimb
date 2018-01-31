import pytest
import numpy as np
from numpy.testing import assert_allclose
from quimb import (
    issparse,
    eigvals,
    eigvecs,
    groundstate,
    expec,
    singlet,
    seigvals,
    spin_operator,
    pauli,
    controlled,
    ham_heis,
    ham_j1j2,
    zspin_projector,
    up,
    down,
    swap,
    rand_ket,
)


class TestSpinOperator:
    def test_spin_half(self):
        Sx = spin_operator('x', 1 / 2)
        assert_allclose(Sx, [[0.0, 0.5], [0.5, 0.0]])

        Sy = spin_operator('y', 1 / 2)
        assert_allclose(Sy, [[0.0, -0.5j], [0.5j, 0.0]])

        Sz = spin_operator('z', 1 / 2)
        assert_allclose(Sz, [[0.5, 0.0], [0.0, -0.5]])

        Sp = spin_operator('+', 1 / 2)
        assert_allclose(Sp, [[0.0, 1.0], [0.0, 0.0]])

        Sm = spin_operator('-', 1 / 2)
        assert_allclose(Sm, [[0.0, 0.0], [1.0, 0.0]])

    @pytest.mark.parametrize("label", ('x', 'y', 'z'))
    @pytest.mark.parametrize("S", [1, 3 / 2, 2, 5 / 2])
    def test_spin_high(self, label, S):
        D = int(2 * S + 1)
        op = spin_operator(label, S)
        assert_allclose(eigvals(op), np.linspace(-S, S, D), atol=1e-13)


class TestPauli:
    def test_pauli_dim2(self):
        for dir in (1, 'x', 'X',
                    2, 'y', 'Y',
                    3, 'z', 'Z'):
            x = pauli(dir)
            assert_allclose(eigvals(x), [-1, 1])

    def test_pauli_dim3(self):
        for dir in (1, 'x', 'X',
                    2, 'y', 'Y',
                    3, 'z', 'Z'):
            x = pauli(dir, dim=3)
            assert_allclose(eigvals(x), [-1, 0, 1],
                            atol=1e-15)

    def test_pauli_bad_dim(self):
        with pytest.raises(KeyError):
            pauli('x', 4)

    def test_pauli_bad_dir(self):
        with pytest.raises(KeyError):
            pauli('w', 2)


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
        evals = eigvals(h)
        assert_allclose(evals, [-0.75, 0.25, 0.25, 0.25])
        gs = groundstate(h)
        assert_allclose(expec(gs, singlet()), 1.)

    @pytest.mark.parametrize("parallel", [False, True])
    def test_ham_heis_sparse_cyclic_4(self, parallel):
        h = ham_heis(4, sparse=True, cyclic=True, parallel=parallel)
        lk = seigvals(h, 4)
        assert_allclose(lk, [-2, -1, -1, -1])

    def test_ham_heis_bz(self):
        h = ham_heis(2, cyclic=False, b=1)
        evals = eigvals(h)
        assert_allclose(evals, [-3 / 4, -3 / 4, 1 / 4, 5 / 4])

    @pytest.mark.parametrize("stype", ["coo", "csr", "csc", "bsr"])
    def test_sformat_construct(self, stype):
        h = ham_heis(4, sparse=True, stype=stype)
        assert h.format == stype


class TestHamJ1J2:
    def test_ham_j1j2_3_dense(self):
        h = ham_j1j2(3, j2=1.0, cyclic=False)
        h2 = ham_heis(3, cyclic=True)
        assert_allclose(h, h2)

    def test_ham_j1j2_6_sparse_cyc(self):
        h = ham_j1j2(6, j2=0.5, sparse=True, cyclic=True)
        lk = seigvals(h, 5)
        assert_allclose(lk, [-9 / 4, -9 / 4, -7 / 4, -7 / 4, -7 / 4])

    def test_ham_j1j2_4_bz(self):
        h = ham_j1j2(4, j2=0.5, cyclic=True, bz=0)
        lk = seigvals(h, 11)
        assert_allclose(lk, [-1.5, -1.5, -0.5, -0.5, -0.5, -0.5,
                             -0.5, -0.5, -0.5, -0.5, -0.5])
        h = ham_j1j2(4, j2=0.5, cyclic=True, bz=0.05)
        lk = seigvals(h, 11)
        assert_allclose(lk, [-1.5, -1.5, -0.55, -0.55, -0.55,
                             -0.5, -0.5, -0.5, -0.45, -0.45, -0.45])


class TestSpinZProjector:
    @pytest.mark.parametrize("sz", [-2, -1, 0, 1, 2])
    def test_works(self, sz):
        prj = zspin_projector(4, sz)
        h = ham_heis(4)
        h0 = prj @ h @ prj.H
        v0s = eigvecs(h0)
        for v0 in v0s.T:
            vf = prj.H @ v0.T
            prjv = vf @ vf.H
            # Check reconstructed full eigenvectors commute with full ham
            assert_allclose(prjv @ h, h @ prjv, atol=1e-13)
        if sz == 0:
            # Groundstate must be in most symmetric subspace
            gs = groundstate(h)
            gs0 = prj .H @ v0s[:, 0]
            assert_allclose(expec(gs, gs0), 1.0)
            assert_allclose(expec(h, gs0), expec(h, gs))

    def test_raises(self):
        with pytest.raises(ValueError):
            zspin_projector(5, 0)
        with pytest.raises(ValueError):
            zspin_projector(4, 1 / 2)

    @pytest.mark.parametrize("sz", [(-1 / 2, 1 / 2), (3 / 2, 5 / 2)])
    def test_spin_half_double_space(self, sz):
        prj = zspin_projector(5, sz)
        h = ham_heis(5)
        h0 = prj @ h @ prj.H
        v0s = eigvecs(h0)
        for v0 in v0s.T:
            vf = prj.H @ v0.T
            prjv = vf @ vf.H
            # Check reconstructed full eigenvectors commute with full ham
            assert_allclose(prjv @ h, h @ prjv, atol=1e-13)
        if sz == 0:
            # Groundstate must be in most symmetric subspace
            gs = groundstate(h)
            gs0 = prj .H @ v0s[:, 0]
            assert_allclose(expec(gs, gs0), 1.0)
            assert_allclose(expec(h, gs0), expec(h, gs))


class TestSwap:
    @pytest.mark.parametrize("sparse", [False, True])
    def test_swap_qubits(self, sparse):
        a = up() & down()
        s = swap(2, sparse=sparse)
        assert_allclose(s @ a, down() & up())

    @pytest.mark.parametrize("sparse", [False, True])
    def test_swap_higher_dim(self, sparse):
        a = rand_ket(9)
        s = swap(3, sparse=sparse)
        assert_allclose(s @ a, a.reshape([3, 3]).T.reshape([9, 1]))
