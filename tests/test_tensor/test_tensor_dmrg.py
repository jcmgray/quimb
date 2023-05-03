import pytest

import numpy as np
from numpy.testing import assert_allclose

from quimb import (
    ham_heis,
    expec,
    plus,
    is_eigenvector,
    eigh,
    heisenberg_energy,
)

from quimb.tensor import (
    MPS_rand_state,
    MPS_product_state,
    MPS_computational_state,
    MPO_ham_ising,
    MPO_ham_XY,
    MPO_ham_heis,
    MPO_ham_mbl,
    MovingEnvironment,
    DMRG1,
    DMRG2,
    DMRGX,
    SpinHam1D,
)


np.random.seed(42)


class TestMovingEnvironment:
    def test_bsz1_start_left(self):
        tn = MPS_rand_state(6, bond_dim=7)
        env = MovingEnvironment(tn, begin='left', bsz=1)
        assert env.pos == 0
        assert len(env().tensors) == 3
        env.move_right()
        assert env.pos == 1
        assert len(env().tensors) == 3
        env.move_right()
        assert env.pos == 2
        assert len(env().tensors) == 3
        env.move_to(5)
        assert env.pos == 5
        assert len(env().tensors) == 3

    def test_bsz1_start_right(self):
        tn = MPS_rand_state(6, bond_dim=7)
        env = MovingEnvironment(tn, begin='right', bsz=1)
        assert env.pos == 5
        assert len(env().tensors) == 3
        env.move_left()
        assert env.pos == 4
        assert len(env().tensors) == 3
        env.move_left()
        assert env.pos == 3
        assert len(env().tensors) == 3
        env.move_to(0)
        assert env.pos == 0
        assert len(env().tensors) == 3

    def test_bsz2_start_left(self):
        tn = MPS_rand_state(6, bond_dim=7)
        env = MovingEnvironment(tn, begin='left', bsz=2)
        assert len(env().tensors) == 4
        env.move_right()
        assert len(env().tensors) == 4
        env.move_right()
        assert len(env().tensors) == 4
        with pytest.raises(ValueError):
            env.move_to(5)
        env.move_to(4)
        assert env.pos == 4
        assert len(env().tensors) == 4

    def test_bsz2_start_right(self):
        tn = MPS_rand_state(6, bond_dim=7)
        env = MovingEnvironment(tn, begin='right', bsz=2)
        assert env.pos == 4
        assert len(env().tensors) == 4
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
        assert len(env().tensors) == 4

    @pytest.mark.parametrize("n", [20, 19])
    @pytest.mark.parametrize("bsz", [1, 2])
    @pytest.mark.parametrize("ssz", [1 / 2, 1.0])
    def test_cyclic_moving_env_init_left(self, n, bsz, ssz):
        nenv = 2
        p = MPS_rand_state(n, 4, cyclic=True)
        norm = p.H & p
        mes = MovingEnvironment(norm, begin='left', bsz=bsz,
                                cyclic=True, ssz=ssz)
        assert len(mes.envs) == n // 2 + n % 2
        assert mes.pos == 0
        assert len(mes.envs[0].tensors) == 2 * bsz + nenv
        assert len(mes.envs[n // 2 - 1].tensors) == 2 * bsz + 1
        assert n // 2 + n % 2 not in mes.envs
        assert n - 1 not in mes.envs

        for i in range(1, 2 * n):
            mes.move_right()
            assert mes.pos == i % n
            cur_env = mes()
            assert len(cur_env.tensors) == 2 * bsz + nenv
            assert (cur_env ^ all) == pytest.approx(1.0)

    @pytest.mark.parametrize("n", [20, 19])
    @pytest.mark.parametrize("bsz", [1, 2])
    @pytest.mark.parametrize("ssz", [1 / 2, 1.0])
    def test_cyclic_moving_env_init_right(self, n, bsz, ssz):
        p = MPS_rand_state(n, 4, cyclic=True)
        norm = p.H | p
        mes = MovingEnvironment(norm, begin='right', bsz=bsz,
                                cyclic=True, ssz=ssz)
        assert len(mes.envs) == n // 2 + n % 2
        assert mes.pos == n - 1
        assert len(mes.envs[n - 1].tensors) == 2 * bsz + 2
        assert len(mes.envs[n - n // 2].tensors) == 2 * bsz + 1
        assert 0 not in mes.envs
        assert n // 2 - 1 not in mes.envs

        for i in reversed(range(-n, n - 1)):
            mes.move_left()
            assert mes.pos == i % n
            cur_env = mes()
            assert len(cur_env.tensors) == 2 * bsz + 2
            assert (cur_env ^ all) == pytest.approx(1.0)


class TestDMRG1:

    def test_single_explicit_sweep(self):
        h = MPO_ham_heis(5)
        dmrg = DMRG1(h, bond_dims=3)
        assert dmrg._k[0].dtype == float

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
        assert dmrg._k[0].dtype == float
        dmrg._k.align_(dmrg._b)
        assert_allclose(abs(dmrg._b @ dmrg._k), 1)

        assert e1.real < e0.real
        assert e2.real < e1.real
        assert e3.real < e2.real
        assert e4.real < e3.real

    @pytest.mark.parametrize("dense", [False, True])
    @pytest.mark.parametrize("MPO_ham", [MPO_ham_XY, MPO_ham_heis])
    @pytest.mark.parametrize("cyclic", [False, True])
    def test_ground_state_matches(self, dense, MPO_ham, cyclic):
        n = 10

        tol = 3e-2 if cyclic else 1e-4

        h = MPO_ham(n, cyclic=cyclic)
        dmrg = DMRG1(h, bond_dims=[4, 8, 12])
        dmrg.opts['local_eig_ham_dense'] = dense
        dmrg.opts['periodic_segment_size'] = 1.0
        dmrg.opts['periodic_nullspace_fudge_factor'] = 1e-6
        assert dmrg.solve(tol=tol / 10, verbosity=1)
        assert dmrg.state.cyclic == cyclic
        eff_e, mps_gs = dmrg.energy, dmrg.state
        mps_gs_dense = mps_gs.to_qarray()

        assert_allclose(mps_gs_dense.H @ mps_gs_dense, 1.0, rtol=tol)

        h_dense = h.to_qarray()

        # check against dense form
        actual_e, gs = eigh(h_dense, k=1)
        assert_allclose(actual_e, eff_e, rtol=tol)
        assert_allclose(abs(expec(mps_gs_dense, gs)), 1.0, rtol=tol)

        # check against actual MPO_ham
        if MPO_ham is MPO_ham_XY:
            ham_dense = ham_heis(n, cyclic=cyclic,
                                 j=(1.0, 1.0, 0.0), sparse=True)
        elif MPO_ham is MPO_ham_heis:
            ham_dense = ham_heis(n, cyclic=cyclic, sparse=True)

        actual_e, gs = eigh(ham_dense, k=1)
        assert_allclose(actual_e, eff_e, rtol=tol)
        assert_allclose(abs(expec(mps_gs_dense, gs)), 1.0, rtol=tol)

    def test_ising_and_MPS_product_state(self):
        h = MPO_ham_ising(6, bx=2.0, j=0.1)
        dmrg = DMRG1(h, bond_dims=8)
        assert dmrg.solve(verbosity=1)
        eff_e, mps_gs = dmrg.energy, dmrg.state
        mps_gs_dense = mps_gs.to_qarray()
        assert_allclose(mps_gs_dense.H @ mps_gs_dense, 1.0)

        # check against dense
        h_dense = h.to_qarray()
        actual_e, gs = eigh(h_dense, k=1)
        assert_allclose(actual_e, eff_e)
        assert_allclose(abs(expec(mps_gs_dense, gs)), 1.0)

        exp_gs = MPS_product_state([plus()] * 6)
        assert_allclose(abs(exp_gs.H @ mps_gs), 1.0, rtol=1e-3)


class TestDMRG2:
    @pytest.mark.parametrize("dense", [False, True])
    @pytest.mark.parametrize("MPO_ham", [MPO_ham_XY, MPO_ham_heis])
    @pytest.mark.parametrize("cyclic", [False, True])
    def test_matches_exact(self, dense, MPO_ham, cyclic):
        n = 6
        h = MPO_ham(n, cyclic=cyclic)

        tol = 3e-2 if cyclic else 1e-4

        dmrg = DMRG2(h, bond_dims=[4, 8, 12])
        assert dmrg._k[0].dtype == float
        dmrg.opts['local_eig_ham_dense'] = dense
        dmrg.opts['periodic_segment_size'] = 1.0
        dmrg.opts['periodic_nullspace_fudge_factor'] = 1e-6

        assert dmrg.solve(tol=tol / 10, verbosity=1)

        # XXX: need to dispatch SLEPc eigh on real input
        # assert dmrg._k[0].dtype == float

        eff_e, mps_gs = dmrg.energy, dmrg.state
        mps_gs_dense = mps_gs.to_qarray()

        assert_allclose(expec(mps_gs_dense, mps_gs_dense), 1.0, rtol=tol)

        h_dense = h.to_qarray()

        # check against dense form
        actual_e, gs = eigh(h_dense, k=1)
        assert_allclose(actual_e, eff_e, rtol=tol)
        assert_allclose(abs(expec(mps_gs_dense, gs)), 1.0, rtol=tol)

        # check against actual MPO_ham
        if MPO_ham is MPO_ham_XY:
            ham_dense = ham_heis(n, cyclic=cyclic, j=(1.0, 1.0, 0.0))
        elif MPO_ham is MPO_ham_heis:
            ham_dense = ham_heis(n, cyclic=cyclic)

        actual_e, gs = eigh(ham_dense, k=1)
        assert_allclose(actual_e, eff_e, rtol=tol)
        assert_allclose(abs(expec(mps_gs_dense, gs)), 1.0, rtol=tol)

    def test_cyclic_solve_big_with_segmenting(self):
        n = 150
        ham = MPO_ham_heis(n, cyclic=True)
        dmrg = DMRG2(ham, bond_dims=range(10, 30, 2))
        dmrg.opts['periodic_segment_size'] = 1 / 3
        assert dmrg.solve(tol=1, verbosity=2)
        assert dmrg.energy == pytest.approx(heisenberg_energy(n), 1e-3)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64,
                                       np.complex64, np.complex128])
    def test_dtypes(self, dtype):
        H = MPO_ham_heis(8).astype(dtype)
        dmrg = DMRG2(H)
        dmrg.opts['local_eig_backend'] = 'scipy'
        dmrg.solve(max_sweeps=3)
        res_dtype, = {t.dtype for t in dmrg.state}
        assert res_dtype == dtype

    def test_total_size_2(self):
        N = 2
        builder = SpinHam1D(1 / 2)
        for i in range(N - 1):
            builder[i, i + 1] += 1.0, 'Z', 'Z'

        H = builder.build_mpo(N)
        dmrg = DMRG2(H)
        dmrg.solve(verbosity=1)
        assert dmrg.energy == pytest.approx(-1 / 4)

    def test_variable_bond_ham(self):
        import quimb as qu

        HB = SpinHam1D(1 / 2)
        HB[0, 1] += 0.6, 'Z', 'Z'
        HB[1, 2] += 0.7, 'Z', 'Z'
        HB[1, 2] += 0.8, 'X', 'X'
        HB[2, 3] += 0.9, 'Y', 'Y'

        H_mpo = HB.build_mpo(4)
        H_sps = HB.build_sparse(4)

        assert H_mpo.bond_sizes() == [3, 4, 3]

        Sx, Sy, Sz = map(qu.spin_operator, 'xyz')
        H_explicit = (
            qu.ikron(0.6 * Sz & Sz, [2, 2, 2, 2], [0, 1]) +
            qu.ikron(0.7 * Sz & Sz, [2, 2, 2, 2], [1, 2]) +
            qu.ikron(0.8 * Sx & Sx, [2, 2, 2, 2], [1, 2]) +
            qu.ikron(0.9 * Sy & Sy, [2, 2, 2, 2], [2, 3])
        )

        assert_allclose(H_explicit, H_mpo.to_qarray())
        assert_allclose(H_explicit, H_sps.A)


class TestDMRGX:

    def test_explicit_sweeps(self):
        n = 8
        chi = 16
        ham = MPO_ham_mbl(n, dh=4, seed=42)
        p0 = MPS_rand_state(n, 2).expand_bond_dimension(chi)

        b0 = p0.H
        p0.align_(ham, b0)
        en0 = (p0 & ham & b0) ^ ...
        dmrgx = DMRGX(ham, p0, chi)
        dmrgx.sweep_right()
        en1 = dmrgx.sweep_left(canonize=False)
        assert en0 != en1

        dmrgx.sweep_right(canonize=False)
        en = dmrgx.sweep_right(canonize=True)

        # check normalized
        assert_allclose(dmrgx._k.H @ dmrgx._k, 1.0)

        k = dmrgx._k.to_qarray()
        h = ham.to_qarray()
        el, ev = eigh(h)

        # check variance very low
        assert np.abs((k.H @ h @ h @ k) - (k.H @ h @ k)**2) < 1e-12

        # check exactly one eigenvalue matched well
        assert np.sum(np.abs(el - en) < 1e-12) == 1

        # check exactly one eigenvector is matched with high fidelity
        ovlps = (ev.H @ k).A**2
        big_ovlps = ovlps[ovlps > 1e-12]
        assert_allclose(big_ovlps, [1])

        # check fully
        assert is_eigenvector(k, h, tol=1e-10)

    def test_solve_bigger(self):
        n = 14
        chi = 16
        ham = MPO_ham_mbl(n, dh=8, seed=42)
        p0 = MPS_computational_state('00110111000101')
        dmrgx = DMRGX(ham, p0, chi)
        assert dmrgx.solve(tol=1e-5, sweep_sequence='R')
        assert dmrgx.state[0].dtype == float
