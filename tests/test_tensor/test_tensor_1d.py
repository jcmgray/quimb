import pytest

import numpy as np
from numpy.testing import assert_allclose

import quimb as qu
from quimb.tensor import (
    MatrixProductState, MatrixProductOperator, tensor_network_align,
    MPS_rand_state, MPO_identity, MPO_identity_like, MPO_zeros, MPO_zeros_like,
    MPO_rand, MPO_rand_herm, MPO_ham_heis, MPS_neel_state, MPS_zero_state,
    bonds, MPS_computational_state, Dense1D)
from quimb.tensor.tensor_core import oset


dtypes = ['float32', 'float64', 'complex64', 'complex128']


class TestMatrixProductState:
    def test_matrix_product_state(self):
        tensors = ([np.random.rand(5, 2)] +
                   [np.random.rand(5, 5, 2)
                    for _ in range(3)] + [np.random.rand(5, 2)])
        mps = MatrixProductState(tensors)
        mps.check()
        assert len(mps.tensors) == 5
        nmps = mps.reindex_sites('foo{}', inplace=False, where=slice(0, 3))
        assert nmps.site_ind_id == "k{}"
        assert isinstance(nmps, MatrixProductState)
        assert set(nmps.outer_inds()) == {'foo0', 'foo1', 'foo2', 'k3', 'k4'}
        assert set(mps.outer_inds()) == {'k0', 'k1', 'k2', 'k3', 'k4'}
        mps.site_ind_id = 'foo{}'
        assert set(
            mps.outer_inds()) == {'foo0', 'foo1', 'foo2', 'foo3', 'foo4'}
        assert mps.site_inds == ('foo0', 'foo1', 'foo2', 'foo3', 'foo4')
        assert mps.site_ind_id == 'foo{}'
        mps.show()

    @pytest.mark.parametrize(
        "dtype", [float, complex, np.complex128, np.float64, 'raise'])
    def test_rand_mps_dtype(self, dtype):
        if dtype == 'raise':
            with pytest.raises(TypeError):
                MPS_rand_state(10, 7, dtype=dtype)
        else:
            p = MPS_rand_state(10, 7, dtype=dtype)
            assert p[0].dtype == dtype
            assert p[7].dtype == dtype

    def test_trans_invar(self):
        with pytest.raises(ValueError):
            psi = MPS_rand_state(10, 7, cyclic=False, trans_invar=True)

        psi = MPS_rand_state(10, 7, cyclic=True, trans_invar=True)
        z0 = psi.expec(psi.gate(qu.pauli('Z'), 0, contract=True))
        z3 = psi.expec(psi.gate(qu.pauli('Z'), 0, contract=True))
        z7 = psi.expec(psi.gate(qu.pauli('Z'), 0, contract=True))

        assert_allclose(z0, z3)
        assert_allclose(z3, z7)

    def test_from_dense(self):
        L = 8
        psi = qu.rand_ket(2**L)
        mps = MatrixProductState.from_dense(psi, dims=[2] * L)
        assert mps.tags == oset(f'I{i}' for i in range(L))
        assert mps.site_inds == tuple(f'k{i}' for i in range(L))
        assert mps.L == L
        assert mps.bond_sizes() == [2, 4, 8, 16, 8, 4, 2]
        mpod = mps.to_qarray()
        assert qu.expec(mpod, psi) == pytest.approx(1)

    def test_from_dense_low_rank(self):
        L = 6
        psi = qu.ghz_state(L)
        mps = MatrixProductState.from_dense(psi, dims=[2] * L)
        assert mps.tags == oset(f'I{i}' for i in range(L))
        assert mps.site_inds == tuple(f'k{i}' for i in range(L))
        assert mps.L == L
        assert mps.bond_sizes() == [2, 2, 2, 2, 2]
        mpod = mps.to_qarray()
        assert qu.expec(mpod, psi) == pytest.approx(1)

    def test_left_canonize_site(self):
        a = np.random.randn(7, 2) + 1.0j * np.random.randn(7, 2)
        b = np.random.randn(7, 7, 2) + 1.0j * np.random.randn(7, 7, 2)
        c = np.random.randn(7, 2) + 1.0j * np.random.randn(7, 2)
        mps = MatrixProductState([a, b, c], site_tag_id="I{}")

        mps.left_canonize_site(0)
        assert mps['I0'].shape == (2, 2)
        assert mps['I0'].tags == oset(('I0',))
        assert mps['I1'].tags == oset(('I1',))

        U = (mps['I0'].data)
        assert_allclose(U.conj().T @ U, np.eye(2), atol=1e-13)
        assert_allclose(U @ U.conj().T, np.eye(2), atol=1e-13)

        # combined two site contraction is identity also
        mps.left_canonize_site(1)
        ptn = (mps.H & mps) ^ ['I0', 'I1']
        assert_allclose(ptn['I1'].data, np.eye(4), atol=1e-13)

        # try normalizing the state
        mps['I2'] /= mps['I2'].norm()

        assert_allclose(abs(mps.H @ mps), 1.0)

    def test_right_canonize_site(self):
        a = np.random.randn(7, 2) + 1.0j * np.random.randn(7, 2)
        b = np.random.randn(7, 7, 2) + 1.0j * np.random.randn(7, 7, 2)
        c = np.random.randn(7, 2) + 1.0j * np.random.randn(7, 2)
        mps = MatrixProductState([a, b, c], site_tag_id="I{}")

        mps.right_canonize_site(2)
        assert mps['I2'].shape == (2, 2)
        assert mps['I2'].tags == oset(('I2',))
        assert mps['I1'].tags == oset(('I1',))

        U = (mps['I2'].data)
        assert_allclose(U.conj().T @ U, np.eye(2), atol=1e-13)
        assert_allclose(U @ U.conj().T, np.eye(2), atol=1e-13)

        # combined two site contraction is identity also
        mps.right_canonize_site(1)
        ptn = (mps.H & mps) ^ ['I1', 'I2']
        assert_allclose(ptn['I1'].data, np.eye(4), atol=1e-13)

        # try normalizing the state
        mps['I0'] /= mps['I0'].norm()

        assert_allclose(mps.H @ mps, 1)

    def test_rand_mps_left_canonize(self):
        n = 10
        k = MPS_rand_state(
            n, 10, site_tag_id="foo{}", tags='bar', normalize=False)
        k.left_canonize(normalize=True)

        assert k.count_canonized() == (9, 0)

        assert_allclose(k.H @ k, 1)
        p_tn = (k.H & k) ^ slice(0, 9)
        assert_allclose(p_tn['foo8'].data, np.eye(10), atol=1e-13)

    def test_rand_mps_left_canonize_with_bra(self):
        n = 10
        k = MPS_rand_state(
            n, 10, site_tag_id="foo{}", tags='bar', normalize=False)
        b = k.H
        k.left_canonize(normalize=True, bra=b)
        assert_allclose(b @ k, 1)
        p_tn = (b & k) ^ slice(0, 9)
        assert_allclose(p_tn['foo8'].data, np.eye(10), atol=1e-13)

    def test_rand_mps_right_canonize(self):
        n = 10
        k = MPS_rand_state(
            n, 10, site_tag_id="foo{}", tags='bar', normalize=False)
        k.right_canonize(normalize=True)
        assert_allclose(k.H @ k, 1)
        p_tn = (k.H & k) ^ slice(..., 0, -1)
        assert_allclose(p_tn['foo1'].data, np.eye(10), atol=1e-13)

    def test_rand_mps_right_canonize_with_bra(self):
        n = 10
        k = MPS_rand_state(
            n, 10, site_tag_id="foo{}", tags='bar', normalize=False)
        b = k.H
        k.right_canonize(normalize=True, bra=b)
        assert_allclose(b @ k, 1)
        p_tn = (b & k) ^ slice(..., 0, -1)
        assert_allclose(p_tn['foo1'].data, np.eye(10), atol=1e-13)

    def test_rand_mps_mixed_canonize(self):
        n = 10
        rmps = MPS_rand_state(
            n, 10, site_tag_id="foo{}", tags='bar', normalize=True)

        # move to the center
        rmps.canonize(4)
        assert rmps.count_canonized() == (4, 5)
        assert_allclose(rmps.H @ rmps, 1)
        p_tn = (rmps.H & rmps) ^ slice(0, 4) ^ slice(..., 4, -1)
        assert_allclose(p_tn['foo3'].data, np.eye(10), atol=1e-13)
        assert_allclose(p_tn['foo5'].data, np.eye(10), atol=1e-13)

        # try shifting to the right
        rmps.shift_orthogonality_center(current=4, new=8)
        assert_allclose(rmps.H @ rmps, 1)
        p_tn = (rmps.H & rmps) ^ slice(0, 8) ^ slice(..., 8, -1)
        assert_allclose(p_tn['foo7'].data, np.eye(4), atol=1e-13)
        assert_allclose(p_tn['foo9'].data, np.eye(2), atol=1e-13)

        # try shifting to the left
        rmps.shift_orthogonality_center(current=8, new=6)
        assert_allclose(rmps.H @ rmps, 1)
        p_tn = (rmps.H & rmps) ^ slice(0, 6) ^ slice(..., 6, -1)
        assert_allclose(p_tn['foo5'].data, np.eye(10), atol=1e-13)
        assert_allclose(p_tn['foo7'].data, np.eye(8), atol=1e-13)

    @pytest.mark.parametrize("dtype", dtypes)
    def test_canonize_and_calc_current_orthog_center(self, dtype):
        p = MPS_rand_state(20, 3, dtype=dtype)
        co = p.calc_current_orthog_center()
        assert co == (0, 19)
        p.canonize((5, 15), co)
        co = p.calc_current_orthog_center()
        assert co == (5, 15)
        p.canonize((8, 11), co)
        co = p.calc_current_orthog_center()
        assert co == (8, 11)
        assert p.dtype == dtype

    def test_can_change_data(self):
        p = MPS_rand_state(3, 10)
        assert_allclose(p.H @ p, 1)
        p[1].modify(data=np.random.randn(10, 10, 2))
        assert abs(p.H @ p - 1) > 1e-13

    def test_can_change_data_using_subnetwork(self):
        p = MPS_rand_state(3, 10)
        pH = p.H
        p.add_tag('__ket__')
        pH.add_tag('__bra__')
        tn = p | pH
        assert_allclose((tn ^ ...), 1)
        assert_allclose(tn[('__ket__', 'I1')].data, tn[('__bra__',
                                                        'I1')].data.conj())
        p[1].modify(data=np.random.randn(10, 10, 2))
        assert abs((tn ^ ...) - 1) > 1e-13
        assert not np.allclose(tn[('__ket__', 'I1')].data,
                               tn[('__bra__', 'I1')].data.conj())

    def test_adding_mps(self):
        p = MPS_rand_state(10, 7)
        assert max(p['I4'].shape) == 7
        p2 = p + p
        assert max(p2['I4'].shape) == 14
        assert_allclose(p2.H @ p, 2)
        p += p
        assert max(p['I4'].shape) == 14
        assert_allclose(p.H @ p, 4)

    @pytest.mark.parametrize("method", ['svd', 'eig'])
    @pytest.mark.parametrize('cutoff_mode', ['abs', 'rel', 'sum2'])
    def test_compress_mps(self, method, cutoff_mode):
        n = 10
        chi = 7
        p = MPS_rand_state(n, chi)
        assert max(p['I4'].shape) == chi
        p2 = p + p
        assert max(p2['I4'].shape) == chi * 2
        assert_allclose(p2.H @ p, 2)
        p2.left_compress(method=method, cutoff=1e-6, cutoff_mode=cutoff_mode)
        assert max(p2['I4'].shape) == chi
        assert_allclose(p2.H @ p, 2)
        assert p2.count_canonized() == (n - 1, 0)

    def test_compress_mps_right(self):
        p = MPS_rand_state(10, 7)
        assert max(p['I4'].shape) == 7
        p2 = p + p
        assert max(p2['I4'].shape) == 14
        assert_allclose(p2.H @ p, 2)
        p2.right_compress()
        assert max(p2['I4'].shape) == 7
        assert_allclose(p2.H @ p, 2)

    @pytest.mark.parametrize("method", ['svd', 'eig'])
    def test_compress_trim_max_bond(self, method):
        p0 = MPS_rand_state(20, 20)
        p = p0.copy()
        p.compress(method=method)
        assert max(p['I4'].shape) == 20
        p.compress(max_bond=13, method=method)
        assert max(p['I4'].shape) == 13
        assert_allclose(p.H @ p, p0.H @ p0)

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

    def test_compress_site(self):
        psi = MPS_rand_state(10, 7)
        psi.compress_site(3, max_bond=1)
        assert psi.bond_sizes() == [2, 4, 1, 1, 7, 7, 7, 4, 2]
        assert psi.calc_current_orthog_center() == (3, 3)

        psi = MPS_rand_state(10, 7)
        psi.compress_site(0, max_bond=1)
        assert psi.bond_sizes() == [1, 7, 7, 7, 7, 7, 7, 4, 2]
        assert psi.calc_current_orthog_center() == (0, 0)

        psi = MPS_rand_state(10, 7)
        psi.compress_site(9, max_bond=1)
        assert psi.bond_sizes() == [2, 4, 7, 7, 7, 7, 7, 7, 1]
        assert psi.calc_current_orthog_center() == (9, 9)

    @pytest.mark.parametrize("method", ['svd', 'eig'])
    @pytest.mark.parametrize("form", ['left', 'right', 'raise'])
    def test_add_and_compress_mps(self, method, form):
        p = MPS_rand_state(10, 7)
        assert max(p['I4'].shape) == 7

        if form == 'raise':
            with pytest.raises(ValueError):
                p.add_MPS(
                    p, compress=True, method=method, form=form, cutoff=1e-6)
            return

        p2 = p.add_MPS(p, compress=True, method=method, form=form, cutoff=1e-6)
        assert max(p2['I4'].shape) == 7
        assert_allclose(p2.H @ p, 2, rtol=1e-5)

    def test_subtract(self):
        a, b, c = (MPS_rand_state(10, 7) for _ in 'abc')
        ab = a.H @ b
        ac = a.H @ c
        abmc = a.H @ (b - c)
        assert_allclose(ab - ac, abmc)

    def test_subtract_inplace(self):
        a, b, c = (MPS_rand_state(10, 7) for _ in 'abc')
        ab = a.H @ b
        ac = a.H @ c
        b -= c
        abmc = a.H @ b
        assert_allclose(ab - ac, abmc)

    def test_amplitude(self):
        mps = MPS_rand_state(10, 7)
        k = mps.to_qarray()
        idx = np.random.randint(0, k.shape[0])
        c_b = mps.amplitude(f'{idx:0>10b}')
        assert k[idx, 0] == pytest.approx(c_b)

    def test_schmidt_values_entropy_gap_simple(self):
        n = 12
        p = MPS_rand_state(n, 16)
        p.right_canonize()
        svns = []
        sgs = []
        for i in range(1, n):
            sgs.append(p.schmidt_gap(i, cur_orthog=i - 1))
            svns.append(p.entropy(i, cur_orthog=i))

        pd = p.to_qarray()
        ex_svns = [
            qu.entropy_subsys(pd, [2] * n, range(i)) for i in range(1, n)
        ]
        ex_sgs = [qu.schmidt_gap(pd, [2] * n, range(i)) for i in range(1, n)]
        assert_allclose(ex_svns, svns)
        assert_allclose(ex_sgs, sgs)

    def test_magnetization(self):
        binary = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        p = MPS_computational_state(binary)
        mzs = [p.magnetization(i) for i in range(len(binary))]
        assert_allclose(mzs, 0.5 - np.array(binary))

    @pytest.mark.parametrize("rescale", [False, True])
    @pytest.mark.parametrize(
        "keep", [(2, 3, 4, 6, 8), slice(-2, 4), slice(3, -1, -1), [1]])
    def test_partial_trace(self, rescale, keep):
        n = 10
        p = MPS_rand_state(n, 7)
        r = p.ptr(keep=keep, upper_ind_id='u{}', rescale_sites=rescale)
        rd = r.to_qarray()
        if isinstance(keep, slice):
            keep = p.slice2sites(keep)
        else:
            if rescale:
                if keep == [1]:
                    assert r.lower_inds_present == ('u0',)
                    assert r.upper_inds_present == ('k0',)
                else:
                    assert r.lower_inds_present == ('u0', 'u1', 'u2',
                                                   'u3', 'u4')
                    assert r.upper_inds_present == ('k0', 'k1', 'k2',
                                                   'k3', 'k4')
            else:
                if keep == [1]:
                    assert r.lower_inds_present == ('u1',)
                    assert r.upper_inds_present == ('k1',)
                else:
                    assert r.lower_inds_present == ('u2', 'u3', 'u4',
                                                    'u6', 'u8')
                    assert r.upper_inds_present == ('k2', 'k3', 'k4',
                                                    'k6', 'k8')
        assert_allclose(r.trace(), 1.0)
        assert qu.isherm(rd)
        pd = p.to_qarray()
        rdd = pd.ptr([2] * n, keep=keep)
        assert_allclose(rd, rdd)

    def test_bipartite_schmidt_state(self):
        psi = MPS_rand_state(16, 5)
        psid = psi.to_qarray()
        eln = qu.logneg(psid, [2**7, 2**9])

        s_d_ket = psi.bipartite_schmidt_state(7, get='ket-dense')
        ln_d_ket = qu.logneg(s_d_ket, [5, 5])
        assert_allclose(eln, ln_d_ket, rtol=1e-5)

        s_d_rho = psi.bipartite_schmidt_state(7, get='rho-dense')
        ln_d_rho = qu.logneg(s_d_rho, [5, 5])
        assert_allclose(eln, ln_d_rho, rtol=1e-5)

        T_s_ket = psi.bipartite_schmidt_state(7, get='ket')
        assert set(T_s_ket.inds) == {'kA', 'kB'}
        assert_allclose(T_s_ket.H @ T_s_ket, 1.0)

        T_s_rho = psi.bipartite_schmidt_state(7, get='rho')
        assert set(T_s_rho.outer_inds()) == {'kA', 'kB', 'bA', 'bB'}
        assert_allclose(T_s_rho.H @ T_s_rho, 1.0)

    @pytest.mark.parametrize(
        "method", ['isvd', 'svds', ('isvd', 'eigsh'), ('isvd', 'cholesky')])
    @pytest.mark.parametrize("cyclic", [True, False])
    @pytest.mark.parametrize(
        "sysa", [range(0, 10),
                 range(10, 20),
                 range(20, 30),
                 range(0, 30)])
    @pytest.mark.parametrize(
        "sysb", [range(30, 40),
                 range(40, 50),
                 range(50, 60),
                 range(30, 60)])
    def test_partial_trace_compress(self, method, cyclic, sysa, sysb):
        k = MPS_rand_state(60, 5, cyclic=cyclic)
        kws = dict(sysa=sysa, sysb=sysb, eps=1e-6, method=method, verbosity=2)
        rhoc_ab = k.partial_trace_compress(**kws)
        assert set(rhoc_ab.outer_inds()) == {'kA', 'kB', 'bA', 'bB'}
        inds = ['kA', 'kB'], ['bA', 'bB']
        x = rhoc_ab.trace(*inds)
        assert_allclose(1.0, x, rtol=1e-3)

    @pytest.mark.parametrize("cyclic", [True, False])
    def test_known_bad_case(self, cyclic):
        k = MPS_rand_state(5, 10, cyclic=cyclic)
        rhoc_ab = k.partial_trace_compress(sysa=range(2), sysb=range(2, 4))
        inds = ['kA', 'kB'], ['bA', 'bB']
        x = rhoc_ab.trace(*inds)
        assert_allclose(1.0, x, rtol=1e-3)

    @pytest.mark.parametrize("block", [
        0, 20, 39,
        slice(0, 5),
        slice(20, 25),
        slice(35, 40),
        slice(38, 42),
        slice(-3, 2)
    ])
    @pytest.mark.parametrize("dtype", [float, complex])
    def test_canonize_cyclic(self, dtype, block):
        k = MPS_rand_state(40, 10, dtype=dtype, cyclic=True)
        b = k.H
        k.add_tag('KET')
        b.add_tag('BRA')
        kb = (b | k)

        assert not np.allclose(k[block].H @ k[block], 1.0)
        assert not np.allclose(b[block].H @ b[block], 1.0)
        k.canonize_cyclic(block, bra=b)
        assert_allclose(k[block].H @ k[block], 1.0, rtol=2e-4)
        assert_allclose(b[block].H @ b[block], 1.0, rtol=2e-4)

        ii = kb.select(block, which='!any') ^ all

        if isinstance(block, slice):
            start, stop = block.start, block.stop
        else:
            start, stop = block, block + 1

        assert len(kb.select_tensors(block, 'any')) == 2 * (stop - start)

        ul, = bonds(kb[k.site_tag(start - 1), 'BRA'],
                    kb[k.site_tag(start), 'BRA'])
        ur, = bonds(kb[k.site_tag(stop - 1), 'BRA'],
                    kb[k.site_tag(stop), 'BRA'])
        ll, = bonds(kb[k.site_tag(start - 1), 'KET'],
                    kb[k.site_tag(start), 'KET'])
        lr, = bonds(kb[k.site_tag(stop - 1), 'KET'],
                    kb[k.site_tag(stop), 'KET'])

        ii = ii.to_qarray((ul, ur), (ll, lr))
        assert_allclose(ii, np.eye(ii.shape[0]), rtol=0.001, atol=0.001)

    @pytest.mark.parametrize("bsz", [1, 2])
    @pytest.mark.parametrize("propagate_tags", [False, True])
    @pytest.mark.parametrize("contract", [False, True])
    def test_gate_no_contract(self, bsz, propagate_tags, contract):
        p = MPS_rand_state(5, 7, tags={'PSI0'})
        q = p.copy()
        G = qu.rand_uni(2**bsz)
        p = p.gate_(G, where=[i for i in range(2, 2 + bsz)], tags='G',
                    contract=contract, propagate_tags=propagate_tags)
        TG = p['G']
        if propagate_tags or contract:
            assert p.site_tag(2) in TG.tags
        assert ('PSI0' in TG.tags) == (propagate_tags is True) or contract
        assert (p.H & p) ^ all == pytest.approx(1.0)
        assert abs((q.H & p) ^ all) < 1.0
        assert len(p.tensors) == 6 - int(contract) * bsz
        assert set(p.outer_inds()) == {f'k{i}' for i in range(5)}

    @pytest.mark.parametrize("propagate_tags", [False, 'sites',
                                                'register', True])
    def test_gate_split_gate(self, propagate_tags):
        p = MPS_rand_state(5, 7, tags={'PSI0'})
        q = p.copy()
        G = qu.CNOT()
        p = p.gate_(G, where=[i for i in range(2, 4)], tags='G',
                    contract='split-gate', propagate_tags=propagate_tags)

        TG = sorted(p['G'], key=lambda t: sorted(t.tags))

        if propagate_tags is False:
            assert TG[0].tags == oset(('G',))
            assert TG[1].tags == oset(('G',))

        elif propagate_tags == 'register':
            assert TG[0].tags == oset(['G', 'I2'])
            assert TG[1].tags == oset(['G', 'I3'])

        elif propagate_tags == 'sites':
            assert TG[0].tags == oset(['G', 'I2', 'I3'])
            assert TG[1].tags == oset(['G', 'I2', 'I3'])

        elif propagate_tags is True:
            assert TG[0].tags == oset(['PSI0', 'G', 'I2', 'I3'])
            assert TG[1].tags == oset(['PSI0', 'G', 'I2', 'I3'])

        assert (p.H & p) ^ all == pytest.approx(1.0)
        assert abs((q.H & p) ^ all) < 1.0
        assert len(p.tensors) == 7
        assert set(p.outer_inds()) == {f'k{i}' for i in range(5)}

    def test_gate_swap_and_split_bond_sizes(self):
        n = 10
        p = MPS_computational_state('0' * n)
        assert p.bond_sizes() == [1] * (n - 1)
        G = qu.rand_uni(4)
        p.gate_(G, (1, n - 2), contract='swap+split')
        assert p.bond_sizes() == [1] + [2] * (n - 3) + [1]

    def test_gate_swap_and_split_matches(self):
        k = MPS_rand_state(6, 7)
        kr = k.copy()

        gates = [qu.rand_uni(4) for _ in range(3)]
        wheres = [(0, 5), (3, 2), (4, 1)]

        for G, (i, j) in zip(gates, wheres):
            k.gate_(G, (i, j), contract='swap+split')
            kr.gate_inds_(G, (k.site_ind(i), k.site_ind(j)))

        assert_allclose(k.to_dense(), kr.to_dense())

    def test_flip(self):
        p = MPS_rand_state(5, 3)
        pf = p.flip()
        # we want a single index per dimension, not all combined into one
        inds = [[ix] for ix in p.site_inds]
        assert_allclose(
            p.to_qarray(*inds),
            pf.to_qarray(*inds).transpose()
        )

    def test_correlation(self):
        ghz = (MPS_computational_state('0000') +
               MPS_computational_state('1111')) / 2**0.5

        assert ghz.correlation(qu.pauli('Z'), 0, 1) == pytest.approx(1.0)
        assert ghz.correlation(qu.pauli('Z'), 1, 2) == pytest.approx(1.0)
        assert ghz.correlation(qu.pauli('Z'), 3, 1) == pytest.approx(1.0)
        assert ghz.correlation(
            qu.pauli('Z'), 3, 1, B=qu.pauli('Y')) == pytest.approx(0.0)

        assert ghz.H @ ghz == pytest.approx(1.0)

    def test_gate_split(self):
        psi = MPS_rand_state(10, 3)
        psi2 = psi.copy()
        G = qu.eye(2) & qu.eye(2)
        psi.gate_split_(G, (2, 3), cutoff=0)
        assert psi.bond_size(2, 3) == 6
        assert psi.H @ psi2 == pytest.approx(1.0)

        # check a unitary application
        G = qu.rand_uni(2**2)
        psi.gate_split_(G, (7, 8))
        psi.compress()
        assert psi.bond_size(2, 3) == 3
        assert psi.bond_size(7, 8) > 3
        assert psi.H @ psi == pytest.approx(1.0)
        assert abs(psi2.H @ psi) < 1.0

        # check matches dense application of gate
        psid = psi2.to_qarray()
        Gd = qu.ikron(G, [2] * 10, (7, 8))
        assert psi.to_qarray().H @ (Gd @ psid) == pytest.approx(1.0)

    def test_swap_far_sites(self):
        psi = MPS_rand_state(7, 2)
        for i, j in [(0, 6), (6, 1), (5, 2)]:
            k1 = psi.to_qarray([
                psi.site_ind(
                    j if site == i else
                    i if site == j else
                    site
                ) for site in psi.sites
            ])
            k2 = psi.swap_sites_with_compress(i, j).to_qarray()
            assert qu.fidelity(k1, k2) == pytest.approx(1.0)

    def test_swap_gating(self):
        psi0 = MPS_rand_state(20, 5)
        CNOT = qu.controlled('not')
        psi0XX = psi0.gate(CNOT, (4, 13))
        psi0XX_s = psi0.gate_with_auto_swap(CNOT, (4, 13))
        assert psi0XX.H @ psi0XX_s == pytest.approx(1.0)

    def test_auto_split_detection(self):
        psi0 = MPS_computational_state('00')
        CNOT = qu.controlled('not')
        ISWAP = qu.iswap()
        G = qu.rand_uni(4)

        opts = {'contract': 'auto-split-gate', 'where': (0, 1)}

        psi_cnot = psi0.gate(CNOT, **opts)
        psi_iswap = psi0.gate(ISWAP, **opts)
        psi_G = psi0.gate(G, **opts)

        assert (psi_cnot.max_bond() ==
                psi_iswap.max_bond() ==
                psi_G.max_bond() == 2)

        assert len(psi_cnot.tensors) == len(psi_iswap.tensors) == 4
        assert len(psi_G.tensors) == 3

    @pytest.mark.parametrize('cur_orthog', (None, 3))
    @pytest.mark.parametrize('site', (0, 5, 9))
    @pytest.mark.parametrize('outcome', (None, 2))
    @pytest.mark.parametrize('renorm', (True, False))
    @pytest.mark.parametrize('remove', (True, False))
    def test_mps_measure(self, cur_orthog, site, outcome, renorm, remove):
        psi = MPS_rand_state(10, 7, phys_dim=3, dtype=complex)
        if cur_orthog:
            psi.canonize(cur_orthog)
        outcome, psim = psi.measure(
            site, outcome=outcome, cur_orthog=cur_orthog,
            renorm=renorm, remove=remove)
        newL = 10 - int(remove)
        assert psim.L == newL
        assert psim.num_tensors == newL
        assert set(psim.site_tags) == {f'I{i}' for i in range(newL)}
        assert set(psim.site_inds) == {f'k{i}' for i in range(newL)}
        if renorm:
            assert psim.H @ psim == pytest.approx(1.0)
        else:
            assert 0.0 < psim.H @ psim < 1.0
        new_can_cen = min(site, newL - 1)
        t = psim[new_can_cen]
        if renorm:
            assert t.H @ t == pytest.approx(1.0)
        else:
            0.0 < t.H @ t < 1.0

    def test_measure_known_outcome(self):
        mps = MPS_computational_state('010101')
        assert mps.measure_(3, get='outcome') == 1

    def test_permute_arrays(self):
        mps = MPS_rand_state(7, 5)
        k0 = mps.to_qarray()
        mps.canonize(3)
        mps.permute_arrays('prl')
        assert mps[0].shape == (2, 2)
        assert mps[1].shape == (2, 4, 2)
        assert mps[2].shape == (2, 5, 4)
        kf = mps.to_qarray()
        assert qu.fidelity(k0, kf) == pytest.approx(1.0)


class TestMatrixProductOperator:

    @pytest.mark.parametrize("cyclic", [False, True])
    def test_matrix_product_operator(self, cyclic):

        end_shape = (5, 5, 2, 2) if cyclic else (5, 2, 2)

        tensors = ([np.random.rand(*end_shape)] +
                   [np.random.rand(5, 5, 2, 2)
                    for _ in range(3)] + [np.random.rand(*end_shape)])
        mpo = MatrixProductOperator(tensors)

        mpo.show()
        assert len(mpo.tensors) == 5
        assert mpo.upper_inds == ('k0', 'k1', 'k2', 'k3', 'k4')
        assert mpo.lower_inds == ('b0', 'b1', 'b2', 'b3', 'b4')
        op = mpo ^ ...
        # this would rely on left to right contraction if not in set form
        assert set(op.inds) == {
            'k0', 'b0', 'k1', 'b1', 'k2', 'b2', 'k3', 'b3', 'k4', 'b4'
        }

        assert set(mpo.site_tags) == {f'I{i}' for i in range(5)}
        assert all(f'I{i}' in mpo.tags for i in range(5))
        mpo.site_tag_id = 'TEST1,{}'
        assert set(mpo.site_tags) == {f'TEST1,{i}' for i in range(5)}
        assert not any(f'I{i}' in mpo.tags for i in range(5))
        assert all(f'TEST1,{i}' in mpo.tags for i in range(5))

    @pytest.mark.parametrize("cyclic", [False, True])
    def test_compress_mpo(self, cyclic):
        A = MPO_rand(12, 5, cyclic=cyclic)
        assert all(b == 5 for b in A.bond_sizes())
        A.expand_bond_dimension(10)
        assert all(b == 10 for b in A.bond_sizes())
        A.compress()
        assert all(b in (4, 5) for b in A.bond_sizes())

    def test_add_mpo(self):
        h = MPO_rand_herm(12, 5)
        h2 = h + h
        assert max(h2[6].shape) == 10
        t = h.trace()
        t2 = h2.trace()
        assert_allclose(2 * t, t2)

    def test_adding_mpo(self):
        h = MPO_ham_heis(6)
        hd = h.to_qarray()
        assert_allclose(h @ h.H, (hd @ hd.H).tr())
        h2 = h + h
        assert_allclose(h2 @ h2.H, (hd @ hd.H).tr() * 4)
        h2.right_compress()
        assert_allclose(h2 @ h2.H, (hd @ hd.H).tr() * 4)
        assert max(h2['I3'].shape) == 5

    @pytest.mark.parametrize("cyclic", (False, True))
    def test_subtract_mpo(self, cyclic):
        a, b = MPO_rand(13, 7, cyclic=cyclic), MPO_rand(13, 7, cyclic=cyclic)
        x1 = a.trace() - b.trace()
        assert_allclose(x1, (a - b).trace())
        a -= b
        assert_allclose(x1, a.trace())

    @pytest.mark.parametrize("cyclic", (False, True))
    @pytest.mark.parametrize("rand_strength", (0, 1e-9))
    def test_expand_mpo(self, cyclic, rand_strength):
        h = MPO_ham_heis(12, cyclic=cyclic)
        assert h[0].dtype == float
        he = h.expand_bond_dimension(13, rand_strength=rand_strength)
        assert h[0].dtype == float
        assert max(he[6].shape) == 13

        if cyclic:
            assert he.bond_size(0, -1) == 13

        t = h.trace()
        te = he.trace()
        assert_allclose(t, te)

    @pytest.mark.parametrize("cyclic", (False, True))
    @pytest.mark.parametrize("rand_strength", (0, 1e-9))
    def test_expand_mpo_limited(self, cyclic, rand_strength):
        h = MPO_ham_heis(12, cyclic=cyclic)
        he = h.expand_bond_dimension(3, rand_strength=rand_strength)
        # should do nothing
        assert max(he[6].shape) == 5

    def test_mpo_identity(self):
        k = MPS_rand_state(13, 7)
        b = MPS_rand_state(13, 7)
        o1 = k @ b
        i = MPO_identity(13)
        k, i, b = tensor_network_align(k, i, b)
        o2 = (k & i & b) ^ ...
        assert_allclose(o1, o2)

    @pytest.mark.parametrize("cyclic", [False, True])
    @pytest.mark.parametrize("dtype", (complex, float))
    def test_mpo_rand_herm_and_trace(self, dtype, cyclic):
        op = MPO_rand_herm(
            20, bond_dim=5, phys_dim=3, dtype=dtype, cyclic=cyclic)
        assert_allclose(op.H @ op, 1.0)
        tr_val = op.trace()
        assert tr_val != 0.0
        assert_allclose(tr_val.imag, 0.0, atol=1e-14)

    @pytest.mark.parametrize("cyclic", [False, True])
    def test_mpo_rand_herm_trace_and_identity_like(self, cyclic):
        op = MPO_rand_herm(
            20, bond_dim=5, phys_dim=3, upper_ind_id='foo{}', cyclic=cyclic)
        t = op.trace()
        assert t != 0.0
        Id = MPO_identity_like(op)
        assert_allclose(Id.trace(), 3**20)
        Id[0] *= 3 / 3**20
        op += Id
        assert_allclose(op.trace(), t + 3)

    def test_partial_transpose(self):
        p = MPS_rand_state(8, 10)
        r = p.ptr([2, 3, 4, 5, 6, 7])
        rd = r.to_qarray()

        assert qu.isherm(rd)
        assert qu.ispos(rd)

        rpt = r.partial_transpose([0, 1, 2])
        rptd = rpt.to_qarray()

        upper_inds = tuple(f'b{i}' for i in range(6))
        lower_inds = tuple(f'k{i}' for i in range(6))
        outer_inds = rpt.outer_inds()
        assert all(i in outer_inds for i in upper_inds + lower_inds)

        assert qu.isherm(rptd)
        assert not qu.ispos(rptd)

    def test_upper_lower_ind_id_guard(self):
        A = MPO_rand(8, 5)
        with pytest.raises(ValueError):
            A.upper_ind_id = 'b{}'
        with pytest.raises(ValueError):
            A.lower_ind_id = 'k{}'

    @pytest.mark.parametrize("cyclic", (False, True))
    def test_apply_mpo(self, cyclic):
        A = MPO_rand(8, 5, cyclic=cyclic)
        B = MPO_rand(
            8, 5, upper_ind_id='q{}', lower_ind_id='w{}', cyclic=cyclic)
        C = A.apply(B)
        assert C.max_bond() == 25
        assert C.upper_ind_id == 'q{}'
        assert C.lower_ind_id == 'w{}'
        Ad, Bd, Cd = A.to_qarray(), B.to_qarray(), C.to_qarray()
        assert_allclose(Ad @ Bd, Cd)

    @pytest.mark.parametrize("cyclic", (False, True))
    @pytest.mark.parametrize("site_ind_id", ('k{}', 'test{}'))
    def test_apply_mps(self, cyclic, site_ind_id):
        A = MPO_rand(8, 5, cyclic=cyclic)
        x = MPS_rand_state(8, 4, site_ind_id=site_ind_id, cyclic=cyclic)
        y = A.apply(x)
        assert y.max_bond() == 20
        assert isinstance(y, MatrixProductState)
        assert len(y.tensors) == 8
        assert y.site_ind_id == site_ind_id
        Ad, xd, yd = A.to_qarray(), x.to_qarray(), y.to_qarray()
        assert_allclose(Ad @ xd, yd)

    def test_permute_arrays(self):
        mpo = MPO_rand(4, 3)
        A0 = mpo.to_qarray()
        mpo.permute_arrays('drul')
        assert mpo[0].shape == (2, 3, 2)
        assert mpo[1].shape == (2, 3, 2, 3)
        Af = mpo.to_qarray()
        assert_allclose(A0, Af)

# --------------------------------------------------------------------------- #
#                         Test specific 1D instances                          #
# --------------------------------------------------------------------------- #


class TestSpecificStatesOperators:
    @pytest.mark.parametrize("cyclic", [False, True])
    def test_rand_ket_mps(self, cyclic):
        n = 10
        rmps = MPS_rand_state(
            n, 10, site_tag_id="foo{}", tags='bar', cyclic=cyclic)
        assert rmps[0].tags == oset(['foo0', 'bar'])
        assert rmps[3].tags == oset(['foo3', 'bar'])
        assert rmps[-1].tags == oset(['foo9', 'bar'])

        rmpsH_rmps = rmps.H & rmps
        assert len(rmpsH_rmps.tag_map['foo0']) == 2
        assert len(rmpsH_rmps.tag_map['bar']) == n * 2

        assert_allclose(rmps.H @ rmps, 1)
        c = (rmps.H & rmps) ^ slice(0, 5) ^ slice(9, 4, -1) ^ slice(4, 6)
        assert_allclose(c, 1)

        assert rmps[0].data.ndim == (3 if cyclic else 2)
        assert rmps[-1].data.ndim == (3 if cyclic else 2)

    def test_mps_computation_state(self):
        p = MPS_neel_state(10)
        pd = qu.neel_state(10)
        assert_allclose(p.to_qarray(), pd)

    def test_zero_state(self):
        z = MPS_zero_state(21, 7)
        p = MPS_rand_state(21, 13)
        assert_allclose(p.H @ z, 0.0)
        assert_allclose(p.H @ p, 1.0)
        zp = z + p
        assert max(zp[13].shape) == 20
        assert_allclose(zp.H @ p, 1.0)

    @pytest.mark.parametrize("cyclic", [False, True])
    @pytest.mark.parametrize("j", [7 / 11, 1, (0.2, 0.3, 0.4)])
    @pytest.mark.parametrize("bz", [0, 7 / 11, 1])
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_mpo_site_ham_heis(self, cyclic, j, bz, n):
        hh_mpo = MPO_ham_heis(n, tags=['foo'], cyclic=cyclic, j=j, bz=bz)
        assert hh_mpo[0].tags == oset(['I0', 'foo'])
        assert hh_mpo[1].tags == oset(['I1', 'foo'])
        assert hh_mpo[-1].tags == oset([f'I{n - 1}', 'foo'])
        assert hh_mpo.shape == (2, ) * 2 * n
        hh_ex = qu.ham_heis(n, cyclic=cyclic, j=j, b=bz)
        assert_allclose(
            qu.eigvalsh(hh_ex), qu.eigvalsh(hh_mpo.to_qarray()), atol=1e-13)

    def test_mpo_zeros(self):
        mpo0 = MPO_zeros(10)
        assert mpo0.trace() == 0.0
        assert mpo0.H @ mpo0 == 0.0

    @pytest.mark.parametrize("cyclic", (False, True))
    def test_mpo_zeros_like(self, cyclic):
        A = MPO_rand(10, 7, phys_dim=3, normalize=False, cyclic=cyclic)
        Z = MPO_zeros_like(A)
        assert A @ Z == 0.0
        assert Z.cyclic == cyclic
        x1 = A.trace()
        x2 = (A + Z).trace()
        assert_allclose(x1, x2)


class TestDense1D:

    def test_simple(self):
        n = 10
        d_psi = qu.computational_state('0' * n)

        t_psi = Dense1D(d_psi)
        assert set(t_psi.outer_inds()) == {f'k{i}' for i in range(n)}
        assert t_psi.tags == oset(f'I{i}' for i in range(n))

        for i in range(n):
            assert t_psi.H @ t_psi.gate(qu.pauli('Z'), i) == pytest.approx(1)

        for i in range(n):
            t_psi.gate_(qu.hadamard(), i)

        assert len(t_psi.tensors) == n + 1

        # should have '++++++++++'
        assert t_psi.H @ t_psi == pytest.approx(1)
        for i in range(n):
            assert t_psi.H @ t_psi.gate(qu.pauli('X'), i) == pytest.approx(1)

    def test_rand(self):
        t_psi = Dense1D.rand(7, dtype='complex64')
        assert t_psi.shape == (2,) * 7
        assert t_psi.dtype == 'complex64'
        assert (t_psi.H @ t_psi) == pytest.approx(1.0)
