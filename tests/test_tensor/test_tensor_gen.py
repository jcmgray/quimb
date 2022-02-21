import pytest
import numpy as np
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn


class TestGeometries:

    @pytest.mark.parametrize('cyclic', [False, True])
    @pytest.mark.parametrize("edge_fn,shape,percell,coordination", [
        (qtn.edges_2d_square, (3, 3), 1, 4),
        (qtn.edges_2d_hexagonal, (3, 3), 2, 3),
        (qtn.edges_2d_kagome, (3, 3), 3, 4),
        (qtn.edges_2d_triangular, (3, 3), 1, 6),
        (qtn.edges_2d_triangular_rectangular, (3, 3), 2, 6),
        (qtn.edges_3d_cubic, (3, 3, 3), 1, 6),
        (qtn.edges_3d_pyrochlore, (3, 3, 3), 4, 6),
        (qtn.edges_3d_diamond, (3, 3, 3), 2, 4),
        (qtn.edges_3d_diamond_cubic, (2, 2, 2), 8, 4),
    ])
    def test_basic(self, cyclic, edge_fn, shape, percell, coordination):
        edges = edge_fn(*shape, cyclic=cyclic)
        tn = qtn.TN_rand_from_edges(edges, D=2)
        assert tn.num_tensors == qu.prod(shape) * percell
        assert max(t.ndim for t in tn) == coordination


class TestSpinHam1D:

    @pytest.mark.parametrize("cyclic", [False, True])
    def test_var_terms(self, cyclic):
        n = 8
        Hd = qu.ham_mbl(n, dh=0.77, seed=42, cyclic=cyclic)
        Ht = qtn.MPO_ham_mbl(n, dh=0.77, seed=42, cyclic=cyclic).to_dense()
        assert_allclose(Hd, Ht)

    @pytest.mark.parametrize("var_two", ['none', 'some', 'only'])
    @pytest.mark.parametrize("var_one", ['some', 'only', 'only-some',
                                         'def-only', 'none'])
    def test_specials(self, var_one, var_two):
        K1 = qu.rand_herm(2**1)

        n = 10
        HB = qtn.SpinHam1D(S=1 / 2)

        if var_two == 'some':
            HB += 1, K1, K1
            HB[4, 5] += 1, K1, K1
            HB[7, 8] += 1, K1, K1
        elif var_two == 'only':
            for i in range(n - 1):
                HB[i, i + 1] += 1, K1, K1
        else:
            HB += 1, K1, K1

        if var_one == 'some':
            HB += 1, K1
            HB[2] += 1, K1
            HB[3] += 1, K1
        elif var_one == 'only':
            for i in range(n - 1):
                HB[i] += 1, K1
        elif var_one == 'only-some':
            HB[1] += 1, K1
        elif var_one == 'def-only':
            HB += 1, K1

        HB.build_local_ham(n)
        H_mpo = HB.build_mpo(n)
        H_sps = HB.build_sparse(n)

        assert_allclose(H_mpo.to_dense(), H_sps.A)

    def test_no_default_term(self):
        N = 10
        builder = qtn.SpinHam1D(1 / 2)

        for i in range(N - 1):
            builder[i, i + 1] += 1.0, 'Z', 'Z'

        H = builder.build_mpo(N)

        dmrg = qtn.DMRG2(H)
        dmrg.solve(verbosity=1)

        assert dmrg.energy == pytest.approx(-2.25)


class TestMPSSpecificStates:

    def test_site_varying_phys_dim(self):
        k = qtn.MPS_rand_state(5, 4, phys_dim=[2, 3, 3, 2, 5])
        assert k.H @ k == pytest.approx(1.0)
        assert k.outer_dims_inds() == (
            (2, 'k0'), (3, 'k1'), (3, 'k2'), (2, 'k3'), (5, 'k4'),
        )

    @pytest.mark.parametrize("dtype", ['float32', 'complex64'])
    def test_ghz_state(self, dtype):
        mps = qtn.MPS_ghz_state(5, dtype=dtype)
        assert mps.dtype == dtype
        psi = qu.ghz_state(5, dtype=dtype)
        assert mps.H @ mps == pytest.approx(1.0)
        assert mps.bond_sizes() == [2, 2, 2, 2]
        assert qu.fidelity(psi, mps.to_dense()) == pytest.approx(1.0)

    @pytest.mark.parametrize("dtype", ['float32', 'complex64'])
    def test_w_state(self, dtype):
        mps = qtn.MPS_w_state(5, dtype=dtype)
        assert mps.dtype == dtype
        psi = qu.w_state(5, dtype=dtype)
        assert mps.H @ mps == pytest.approx(1.0)
        assert mps.bond_sizes() == [2, 2, 2, 2]
        assert qu.fidelity(psi, mps.to_dense()) == pytest.approx(1.0)

    def test_computational_state(self):
        mps = qtn.MPS_computational_state('01+-')
        assert_allclose(mps.to_dense(),
                        qu.up() & qu.down() & qu.plus() & qu.minus())


class TestMatrixProductOperatorSpecifics:

    def test_MPO_product_operator(self):
        psis = [qu.rand_ket(2) for _ in range(5)]
        ops = [qu.rand_matrix(2) for _ in range(5)]
        psif = qu.kron(*ops) @ qu.kron(*psis)
        mps = qtn.MPS_product_state(psis)
        mpo = qtn.MPO_product_operator(ops)
        assert mpo.bond_sizes() == [1, 1, 1, 1]
        mpsf = mpo.apply(mps)
        assert_allclose(mpsf.to_dense(), psif)


class TestGenericTN:

    def test_TN_rand_reg(self):
        n = 6
        reg = 3
        D = 2
        tn = qtn.TN_rand_reg(n, reg, D=D)
        assert tn.outer_inds() == ()
        assert tn.max_bond() == D
        assert {t.ndim for t in tn} == {reg}
        ket = qtn.TN_rand_reg(n, reg, D=2, phys_dim=2)
        assert set(ket.outer_inds()) == {f'k{i}' for i in range(n)}
        assert ket.max_bond() == D

    @pytest.mark.parametrize('Lx', [3])
    @pytest.mark.parametrize('Ly', [2, 4])
    @pytest.mark.parametrize('beta', [0.13, 0.44])
    @pytest.mark.parametrize('j', [-1.0, +1.0])
    @pytest.mark.parametrize('h', [0.0, 0.1])
    @pytest.mark.parametrize('cyclic',
                             [False, True, (False, True), (True, False)])
    def test_2D_classical_ising_model(self, Lx, Ly, beta, j, h, cyclic):
        tn = qtn.TN2D_classical_ising_partition_function(
            Lx, Ly, beta=beta, j=j, h=h, cyclic=cyclic)
        htn = qtn.HTN2D_classical_ising_partition_function(
            Lx, Ly, beta=beta, j=j, h=h, cyclic=cyclic)
        Z1 = tn.contract(all, output_inds=())
        Z2 = htn.contract(all, output_inds=())
        assert Z1 == pytest.approx(Z2)

        if not cyclic:
            # skip cyclic as nx has no multibonds for L=2
            import networkx as nx
            G = nx.lattice.grid_graph((Lx, Ly))
            Z3 = qtn.TN_classical_partition_function_from_edges(
                G.edges, beta=beta, j=j, h=h
            ).contract(all, output_inds=())
            assert Z2 == pytest.approx(Z3)
            Z4 = qtn.HTN_classical_partition_function_from_edges(
                G.edges, beta=beta, j=j, h=h
            ).contract(all, output_inds=())
            assert Z3 == pytest.approx(Z4)

    @pytest.mark.parametrize('Lx', [2])
    @pytest.mark.parametrize('Ly', [3])
    @pytest.mark.parametrize('Lz', [4])
    @pytest.mark.parametrize('beta', [0.13, 1 / 4.5])
    @pytest.mark.parametrize('j', [-1.0, +1.0])
    @pytest.mark.parametrize('h', [0.0, 0.1])
    @pytest.mark.parametrize('cyclic',
                             [False, True,
                              (False, True, False), (True, False, True)])
    def test_3D_classical_ising_model(self, Lx, Ly, Lz, beta, j, h, cyclic):
        tn = qtn.TN3D_classical_ising_partition_function(
            Lx, Ly, Lz, beta=beta, j=j, h=h, cyclic=cyclic)
        htn = qtn.HTN3D_classical_ising_partition_function(
            Lx, Ly, Lz, beta=beta, j=j, h=h, cyclic=cyclic)
        Z1 = tn.contract(all, output_inds=())
        Z2 = htn.contract(all, output_inds=())
        assert Z1 == pytest.approx(Z2)

        if not cyclic:
            # skip cyclic as nx has no multibonds for L=2
            import networkx as nx
            G = nx.lattice.grid_graph((Lx, Ly, Lz))
            Z3 = qtn.TN_classical_partition_function_from_edges(
                G.edges, beta=beta, j=j, h=h
            ).contract(all, output_inds=())
            assert Z2 == pytest.approx(Z3)
            Z4 = qtn.HTN_classical_partition_function_from_edges(
                G.edges, beta=beta, j=j, h=h
            ).contract(all, output_inds=())
            assert Z3 == pytest.approx(Z4)

    def test_2d_classical_ising_varying_j(self):
        L = 5
        beta = 0.3
        edges = qtn.edges_2d_square(L, L)
        np.random.seed(666)
        js = {
            edge: np.random.normal()
            for edge in edges
        }
        tn = qtn.TN_classical_partition_function_from_edges(
            edges, beta=beta, j=lambda i, j: js[i, j])
        assert tn.dtype == 'float64'
        x0 = tn.contract(all, output_inds=())
        tn = qtn.HTN_classical_partition_function_from_edges(
            edges, beta=beta, j=lambda i, j: js[i, j])
        assert tn.dtype == 'float64'
        x1 = tn.contract(all, output_inds=())
        tn = qtn.TN2D_classical_ising_partition_function(
            L, L, beta=beta,  j=lambda i, j: js[i, j])
        assert tn.dtype == 'float64'
        x2 = tn.contract(all, output_inds=())
        tn = qtn.HTN2D_classical_ising_partition_function(
            L, L, beta=beta,  j=lambda i, j: js[i, j])
        assert tn.dtype == 'float64'
        x3 = tn.contract(all, output_inds=())
        assert x0 == pytest.approx(x1)
        assert x1 == pytest.approx(x2)
        assert x2 == pytest.approx(x3)

    def test_3d_classical_ising_varying_j(self):
        L = 3
        beta = 0.3
        edges = qtn.edges_3d_cubic(L, L, L)
        np.random.seed(666)
        js = {
            edge: np.random.normal()
            for edge in edges
        }
        tn = qtn.TN_classical_partition_function_from_edges(
            edges, beta=beta, j=lambda i, j: js[i, j])
        assert tn.dtype == 'float64'
        x0 = tn.contract(all, output_inds=())
        tn = qtn.HTN_classical_partition_function_from_edges(
            edges, beta=beta, j=lambda i, j: js[i, j])
        assert tn.dtype == 'float64'
        x1 = tn.contract(all, output_inds=())
        tn = qtn.TN3D_classical_ising_partition_function(
            L, L, L, beta=beta,  j=lambda i, j: js[i, j])
        assert tn.dtype == 'float64'
        x2 = tn.contract(all, output_inds=())
        tn = qtn.HTN3D_classical_ising_partition_function(
            L, L, L, beta=beta,  j=lambda i, j: js[i, j])
        assert tn.dtype == 'float64'
        x3 = tn.contract(all, output_inds=())
        assert x0 == pytest.approx(x1)
        assert x1 == pytest.approx(x2)
        assert x2 == pytest.approx(x3)

    def test_tn_dimer_covering(self):
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        tn = qtn.TN_dimer_covering_from_edges(edges, cover_count=1)
        assert tn ^ all == pytest.approx(2.0)
        tn = qtn.TN_dimer_covering_from_edges(edges, cover_count=2)
        assert tn ^ all == pytest.approx(1.0)
        edges = [(0, 1), (1, 2), (2, 0)]
        tn = qtn.TN_dimer_covering_from_edges(edges, cover_count=1)
        assert tn ^ all == pytest.approx(0.0)

    def test_tn2d_fillers(self):
        tn = qtn.TN2D_empty(Lx=2, Ly=2, D=2)
        assert isinstance(tn, qtn.TensorNetwork2D)
        assert (
            (qtn.TN2D_rand(Lx=2, Ly=2, D=2, seed=42) ^ all) ==
            pytest.approx(qtn.TN2D_rand(Lx=2, Ly=2, D=2, seed=42) ^ all)
        )
        tn = qtn.TN2D_with_value(1.0, Lx=2, Ly=3, D=4)
        assert tn ^ all == pytest.approx(qu.prod(tn.ind_sizes().values()))

    def test_tn3d_fillers(self):
        tn = qtn.TN3D_empty(Lx=2, Ly=2, Lz=2, D=2)
        assert isinstance(tn, qtn.TensorNetwork3D)
        assert (
            (qtn.TN3D_rand(Lx=2, Ly=2, Lz=2, D=2, seed=42) ^ all) ==
            pytest.approx(qtn.TN3D_rand(Lx=2, Ly=2, Lz=2, D=2, seed=42) ^ all)
        )
        tn = qtn.TN3D_with_value(1.0, Lx=2, Ly=3, Lz=2, D=2)
        assert tn ^ all == pytest.approx(qu.prod(tn.ind_sizes().values()))
