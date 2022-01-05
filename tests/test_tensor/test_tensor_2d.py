import itertools

import pytest
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn


class TestPEPSConstruct:

    @pytest.mark.parametrize('Lx', [3, 4, 5])
    @pytest.mark.parametrize('Ly', [3, 4, 5])
    def test_basic_rand(self, Lx, Ly):
        psi = qtn.PEPS.rand(Lx, Ly, bond_dim=4)

        assert psi.max_bond() == 4
        assert psi.Lx == Lx
        assert psi.Ly == Ly
        assert len(psi.tensor_map) == Lx * Ly
        assert psi.site_inds == tuple(
            f'k{i},{j}' for i in range(Lx) for j in range(Ly)
        )
        assert psi.site_tags == tuple(
            f'I{i},{j}' for i in range(Lx) for j in range(Ly)
        )

        assert psi.bond_size((1, 1), (1, 2)) == (4)

        for i in range(Lx):
            assert len(psi.select(f'ROW{i}').tensor_map) == Ly
        for j in range(Ly):
            assert len(psi.select(f'COL{j}').tensor_map) == Lx

        for i in range(Lx):
            for j in range(Ly):
                assert psi.phys_dim(i, j) == 2
                assert isinstance(psi[i, j], qtn.Tensor)
                assert isinstance(psi[f'I{i},{j}'], qtn.Tensor)

        if Lx == Ly == 3:
            psi_dense = psi.to_dense(optimize='random-greedy')
            assert psi_dense.shape == (512, 1)

        psi.show()
        assert f'Lx={Lx}' in psi.__str__()
        assert f'Lx={Lx}' in psi.__repr__()

    def test_flatten(self):
        psi = qtn.PEPS.rand(3, 5, 3, seed=42)
        norm = psi.H & psi
        assert len(norm.tensors) == 30
        norm.flatten_()
        assert len(norm.tensors) == 15
        assert norm.max_bond() == 9

    def test_add_peps(self):
        pa = qtn.PEPS.rand(3, 4, 2)
        pb = qtn.PEPS.rand(3, 4, 3)
        pc = qtn.PEPS.rand(3, 4, 4)
        pab = pa + pb
        assert pab.max_bond() == 5
        assert pab @ pc == pytest.approx(pa @ pc + pb @ pc)

    @pytest.mark.parametrize('Lx', [3, 4, 5])
    @pytest.mark.parametrize('Ly', [3, 4, 5])
    def test_bond_coordinates(self, Lx, Ly):
        psi = qtn.PEPS.rand(Lx, Ly, bond_dim=1)
        all_bonds = tuple(psi.gen_bond_coos())
        assert len(all_bonds) == 2 * Lx * Ly - Lx - Ly
        he = tuple(psi.gen_horizontal_even_bond_coos())
        ho = tuple(psi.gen_horizontal_odd_bond_coos())
        ve = tuple(psi.gen_vertical_even_bond_coos())
        vo = tuple(psi.gen_vertical_odd_bond_coos())
        for p in (he, ho, ve, vo):
            assert len(set(p)) == len(p)
            # check there is no overlap at all
            sites = tuple(itertools.chain.from_iterable(he))
            assert len(set(sites)) == len(sites)
        # check all coordinates are generated
        assert set(itertools.chain(he, ho, ve, vo)) == set(all_bonds)

    @pytest.mark.parametrize('where', [
        [(0, 0)], [(0, 1)], [(0, 2)], [(2, 2)],
        [(3, 2)], [(3, 1)], [(3, 0)], [(2, 0)], [(1, 1)],
    ])
    @pytest.mark.parametrize('contract', [False, True])
    def test_gate_2d_single_site(self, where, contract):
        Lx = 4
        Ly = 3
        D = 2

        psi = qtn.PEPS.rand(Lx, Ly, bond_dim=D, seed=42, dtype=complex)
        psi_d = psi.to_dense()
        G = qu.rand_matrix(2)

        # compute the exact dense reference
        dims = [[2] * Ly] * Lx
        IGI = qu.ikron(G, dims, where, sparse=True)
        xe = (psi_d.H @ IGI @ psi_d).item()

        tn = psi.H & psi.gate(G, where, contract=contract)
        assert len(tn.tensors) == 2 * Lx * Ly + int(not contract)

        assert tn ^ all == pytest.approx(xe)

    @pytest.mark.parametrize(
        'contract', [False, True, 'split', 'reduce-split'])
    @pytest.mark.parametrize('where', [
        [(1, 1), (2, 1)], [(3, 2), (2, 2)],
        [(0, 0), (1, 1)], [(3, 1), (1, 2)]
    ])
    def test_gate_2d_two_site(self, where, contract):
        Lx = 4
        Ly = 3
        D = 2

        psi = qtn.PEPS.rand(Lx, Ly, bond_dim=D, seed=42, dtype=complex)
        psi_d = psi.to_dense()

        # ikron can't tensor operators across non-adjacent subsytems
        # so we explicitly construct the gate as a sum of tensor components
        G_comps = [(qu.rand_matrix(2), qu.rand_matrix(2)) for _ in range(4)]
        G = sum(A & B for A, B in G_comps)

        # compute the exact dense reference
        dims = [[2] * Ly] * Lx
        IGI = sum(qu.ikron([A, B], dims, where, sparse=True)
                  for A, B in G_comps)

        xe = (psi_d.H @ IGI @ psi_d).item()

        tn = psi.H & psi.gate(G, where, contract=contract)
        change = {False: 1, True: -1, 'split': 0, 'reduce-split': 0}[contract]
        assert len(tn.tensors) == 2 * Lx * Ly + change

        assert tn ^ all == pytest.approx(xe)

    @pytest.mark.parametrize('propagate_tags',
                             [False, True, 'sites', 'register'])
    def test_gate_propagate_tags(self, propagate_tags):
        Lx = 4
        Ly = 3
        D = 1
        psi = qtn.PEPS.rand(Lx, Ly, D, tags='PSI0')
        psi.gate_(qu.rand_uni(4), [(1, 1), (1, 2)], tags='G1',
                  propagate_tags=propagate_tags)
        psi.gate_(qu.rand_uni(4), [(1, 2), (3, 2)], tags='G2',
                  propagate_tags=propagate_tags)
        if propagate_tags is False:
            assert set(psi['G1'].tags) == {'G1'}
            assert set(psi['G2'].tags) == {'G2'}
        if propagate_tags is True:
            tgs1 = {'I1,1', 'I1,2', 'G1', 'PSI0', 'COL1', 'COL2', 'ROW1'}
            assert set(psi['G1'][0].tags) == tgs1
            assert set(psi['G2'].tags) == tgs1 | {'G2', 'I3,2', 'ROW3', 'COL2'}
        if propagate_tags == 'sites':
            assert set(psi['G1'].tags) == {'G1', 'I1,1', 'I1,2'}
            assert set(psi['G2'].tags) == {'G2', 'I1,1', 'I1,2', 'I3,2'}
        if propagate_tags == 'register':
            assert set(psi['G1'].tags) == {'G1', 'I1,1', 'I1,2'}
            assert set(psi['G2'].tags) == {'G2', 'I1,2', 'I3,2'}


class Test2DContract:

    def test_contract_2d_one_layer_boundary(self):
        psi = qtn.PEPS.rand(4, 4, 3, seed=42)
        norm = psi.make_norm()
        xe = norm.contract(all, optimize='auto-hq')
        xt = norm.contract_boundary(max_bond=9)
        assert xt == pytest.approx(xe, rel=1e-2)

    def test_contract_2d_two_layer_boundary(self):
        psi = qtn.PEPS.rand(4, 4, 3, seed=42, tags='KET')
        norm = psi.make_norm()
        xe = norm.contract(all, optimize='auto-hq')
        xt = norm.contract_boundary(max_bond=27, layer_tags=['KET', 'BRA'])
        assert xt == pytest.approx(xe, rel=1e-2)

    def test_contract_2d_full_bond(self):
        psi = qtn.PEPS.rand(4, 4, 3, seed=42, tags='KET')
        norm = psi.make_norm()
        xe = norm.contract(all, optimize='auto-hq')
        xt = norm.contract_boundary(max_bond=27, mode='full-bond')
        assert xt == pytest.approx(xe, rel=1e-2)

    @pytest.mark.parametrize("mode,two_layer", [
        ('mps', False),
        ('mps', True),
        ('full-bond', False),
    ])
    def test_compute_row_envs(self, mode, two_layer):
        psi = qtn.PEPS.rand(5, 4, 2, seed=42, tags='KET')
        norm = psi.make_norm()
        ex = norm.contract(all)

        if two_layer:
            compress_opts = {'cutoff': 1e-6, 'max_bond': 12, 'mode': mode,
                             'layer_tags': ['KET', 'BRA']}
        else:
            compress_opts = {'cutoff': 1e-6, 'max_bond': 8, 'mode': mode}
        row_envs = norm.compute_row_environments(**compress_opts)

        for i in range(norm.Lx):
            norm_i = (
                row_envs['bottom', i] &
                norm.select(norm.row_tag(i)) &
                row_envs['top', i]
            )
            x = norm_i.contract(all)
            assert x == pytest.approx(ex, rel=1e-2)

    @pytest.mark.parametrize("mode,two_layer", [
        ('mps', False),
        ('mps', True),
        ('full-bond', False),
    ])
    def test_compute_col_envs(self, mode, two_layer):
        psi = qtn.PEPS.rand(4, 5, 2, seed=42, tags='KET')
        norm = psi.retag({'KET': 'BRA'}).H | psi
        ex = norm.contract(all)

        if two_layer:
            compress_opts = {'cutoff': 1e-6, 'max_bond': 12, 'mode': mode,
                             'layer_tags': ['KET', 'BRA']}
        else:
            compress_opts = {'cutoff': 1e-6, 'max_bond': 8, 'mode': mode}
        col_envs = norm.compute_col_environments(**compress_opts)

        for j in range(norm.Lx):
            norm_j = (
                col_envs['left', j] &
                norm.select(norm.col_tag(j)) &
                col_envs['right', j]
            )
            x = norm_j.contract(all)
            assert x == pytest.approx(ex, rel=1e-2)

    def test_normalize(self):
        psi = qtn.PEPS.rand(4, 5, 2, seed=42)
        norm = (psi.H | psi).contract(all)
        assert norm != pytest.approx(1.0)
        psi.normalize_(balance_bonds=True, equalize_norms=True, cutoff=2e-3)
        norm = (psi.H | psi).contract(all)
        assert norm == pytest.approx(1.0, rel=0.01)

    @pytest.mark.parametrize('normalized', [False, True])
    @pytest.mark.parametrize('mode', ['mps', 'full-bond'])
    def test_compute_local_expectation_one_sites(self, mode, normalized):
        peps = qtn.PEPS.rand(4, 3, 2, seed=42, dtype='complex')

        # reference
        k = peps.to_dense()
        if normalized:
            qu.normalize(k)
        coos = list(itertools.product([0, 2, 3], [0, 1, 2]))
        terms = {coo: qu.rand_matrix(2) for coo in coos}
        dims = [[2] * 3] * 4
        A = sum(qu.ikron(A, dims, [coo], sparse=True)
                for coo, A in terms.items())
        ex = qu.expec(A, k)

        opts = dict(cutoff=2e-3, max_bond=9, contract_optimize='random-greedy')
        e = peps.compute_local_expectation(
            terms, mode=mode, normalized=normalized, **opts)

        assert e == pytest.approx(ex, rel=1e-2)

    @pytest.mark.parametrize('normalized', [False, True])
    @pytest.mark.parametrize('mode', ['mps', 'full-bond'])
    def test_compute_local_expectation_two_sites(self, mode, normalized):
        H = qu.ham_heis_2D(4, 3, sparse=True)
        Hij = qu.ham_heis(2, cyclic=False)

        peps = qtn.PEPS.rand(4, 3, 2, seed=42)
        k = peps.to_dense()

        if normalized:
            qu.normalize(k)
        ex = qu.expec(H, k)

        opts = dict(
            mode=mode,
            normalized=normalized,
            cutoff=2e-3,
            max_bond=16,
            contract_optimize='random-greedy'
        )

        # compute 2x1 and 1x2 plaquettes separately
        hterms = {coos: Hij for coos in peps.gen_horizontal_bond_coos()}
        vterms = {coos: Hij for coos in peps.gen_vertical_bond_coos()}

        he = peps.compute_local_expectation(hterms, **opts)
        ve = peps.compute_local_expectation(vterms, **opts)

        assert he + ve == pytest.approx(ex, rel=1e-2)

        # compute all terms in 2x2 plaquettes
        terms_all = {**hterms, **vterms}
        e = peps.compute_local_expectation(terms_all, autogroup=False, **opts)

        assert e == pytest.approx(ex, rel=1e-2)


class TestPEPO:

    @pytest.mark.parametrize('Lx', [3, 4, 5])
    @pytest.mark.parametrize('Ly', [3, 4, 5])
    def test_basic_rand(self, Lx, Ly):
        X = qtn.PEPO.rand_herm(Lx, Ly, bond_dim=4)

        assert X.max_bond() == 4
        assert X.Lx == Lx
        assert X.Ly == Ly
        assert len(X.tensor_map) == Lx * Ly
        assert X.upper_inds == tuple(
            f'k{i},{j}' for i in range(Lx) for j in range(Ly)
        )
        assert X.lower_inds == tuple(
            f'b{i},{j}' for i in range(Lx) for j in range(Ly)
        )
        assert X.site_tags == tuple(
            f'I{i},{j}' for i in range(Lx) for j in range(Ly)
        )

        assert X.bond_size((1, 1), (1, 2)) == (4)

        for i in range(Lx):
            assert len(X.select(f'ROW{i}').tensor_map) == Ly
        for j in range(Ly):
            assert len(X.select(f'COL{j}').tensor_map) == Lx

        for i in range(Lx):
            for j in range(Ly):
                assert X.phys_dim(i, j) == 2
                assert isinstance(X[i, j], qtn.Tensor)
                assert isinstance(X[f'I{i},{j}'], qtn.Tensor)

        if Lx == Ly == 3:
            X_dense = X.to_dense(optimize='random-greedy')
            assert X_dense.shape == (512, 512)
            assert qu.isherm(X_dense)

        X.show()
        assert f'Lx={Lx}' in X.__str__()
        assert f'Lx={Lx}' in X.__repr__()

    def test_add_pepo(self):
        pa = qtn.PEPO.rand(3, 4, 2)
        pb = qtn.PEPO.rand(3, 4, 3)
        pc = qtn.PEPO.rand(3, 4, 4)
        pab = pa + pb
        assert pab.max_bond() == 5
        assert pab @ pc == pytest.approx(pa @ pc + pb @ pc)

    def test_apply_pepo(self):
        A = qtn.PEPO.rand(Lx=3, Ly=2, bond_dim=2, seed=1)
        x = qtn.PEPS.rand(Lx=3, Ly=2, bond_dim=2, seed=0)
        y = A.apply(x)
        assert y.num_indices == x.num_indices
        Ad = A.to_dense()
        xd = x.to_dense()
        yd = y.to_dense()
        assert_allclose(Ad @ xd, yd)
        yc = A.apply(x, compress=True, max_bond=3)
        assert yc.max_bond() == 3


class TestMisc:

    def test_calc_plaquette_sizes(self):
        from quimb.tensor.tensor_2d import calc_plaquette_sizes
        H2 = {None: qu.ham_heis(2)}
        ham = qtn.LocalHam2D(10, 10, H2)
        assert calc_plaquette_sizes(ham.terms.keys()) == ((1, 2), (2, 1))
        assert (calc_plaquette_sizes(ham.terms.keys(), autogroup=False) ==
                ((2, 2),))
        H2[(1, 1), (2, 2)] = 0.5 * qu.ham_heis(2)
        ham = qtn.LocalHam2D(10, 10, H2)
        assert calc_plaquette_sizes(ham.terms.keys()) == ((2, 2),)
        H2[(2, 2), (2, 4)] = 0.25 * qu.ham_heis(2)
        H2[(2, 4), (4, 4)] = 0.25 * qu.ham_heis(2)
        ham = qtn.LocalHam2D(10, 10, H2)
        assert (calc_plaquette_sizes(ham.terms.keys()) ==
                ((1, 3), (2, 2), (3, 1)))
        assert (calc_plaquette_sizes(ham.terms.keys(), autogroup=False) ==
                ((3, 3),))

    def test_calc_plaquette_map(self):
        from quimb.tensor.tensor_2d import calc_plaquette_map
        plaquettes = [
            # 2x2 plaquette covering all sites
            ((0, 0), (2, 2)),
            # horizontal plaquettes
            ((0, 0), (1, 2)),
            ((1, 0), (1, 2)),
            # vertical plaquettes
            ((0, 0), (2, 1)),
            ((0, 1), (2, 1)),
        ]
        assert (
            calc_plaquette_map(plaquettes) ==
            {(0, 0): ((0, 0), (2, 1)),
             (0, 1): ((0, 1), (2, 1)),
             (1, 0): ((1, 0), (1, 2)),
             (1, 1): ((1, 0), (1, 2)),
             ((0, 0), (0, 1)): ((0, 0), (1, 2)),
             ((0, 0), (1, 0)): ((0, 0), (2, 1)),
             ((0, 0), (1, 1)): ((0, 0), (2, 2)),
             ((0, 1), (1, 0)): ((0, 0), (2, 2)),
             ((0, 1), (1, 1)): ((0, 1), (2, 1)),
             ((1, 0), (1, 1)): ((1, 0), (1, 2))}
        )
