import pytest
import numpy as np
import itertools
from quimb.tensor.block_interface import BondInfo, SparseFermionTensor, U11, U1, Z4, Z2
from quimb.tensor.fermion_gen import gen_mf_peps

@pytest.fixture(scope='class')
def u11setup(request):
    bond = BondInfo({U11(0):1, U11(2): 1, U11(1,-1):1, U11(1,1):1})
    G = SparseFermionTensor.random((bond, bond), pattern="+-").to_flat()
    Hij = SparseFermionTensor.random((bond,)*4, pattern="++--").to_flat()
    request.cls.G = G
    request.cls.Hij = Hij
    request.cls.Lx = Lx = 3
    request.cls.Ly = Ly = 3
    request.cls.state_array = state_array = np.random.randint(0, 4, Lx*Ly).reshape(Lx, Ly)
    request.cls.peps = gen_mf_peps(state_array, symmetry="U11")
    for itsr in request.cls.peps.tensor_map.values():
        itsr.data.data *= np.random.random(itsr.data.data.size) * 5

@pytest.fixture(scope='class')
def u1setup(request):
    bond = BondInfo({U1(0):1, U1(1): 2, U1(2):1})
    G = SparseFermionTensor.random((bond, bond), pattern="+-").to_flat()
    Hij = SparseFermionTensor.random((bond,)*4, pattern="++--").to_flat()
    request.cls.G = G
    request.cls.Hij = Hij
    request.cls.Lx = Lx = 3
    request.cls.Ly = Ly = 3
    request.cls.state_array = state_array = np.random.randint(0, 4, Lx*Ly).reshape(Lx, Ly)
    request.cls.peps = gen_mf_peps(state_array, symmetry="U1")
    for itsr in request.cls.peps.tensor_map.values():
        itsr.data.data *= np.random.random(itsr.data.data.size) * 5

@pytest.fixture(scope='class')
def z4setup(request):
    bond = BondInfo({Z4(0):2, Z4(1): 2})
    G = SparseFermionTensor.random((bond, bond), pattern="+-").to_flat()
    Hij = SparseFermionTensor.random((bond,)*4, pattern="++--").to_flat()
    request.cls.G = G
    request.cls.Hij = Hij
    request.cls.Lx = Lx = 3
    request.cls.Ly = Ly = 3
    request.cls.state_array = state_array = np.random.randint(0, 4, Lx*Ly).reshape(Lx, Ly)
    request.cls.peps = gen_mf_peps(state_array, symmetry="z4")
    for itsr in request.cls.peps.tensor_map.values():
        itsr.data.data *= np.random.random(itsr.data.data.size) * 5

@pytest.fixture(scope='class')
def z2setup(request):
    bond = BondInfo({Z2(0):2, Z2(1): 2})
    G = SparseFermionTensor.random((bond, bond), pattern="+-").to_flat()
    Hij = SparseFermionTensor.random((bond,)*4, pattern="++--").to_flat()
    request.cls.G = G
    request.cls.Hij = Hij
    request.cls.Lx = Lx = 3
    request.cls.Ly = Ly = 3
    request.cls.state_array = state_array = np.random.randint(0, 4, Lx*Ly).reshape(Lx, Ly)
    request.cls.peps = gen_mf_peps(state_array, symmetry="z2")
    for itsr in request.cls.peps.tensor_map.values():
        itsr.data.data *= np.random.random(itsr.data.data.size) * 5

@pytest.mark.usefixtures('u11setup')
class TestPEPS_U11:
    @pytest.mark.parametrize('where', [
        (0, 0), (0, 1), (0, 2), (2, 0),
        (1, 0), (1, 1), (1, 2), (2, 1)
    ])
    @pytest.mark.parametrize('contract', [False, True])
    def test_gate_2d_single_site(self, where, contract):
        bond = BondInfo({U11(0):1, U11(2): 1, U11(1,-1):1, U11(1,1):1})
        G = self.G
        Lx = 3
        Ly = 3
        psi = self.peps
        xe = psi.compute_local_expectation({where:  G})
        tn = psi.H & psi.gate(G, where, contract=contract)
        assert len(tn.tensors) == 2 * Lx * Ly + int(not contract)
        assert tn ^ all == pytest.approx(xe)

    @pytest.mark.parametrize(
        'contract', [False, True, 'split', 'reduce-split'])
    @pytest.mark.parametrize('where', [
        [(1, 1), (2, 1)], [(2, 1), (2, 2)]
    ])
    def test_gate_2d_two_site(self, where, contract):
        Hij = self.Hij
        psi = self.peps
        xe = psi.compute_local_expectation({tuple(where):  Hij})
        tn = psi.H & psi.gate(Hij, tuple(where), contract=contract)
        change = {False: 1, True: -1, 'split': 0, 'reduce-split': 0}[contract]
        assert len(tn.tensors) == 2 * self.Lx * self.Ly + change
        assert tn ^ all == pytest.approx(xe)

    def test_contract_2d_one_layer_boundary(self):
        psi = self.peps
        norm = psi.make_norm()
        xe = norm.contract(all, optimize='auto-hq')
        xt = norm.contract_boundary(max_bond=6)
        assert xt == pytest.approx(xe, rel=1e-2)

    def test_contract_2d_two_layer_boundary(self):
        psi = self.peps
        norm = psi.make_norm()
        xe = norm.contract(all, optimize='auto-hq')
        xt = norm.contract_boundary(max_bond=6, layer_tags=['KET', 'BRA'])
        assert xt == pytest.approx(xe, rel=1e-2)

    @pytest.mark.parametrize("two_layer", [False, True])
    def test_compute_row_envs(self, two_layer):
        psi = self.peps
        norm = psi.make_norm()
        ex = norm.contract(all)
        if two_layer:
            compress_opts = {'cutoff': 1e-6, 'max_bond': 12,
                             'layer_tags': ['KET', 'BRA']}
        else:
            compress_opts = {'cutoff': 1e-6, 'max_bond': 8}
        row_envs = norm.compute_row_environments(**compress_opts)

        for i in range(norm.Lx):
            norm_i = (
                row_envs['below', i] &
                row_envs['mid', i] &
                row_envs['above', i]
            )
            x = norm_i.contract(all)
            assert x == pytest.approx(ex, rel=1e-2)

    @pytest.mark.parametrize("two_layer", [False, True])
    def test_compute_col_envs(self, two_layer):
        psi = self.peps
        norm = psi.make_norm()
        ex = norm.contract(all)
        if two_layer:
            compress_opts = {'cutoff': 1e-6, 'max_bond': 12,
                             'layer_tags': ['KET', 'BRA']}
        else:
            compress_opts = {'cutoff': 1e-6, 'max_bond': 8}
        row_envs = norm.compute_col_environments(**compress_opts)

        for i in range(norm.Ly):
            norm_i = (
                row_envs['left', i] &
                row_envs['mid', i] &
                row_envs['right', i]
            )
            x = norm_i.contract(all)
            assert x == pytest.approx(ex, rel=1e-2)

    def test_normalize(self):
        psi = self.peps
        norm = psi.make_norm().contract(all)
        assert norm != pytest.approx(1.0)
        psi.normalize_(balance_bonds=True, equalize_norms=True, cutoff=2e-3)
        norm = psi.make_norm().contract(all)
        assert norm == pytest.approx(1.0, rel=1e-2)

    def test_compute_local_expectation_one_sites(self):
        peps = self.peps
        coos = list(itertools.product(range(self.Lx), range(self.Ly)))
        terms = {coo: self.G for coo in coos}

        expecs = peps.compute_local_expectation(
            terms,
            normalized=True,
            return_all=True)

        norm = peps.compute_norm()
        for where, G in terms.items():
            ket = peps.copy()
            ket.add_tag("KET")
            bra = ket.H
            bra.retag({"KET": "BRA"})
            bra.mangle_inner_("*")
            ket.gate_(G, where)
            tn = ket & bra
            out = tn.contract_boundary(max_bond=12)
            assert out == pytest.approx(expecs[where][0], rel=1e-2)
            assert norm == pytest.approx(expecs[where][1], rel=1e-2)

    def test_compute_local_expectation_two_sites(self):
        normalized=True
        peps = self.peps
        Hij = self.Hij
        hterms = {coos: Hij for coos in peps.gen_horizontal_bond_coos()}
        vterms = {coos: Hij for coos in peps.gen_vertical_bond_coos()}

        opts = dict(cutoff=2e-3, max_bond=12, contract_optimize='random-greedy')
        norm = peps.compute_norm(max_bond=12, cutoff=2e-3)
        he = peps.compute_local_expectation(
            hterms, normalized=normalized, return_all=True, **opts)
        ve = peps.compute_local_expectation(
            vterms, normalized=normalized, return_all=True, **opts)

        for where, G in hterms.items():
            ket = peps.copy()
            ket.add_tag("KET")
            bra = ket.H
            bra.retag({"KET": "BRA"})
            bra.mangle_inner_("*")
            ket.gate_(G, where, contract="reduce-split")
            tn = ket & bra
            out = tn.contract_boundary(max_bond=12, cutoff=2e-3)
            assert out == pytest.approx(he[where][0], rel=1e-2)
            assert norm == pytest.approx(he[where][1], rel=1e-2)

        for where, G in vterms.items():
            ket = peps.copy()
            ket.add_tag("KET")
            bra = ket.H
            bra.retag({"KET": "BRA"})
            bra.mangle_inner_("*")
            ket.gate_(G, where, contract="split")
            tn = ket & bra
            out = tn.contract_boundary(max_bond=12, cutoff=2e-3)
            assert out == pytest.approx(ve[where][0], rel=1e-2)
            assert norm == pytest.approx(ve[where][1], rel=1e-2)

@pytest.mark.usefixtures('u1setup')
class TestPEPS_U1(TestPEPS_U11):
    pass

@pytest.mark.usefixtures('z4setup')
class TestPEPS_Z4(TestPEPS_U11):
    pass

@pytest.mark.usefixtures('z2setup')
class TestPEPS_Z2(TestPEPS_U11):
    pass
