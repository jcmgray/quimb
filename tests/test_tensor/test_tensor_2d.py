import pytest

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


class Test2DContract:

    def test_contract_2d_one_layer_boundary(self):
        psi = qtn.PEPS.rand(4, 4, 3, seed=42)
        norm = psi.H & psi

        # hopefully uneccesary at some point
        norm.view_as_(qtn.TensorNetwork2D, like=psi)

        # flatten
        for i, j in norm.gen_site_coos():
            norm ^= (i, j)

        xe = norm.contract(all, optimize='auto-hq')
        xt = qtn.contract_2d_one_layer_boundary(norm, max_bond=9)
        assert xt == pytest.approx(xe, rel=1e-2)

    def test_contract_2d_two_layer_boundary(self):
        psi = qtn.PEPS.rand(4, 4, 3, seed=42, tags='KET')
        norm = psi.retag({'KET': 'BRA'}).H & psi

        # hopefully uneccesary at some point
        norm.view_as_(qtn.TensorNetwork2D, like=psi)

        xe = norm.contract(all, optimize='auto-hq')
        xt = qtn.contract_2d_two_layer_boundary(norm, max_bond=18)
        assert xt == pytest.approx(xe, rel=1e-2)
