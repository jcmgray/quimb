import pytest
import autoray as ar

import quimb as qu
import quimb.tensor as qtn


class TestTensorNetwork3D:
    def test_cyclic_basic(self):
        tn = qtn.TN3D_empty(Lx=3, Ly=4, Lz=5, D=2, cyclic=True)
        assert tn.is_cyclic_x()
        assert tn.is_cyclic_y()
        assert tn.is_cyclic_z()
        assert tn.num_indices == 3 * tn.nsites
        tn = qtn.TN3D_empty(Lx=3, Ly=4, Lz=5, D=2, cyclic=(False, False, True))
        assert not tn.is_cyclic_x()
        assert not tn.is_cyclic_y()
        assert tn.is_cyclic_z()
        assert tn.num_indices == 3 * tn.nsites - (tn.Lx * tn.Lz) - (
            tn.Ly * tn.Lz
        )
        tn = qtn.TN3D_empty(Lx=3, Ly=4, Lz=5, D=2, cyclic=(False, True, False))
        assert not tn.is_cyclic_x()
        assert tn.is_cyclic_y()
        assert not tn.is_cyclic_z()
        assert tn.num_indices == 3 * tn.nsites - (tn.Lx * tn.Ly) - (
            tn.Ly * tn.Lz
        )
        tn = qtn.TN3D_empty(Lx=3, Ly=4, Lz=5, D=2, cyclic=(True, False, False))
        assert tn.is_cyclic_x()
        assert not tn.is_cyclic_y()
        assert not tn.is_cyclic_z()
        assert tn.num_indices == 3 * tn.nsites - (tn.Lx * tn.Ly) - (
            tn.Lx * tn.Lz
        )


class Test3DManualContract:
    @pytest.mark.parametrize("canonize", [False, True])
    def test_contract_boundary_ising_model(self, canonize):
        L = 5
        beta = 0.3
        fex = -2.7654417752878
        tn = qtn.TN3D_classical_ising_partition_function(L, L, L, beta=beta)
        Z = tn.contract_boundary(max_bond=8, canonize=canonize)
        f = -qu.log(Z) / (L**3 * beta)
        assert f == pytest.approx(fex, rel=1e-3)

    @pytest.mark.parametrize("dims", [(10, 4, 3), (4, 3, 10), (3, 10, 4)])
    def test_contract_boundary_stopping_criterion(self, dims):
        tn = qtn.TN3D_from_fill_fn(
            lambda shape: ar.lazy.Variable(shape=shape, backend="numpy"),
            *dims,
            D=2,
        )
        tn.contract_boundary_(
            4, cutoff=0.0, final_contract=False, progbar=True
        )
        assert tn.max_bond() == 4
        assert 32 <= tn.num_tensors <= 40

    @pytest.mark.parametrize("lazy", [False, True])
    def test_coarse_grain_basics(self, lazy):
        tn = qtn.TN3D_from_fill_fn(
            lambda shape: ar.lazy.Variable(shape, backend="numpy"),
            Lx=6,
            Ly=7,
            Lz=8,
            D=2,
        )
        tncg = tn.coarse_grain_hotrg("x", max_bond=3, cutoff=0.0, lazy=lazy)
        assert (tncg.Lx, tncg.Ly, tncg.Lz) == (3, 7, 8)
        assert not tncg.outer_inds()
        assert tncg.max_bond() == 3
        assert "I4,0,0" not in tncg.tag_map
        assert "X3" not in tncg.tag_map

        tncg = tn.coarse_grain_hotrg("y", max_bond=3, cutoff=0.0, lazy=lazy)
        assert (tncg.Lx, tncg.Ly, tncg.Lz) == (6, 4, 8)
        assert not tncg.outer_inds()
        assert tncg.max_bond() == 3
        assert "I0,5,0" not in tncg.tag_map
        assert "Y4" not in tncg.tag_map

        tncg = tn.coarse_grain_hotrg("z", max_bond=3, cutoff=0.0, lazy=lazy)
        assert (tncg.Lx, tncg.Ly, tncg.Lz) == (6, 7, 4)
        assert "I0,0,5" not in tncg.tag_map
        assert "Z4" not in tncg.tag_map

    def test_contract_hotrg_ising_model(self):
        L = 5
        beta = 0.3
        fex = -2.7654417752878
        tn = qtn.TN3D_classical_ising_partition_function(L, L, L, beta=beta)
        tn.contract_hotrg_(max_bond=4, progbar=True, equalize_norms=1.0)
        Z = tn.item() * 10**tn.exponent
        f = -qu.log(Z) / (L**3 * beta)
        assert f == pytest.approx(fex, rel=1e-2)

    @pytest.mark.parametrize("cyclicx", [False, True])
    @pytest.mark.parametrize("cyclicy", [False, True])
    @pytest.mark.parametrize("cyclicz", [False, True])
    @pytest.mark.parametrize("mode", ["hotrg", "ctmrg"])
    def test_contract_cyclic(self, cyclicx, cyclicy, cyclicz, mode):
        Lx, Ly, Lz = 3, 4, 5
        chi = 3
        tn = qtn.TN3D_from_fill_fn(
            lambda shape: ar.lazy.Variable(shape=shape, backend="numpy"),
            Lx,
            Ly,
            Lz,
            D=2,
            cyclic=(cyclicx, cyclicy, cyclicz),
        )
        if mode == "hotrg":
            lZ = tn.contract_hotrg(max_bond=chi, cutoff=0.0)
        elif mode == "ctmrg":
            lZ = tn.contract_ctmrg(max_bond=chi, cutoff=0.0)

        if any((cyclicx, cyclicy, cyclicz)):
            assert lZ.history_max_size() < 2**16
        else:
            assert lZ.history_max_size() < 2**13
