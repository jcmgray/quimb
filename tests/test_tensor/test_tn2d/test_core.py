import itertools

import autoray as ar
import numpy as np
import pytest
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.environments import all_blocks

from .. import (
    bond_orientations,
    make_symmetric_2d_tn,
    requires_symmray,
    symmetry_cases,
)


class TestPEPSConstruct:
    @pytest.mark.parametrize("Lx", [3, 4, 5])
    @pytest.mark.parametrize("Ly", [3, 4, 5])
    def test_basic_rand(self, Lx, Ly):
        psi = qtn.PEPS.rand(Lx, Ly, bond_dim=4)

        assert psi.max_bond() == 4
        assert psi.Lx == Lx
        assert psi.Ly == Ly
        assert len(psi.tensor_map) == Lx * Ly
        assert psi.site_inds == tuple(
            f"k{i},{j}" for i in range(Lx) for j in range(Ly)
        )
        assert psi.site_tags == tuple(
            f"I{i},{j}" for i in range(Lx) for j in range(Ly)
        )

        assert psi.bond_size((1, 1), (1, 2)) == (4)

        for i in range(Lx):
            assert len(psi.select(f"X{i}").tensor_map) == Ly
        for j in range(Ly):
            assert len(psi.select(f"Y{j}").tensor_map) == Lx

        for i in range(Lx):
            for j in range(Ly):
                assert psi.phys_dim(i, j) == 2
                assert isinstance(psi[i, j], qtn.Tensor)
                assert isinstance(psi[f"I{i},{j}"], qtn.Tensor)

        if Lx == Ly == 3:
            psi_dense = psi.to_qarray(optimize="auto-hq")
            assert psi_dense.shape == (512, 1)

        psi.show()
        assert f"Lx={Lx}" in psi.__str__()
        assert f"Lx={Lx}" in psi.__repr__()

    def test_cyclic_edge_cases(self):
        peps = qtn.PEPS.rand(3, 3, bond_dim=1, cyclic=True)
        assert peps.is_cyclic_x()
        assert peps.is_cyclic_y()
        assert peps.num_indices == peps.num_tensors * 3

    def test_cyclic_length_two_keeps_boundary_bonds(self):
        peps = qtn.PEPS.rand(2, 3, bond_dim=1, cyclic=(True, False))
        assert not peps.is_cyclic_y()
        assert len(peps[0, 1].bonds(peps[1, 1])) == 2
        assert peps.num_indices == 16

    def test_cyclic_two_by_two_creates_all_gauges(self):
        peps = qtn.PEPS.rand(2, 2, bond_dim=3, cyclic=True)
        gauges = {}
        peps.gauge_all_simple_(
            max_iterations=1,
            gauges=gauges,
            fuse_multibonds=False,
        )
        assert len(gauges) == 8
        assert all(gauge.shape == (3,) for gauge in gauges.values())

    @pytest.mark.parametrize(
        "Lx,Ly,cyclic,site,repeat",
        [
            (1, 3, (True, False), (0, 1), (0, 2)),
            (3, 1, (False, True), (1, 0), (1, 3)),
        ],
    )
    def test_cyclic_length_one_keeps_self_bond(
        self, Lx, Ly, cyclic, site, repeat
    ):
        peps = qtn.PEPS.rand(Lx, Ly, bond_dim=1, cyclic=cyclic)
        tensor = peps[site]
        assert tensor.inds[repeat[0]] == tensor.inds[repeat[1]]
        assert peps.num_indices == 8

    def test_zeros(self):
        peps = qtn.PEPS.zeros(3, 3, cyclic=True, bond_dim=1)
        assert peps.num_tensors == 9
        assert peps.num_indices == 27
        assert_allclose(peps.to_dense(), np.zeros([512, 1]))

    def test_flatten(self):
        from quimb.tensor.tn2d.core import TensorNetwork2DFlat

        psi = qtn.PEPS.rand(3, 5, 3, seed=42)
        norm = psi.H & psi
        assert len(norm.tensors) == 30
        norm.flatten_()
        assert len(norm.tensors) == 15
        assert norm.max_bond() == 9
        assert norm.__class__ == TensorNetwork2DFlat

    def test_add_peps(self):
        pa = qtn.PEPS.rand(3, 4, 2)
        pb = qtn.PEPS.rand(3, 4, 3)
        pc = qtn.PEPS.rand(3, 4, 4)
        pab = pa + pb
        assert pab.max_bond() == 5
        assert pab @ pc == pytest.approx(pa @ pc + pb @ pc)

    @pytest.mark.parametrize("Lx", [3, 4, 5])
    @pytest.mark.parametrize("Ly", [3, 4, 5])
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

    @pytest.mark.parametrize(
        "where",
        [
            [(0, 0)],
            [(0, 1)],
            [(0, 2)],
            [(2, 2)],
            [(3, 2)],
            [(3, 1)],
            [(3, 0)],
            [(2, 0)],
            [(1, 1)],
        ],
    )
    @pytest.mark.parametrize("contract", [False, True])
    def test_gate_2d_single_site(self, where, contract):
        Lx = 4
        Ly = 3
        D = 2

        psi = qtn.PEPS.rand(Lx, Ly, bond_dim=D, seed=42, dtype=complex)
        psi_d = psi.to_qarray()
        G = qu.rand_matrix(2)

        # compute the exact dense reference
        dims = [[2] * Ly] * Lx
        IGI = qu.ikron(G, dims, where, sparse=True)
        xe = (psi_d.H @ IGI @ psi_d).item()

        tn = psi.H & psi.gate(G, where, contract=contract)
        assert len(tn.tensors) == 2 * Lx * Ly + int(not contract)

        assert tn ^ all == pytest.approx(xe)

    @pytest.mark.parametrize(
        "contract", [False, True, "split", "reduce-split"]
    )
    @pytest.mark.parametrize(
        "where",
        [
            [(1, 1), (2, 1)],
            [(3, 2), (2, 2)],
            # XXX: long range, disable for now
            # [(0, 0), (1, 1)],
            # [(3, 1), (1, 2)],
        ],
    )
    def test_gate_2d_two_site(self, where, contract):
        Lx = 4
        Ly = 3
        D = 2

        psi = qtn.PEPS.rand(Lx, Ly, bond_dim=D, seed=42, dtype=complex)
        psi_d = psi.to_qarray()

        # ikron can't tensor operators across non-adjacent subsytems
        # so we explicitly construct the gate as a sum of tensor components
        G_comps = [(qu.rand_matrix(2), qu.rand_matrix(2)) for _ in range(4)]
        G = sum(A & B for A, B in G_comps)

        # compute the exact dense reference
        dims = [[2] * Ly] * Lx
        IGI = sum(
            qu.ikron([A, B], dims, where, sparse=True) for A, B in G_comps
        )

        xe = (psi_d.H @ IGI @ psi_d).item()

        tn = psi.H & psi.gate(G, where, contract=contract)
        change = {False: 1, True: -1, "split": 0, "reduce-split": 0}[contract]
        assert len(tn.tensors) == 2 * Lx * Ly + change

        assert tn ^ all == pytest.approx(xe)

    @pytest.mark.parametrize(
        "propagate_tags", [False, True, "sites", "register"]
    )
    def test_gate_propagate_tags(self, propagate_tags):
        Lx = 4
        Ly = 3
        D = 1
        psi = qtn.PEPS.rand(Lx, Ly, D, tags="PSI0")
        psi.gate_(
            qu.rand_uni(4),
            [(1, 1), (1, 2)],
            tags="G1",
            propagate_tags=propagate_tags,
        )
        psi.gate_(
            qu.rand_uni(4),
            [(1, 2), (3, 2)],
            tags="G2",
            propagate_tags=propagate_tags,
        )
        if propagate_tags is False:
            assert set(psi["G1"].tags) == {"G1"}
            assert set(psi["G2"].tags) == {"G2"}
        if propagate_tags is True:
            tgs1 = {"I1,1", "I1,2", "G1", "PSI0", "Y1", "Y2", "X1"}
            assert set(psi["G1"][0].tags) == tgs1
            assert set(psi["G2"].tags) == tgs1 | {"G2", "I3,2", "X3", "Y2"}
        if propagate_tags == "sites":
            assert set(psi["G1"].tags) == {"G1", "I1,1", "I1,2"}
            assert set(psi["G2"].tags) == {"G2", "I1,1", "I1,2", "I3,2"}
        if propagate_tags == "register":
            assert set(psi["G1"].tags) == {"G1", "I1,1", "I1,2"}
            assert set(psi["G2"].tags) == {"G2", "I1,2", "I3,2"}


class Test2DContract:
    @pytest.mark.parametrize("method", ["mps", "projector", "full-bond"])
    def test_contract_boundary(self, method):
        # make a large but cheap and easy (mostly positive) TN
        rng = np.random.default_rng(42)
        tn = qtn.TN2D_from_fill_fn(
            lambda shape: rng.uniform(low=-0.1, size=shape),
            Lx=8,
            Ly=8,
            D=2,
        )
        Zex = tn.contract(...)
        Z = tn.contract_boundary(max_bond=4, method=method)
        assert Z == pytest.approx(Zex, rel=1e-3)

    def test_contract_2d_one_layer_boundary(self):
        psi = qtn.PEPS.rand(4, 4, 3, seed=42)
        norm = psi.make_norm()
        xe = norm.contract(all, optimize="auto-hq")
        xt = norm.contract_boundary(max_bond=9)
        assert xt == pytest.approx(xe, rel=1e-2)

    def test_contract_2d_two_layer_boundary(self):
        psi = qtn.PEPS.rand(4, 4, 3, seed=42, tags="KET")
        norm = psi.make_norm()
        xe = norm.contract(all, optimize="auto-hq")
        xt = norm.contract_boundary(max_bond=27, layer_tags=["KET", "BRA"])
        assert xt == pytest.approx(xe, rel=1e-2)

    def test_contract_boundary_mode_deprecated(self):
        tn = qtn.TN2D_rand(4, 4, 2, seed=42)
        expected = tn.contract_boundary(max_bond=4, method="mps")
        with pytest.warns(FutureWarning, match="method"):
            Z = tn.contract_boundary(max_bond=4, mode="mps")
        assert Z == pytest.approx(expected)

    def test_contract_boundary_full_bond_similarity_method(self):
        # large enough bonds that the full bond compression is needed
        tn = qtn.TN2D_rand(4, 4, 3, seed=42)
        opts = {"max_bond": 4, "method": "full-bond"}
        expected = tn.contract_boundary(
            compress_opts={"method": "svd"}, **opts
        )
        Z = tn.contract_boundary(similarity_method="svd", **opts)
        assert Z == pytest.approx(expected)
        # check the option reaches the similarity decomposition
        with pytest.raises(KeyError):
            tn.contract_boundary(similarity_method="unknown", **opts)
        # previously ``method`` selected the similarity decomposition
        with pytest.warns(FutureWarning, match="method"):
            Z = tn.contract_boundary(
                max_bond=4, mode="full-bond", method="svd"
            )
        assert Z == pytest.approx(expected)

    @pytest.mark.parametrize("two_layer", [False, True])
    def test_contract_2d_boundary_via_1d(self, two_layer):
        psi = qtn.PEPS.rand(4, 4, 3, seed=42, tags="KET")
        norm = psi.make_norm()
        xe = norm.contract(all, optimize="auto-hq")

        # retain a virtual view to detect leaked tags
        view = norm.select(norm.x_tag(0))

        layer_tags = ["KET", "BRA"] if two_layer else None
        norm.contract_boundary_(
            max_bond=27, method="dm", layer_tags=layer_tags
        )
        assert norm.contract(all) == pytest.approx(xe, rel=5e-2)
        assert not any(tag.startswith("__ST") for tag in view.tag_map)

    def test_contract_2d_full_bond(self):
        psi = qtn.PEPS.rand(4, 4, 3, seed=42, tags="KET")
        norm = psi.make_norm()
        xe = norm.contract(all, optimize="auto-hq")
        xt = norm.contract_boundary(max_bond=27, method="full-bond")
        assert xt == pytest.approx(xe, rel=1e-2)

    @pytest.mark.parametrize("dims", [(10, 4), (4, 10)])
    def test_contract_boundary_stopping_criterion(self, dims):
        tn = qtn.TN2D_from_fill_fn(
            lambda shape: ar.lazy.Variable(shape=shape, backend="numpy"),
            *dims,
            D=2,
        )
        tn.contract_ctmrg_(4, cutoff=0.0, final_contract=False, progbar=True)
        assert tn.max_bond() == 4
        assert 16 <= tn.num_tensors <= 20

    @pytest.mark.parametrize("lazy", [False, True])
    def test_coarse_grain_basics(self, lazy):
        tn = qtn.TN2D_from_fill_fn(
            lambda shape: ar.lazy.Variable(shape, backend="numpy"),
            Lx=6,
            Ly=7,
            D=2,
        )
        tncg = tn.coarse_grain_hotrg("x", max_bond=3, cutoff=0.0, lazy=lazy)
        assert (tncg.Lx, tncg.Ly) == (3, 7)
        assert not tncg.outer_inds()
        assert tncg.max_bond() == 3
        assert "I4,0" not in tncg.tag_map
        assert "X5" not in tncg.tag_map

        tncg = tn.coarse_grain_hotrg("y", max_bond=3, cutoff=0.0, lazy=lazy)
        assert (tncg.Lx, tncg.Ly) == (6, 4)
        assert not tncg.outer_inds()
        assert tncg.max_bond() == 3
        assert "I0,5" not in tncg.tag_map
        assert "Y6" not in tncg.tag_map

    def test_contract_hotrg(self):
        tn = qtn.TN2D_classical_ising_partition_function(16, 16, 0.44)
        tn.contract_hotrg_(max_bond=5, progbar=True, equalize_norms=1.0)
        Zap = tn.item() * 10**tn.exponent
        assert Zap == pytest.approx(8.459419593253275e100, rel=2e-3)

    def test_contract_hotrg_two_layer_rand_peps(self):
        rng = np.random.default_rng(42)
        psi = qtn.PEPS.from_fill_fn(
            lambda shape: rng.uniform(low=-0.1, size=shape),
            Lx=7,
            Ly=5,
            bond_dim=2,
        )
        norm = psi.make_norm()
        xe = norm.contract(all, optimize="auto-hq")
        xt = norm.contract_hotrg(max_bond=5)
        assert xt == pytest.approx(xe, rel=1e-4)

    @pytest.mark.parametrize(
        "bond_dim,max_bond,dist,tolerance",
        [
            pytest.param(2, None, "normal", 1e-10, id="untruncated"),
            # local rank is up to 16, so max_bond=8 truncates
            pytest.param(4, 8, "uniform", 0.1, id="truncated"),
        ],
    )
    @requires_symmray
    @pytest.mark.parametrize("symmetry", symmetry_cases)
    @pytest.mark.parametrize("bond_orientation", bond_orientations)
    @pytest.mark.parametrize("direction", ["x", "y"])
    def test_contract_hotrg_symmetric(
        self,
        symmetry,
        bond_orientation,
        direction,
        bond_dim,
        max_bond,
        dist,
        tolerance,
    ):
        """Check exact and truncated HOTRG contraction.

        Cover tensor parity, bond orientation, and coarse-graining direction.
        Uniform data gives a stable approximation when truncating.
        """
        seed = qu.utils.hash_kwargs_to_int(
            method="hotrg",
            symmetry=symmetry,
            bond_orientation=bond_orientation,
            direction=direction,
            bond_dim=bond_dim,
            max_bond=max_bond,
            dist=dist,
        )
        tn = make_symmetric_2d_tn(
            symmetry,
            duals=bond_orientation,
            bond_dim=bond_dim,
            seed=seed,
            dist=dist,
        )

        expected = tn.contract(all, optimize="auto-hq")
        value = tn.contract_hotrg(
            max_bond=max_bond,
            cutoff=0.0,
            sequence=(direction,),
            reduce_opts={"method": "eigh"},
        )
        assert value == pytest.approx(expected, rel=tolerance)

    def test_ising_accuracy_regression(self):
        tn = qtn.TN2D_classical_ising_partition_function(16, 16, 0.44)
        for s in [("xmin",), ("xmax",), ("ymin",), ("ymax",)]:
            Zap = tn.contract_boundary(max_bond=8, sequence=s)
            assert Zap == pytest.approx(8.459419593253275e100, rel=2.2e-7)
        for s in [("xmin", "xmax"), ("ymin", "ymax")]:
            Zap = tn.contract_boundary(max_bond=8, sequence=s)
            assert Zap == pytest.approx(8.459419593253275e100, rel=3.9e-9)

    @pytest.mark.parametrize("mode", ["mps", "ctmrg", "hotrg"])
    @pytest.mark.parametrize("builder", ["loop", "cactus"])
    def test_cdl_rand_large(self, mode, builder):
        build_fn = {
            "loop": qtn.TN2D_rand_hidden_loop,
            "cactus": qtn.TN2D_rand_hidden_cactus,
        }[builder]
        tn = build_fn(10, 10, seed=42, contract_sites=False)
        Zex = tn.contract(...)
        tn = build_fn(10, 10, seed=42, contract_sites=True)

        if mode == "mps":
            Z = tn.contract_boundary(max_bond=16)
        elif mode == "ctmrg":
            Z = tn.contract_ctmrg(max_bond=16)
        elif mode == "hotrg":
            Z = tn.contract_hotrg(max_bond=16)

        assert Z == pytest.approx(Zex, rel=1e-1)

    @pytest.mark.parametrize(
        "method,two_layer",
        [
            ("mps", False),
            ("mps", True),
            ("full-bond", False),
        ],
    )
    def test_compute_x_envs(self, method, two_layer):
        psi = qtn.PEPS.rand(5, 4, 2, seed=42, tags="KET")
        norm = psi.make_norm()
        ex = norm.contract(all)

        if two_layer:
            compress_opts = {
                "cutoff": 1e-6,
                "max_bond": 12,
                "method": method,
                "layer_tags": ["KET", "BRA"],
            }
        else:
            compress_opts = {"cutoff": 1e-6, "max_bond": 8, "method": method}
        row_envs = norm.compute_x_environments(**compress_opts)

        for i in range(norm.Lx):
            norm_i = (
                row_envs["xmin", i]
                & norm.select(norm.x_tag(i))
                & row_envs["xmax", i]
            )
            x = norm_i.contract(all)
            assert x == pytest.approx(ex, rel=1e-2)

    @pytest.mark.parametrize(
        "method,two_layer",
        [
            ("mps", False),
            ("mps", True),
            ("full-bond", False),
        ],
    )
    def test_compute_y_envs(self, method, two_layer):
        psi = qtn.PEPS.rand(4, 5, 2, seed=42, tags="KET")
        norm = psi.retag({"KET": "BRA"}).H | psi
        ex = norm.contract(all)

        if two_layer:
            compress_opts = {
                "cutoff": 1e-6,
                "max_bond": 12,
                "method": method,
                "layer_tags": ["KET", "BRA"],
            }
        else:
            compress_opts = {"cutoff": 1e-6, "max_bond": 8, "method": method}
        col_envs = norm.compute_y_environments(**compress_opts)

        for j in range(norm.Lx):
            norm_j = (
                col_envs["ymin", j]
                & norm.select(norm.y_tag(j))
                & col_envs["ymax", j]
            )
            x = norm_j.contract(all)
            assert x == pytest.approx(ex, rel=1e-2)

    @pytest.mark.parametrize("initial_exponent", [False, True])
    @pytest.mark.parametrize("equalize_norms", [False, 1.0, True])
    def test_compute_envs_with_equalize_norms(
        self, initial_exponent, equalize_norms
    ):
        tn = qtn.TN2D_rand(5, 5, 2, dist="uniform", seed=42)
        if initial_exponent:
            tn.equalize_norms_(1.0)
        zex = tn.contract()
        envs = {}
        tn.compute_xmin_environments(
            max_bond=8,
            cutoff=0.0,
            envs=envs,
            equalize_norms=equalize_norms,
        )
        tn.compute_xmax_environments(
            max_bond=8,
            cutoff=0.0,
            envs=envs,
            equalize_norms=equalize_norms,
        )
        # test with full strips
        for i in range(tn.Lx):
            tni = (
                envs["xmin", i]
                | tn.select(tn.x_tag(i), with_exponent=True)
                | envs["xmax", i]
            )
            zi = tni.contract()
            assert zi == pytest.approx(zex, rel=1e-5)
        # test with direct overlap of envs
        for i in range(tn.Lx - 1):
            tni = envs["xmin", i + 1] | envs["xmax", i]
            # need to add back the original exponent in this case
            tni.exponent += tn.exponent
            zi = tni.contract()
            assert zi == pytest.approx(zex, rel=1e-5)

    def test_normalize(self):
        psi = qtn.PEPS.rand(4, 5, 2, seed=42)
        norm = (psi.H | psi).contract(all)
        assert norm != pytest.approx(1.0)
        psi.normalize_(balance_bonds=True, equalize_norms=True, cutoff=2e-3)
        norm = (psi.H | psi).contract(all)
        assert norm == pytest.approx(1.0, rel=0.01)

    @pytest.mark.parametrize("normalized", [False, True])
    @pytest.mark.parametrize("method", ["mps", "full-bond"])
    def test_compute_local_expectation_one_sites(self, method, normalized):
        peps = qtn.PEPS.rand(4, 3, 2, seed=42, dtype="complex")

        # reference
        k = peps.to_qarray()
        if normalized:
            qu.normalize(k)
        coos = list(itertools.product([0, 2, 3], [0, 1, 2]))
        terms = {coo: qu.rand_matrix(2) for coo in coos}
        dims = [[2] * 3] * 4
        A = sum(
            qu.ikron(A, dims, [coo], sparse=True) for coo, A in terms.items()
        )
        ex = qu.expec(A, k)

        opts = {"cutoff": 2e-3, "max_bond": 9, "contract_optimize": "auto-hq"}
        e = peps.compute_local_expectation(
            terms, method=method, normalized=normalized, **opts
        )

        assert e == pytest.approx(ex, rel=1e-2)

    @pytest.mark.parametrize("normalized", [False, True])
    @pytest.mark.parametrize("method", ["mps", "full-bond"])
    def test_compute_local_expectation_two_sites(self, method, normalized):
        H = qu.ham_heis_2D(4, 3, sparse=True)
        Hij = qu.ham_heis(2, cyclic=False)

        peps = qtn.PEPS.rand(4, 3, 2, seed=42)
        k = peps.to_qarray()

        if normalized:
            qu.normalize(k)
        ex = qu.expec(H, k)

        opts = {
            "method": method,
            "normalized": normalized,
            "cutoff": 2e-3,
            "max_bond": 16,
            "contract_optimize": "auto-hq",
        }

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

    def test_cyclic_basic(self):
        tn = qtn.TN2D_rand(Lx=3, Ly=4, D=2, cyclic=True)
        assert tn.is_cyclic_x()
        assert tn.is_cyclic_y()
        assert tn.num_indices == 2 * 3 * 4
        tn = qtn.TN2D_rand(Lx=3, Ly=4, D=2, cyclic=(False, True))
        assert not tn.is_cyclic_x()
        assert tn.is_cyclic_y()
        assert tn.num_indices == 2 * 3 * 4 - 4
        tn = qtn.TN2D_rand(Lx=3, Ly=4, D=2, cyclic=(True, False))
        assert tn.is_cyclic_x()
        assert not tn.is_cyclic_y()
        assert tn.num_indices == 2 * 3 * 4 - 3
        tn = qtn.TN2D_rand(Lx=3, Ly=4, D=2, cyclic=(False, False))
        assert not tn.is_cyclic_x()
        assert not tn.is_cyclic_y()
        assert tn.num_indices == 2 * 3 * 4 - 7

    @pytest.mark.parametrize("schedule", ["tree", "cut"])
    @pytest.mark.parametrize("direction", ["x", "y"])
    def test_compute_block_environments(self, direction, schedule):
        tn = qtn.TN2D_rand(4, 4, 2, cyclic=True, seed=42)
        tn.equalize_norms_(1.0)
        expected = tn.contract()
        environments = tn.compute_block_environments(
            direction,
            all_blocks(4, 2),
            max_bond=64,
            cutoff=1e-10,
            schedule=schedule,
        )

        for (start, _), environment in environments.items():
            if direction == "x":
                tags = [tn.x_tag(start), tn.x_tag(start + 1)]
            else:
                tags = [tn.y_tag(start), tn.y_tag(start + 1)]
            actual = (tn.select_any(tags) | environment).contract()
            assert actual == pytest.approx(expected)

    @pytest.mark.parametrize("schedule", ["tree", "cut"])
    def test_compute_block_environments_layered(self, schedule):
        peps = qtn.PEPS.rand(3, 3, 2, cyclic=True, seed=42)
        norm = peps.make_norm()
        expected = norm.contract()
        environment = norm.compute_block_environments(
            "x",
            max_bond=256,
            blocks=((0, 1),),
            cutoff=1e-10,
            schedule=schedule,
            layer_tags=("KET", "BRA"),
        )[0, 1]

        target = norm.select(norm.x_tag(0), virtual=False)
        assert (target | environment).contract() == pytest.approx(expected)

    @pytest.mark.parametrize("compress_fn", [None, "ag", "1d"])
    def test_compute_block_environments_compress_fn(self, compress_fn):
        # periodic along x, so each x environment is an open chain along y
        tn = qtn.TN2D_rand(4, 3, 2, cyclic=(True, False), seed=42)
        expected = tn.contract()
        environment = tn.compute_block_environments(
            "x",
            ((0, 1),),
            max_bond=64,
            cutoff=1e-10,
            compress_fn=compress_fn,
        )[0, 1]
        actual = (tn.select(tn.x_tag(0)) | environment).contract()
        assert actual == pytest.approx(expected)

        with pytest.raises(ValueError, match="compress_fn"):
            tn.compute_block_environments(
                "x", ((0, 1),), max_bond=4, compress_fn="3d"
            )

    def test_compute_block_environments_invalid_layers(self):
        tn = qtn.TN2D_rand(3, 3, 2, cyclic=True, seed=42)
        with pytest.raises(ValueError, match="layer_tags"):
            tn.compute_block_environments(
                "x",
                max_bond=16,
                blocks=((0, 1),),
                layer_tags=("KET", "BRA"),
            )

    @pytest.mark.parametrize("second_schedule", ["tree", "cut"])
    @pytest.mark.parametrize("schedule", ["tree", "cut"])
    @pytest.mark.parametrize("first_contract", ["x", "y"])
    @pytest.mark.parametrize("cyclic", [True, (True, False), (False, True)])
    def test_compute_plaquette_environments_via_envs(
        self, first_contract, cyclic, schedule, second_schedule
    ):
        tn = qtn.TN2D_rand(4, 4, 2, cyclic=cyclic, seed=42)
        tn.equalize_norms_(1.0)
        expected = tn.contract()
        cyclic_x, cyclic_y = (
            (cyclic, cyclic) if isinstance(cyclic, bool) else cyclic
        )
        i = 3 if cyclic_x else 1
        j = 3 if cyclic_y else 1
        environments = tn.compute_plaquette_environments_via_envs(
            2,
            2,
            max_bond=64,
            starts=((i, j),),
            cyclic=cyclic,
            cutoff=1e-10,
            first_contract=first_contract,
            schedule=schedule,
            second_schedule=second_schedule,
        )

        environment = environments[(i, j), (2, 2)]
        tags = [
            tn.site_tag((i + di) % 4, (j + dj) % 4)
            for di in range(2)
            for dj in range(2)
        ]
        actual = (tn.select_any(tags) | environment).contract()
        assert actual == pytest.approx(expected)

    def test_compute_selected_plaquette_environments_via_envs(self):
        tn = qtn.TN2D_rand(3, 3, 2, cyclic=True, seed=42)
        starts = ((0, 0), (1, 1))
        environments = tn.compute_plaquette_environments_via_envs(
            1,
            1,
            max_bond=64,
            starts=starts,
            cutoff=1e-10,
        )

        assert set(environments) == {(start, (1, 1)) for start in starts}
        with pytest.raises(ValueError):
            tn.compute_plaquette_environments_via_envs(
                x_bsz=0,
                max_bond=4,
                starts=(),
            )
        with pytest.raises(ValueError):
            tn.compute_block_environments(
                "x", blocks=((tn.Lx, 1),), max_bond=4
            )

    @pytest.mark.parametrize("schedule", ["tree", "cut"])
    @pytest.mark.parametrize("normalized", [False, True, "return"])
    def test_compute_local_expectation_via_envs(self, normalized, schedule):
        peps = qtn.PEPS.rand(3, 3, 2, cyclic=True, seed=42, dtype="complex128")
        peps.equalize_norms_(1.0)
        terms = {
            (2, 2): qu.rand_herm(2),
            (0, 0): qu.rand_herm(2),
            ((2, 0), (0, 0)): qu.rand_herm(4),
            ((0, 0), (2, 0)): qu.rand_herm(4),
            ((1, 2), (1, 0)): qu.rand_herm(4),
        }
        opts = {"max_bond": 256, "cutoff": 1e-10}
        expecs_ex = {
            where: peps.local_expectation_exact(
                G, where, normalized=normalized
            )
            for where, G in terms.items()
        }
        expecs = peps.compute_local_expectation_via_envs(
            terms,
            normalized=normalized,
            return_all=True,
            schedule=schedule,
            **opts,
        )
        for where in terms:
            if normalized == "return":
                expec, nfactor = expecs[where]
                expec_ex, nfactor_ex = expecs_ex[where]
                assert nfactor == pytest.approx(nfactor_ex)
                assert expec == pytest.approx(expec_ex)
            else:
                assert expecs[where] == pytest.approx(expecs_ex[where])

        # the default method for a periodic PEPS
        total = peps.compute_local_expectation(
            terms, normalized=normalized, schedule=schedule, **opts
        )
        if normalized == "return":
            expected = sum(e / n for e, n in expecs_ex.values())
        else:
            expected = sum(expecs_ex.values())
        assert total == pytest.approx(expected)

    @pytest.mark.parametrize("get", ["matrix", "array", "tensor"])
    @pytest.mark.parametrize("cyclic", [False, True, (True, False)])
    def test_compute_partial_traces_via_envs(self, cyclic, get):
        peps = qtn.PEPS.rand(
            3, 4, 2, cyclic=cyclic, seed=42, dtype="complex128"
        )
        peps.equalize_norms_(1.0)
        # share one first sweep across plaquette sizes
        wheres = [
            (1, 1),
            ((0, 1), (0, 2)),
            ((2, 1), (1, 1)),
            ((0, 0), (1, 1)),
            ((1, 3), (1, 1)),
        ]
        if peps.is_cyclic_x():
            wheres.append(((2, 0), (0, 0)))
        rhos = peps.compute_partial_traces_via_envs(
            wheres, max_bond=256, cutoff=1e-10, get=get
        )
        assert set(rhos) == set(wheres)
        for where in wheres:
            rho = rhos[where]
            expected = peps.partial_trace_exact(where, get=get)
            if get == "tensor":
                assert rho.inds == expected.inds
                rho, expected = rho.data, expected.data
            assert rho == pytest.approx(expected)

    @pytest.mark.parametrize(
        "first_contract,autogroup,diagonal,expected_sweeps",
        [
            # 1x2 and 2x1 plaquettes each leave strips of width one
            (None, True, False, {"x": {1}, "y": {1}}),
            # one direction leaves strips of width two for 2x1 plaquettes
            ("x", True, False, {"x": {1, 2}}),
            # a single 2x2 plaquette size
            (None, False, False, {"x": {2}}),
            # the 2x2 plaquettes needed anyway cover the 1x2 and 2x1 sites
            (None, True, True, {"x": {2}}),
        ],
    )
    def test_compute_partial_traces_via_envs_sweeps(
        self, monkeypatch, first_contract, autogroup, diagonal, expected_sweeps
    ):
        from quimb.tensor.tn2d.core import TensorNetwork2D

        sweeps = {}
        gen_block_environments = TensorNetwork2D.gen_block_environments

        def spy(self, direction, blocks, *args, **kwargs):
            sweeps[direction] = {size for _, size in blocks}
            return gen_block_environments(
                self, direction, blocks, *args, **kwargs
            )

        monkeypatch.setattr(TensorNetwork2D, "gen_block_environments", spy)

        peps = qtn.PEPS.rand(3, 3, 2, seed=42, dtype="complex128")
        wheres = [((0, 0), (0, 1)), ((1, 1), (2, 1)), ((2, 1), (2, 2))]
        if diagonal:
            wheres.append(((1, 1), (2, 2)))
        rhos = peps.compute_partial_traces_via_envs(
            wheres,
            max_bond=64,
            cutoff=1e-10,
            first_contract=first_contract,
            autogroup=autogroup,
        )
        assert sweeps == expected_sweeps
        for where in wheres:
            expected = peps.partial_trace_exact(where)
            assert rhos[where] == pytest.approx(expected)

    @pytest.mark.parametrize(
        "second_dense,expected_nexact",
        # strip widths are one for 1x3 and two for 2x2
        [(None, 1), (True, 2), (False, 0)],
    )
    @pytest.mark.parametrize("cyclic", [False, True, (True, False)])
    def test_compute_partial_traces_via_envs_second_dense(
        self, monkeypatch, cyclic, second_dense, expected_nexact
    ):
        from quimb.tensor.tn2d import core

        nexact = 0
        gen_exact_environments = core.gen_exact_environments

        def spy(*args, **kwargs):
            nonlocal nexact
            nexact += 1
            return gen_exact_environments(*args, **kwargs)

        monkeypatch.setattr(core, "gen_exact_environments", spy)

        peps = qtn.PEPS.rand(
            3, 3, 2, cyclic=cyclic, seed=42, dtype="complex128"
        )
        peps.equalize_norms_(1.0)
        # neither size contains the other, so both are kept
        wheres = [((0, 0), (0, 1), (0, 2)), ((1, 1), (2, 2))]
        rhos = peps.compute_partial_traces_via_envs(
            wheres, max_bond=256, cutoff=1e-10, second_dense=second_dense
        )
        assert nexact == expected_nexact
        for where in wheres:
            expected = peps.partial_trace_exact(where)
            assert rhos[where] == pytest.approx(expected)

    def test_compute_partial_traces_via_envs_contract_opts(self):
        peps = qtn.PEPS.rand(3, 3, 2, seed=42, dtype="complex128")
        # contract strips of width one and two exactly
        wheres = [((0, 0), (0, 1), (0, 2)), ((1, 1), (2, 2))]
        rhos = peps.compute_partial_traces_via_envs(
            wheres,
            max_bond=64,
            second_dense=True,
            contract_opts={"optimize": "greedy"},
        )
        for where in wheres:
            expected = peps.partial_trace_exact(where)
            assert rhos[where] == pytest.approx(expected)

    @pytest.mark.parametrize("get", ["matrix", "array", "tensor"])
    @pytest.mark.parametrize("autogroup", [False, True])
    def test_compute_partial_traces_boundary(self, autogroup, get):
        peps = qtn.PEPS.rand(3, 3, 2, seed=42, dtype="complex128")
        wheres = [(1, 1), ((0, 1), (0, 2)), ((2, 1), (1, 1))]
        rhos = peps.compute_partial_traces(
            wheres, max_bond=64, cutoff=0.0, autogroup=autogroup, get=get
        )
        assert set(rhos) == set(wheres)
        for where in wheres:
            rho = rhos[where]
            expected = peps.partial_trace_exact(where, get=get)
            if get == "tensor":
                assert rho.inds == expected.inds
                rho, expected = rho.data, expected.data
            assert rho == pytest.approx(expected)

    @pytest.mark.parametrize("route", ["boundary", "envs"])
    @pytest.mark.parametrize(
        "where", [((2, 1), (1, 1)), [(2, 1), (1, 1)], [[2, 1], [1, 1]]]
    )
    def test_partial_trace_and_local_expectation(self, route, where):
        peps = qtn.PEPS.rand(3, 3, 2, seed=42, dtype="complex128")
        opts = {"max_bond": 64, "cutoff": 1e-10, "route": route}
        G = qu.rand_herm(4, seed=7)

        rho = peps.partial_trace([[2, 1], [1, 1]], **opts)
        assert rho == pytest.approx(peps.partial_trace_exact(where))

        expec = peps.local_expectation(G, where, **opts)
        assert expec == pytest.approx(peps.local_expectation_exact(G, where))

        rho, nfactor = peps.partial_trace((1, 1), normalized="return", **opts)
        rho_ex, nfactor_ex = peps.partial_trace_exact(
            (1, 1), normalized="return"
        )
        assert nfactor == pytest.approx(nfactor_ex)
        assert rho == pytest.approx(rho_ex)

    @requires_symmray
    @pytest.mark.parametrize("autogroup", [False, True])
    @pytest.mark.parametrize("fermionic", [False, True])
    def test_compute_local_expectation_boundary_symmray(
        self, fermionic, autogroup
    ):
        import symmray as sr

        peps = sr.PEPS_abelian_rand(
            "Z2",
            3,
            3,
            bond_dim=2,
            fermionic=fermionic,
            subsizes="equal",
            dtype="complex128",
            seed=42,
        )
        edges = [
            ((0, 0), (0, 1)),
            ((1, 1), (2, 1)),
            ((2, 1), (2, 2)),
            ((0, 2), (1, 2)),
        ]
        if fermionic:
            terms = sr.ham_fermi_hubbard_spinless_from_edges("Z2", edges)
        else:
            terms = sr.ham_heisenberg_from_edges("Z2", edges)

        expected = sum(
            peps.local_expectation_exact(G, where)
            for where, G in terms.items()
        )
        # the default route for an open PEPS
        actual = peps.compute_local_expectation(
            terms, max_bond=16, cutoff=0.0, autogroup=autogroup
        )
        assert actual == pytest.approx(expected)

    @requires_symmray
    @pytest.mark.parametrize("schedule", ["tree", "cut"])
    @pytest.mark.parametrize("fermionic", [False, True])
    def test_compute_local_expectation_via_envs_symmray(
        self, fermionic, schedule
    ):
        import symmray as sr

        peps = sr.PEPS_abelian_rand(
            "Z2",
            3,
            3,
            bond_dim=2,
            cyclic=True,
            fermionic=fermionic,
            subsizes="equal",
            dtype="complex128",
            seed=42,
        )
        # include bonds across both periodic boundaries
        edges = [
            ((0, 0), (0, 1)),
            ((2, 0), (0, 0)),
            ((1, 2), (1, 0)),
            ((0, 0), (2, 0)),
        ]
        if fermionic:
            terms = sr.ham_fermi_hubbard_spinless_from_edges("Z2", edges)
        else:
            terms = sr.ham_heisenberg_from_edges("Z2", edges)

        expected = sum(
            peps.local_expectation_exact(G, where)
            for where, G in terms.items()
        )
        actual = peps.compute_local_expectation_via_envs(
            terms,
            max_bond=256,
            cutoff=1e-10,
            schedule=schedule,
        )
        assert actual == pytest.approx(expected)

    @pytest.mark.parametrize("direction", ["x", "y"])
    @pytest.mark.parametrize(
        "shape,cyclic",
        [
            ((4, 5), (True, False)),
            ((4, 5), (False, True)),
            ((2, 3), (True, True)),
        ],
    )
    def test_compute_block_environments_geometry(
        self, shape, cyclic, direction
    ):
        tn = qtn.TN2D_rand(*shape, 2, cyclic=cyclic, seed=42)
        expected = tn.contract()
        if direction == "x":
            tag, is_cyclic, L = tn.x_tag(0), cyclic[0], shape[0]
        else:
            tag, is_cyclic, L = tn.y_tag(0), cyclic[1], shape[1]
        environment = tn.compute_block_environments(
            direction,
            # each of the L - 1 contracted lines contributes a bond of size 2
            max_bond=2 ** (L - 1),
            blocks=((0, 1),),
            cyclic=is_cyclic,
            cutoff=0.0,
        )[0, 1]
        actual = (tn.select(tag) | environment).contract()
        assert actual == pytest.approx(expected)

    @pytest.mark.parametrize("schedule", ["tree", "cut"])
    def test_compute_block_environments_equalize_norms(self, schedule):
        tn = qtn.TN2D_rand(4, 4, 2, cyclic=True, seed=42)
        tn.equalize_norms_(1.0)
        expected = tn.contract()
        environment = tn.compute_block_environments(
            "x",
            max_bond=64,
            blocks=((0, 1),),
            cutoff=1e-10,
            equalize_norms=True,
            schedule=schedule,
        )[0, 1]
        actual = (tn.select(tn.x_tag(0)) | environment).contract()
        assert actual == pytest.approx(expected)

    def test_compute_block_environments_callable(self):
        from quimb.tensor.tnag.compress import (
            tensor_network_ag_compress_local_early,
        )

        calls = []

        def compressor(tn, **kwargs):
            calls.append(tn.num_tensors)
            return tensor_network_ag_compress_local_early(tn, **kwargs)

        tn = qtn.TN2D_rand(4, 4, 2, cyclic=True, seed=42)
        tn.compute_block_environments(
            "x",
            max_bond=16,
            blocks=((0, 2),),
            cutoff=0.0,
            method=compressor,
        )
        # the first row is used as is, then compressed with the second
        assert calls == [8]

    @pytest.mark.parametrize("method", ["local-early", "projector"])
    def test_compute_block_environments_compressed(self, method):
        tn = qtn.TN2D_rand(5, 5, 2, cyclic=True, seed=42, dist="uniform")
        expected = tn.contract()
        environment = tn.compute_block_environments(
            "x",
            max_bond=4,
            blocks=((0, 1),),
            cutoff=1e-10,
            method=method,
        )[0, 1]
        actual = (tn.select(tn.x_tag(0)) | environment).contract()
        assert actual == pytest.approx(expected, rel=1e-3)

    @pytest.mark.parametrize("strip_exponent", [False, True])
    @pytest.mark.parametrize("equalize_norms", [False, 1.0, True])
    @pytest.mark.parametrize("inplace", [False, True])
    def test_contract_boundary_strip_exponent(
        self, strip_exponent, equalize_norms, inplace
    ):
        tn = qtn.TN2D_classical_ising_partition_function(16, 16, 0.44)
        Zex = 8.459419593253275e100

        if inplace:
            tnc = tn.copy()
            tnc.contract_boundary_(
                max_bond=8,
                strip_exponent=strip_exponent,
                equalize_norms=equalize_norms,
                final_contract=False,
            )
            Z = tnc.contract(...)
        else:
            Z = tn.contract_boundary(
                max_bond=8,
                strip_exponent=strip_exponent,
                equalize_norms=equalize_norms,
            )
            if strip_exponent:
                Z = Z[0] * 10 ** Z[1]

        assert Z == pytest.approx(Zex, rel=1e-6)

    @pytest.mark.parametrize("strip_exponent", [False, True])
    @pytest.mark.parametrize("equalize_norms", [False, 1.0, True])
    @pytest.mark.parametrize("inplace", [False, True])
    def test_contract_hotrg_strip_exponent(
        self, strip_exponent, equalize_norms, inplace
    ):
        tn = qtn.TN2D_classical_ising_partition_function(16, 16, 0.44)
        Zex = 8.459419593253275e100

        if inplace:
            tnc = tn.copy()
            tnc.contract_hotrg_(
                max_bond=5,
                strip_exponent=strip_exponent,
                equalize_norms=equalize_norms,
                final_contract=False,
            )
            Z = tnc.contract(...)
        else:
            Z = tn.contract_hotrg(
                max_bond=5,
                strip_exponent=strip_exponent,
                equalize_norms=equalize_norms,
            )
            if strip_exponent:
                Z = Z[0] * 10 ** Z[1]

        assert Z == pytest.approx(Zex, rel=2e-3)

    @pytest.mark.parametrize("strip_exponent", [False, True])
    @pytest.mark.parametrize("equalize_norms", [False, 1.0, True])
    @pytest.mark.parametrize("inplace", [False, True])
    def test_contract_ctmrg_strip_exponent(
        self, strip_exponent, equalize_norms, inplace
    ):
        tn = qtn.TN2D_classical_ising_partition_function(16, 16, 0.44)
        Zex = 8.459419593253275e100

        if inplace:
            tnc = tn.copy()
            tnc.contract_ctmrg_(
                max_bond=8,
                strip_exponent=strip_exponent,
                equalize_norms=equalize_norms,
                final_contract=False,
            )
            Z = tnc.contract(...)
        else:
            Z = tn.contract_ctmrg(
                max_bond=8,
                strip_exponent=strip_exponent,
                equalize_norms=equalize_norms,
            )
            if strip_exponent:
                Z = Z[0] * 10 ** Z[1]

        assert Z == pytest.approx(Zex, rel=1e-2)

    @pytest.mark.parametrize("cyclicx", [False, True])
    @pytest.mark.parametrize("cyclicy", [False, True])
    @pytest.mark.parametrize("mode", ["mps", "hotrg", "ctmrg"])
    def test_cyclic_contract(self, cyclicx, cyclicy, mode):
        Lx = 5
        Ly = 6
        D = 2
        chi = 3
        tn = qtn.TN2D_rand(
            Lx,
            Ly,
            D,
            cyclic=(cyclicx, cyclicy),
            seed=42,
            dist="uniform",
        )
        Zex = tn.contract(...)
        if mode == "hotrg":
            Z = tn.contract_hotrg(chi)
        elif mode == "ctmrg":
            Z = tn.contract_ctmrg(chi)
        else:
            Z = tn.contract_boundary(chi, method=mode)
        assert abs(1 - Z / Zex) < 1e-3


class TestPEPO:
    @pytest.mark.parametrize("Lx", [3, 4, 5])
    @pytest.mark.parametrize("Ly", [3, 4, 5])
    def test_basic_rand(self, Lx, Ly):
        X = qtn.PEPO.rand_herm(Lx, Ly, bond_dim=4)

        assert X.max_bond() == 4
        assert X.Lx == Lx
        assert X.Ly == Ly
        assert len(X.tensor_map) == Lx * Ly
        assert X.upper_inds == tuple(
            f"k{i},{j}" for i in range(Lx) for j in range(Ly)
        )
        assert X.lower_inds == tuple(
            f"b{i},{j}" for i in range(Lx) for j in range(Ly)
        )
        assert X.site_tags == tuple(
            f"I{i},{j}" for i in range(Lx) for j in range(Ly)
        )

        assert X.bond_size((1, 1), (1, 2)) == (4)

        for i in range(Lx):
            assert len(X.select(f"X{i}").tensor_map) == Ly
        for j in range(Ly):
            assert len(X.select(f"Y{j}").tensor_map) == Lx

        for i in range(Lx):
            for j in range(Ly):
                assert X.phys_dim(i, j) == 2
                assert isinstance(X[i, j], qtn.Tensor)
                assert isinstance(X[f"I{i},{j}"], qtn.Tensor)

        if Lx == Ly == 3:
            X_dense = X.to_qarray(optimize="auto-hq")
            assert X_dense.shape == (512, 512)
            assert qu.isherm(X_dense)

        X.show()
        assert f"Lx={Lx}" in X.__str__()
        assert f"Lx={Lx}" in X.__repr__()

    def test_cyclic_length_two_keeps_boundary_bonds(self):
        X = qtn.PEPO.rand(2, 3, bond_dim=1, cyclic=(True, False))
        assert not X.is_cyclic_y()
        assert len(X[0, 1].bonds(X[1, 1])) == 2
        assert X.num_indices == 22

    @pytest.mark.parametrize(
        "Lx,Ly,cyclic,site,repeat",
        [
            (1, 3, (True, False), (0, 1), (0, 2)),
            (3, 1, (False, True), (1, 0), (1, 3)),
        ],
    )
    def test_cyclic_length_one_keeps_self_bond(
        self, Lx, Ly, cyclic, site, repeat
    ):
        X = qtn.PEPO.rand(Lx, Ly, bond_dim=1, cyclic=cyclic)
        tensor = X[site]
        assert tensor.inds[repeat[0]] == tensor.inds[repeat[1]]
        assert X.num_indices == 11

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
        Ad = A.to_qarray()
        xd = x.to_qarray()
        yd = y.to_qarray()
        assert_allclose(Ad @ xd, yd)
        yc = A.apply(x, compress=True, max_bond=3)
        assert yc.max_bond() == 3

    def test_pepo_product_operator(self):
        Lx = Ly = 2
        N = Lx * Ly
        Z = qu.pauli("Z")
        arrays = [[Z, Z], [Z, Z]]
        P = qtn.PEPO_product_operator(arrays)
        flat = [arrays[i][j] for i in range(Lx) for j in range(Ly)]
        expected = qu.ikron(flat, dims=[2] * N, inds=range(N))
        assert_allclose(P.to_qarray(), expected)

        I2 = np.eye(2)
        arrays_i = [[I2, I2], [I2, I2]]
        Pi = qtn.PEPO_product_operator(arrays_i)
        assert_allclose(Pi.to_qarray(), np.eye(2**N))

    @pytest.mark.parametrize("cyclic", [True, (True, False), (False, True)])
    def test_pepo_product_operator_cyclic(self, cyclic):
        # 3x3 is the smallest grid where shape-inference can distinguish
        # cyclic in each direction (mixed cyclicity on 2x2 is ambiguous)
        Lx = Ly = 3
        N = Lx * Ly
        X = qu.pauli("X")
        Z = qu.pauli("Z")
        arrays = [
            [X if (i + j) % 2 else Z for j in range(Ly)] for i in range(Lx)
        ]

        P = qtn.PEPO_product_operator(arrays, cyclic=cyclic)

        # confirm boundary conditions match explicit request
        try:
            cx_expected, cy_expected = cyclic
        except TypeError:
            cx_expected = cy_expected = cyclic
        assert P.is_cyclic_x() == cx_expected
        assert P.is_cyclic_y() == cy_expected

        # dense should be the Kronecker product regardless of BC (bond
        # dim 1 cyclic wraps add no new structure to the dense operator)
        flat = [arrays[i][j] for i in range(Lx) for j in range(Ly)]
        expected = qu.ikron(flat, dims=[2] * N, inds=range(N))
        assert_allclose(P.to_qarray(), expected)

    def test_apply_pepo_to_pepo(self):
        A = qtn.PEPO.rand(Lx=3, Ly=2, bond_dim=2, seed=2)
        B = qtn.PEPO.rand(Lx=3, Ly=2, bond_dim=2, seed=3)
        C = A.apply(B)
        assert isinstance(C, qtn.PEPO)
        Ad, Bd, Cd = A.to_qarray(), B.to_qarray(), C.to_qarray()
        assert_allclose(Ad @ Bd, Cd)
        Cc = A.apply(B, compress=True, max_bond=4)
        assert Cc.max_bond() == 4

    def test_apply_pepo_inplace_consumes_acting(self):
        # inplace=True should leave `other` untouched and consume `self`
        A = qtn.PEPO.rand(Lx=2, Ly=2, bond_dim=2, seed=11)
        x = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=12)

        Ad = A.to_qarray()
        xd_before = x.to_qarray()

        y = A.apply_(x)

        # other (x) must be untouched
        assert_allclose(x.to_qarray(), xd_before)
        # result equals A @ x
        assert_allclose(y.to_qarray(), Ad @ xd_before)

    def test_pepo_trace(self):
        P = qtn.PEPO.rand(Lx=2, Ly=2, bond_dim=2, seed=4)
        assert_allclose(P.trace(), np.trace(P.to_qarray()))

    def test_pepo_partial_transpose_involution(self):
        P = qtn.PEPO.rand(Lx=3, Ly=2, bond_dim=2, seed=5)
        site = (0, 1)
        P1 = P.partial_transpose(site)
        P2 = P1.partial_transpose(site)
        assert_allclose(P.to_qarray(), P2.to_qarray())
        P3 = P.partial_transpose((site,))
        assert_allclose(P1.to_qarray(), P3.to_qarray())

    def test_pepo_partial_transpose_vs_dense(self):
        # non-involution check: compare PEPO partial_transpose to the
        # dense reference qu.partial_transpose on a non-hermitian PEPO
        Lx, Ly = 2, 3
        P = qtn.PEPO.rand(Lx=Lx, Ly=Ly, bond_dim=2, seed=7)
        N = Lx * Ly
        dims = [2] * N

        # test a few different subsystem choices
        for sysa_sites in [
            [(0, 0)],
            [(0, 1), (1, 2)],
            [(0, 0), (0, 1), (0, 2)],
        ]:
            # gen_site_coos for TN2D is product(range(Lx), range(Ly)), so
            # the flat index for to_qarray is i*Ly + j
            sysa_flat = [i * Ly + j for (i, j) in sysa_sites]

            P_pt = P.partial_transpose(sysa_sites)
            expected = qu.partial_transpose(
                P.to_qarray(), dims=dims, sysa=sysa_flat
            )
            assert_allclose(P_pt.to_qarray(), expected)
            # original unchanged
            assert_allclose(P.to_qarray(), P.to_qarray())

    def test_pepo_apply_typeerror(self):
        P = qtn.PEPO.rand(Lx=2, Ly=2, bond_dim=2, seed=6)
        with pytest.raises(
            TypeError,
            match="TensorNetworkGenOperator or TensorNetworkGenVector",
        ):
            P.apply("not a peps")

    def test_build_pepo_propagator_trotterized_identity(self):
        ham = qtn.LocalHam2D(2, 2, qu.rand_herm(4))
        U = ham.build_pepo_propagator_trotterized(0.0)
        d = U.to_qarray().shape[0]
        assert_allclose(U.to_qarray(), np.eye(d))

    def test_build_pepo_propagator_trotterized_accuracy(self):
        ham = qtn.LocalHam2D(2, 2, qu.rand_herm(4, seed=42))
        dims = [2] * 4
        sites = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
        Hd = sum(
            qu.pkron(t, dims, [sites[coo] for coo in w])
            for w, t in ham.items()
        )

        errs = []
        for x in (0.02, 0.01):
            U = ham.build_pepo_propagator_trotterized(x)
            errs.append(np.linalg.norm(U.to_qarray() - qu.expm(x * Hd)))

        # first order: halving x should quarter the error
        assert errs[0] / errs[1] == pytest.approx(4, rel=0.15)

    def test_build_pepo_propagator_trotterized_order(self):
        # only check the plumbing, higher order pepos get expensive fast
        ham = qtn.LocalHam2D(2, 2, qu.rand_herm(4, seed=42))
        U1 = ham.build_pepo_propagator_trotterized(0.1, order=1)
        U2 = ham.build_pepo_propagator_trotterized(0.1, order=2)
        assert U2.max_bond() > U1.max_bond()


class TestMisc:
    def test_calc_plaquette_sizes(self):
        from quimb.tensor.tn2d.core import calc_plaquette_sizes

        H2 = {None: qu.ham_heis(2)}
        ham = qtn.LocalHam2D(10, 10, H2)
        assert calc_plaquette_sizes(ham.terms.keys()) == ((1, 2), (2, 1))
        assert calc_plaquette_sizes(ham.terms.keys(), autogroup=False) == (
            (2, 2),
        )
        H2[(1, 1), (2, 2)] = 0.5 * qu.ham_heis(2)
        ham = qtn.LocalHam2D(10, 10, H2)
        assert calc_plaquette_sizes(ham.terms.keys()) == ((2, 2),)
        H2[(2, 2), (2, 4)] = 0.25 * qu.ham_heis(2)
        H2[(2, 4), (4, 4)] = 0.25 * qu.ham_heis(2)
        ham = qtn.LocalHam2D(10, 10, H2)
        assert calc_plaquette_sizes(ham.terms.keys()) == (
            (1, 3),
            (2, 2),
            (3, 1),
        )
        assert calc_plaquette_sizes(ham.terms.keys(), autogroup=False) == (
            (3, 3),
        )

    def test_calc_plaquette_map(self):
        from quimb.tensor.tn2d.core import calc_plaquette_map

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
        assert calc_plaquette_map(plaquettes) == {
            (0, 0): ((0, 0), (2, 1)),
            (0, 1): ((0, 1), (2, 1)),
            (1, 0): ((1, 0), (1, 2)),
            (1, 1): ((1, 0), (1, 2)),
            ((0, 0), (0, 1)): ((0, 0), (1, 2)),
            ((0, 0), (1, 0)): ((0, 0), (2, 1)),
            ((0, 0), (1, 1)): ((0, 0), (2, 2)),
            ((0, 1), (1, 0)): ((0, 0), (2, 2)),
            ((0, 1), (1, 1)): ((0, 1), (2, 1)),
            ((1, 0), (1, 1)): ((1, 0), (1, 2)),
        }
