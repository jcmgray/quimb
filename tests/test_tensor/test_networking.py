import warnings

import pytest

import quimb.tensor as qtn


def test_istree():
    assert qtn.Tensor().as_network().istree()
    tn = qtn.rand_tensor([2] * 1, ["x"]).as_network()
    assert tn.istree()
    tn |= qtn.rand_tensor([2] * 3, ["x", "y", "z"])
    assert tn.istree()
    tn |= qtn.rand_tensor([2] * 2, ["y", "z"])
    assert tn.istree()
    tn |= qtn.rand_tensor([2] * 2, ["x", "z"])
    assert not tn.istree()


def test_isconnected():
    assert qtn.Tensor().as_network().isconnected()
    tn = qtn.rand_tensor([2] * 1, ["x"]).as_network()
    assert tn.isconnected()
    tn |= qtn.rand_tensor([2] * 3, ["x", "y", "z"])
    assert tn.isconnected()
    tn |= qtn.rand_tensor([2] * 2, ["w", "u"])
    assert not tn.isconnected()
    assert not (qtn.Tensor() | qtn.Tensor()).isconnected()


def test_get_path_between_tids():
    tn = qtn.MPS_rand_state(5, 3)
    path = tn.get_path_between_tids(0, 4)
    assert path.tids == (0, 1, 2, 3, 4)
    path = tn.get_path_between_tids(3, 0)
    assert path.tids == (3, 2, 1, 0)


def test_subgraphs():
    k1 = qtn.MPS_rand_state(6, 7, site_ind_id="a{}")
    k2 = qtn.MPS_rand_state(8, 7, site_ind_id="b{}")
    tn = k1 | k2
    s1, s2 = tn.subgraphs()
    assert {s1.num_tensors, s2.num_tensors} == {6, 8}


def test_gen_paths_loops():
    tn = qtn.TN2D_rand(3, 4, 2)
    loops = tuple(tn.gen_paths_loops())
    assert len(loops) == 6
    assert all(len(loop) == 4 for loop in loops)


def test_gen_paths_loops_intersect():
    tn = qtn.TN2D_empty(5, 4, 2)
    loops = tuple(tn.gen_paths_loops(8, False))
    na = len(loops)
    assert na == len(frozenset(loops))
    assert na == len(frozenset(map(frozenset, loops)))

    loops = tuple(tn.gen_paths_loops(8, True))
    nb = len(loops)
    assert nb == len(frozenset(loops))
    assert nb == len(frozenset(map(frozenset, loops)))
    assert nb > na


def test_gen_inds_connected():
    tn = qtn.TN2D_rand(3, 4, 2)
    patches = tuple(tn.gen_inds_connected(2))
    assert len(patches) == 34


class TestGenGloops:
    # a triangle sharing the single site 2 with a square
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 2)]

    # site 3 dangles, and here the size 3 loops are only found after some
    # size 4 regions have already been queued
    ragged_edges = [
        (0, 1),
        (0, 7),
        (0, 8),
        (1, 2),
        (1, 3),
        (1, 6),
        (1, 7),
        (2, 4),
        (4, 5),
        (4, 8),
        (5, 6),
        (7, 8),
    ]

    # two triangles joined by a path, whose sites lie on no cycle
    dumbbell_edges = [
        (0, 1),
        (1, 2),
        (2, 0),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 5),
    ]

    # a triangle and a square with no bond between them
    split_edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 6), (6, 3)]

    @staticmethod
    def get_gloops(psi, max_size=None, **kwargs):
        gloops = psi.gen_gloops_sites(max_size, **kwargs)
        return sorted(tuple(sorted(gloop)) for gloop in gloops)

    @staticmethod
    def get_gloops_tids(tn, max_size=None, **kwargs):
        gloops = tn.gen_gloops(max_size, **kwargs)
        return sorted(tuple(sorted(gloop)) for gloop in gloops)

    def test_min_stops_at_smallest_loop(self):
        psi = qtn.TN_from_edges_rand(self.edges, D=2, phys_dim=2, seed=42)
        assert self.get_gloops(psi, "min") == [(0, 1, 2)]

    def test_cover_grows_until_all_sites_included(self):
        psi = qtn.TN_from_edges_rand(self.edges, D=2, phys_dim=2, seed=42)
        gloops = self.get_gloops(psi)
        # the triangle alone leaves 3, 4, 5 uncovered
        assert gloops == [(0, 1, 2), (2, 3, 4, 5)]
        assert set.union(*map(set, gloops)) == set(range(6))
        # covering happens at size 4 -> same as requesting that explicitly
        assert gloops == self.get_gloops(psi, 4)

    def test_min_does_not_yield_oversized_loops(self):
        psi = qtn.TN_from_edges_rand(
            self.ragged_edges, D=2, phys_dim=2, seed=42
        )
        gloops = self.get_gloops(psi, "min")
        assert gloops == [(0, 1, 7), (0, 7, 8)]

    def test_dangling_site_ignored(self):
        psi = qtn.TN_from_edges_rand(
            self.ragged_edges, D=2, phys_dim=2, seed=42
        )
        gloops = self.get_gloops(psi)
        # site 3 dangles, so only the other eight sites can be covered
        assert set.union(*map(set, gloops)) == set(range(9)) - {3}
        assert max(map(len, gloops)) == 5

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
    def test_tree_has_no_gloops(self, n):
        # nothing is in the 2-core, so there is nothing to cover at all
        mps = qtn.MPS_rand_state(n, 3)
        with pytest.warns(UserWarning, match="tree like"):
            assert tuple(mps.gen_gloops()) == ()
        # only the automatic size looks for the 2-core
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert tuple(mps.gen_gloops("min")) == ()
            assert tuple(mps.gen_gloops(4)) == ()

    def test_dangling_site_beside_small_loop_ignored(self):
        # site 3 dangles, so covering stops at the triangle
        psi = qtn.TN_from_edges_rand(
            [(0, 1), (1, 2), (2, 0), (0, 3)], D=2, phys_dim=2, seed=42
        )
        assert self.get_gloops(psi) == [(0, 1, 2)]
        assert self.get_gloops(psi, "min") == [(0, 1, 2)]

    def test_uncoverable_target_warns(self):
        # site 4 has two bonds, but the chain has no loop at all
        psi = qtn.TN_from_edges_rand(
            [(i, i + 1) for i in range(9)], D=2, phys_dim=2, seed=42
        )
        with pytest.warns(UserWarning, match="never"):
            assert self.get_gloops(psi, sites=(4,), grow_from="any") == []
        # naming no target instead reports the whole network
        with pytest.warns(UserWarning, match="tree like"):
            assert self.get_gloops(psi) == []

    def test_partly_uncoverable_targets_warn_and_cover_the_rest(self):
        # site 3 dangles, site 0 is in a triangle
        psi = qtn.TN_from_edges_rand(
            [(0, 1), (1, 2), (2, 0), (0, 3)], D=2, phys_dim=2, seed=42
        )
        with pytest.warns(UserWarning, match=r"\(3,\)"):
            gloops = self.get_gloops(psi, sites=(0, 3), grow_from="any")
        assert gloops == [(0, 1, 2)]

    def test_covering_site_on_no_cycle(self):
        psi = qtn.TN_from_edges_rand(
            self.dumbbell_edges, D=2, phys_dim=2, seed=42
        )
        # the path sites can only appear in a loop that spans everything
        gloops = self.get_gloops(psi)
        assert set.union(*map(set, gloops)) == set(range(8))
        assert max(map(len, gloops)) == 8

    def test_lattice_covered_by_smallest_plaquettes(self):
        peps = qtn.PEPS.rand(3, 4, 2)
        gloops = self.get_gloops(peps)
        # every site is in a plaquette -> covering and 'min' agree
        assert gloops == self.get_gloops(peps, "min")
        assert gloops == self.get_gloops(peps, 4)
        assert len(gloops) == 6

    def test_local_grow_from_all_unaffected(self):
        psi = qtn.TN_from_edges_rand(self.edges, D=2, phys_dim=2, seed=42)
        # the first valid loop already contains every target site
        for sites in [(2,), (0, 1), (0, 3)]:
            assert self.get_gloops(psi, sites=sites) == self.get_gloops(
                psi, "min", sites=sites
            )

    def test_local_grow_from_any_covers_each_site(self):
        psi = qtn.TN_from_edges_rand(self.edges, D=2, phys_dim=2, seed=42)
        sites = (0, 3)
        # 'min' stops at the triangle, leaving site 3 uncovered
        assert self.get_gloops(psi, "min", sites=sites, grow_from="any") == [
            (0, 1, 2)
        ]
        assert self.get_gloops(psi, sites=sites, grow_from="any") == [
            (0, 1, 2),
            (2, 3, 4, 5),
        ]

    @pytest.mark.parametrize(
        "grow_from,expected",
        [("alldangle", [(0, 3)]), ("anydangle", [(0,), (3,)])],
    )
    def test_dangle_modes_ignore_covering(self, grow_from, expected):
        # the targets are exempt from the two bond condition, so the seed
        # region is already valid and covering stops there -> the automatic
        # size gives the same as 'min', and no expansion at all
        psi = qtn.TN_from_edges_rand(self.edges, D=2, phys_dim=2, seed=42)
        sites = (0, 3)
        assert self.get_gloops(psi, sites=sites, grow_from=grow_from) == (
            expected
        )
        assert (
            self.get_gloops(psi, "min", sites=sites, grow_from=grow_from)
            == expected
        )
        # an explicit size does expand
        assert (
            len(self.get_gloops(psi, 4, sites=sites, grow_from=grow_from)) > 1
        )

    def test_num_joins_generates_global_gloops(self):
        # joining local gloops needs the global ones, which are generated with
        # the same `max_size` -> the dangling site 3 is ignored by both
        psi = qtn.TN_from_edges_rand(
            self.ragged_edges, D=2, phys_dim=2, seed=42
        )
        sites = (0,)
        assert self.get_gloops(psi, sites=sites, grow_from="any") == [
            (0, 1, 7),
            (0, 7, 8),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert self.get_gloops(
                psi, sites=sites, grow_from="any", num_joins=2
            ) == [(0, 1, 2, 4, 7, 8), (0, 1, 7), (0, 1, 7, 8), (0, 7, 8)]
        assert self.get_gloops(
            psi, "min", sites=sites, grow_from="any", num_joins=2
        ) == [(0, 1, 7), (0, 1, 7, 8), (0, 7, 8)]

    @pytest.mark.parametrize("max_size", [None, "min", 4])
    @pytest.mark.parametrize("grow_from", ["all", "any"])
    def test_empty_sites(self, max_size, grow_from):
        # no targets -> no loops, rather than one empty loop
        psi = qtn.TN_from_edges_rand(self.edges, D=2, phys_dim=2, seed=42)
        assert (
            self.get_gloops(psi, max_size, sites=(), grow_from=grow_from) == []
        )

    @pytest.mark.parametrize("max_size", [0, -1])
    def test_non_positive_max_size_yields_nothing(self, max_size):
        # no error, unlike an automatic size that cannot cover
        psi = qtn.TN_from_edges_rand(self.edges, D=2, phys_dim=2, seed=42)
        assert self.get_gloops(psi, max_size) == []

    @pytest.mark.parametrize("max_size", ["MIN", "cover", "auto"])
    def test_invalid_max_size_raises(self, max_size):
        psi = qtn.TN_from_edges_rand(self.edges, D=2, phys_dim=2, seed=42)
        with pytest.raises(ValueError):
            self.get_gloops(psi, max_size)

    @pytest.mark.parametrize("max_size", [None, "min", 4])
    def test_double_bond_is_a_loop(self, max_size):
        # two bonds between two tensors -> a valid size 2 loop
        tn = qtn.TensorNetwork(
            [
                qtn.rand_tensor((2, 2), inds=["a", "b"]),
                qtn.rand_tensor((2, 2), inds=["a", "b"]),
            ]
        )
        assert self.get_gloops_tids(tn, max_size) == [
            tuple(sorted(tn.tensor_map))
        ]

    def test_hyper_index_is_not_a_loop(self):
        # three tensors on one index have a single bond each, not two
        tn = qtn.TensorNetwork(
            [qtn.rand_tensor((2,), inds=["h"]) for _ in range(3)]
        )
        assert self.get_gloops_tids(tn, "min") == []
        with pytest.warns(UserWarning, match="tree like"):
            assert self.get_gloops_tids(tn) == []

    @pytest.mark.parametrize("max_size", [None, "min", 4])
    def test_empty_network(self, max_size):
        assert self.get_gloops_tids(qtn.TensorNetwork([]), max_size) == []

    def test_disconnected_components(self):
        psi = qtn.TN_from_edges_rand(
            self.split_edges, D=2, phys_dim=2, seed=42
        )
        # each component is reached separately
        assert self.get_gloops(psi) == [(0, 1, 2), (3, 4, 5, 6)]
        assert self.get_gloops(psi, "min") == [(0, 1, 2)]
        # a region is only checked bond by bond, so with `grow_from='all'` the
        # union of one loop per component is accepted as a single loop
        assert self.get_gloops(psi, sites=(0, 3)) == [(0, 1, 2, 3, 4, 5, 6)]
        assert self.get_gloops(psi, sites=(0, 3), grow_from="any") == [
            (0, 1, 2),
            (3, 4, 5, 6),
        ]

    def test_global_generation_ignores_dangling_sites_quietly(self):
        # a dangling site is only reported if it was named as a target
        psi = qtn.TN_from_edges_rand(
            self.ragged_edges, D=2, phys_dim=2, seed=42
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert len(self.get_gloops(psi)) == 6


def test_connected_bipartitions():
    tn = qtn.TN_rand_reg(6, 3, 2)
    for pa, pb in tn.connected_bipartitions():
        assert pa | pb == frozenset(tn.tensor_map)
        assert not (pa & pb)
        assert tn._select_tids(pa).isconnected()
        assert tn._select_tids(pb).isconnected()
