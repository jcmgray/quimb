import math

import numpy as np
import pytest

import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tn2dinf.core import (
    PEPSInfinite2D,
    TensorNetworkInfinite2DFlat,
)
from quimb.tensor.tn2dinf.geometry import (
    GeometryInfinite2D,
    get_bond_sorted,
    get_bond_type,
)
from quimb.tensor.tn2dinf.tebd import (
    LocalHamInfinite2D,
    SimpleUpdateInfinite2D,
)

geom_square_2x2 = GeometryInfinite2D.square()
geom_square_2x2_nnn = GeometryInfinite2D.square(couplings=2)
edges_inf_2d_square_2x2 = geom_square_2x2.bond_types
edges_inf_2d_square_2x2_nnn = geom_square_2x2_nnn.bond_types


def test_public_exports():
    # the package surface is re-exported on the top-level quimb.tensor namespace
    assert qtn.GeometryInfinite2D is GeometryInfinite2D
    assert qtn.TensorNetworkInfinite2DFlat is TensorNetworkInfinite2DFlat
    assert qtn.PEPSInfinite2D is PEPSInfinite2D
    assert qtn.LocalHamInfinite2D is LocalHamInfinite2D
    assert qtn.SimpleUpdateInfinite2D is SimpleUpdateInfinite2D


class TestGetBondType:
    def test_order_invariant(self):
        for a, b in GeometryInfinite2D.square(3, couplings=3).bond_types:
            assert get_bond_type(a, b) == get_bond_type(b, a)

    def test_starts_in_origin_cell(self):
        for edges in (edges_inf_2d_square_2x2, edges_inf_2d_square_2x2_nnn):
            for a, b in edges:
                (cell_a, _), _ = get_bond_type(a, b)
                assert cell_a == (0, 0)

    def test_within_cell_ordered_by_site_type(self):
        bt = get_bond_type(((0, 0), (1, 1)), ((0, 0), (0, 1)))
        (cell_a, sta), (cell_b, stb) = bt
        assert cell_a == (0, 0) and cell_b == (0, 0)
        assert sta <= stb

    def test_is_sorted_pair(self):
        # canonical form: first endpoint at the origin cell, and the pair sorted
        for edges in (edges_inf_2d_square_2x2, edges_inf_2d_square_2x2_nnn):
            for a, b in edges:
                first, second = get_bond_type(a, b)
                assert first[0] == (0, 0)
                assert first < second

    def test_dx_precedence(self):
        # cells compare dx-first: an up-left edge (dx < 0) flips to down-right
        bt = get_bond_type(((0, 0), "A"), ((-1, 1), "B"))
        assert bt == (((0, 0), "B"), ((1, -1), "A"))
        assert get_bond_type(((-1, 1), "B"), ((0, 0), "A")) == bt

    def test_idempotent(self):
        g = GeometryInfinite2D.square()
        for bt in g.bond_types:
            a, b = bt
            assert get_bond_type(a, b) == bt

    def test_rejects_distant_cells(self):
        with pytest.raises(ValueError):
            get_bond_type(((0, 0), (0, 0)), ((2, 0), (0, 0)))
        with pytest.raises(ValueError):
            get_bond_type(((0, 0), (0, 0)), ((0, -2), (1, 1)))


class TestGetBondSorted:
    def test_order_invariant(self):
        a, b = ((0, 0), (0, 1)), ((1, 0), (0, 0))
        assert get_bond_sorted(a, b) == get_bond_sorted(b, a)
        lo, hi = get_bond_sorted(a, b)
        assert lo <= hi


class TestGeometryInfinite2D:
    def test_site_and_bond_type_counts(self):
        g = GeometryInfinite2D.square()
        assert len(g.site_types) == 4
        assert len(g.bond_types) == 8

    def test_types_are_canonically_sorted(self):
        g = GeometryInfinite2D.square()
        assert g.site_types == tuple(sorted(g.site_types))
        assert g.bond_types == tuple(sorted(g.bond_types))

    def test_dedup_matches_manual_canonicalization(self):
        # __init__ dedup must agree with canonicalizing each edge by hand
        for edges in (
            edges_inf_2d_square_2x2,
            edges_inf_2d_square_2x2_nnn,
        ):
            manual = {get_bond_type(a, b) for a, b in edges}
            g = GeometryInfinite2D(edges)
            assert set(g.bond_types) == manual
            assert len(g.bond_types) == len(manual)

    def test_get_site_neighbors_concrete(self):
        g = GeometryInfinite2D.square()
        nbrs = set(g.get_site_neighbors(((0, 0), (0, 0))))
        assert nbrs == {
            ((0, 0), (0, 1)),
            ((0, 0), (1, 0)),
            ((0, -1), (0, 1)),
            ((-1, 0), (1, 0)),
        }

    def test_neighbors_reciprocal(self):
        g = GeometryInfinite2D.square()
        for st in g.site_types:
            site = ((0, 0), st)
            for nb in g.get_site_neighbors(site):
                assert site in set(g.get_site_neighbors(nb))

    def test_neighbors_translation_invariant(self):
        g = GeometryInfinite2D.square()
        base = set(g.get_site_neighbors(((0, 0), (1, 1))))
        shifted = set(g.get_site_neighbors(((3, -2), (1, 1))))
        assert shifted == {((c[0] + 3, c[1] - 2), st) for c, st in base}

    def test_neighbor_entry_count_matches_bond_degree(self):
        g = GeometryInfinite2D.square()
        total = sum(len(v) for v in g.site_type_neighbors.values())
        assert total == 2 * len(g.bond_types)

    def test_covering_sites_include_unit_cell(self):
        g = GeometryInfinite2D.square()
        for st in g.site_types:
            assert ((0, 0), st) in g.covering_sites

    def test_covering_bonds_endpoints_present_and_sorted(self):
        g = GeometryInfinite2D.square()
        for a, b in g.covering_bonds:
            assert a in g.covering_sites
            assert b in g.covering_sites
            assert a <= b

    def test_coordinate_default_is_cell(self):
        g = GeometryInfinite2D.square()  # no basis/positions
        x, y = g.coordinate(((2, 3), (0, 0)))
        assert (x, y) == pytest.approx((2.0, 3.0))

    def test_coordinate_with_positions_and_translation(self):
        g = geom_square_2x2  # square basis + 0-based offsets
        x, y = g.coordinate(((0, 0), (0, 1)))
        assert (x, y) == pytest.approx((0.0, 0.5))
        # +1 cell along a1=(1,0) shifts x by 1
        x2, y2 = g.coordinate(((1, 0), (0, 1)))
        assert (x2, y2) == pytest.approx((1.0, 0.5))

    def test_coordinate_nonsquare_basis(self):
        basis = ((1.0, 0.0), (0.5, math.sqrt(3) / 2))
        g = GeometryInfinite2D.square(basis=basis)
        # cell (0, 1), no offsets -> exactly a2
        x, y = g.coordinate(((0, 1), (0, 0)))
        assert (x, y) == pytest.approx((0.5, math.sqrt(3) / 2))

    def test_self_loop_raises(self):
        edges = ((((0, 0), "A"), ((1, 0), "A")),)
        with pytest.raises(NotImplementedError):
            GeometryInfinite2D(edges)

    def test_repr(self):
        g = GeometryInfinite2D.square()
        r = repr(g)
        assert "GeometryInfinite2D" in r
        assert "site_types=4" in r
        assert "bond_types=8" in r

    def test_cell_size_2x2(self):
        # 2x2 unit cell spans two sub-lattice hops in each direction
        assert geom_square_2x2.get_cell_size() == (2, 2)

    def test_cell_size_asymmetric(self):
        # an nx-by-ny block cell of NN-bonded sites costs (nx, ny) hops to
        # cross (a self-loop-free generalization of the 2x2 construction)
        def block_cell_edges(nx, ny):
            edges = []
            for i in range(nx):
                for j in range(ny):
                    a = ((0, 0), (i, j))
                    xnb = (
                        ((0, 0), (i + 1, j))
                        if i + 1 < nx
                        else ((1, 0), (0, j))
                    )
                    ynb = (
                        ((0, 0), (i, j + 1))
                        if j + 1 < ny
                        else ((0, 1), (i, 0))
                    )
                    edges.append((a, xnb))
                    edges.append((a, ynb))
            return edges

        g = GeometryInfinite2D(block_cell_edges(3, 2))
        assert g.get_cell_size() == (3, 2)

    def test_tiling_for_radius(self):
        g = geom_square_2x2  # dx = dy = 2
        assert g.get_tiling_for_radius(0) == (0, 0)
        assert g.get_tiling_for_radius(2) == (1, 1)
        assert g.get_tiling_for_radius(3) == (2, 2)  # reaches into 2nd cell
        assert g.get_tiling_for_radius(4) == (2, 2)
        # monotonic and never undershoots the radius ball it must contain
        assert g.get_tiling_for_radius(5) >= g.get_tiling_for_radius(4)


class TestSquareBuilder:
    def test_2x2_matches_constant(self):
        g = GeometryInfinite2D.square(2, 2)
        ref = GeometryInfinite2D.square()
        assert set(g.bond_types) == set(ref.bond_types)
        assert set(g.site_types) == set(ref.site_types)

    def test_nnn_matches_constant(self):
        g = GeometryInfinite2D.square(2, 2, couplings=2)
        ref = GeometryInfinite2D(edges_inf_2d_square_2x2_nnn)
        assert set(g.bond_types) == set(ref.bond_types)

    def test_radius_equals_two_shells(self):
        # radius sqrt(2) ~ 1.5 includes NN + NNN, same as couplings=2
        a = GeometryInfinite2D.square(2, 2, radius=1.5).bond_types
        b = GeometryInfinite2D.square(2, 2, couplings=2).bond_types
        assert set(a) == set(b)

    def test_couplings_and_radius_conflict_raises(self):
        with pytest.raises(ValueError):
            GeometryInfinite2D.square(2, 2, couplings=1, radius=1.0)

    def test_float_couplings_points_to_radius(self):
        with pytest.raises(TypeError):
            GeometryInfinite2D.square(2, 2, couplings=1.5)

    def test_explicit_displacements(self):
        g = GeometryInfinite2D.square(3, couplings=[(1, 0), (0, 1)])  # M -> N
        assert len(g.site_types) == 9
        assert set(g.bond_types) == set(
            GeometryInfinite2D.square(3).bond_types
        )

    def test_zero_based_positions(self):
        g = GeometryInfinite2D.square(2, 2)
        assert g.coordinate(((0, 0), (0, 0))) == pytest.approx((0.0, 0.0))
        assert g.coordinate(((0, 0), (1, 1))) == pytest.approx((0.5, 0.5))

    def test_coupling_folds_onto_sublattice_raises(self):
        # 3rd-NN (2, 0) folds onto the same sublattice in a 2x2 cell
        with pytest.raises(ValueError):
            GeometryInfinite2D.square(2, 2, couplings=[(2, 0)])
        # but is representable in a larger cell
        assert GeometryInfinite2D.square(4, 4, couplings=[(2, 0)]).bond_types


class TestPEPSInfinite2D:
    def test_subclasses_tensor_network_infinite_2d(self):
        assert issubclass(PEPSInfinite2D, TensorNetworkInfinite2DFlat)

    def test_base_builds_bare_gen_fragment(self):
        # the base is concrete: an empty bare TensorNetworkGen (no physical leg)
        itn = TensorNetworkInfinite2DFlat(geom_square_2x2)
        assert itn.fragment.num_tensors == 0
        assert isinstance(itn.fragment, qtn.TensorNetworkGen)
        assert not isinstance(itn.fragment, qtn.TensorNetworkGenVector)
        # bond-only leg helpers (no physical leg appended)
        site = ((0, 0), itn.geometry.site_types[0])
        degree = len(list(itn.geometry.get_site_neighbors(site)))
        assert itn.get_site_shape(site, 5) == (5,) * degree
        assert len(itn.get_site_inds(site)) == degree
        assert len(itn.get_site_duals(site)) == degree

    def test_construct_from_geometry_or_edges(self):
        a = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=2)
        b = PEPSInfinite2D.rand(edges_inf_2d_square_2x2, bond_dim=2)
        assert a.site_types == b.site_types
        assert isinstance(b.geometry, GeometryInfinite2D)
        assert a.geometry is geom_square_2x2

    def test_bare_init_is_empty(self):
        # the bare constructor builds an empty scaffold, no tensors / dims
        psi = PEPSInfinite2D(geom_square_2x2)
        assert psi.fragment.num_tensors == 0
        assert psi._sites == set()
        assert not hasattr(psi, "bond_dim")

    def test_rand_populates_block_with_shapes(self):
        psi = PEPSInfinite2D.rand(
            geom_square_2x2, bond_dim=3, phys_dim=2, seed=0
        )
        for site in psi._sites:
            assert psi.fragment[site].shape == psi.get_site_shape(site, 3, 2)

    def test_from_fill_fn_is_fill_fn_first(self):
        # fill_fn is the first positional arg (tensor_builder convention)
        geom = geom_square_2x2
        psi = PEPSInfinite2D.from_fill_fn(np.ones, geom, bond_dim=2)
        for ts in psi.shared_tensors.values():
            for t in ts:
                assert t.data == pytest.approx(1.0)

    def test_get_site_shape(self):
        psi = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=2)
        site = ((0, 0), psi.site_types[0])
        degree = len(list(psi.geometry.get_site_neighbors(site)))
        assert psi.get_site_shape(site, 5, phys_dim=3) == (5,) * degree + (3,)
        # matches the tensor actually built
        assert psi.fragment[site].shape == psi.get_site_shape(site, 2, 2)

    def test_get_site_duals_opposite_across_bond(self):
        psi = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=2)
        geom = psi.geometry
        for st in geom.site_types:
            sa = ((0, 0), st)
            nbrs_a = list(geom.get_site_neighbors(sa))
            duals_a = psi.get_site_duals(sa)
            assert duals_a[-1] is False  # physical leg is a ket
            for i, sb in enumerate(nbrs_a):
                # the two ends of the shared bond carry opposite duals
                j = list(geom.get_site_neighbors(sb)).index(sa)
                assert duals_a[i] != psi.get_site_duals(sb)[j]

    def test_init_populates_neighbor_block(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=3)
        # `.rand` adds the full [-1, 1]^2 block of cells x site_types
        block = {
            ((x, y), st)
            for x in (-1, 0, 1)
            for y in (-1, 0, 1)
            for st in itn.geometry.site_types
        }
        assert itn._sites == block
        assert itn.fragment.num_tensors == len(block)
        for site in block:
            t = itn.fragment[site]
            assert t in itn.shared_tensors[site[1]]
            # one virtual bond per neighbor + one physical index (edge sites
            # keep dangling bonds for neighbors outside the block)
            degree = len(list(itn.geometry.get_site_neighbors(site)))
            assert t.ndim == degree + 1
            assert itn.site_ind_id.format(site) in t.inds
            assert itn.site_type_tag_id.format(site[1]) in t.tags

    def test_add_site_idempotent(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=3)
        st = itn.site_types[0]
        site = ((0, 0), st)
        assert itn.has_fragment_site(site)
        n_sites, n_shared = len(itn._sites), len(itn.shared_tensors[st])
        itn.add_fragment_site(site)  # no-op
        assert len(itn._sites) == n_sites
        assert len(itn.shared_tensors[st]) == n_shared

    def test_add_site_out_of_range_raises(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=3)
        with pytest.raises(ValueError):
            itn.add_fragment_site(((2, 0), itn.site_types[0]))

    def test_same_site_type_shares_data(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=3)
        for st, ts in itn.shared_tensors.items():
            assert len(ts) >= 1
            first = ts[0]
            for t in ts:
                assert t.data is first.data

    def test_get_bond_ind_order_invariant(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=3)
        a = ((0, 0), itn.site_types[0])
        b = next(iter(itn.geometry.get_site_neighbors(a)))
        assert itn.get_bond_ind(a, b) == itn.get_bond_ind(b, a)

    def test_bond_shared_between_neighbors(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=3)
        a = ((0, 0), itn.site_types[0])
        b = next(iter(itn.geometry.get_site_neighbors(a)))

        bix = itn.get_bond_ind(a, b)
        ta = itn.fragment[a]
        tb = itn.fragment[b]
        assert bix in ta.inds and bix in tb.inds
        # an inner index connecting exactly the two tensors
        assert len(itn.fragment.ind_map[bix]) == 2
        # and registered under the bond type
        assert bix in itn.shared_indices[itn.get_bond_type(a, b)]

    def test_fragment_tensor_count_matches_sites(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=3)
        assert itn.fragment.num_tensors == len(itn._sites)

    def test_copy_independent_structure(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=3)
        c = itn.copy()
        assert c.fragment is not itn.fragment
        assert c._sites == itn._sites and c._sites is not itn._sites
        assert c.geometry is itn.geometry  # static config shared
        assert c.max_bond() == itn.max_bond()  # dims live in the tensors
        assert c.site_tag_id == itn.site_tag_id
        assert c.site_ind_id == itn.site_ind_id
        assert c.bond_ind_id == itn.bond_ind_id

    def test_copy_shared_tensors_regrouped_not_aliased(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=3)
        c = itn.copy()
        assert set(c.shared_tensors) == set(itn.shared_tensors)
        for st, ts in c.shared_tensors.items():
            assert len(ts) == len(itn.shared_tensors[st])
            for t in ts:
                # new object, not aliasing the original network's tensor
                assert all(t is not o for o in itn.shared_tensors[st])
                # and it really lives in the copied fragment
                assert any(t is ft for ft in c.fragment)

    def test_copy_modify_isolated_shallow(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=3)
        c = itn.copy()  # deep=False -> arrays shared, but modify is isolated
        st = itn.site_types[0]
        orig = itn.shared_tensors[st][0].data.copy()
        t = c.shared_tensors[st][0]
        t.modify(data=t.data * 0.0)
        assert itn.shared_tensors[st][0].data == pytest.approx(orig)

    def test_copy_shared_indices_value_copy(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=3)
        c = itn.copy()
        assert c.shared_indices == itn.shared_indices
        assert c.shared_indices is not itn.shared_indices
        for bt in c.shared_indices:
            assert c.shared_indices[bt] is not itn.shared_indices[bt]

    def test_copy_deep_independent_data(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=3)
        c = itn.copy(deep=True)
        st = itn.site_types[0]
        assert (
            c.shared_tensors[st][0].data is not itn.shared_tensors[st][0].data
        )

    def test_compute_local_expectation_cluster_sum_and_return_all(self):
        geom = geom_square_2x2
        itn = PEPSInfinite2D.rand(geom, bond_dim=3)
        ham = LocalHamInfinite2D(geom, qu.ham_heis(2))
        gauges = {}
        itn.gauge_all_simple_(gauges=gauges, max_iterations=50, tol=1e-12)

        per_bond = itn.compute_local_expectation_cluster(
            ham, gauges=gauges, return_all=True
        )
        total = itn.compute_local_expectation_cluster(ham, gauges=gauges)
        assert set(per_bond) == set(geom.bond_types)
        assert sum(per_bond.values()) == pytest.approx(total)
        assert abs(np.imag(total)) < 1e-12
        # single-term path matches the corresponding entry
        bt = geom.bond_types[0]
        e1 = itn.local_expectation_cluster(ham.get_gate(bt), bt, gauges=gauges)
        assert e1 == pytest.approx(per_bond[bt])

    def test_cluster_expectation_product_state_is_exact(self):
        # for bond_dim=1 (a product state) the max_distance=0 cluster value
        # must equal the exact two-site expectation
        geom = geom_square_2x2
        itn = PEPSInfinite2D.rand(geom, bond_dim=1)
        ham = LocalHamInfinite2D(geom, qu.ham_heis(2))
        gauges = {}
        itn.gauge_all_simple_(gauges=gauges, max_iterations=20)

        G = qu.ham_heis(2)
        per_bond = itn.compute_local_expectation_cluster(
            ham, gauges=gauges, return_all=True
        )
        for (sa, sb), e_cluster in per_bond.items():
            fa = itn.fragment[sa].data.reshape(-1)
            fb = itn.fragment[sb].data.reshape(-1)
            # 1d amplitude vectors -> np.kron (qu.kron is operator-only)
            v = np.kron(fa, fb)
            exact = (v.conj() @ G @ v) / ((fa.conj() @ fa) * (fb.conj() @ fb))
            assert e_cluster == pytest.approx(exact)

    def test_cluster_trivial_bonds_gauge_independent(self):
        # bond_dim=1 -> trivial bonds -> gauges make no difference
        geom = geom_square_2x2
        itn = PEPSInfinite2D.rand(geom, bond_dim=1)
        ham = LocalHamInfinite2D(geom, qu.ham_heis(2))
        gauges = {}
        itn.gauge_all_simple_(gauges=gauges, max_iterations=20)
        eg = itn.compute_local_expectation_cluster(ham, gauges=gauges)
        en = itn.compute_local_expectation_cluster(ham)
        assert eg == pytest.approx(en)

    def test_partial_trace_cluster_matches_expectation(self):
        geom = geom_square_2x2
        itn = PEPSInfinite2D.rand(geom, bond_dim=3)
        gauges = {}
        itn.gauge_all_simple_(gauges=gauges, max_iterations=50, tol=1e-12)

        G = qu.ham_heis(2)
        bt = geom.bond_types[0]
        rho = itn.partial_trace_cluster(bt, gauges=gauges)
        assert rho.shape == (4, 4)
        assert np.trace(rho) == pytest.approx(1.0)  # normalized
        # Tr[rho G] reproduces the local expectation
        assert np.trace(rho @ G) == pytest.approx(
            itn.local_expectation_cluster(G, bt, gauges=gauges)
        )

    def test_build_fragment_unrestricted_and_tiles_gauges(self):
        geom = geom_square_2x2
        psi = PEPSInfinite2D.rand(geom, bond_dim=3)
        gauges = {}
        psi.gauge_all_simple_(gauges=gauges, max_iterations=20)
        # a region reaching outside the [-1, 1]^2 block that add_fragment_site
        # forbids, plus an isolated far site whose bonds all dangle
        st = geom.site_types[0]
        far = ((3, 0), st)
        sites = psi._region_sites([((0, 0), st)], 1) | {far}

        # build_fragment returns just the tn (no gauges to unpack)
        bare = psi.build_fragment(sites)
        assert isinstance(bare, qtn.TensorNetworkGenVector)
        assert bare.num_tensors == len(sites)
        # each tensor carries its site_type's shared data
        for site in sites:
            assert bare[site].data == pytest.approx(
                psi.shared_tensors[site[1]][0].data
            )

        # with no gauges the pair still unpacks, gauges side just None
        _, no_gauges = psi.build_fragment_with_gauges(sites, None)
        assert no_gauges is None

        # each fragment bond keyed to itself, valued by its bond_type gauge
        frag, frag_gauges = psi.build_fragment_with_gauges(sites, gauges)
        assert frag.num_tensors == len(sites)
        # every bond present is gauged, incl. the far site's dangling bonds
        for site in sites:
            for nb in geom.get_site_neighbors(site):
                ind = psi.get_bond_ind(site, nb)
                canon = psi.get_bond_ind(*psi.get_bond_type(site, nb))
                assert ind in frag_gauges
                assert frag_gauges[ind] == pytest.approx(gauges[canon])

    def test_cluster_max_distance_runs(self):
        # max_distance>0 builds a tiled fragment; values are finite + real
        geom = geom_square_2x2
        psi = PEPSInfinite2D.rand(geom, bond_dim=2)
        ham = LocalHamInfinite2D(geom, qu.ham_heis(2))
        gauges = {}
        psi.gauge_all_simple_(gauges=gauges, max_iterations=50, tol=1e-12)
        for md in (1, 2):
            e = psi.compute_local_expectation_cluster(
                ham, gauges=gauges, max_distance=md
            )
            assert np.isfinite(e) and abs(np.imag(e)) < 1e-10

    def test_cluster_max_distance_translation_invariant(self):
        # tiled fragment + tiled gauges are translation consistent: a shifted
        # `where` gives an identical expectation regardless of convergence
        geom = geom_square_2x2
        psi = PEPSInfinite2D.rand(geom, bond_dim=2)
        ham = LocalHamInfinite2D(geom, qu.ham_heis(2))
        gauges = {}
        psi.gauge_all_simple_(gauges=gauges, max_iterations=50, tol=1e-12)
        bt = geom.bond_types[0]
        G = ham.get_gate(bt)
        bt_shift = tuple(((c[0] + 4, c[1] - 2), s) for c, s in bt)
        v0 = psi.local_expectation_cluster(
            G, bt, gauges=gauges, max_distance=2
        )
        vs = psi.local_expectation_cluster(
            G, bt_shift, gauges=gauges, max_distance=2
        )
        assert v0 == pytest.approx(vs)

    def test_cluster_max_distance_product_state_exact(self):
        # bond_dim=1 product state: any max_distance equals the exact 2-site
        # expectation (trivial environment)
        geom = geom_square_2x2
        psi = PEPSInfinite2D.rand(geom, bond_dim=1)
        gauges = {}
        psi.gauge_all_simple_(gauges=gauges, max_iterations=20)
        G = qu.ham_heis(2)
        bt = geom.bond_types[0]
        sa, sb = bt
        fa = psi.fragment[sa].data.reshape(-1)
        fb = psi.fragment[sb].data.reshape(-1)
        # 1d amplitude vectors -> np.kron (qu.kron is operator-only)
        v = np.kron(fa, fb)
        exact = (v.conj() @ G @ v) / ((fa.conj() @ fa) * (fb.conj() @ fb))
        for md in (0, 1, 2):
            e = psi.local_expectation_cluster(
                G, bt, gauges=gauges, max_distance=md
            )
            assert e == pytest.approx(exact)

    def test_gloop_expand_runs_and_sums(self):
        geom = geom_square_2x2
        psi = PEPSInfinite2D.rand(geom, bond_dim=2)
        ham = LocalHamInfinite2D(geom, qu.ham_heis(2))
        gauges = {}
        psi.gauge_all_simple_(gauges=gauges, max_iterations=50, tol=1e-12)
        bt = geom.bond_types[0]
        v = psi.local_expectation_gloop_expand(
            ham.get_gate(bt), bt, gloops=4, gauges=gauges
        )
        assert np.isfinite(v) and abs(np.imag(v)) < 1e-10
        per = psi.compute_local_expectation_gloop_expand(
            ham, gloops=4, gauges=gauges, return_all=True
        )
        assert set(per) == set(geom.bond_types)
        total = psi.compute_local_expectation_gloop_expand(
            ham, gloops=4, gauges=gauges
        )
        assert sum(per.values()) == pytest.approx(total)

    def test_gloop_expand_explicit_loops(self):
        # explicit loop site-lists are honored (region = union of their sites)
        geom = geom_square_2x2
        psi = PEPSInfinite2D.rand(geom, bond_dim=2)
        ham = LocalHamInfinite2D(geom, qu.ham_heis(2))
        gauges = {}
        psi.gauge_all_simple_(gauges=gauges, max_iterations=20)
        bt = geom.bond_types[0]
        # the unit-cell plaquette as an explicit generalized loop
        plaq = (
            ((0, 0), (0, 0)),
            ((0, 0), (0, 1)),
            ((0, 0), (1, 1)),
            ((0, 0), (1, 0)),
        )
        v = psi.local_expectation_gloop_expand(
            ham.get_gate(bt), bt, gloops=[plaq], gauges=gauges
        )
        assert np.isfinite(v) and abs(np.imag(v)) < 1e-10

    def test_gauge_all_simple_populates_and_normalizes(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=4)
        gauges = {}
        itn.gauge_all_simple_(gauges=gauges, max_iterations=50, tol=1e-12)
        # every covering bond has a gauge, each renormalized to unit norm
        bixs = {itn.get_bond_ind(a, b) for a, b in itn.geometry.covering_bonds}
        assert bixs <= set(gauges)
        for g in gauges.values():
            assert float(np.linalg.norm(g)) == pytest.approx(1.0)

    def test_gauge_all_simple_preserves_shared_invariants(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=4)
        gauges = {}
        itn.gauge_all_simple_(gauges=gauges, max_iterations=20)
        # same site_type -> identical data
        for ts in itn.shared_tensors.values():
            for t in ts:
                assert t.data == pytest.approx(ts[0].data)
        # same bond_type -> identical gauge
        for inds in itn.shared_indices.values():
            ref = gauges[next(iter(inds))]
            for ix in inds:
                assert gauges[ix] == pytest.approx(ref)

    def test_gauge_all_simple_converges(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=4)
        gauges = {}
        itn.gauge_all_simple_(gauges=gauges, max_iterations=50, tol=1e-12)
        before = dict(gauges)
        itn.gauge_all_simple_(gauges=gauges, max_iterations=1)
        assert all(
            np.linalg.norm(gauges[k] - before[k]) < 1e-8 for k in before
        )

    def test_gauge_all_simple_info(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=4)
        # tol > 0 tracks the singular value diff and can stop early
        info = {}
        itn.gauge_all_simple_(max_iterations=50, tol=1e-12, info=info)
        assert info["iterations"] < 50  # converged before the cap
        assert info["max_sdiff"] <= 1e-12
        assert "sdiff" not in info  # internal running diff cleaned up

        # tol == 0 computes no diffs: runs every sweep and reports -1.0
        info = {}
        itn.gauge_all_simple_(max_iterations=3, tol=0.0, info=info)
        assert info["iterations"] == 3
        assert info["max_sdiff"] == -1.0

    def test_gauge_all_simple_not_inplace_leaves_original(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=4)
        st = itn.site_types[0]
        before = itn.shared_tensors[st][0].data.copy()
        out = itn.gauge_all_simple(max_iterations=10)  # copy + reabsorb
        assert out is not itn
        assert itn.shared_tensors[st][0].data == pytest.approx(before)

    def test_gate_simple_preserves_shared_invariants(self):
        geom = geom_square_2x2
        psi = PEPSInfinite2D.rand(geom, bond_dim=2)
        ham = LocalHamInfinite2D(geom, qu.ham_heis(2))
        gauges = {}
        psi.gauge_all_simple_(gauges=gauges, max_iterations=20)
        for bt in ham.get_auto_ordering():
            psi.gate_simple_(
                ham.get_gate_expm(bt, -0.1), bt, gauges=gauges, max_bond=5
            )
        # same site_type -> identical data; same bond_type -> identical gauge
        for ts in psi.shared_tensors.values():
            for t in ts:
                assert t.data == pytest.approx(ts[0].data)
        for inds in psi.shared_indices.values():
            ref = gauges[next(iter(inds))]
            for ix in inds:
                assert gauges[ix] == pytest.approx(ref)

    def test_gate_simple_not_inplace_leaves_tensors(self):
        geom = geom_square_2x2
        psi = PEPSInfinite2D.rand(geom, bond_dim=2)
        ham = LocalHamInfinite2D(geom, qu.ham_heis(2))
        gauges = {}
        psi.gauge_all_simple_(gauges=gauges, max_iterations=20)
        st = psi.site_types[0]
        before = psi.shared_tensors[st][0].data.copy()
        bt = geom.bond_types[0]
        out = psi.gate_simple(
            ham.get_gate_expm(bt, -0.1), bt, gauges=gauges, max_bond=4
        )
        assert out is not psi
        assert psi.shared_tensors[st][0].data == pytest.approx(before)

    def test_gate_simple_lowers_energy(self):
        geom = geom_square_2x2
        rng = np.random.default_rng(0)
        psi = PEPSInfinite2D.from_fill_fn(
            lambda s: rng.standard_normal(s), geom, bond_dim=2
        )
        ham = LocalHamInfinite2D(geom, qu.ham_heis(2))
        gauges = {}
        psi.gauge_all_simple_(gauges=gauges, max_iterations=100, tol=1e-10)
        e0 = (
            psi.compute_local_expectation_cluster(ham, gauges=gauges)
            / ham.nsites
        )
        ordering = ham.get_auto_ordering()
        for _ in range(50):
            for bt in ordering:
                psi.gate_simple_(
                    ham.get_gate_expm(bt, -0.1), bt, gauges=gauges, max_bond=4
                )
        psi.gauge_all_simple_(gauges=gauges, max_iterations=100, tol=1e-10)
        e1 = (
            psi.compute_local_expectation_cluster(ham, gauges=gauges)
            / ham.nsites
        )
        assert e1 < e0

    def test_gate_simple_single_site(self):
        geom = geom_square_2x2
        psi = PEPSInfinite2D.rand(geom, bond_dim=2)
        gauges = {}
        psi.gauge_all_simple_(gauges=gauges, max_iterations=20)
        site = ((0, 0), geom.site_types[0])
        psi.gate_simple_(qu.pauli("Z"), (site,), gauges=gauges)
        st = site[1]
        for t in psi.shared_tensors[st]:
            assert t.data == pytest.approx(psi.shared_tensors[st][0].data)

    def test_gate_simple_long_range(self):
        # NNN Hamiltonian term on an NN PEPS: endpoints aren't directly bonded,
        # so the gate goes via a path of sites (different TN/ham geometries)
        tn_geom = geom_square_2x2
        ham = LocalHamInfinite2D(geom_square_2x2_nnn, qu.ham_heis(2))
        psi = PEPSInfinite2D.rand(tn_geom, bond_dim=2)
        gauges = {}
        psi.gauge_all_simple_(gauges=gauges, max_iterations=20)

        bt = (((0, 0), (0, 0)), ((0, 0), (1, 1)))  # within-cell diagonal
        assert bt in ham.bond_types
        (ta,) = psi.fragment._get_tids_from_inds(psi.fragment.site_ind(bt[0]))
        (tb,) = psi.fragment._get_tids_from_inds(psi.fragment.site_ind(bt[1]))
        assert not psi.fragment.tensor_map[ta].bonds(
            psi.fragment.tensor_map[tb]
        )

        psi.gate_simple_(
            ham.get_gate_expm(bt, -0.1), bt, gauges=gauges, max_bond=4
        )
        for ts in psi.shared_tensors.values():
            for t in ts:
                assert t.data == pytest.approx(ts[0].data)
        for inds in psi.shared_indices.values():
            ref = gauges[next(iter(inds))]
            for ix in inds:
                assert gauges[ix] == pytest.approx(ref)

    def test_max_bond(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=3)
        assert itn.max_bond() == 3

    def test_gauge_all_simple_rejects_fuse_multibonds(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=2)
        with pytest.raises(NotImplementedError):
            itn.gauge_all_simple_(gauges={}, fuse_multibonds=True)
        # the default (False) still works
        itn.gauge_all_simple_(
            gauges={}, fuse_multibonds=False, max_iterations=2
        )

    def test_gauge_simple_insert_matches_conditioning(self):
        # gauges=None conditioning == gauge (left on bonds) then reinsert them
        rng = np.random.default_rng(1)
        a = PEPSInfinite2D.from_fill_fn(
            lambda s: rng.standard_normal(s),
            geom_square_2x2,
            bond_dim=3,
        )
        b = a.copy(deep=True)
        a.gauge_all_simple_(max_iterations=10)  # internal gauges, reabsorbed
        gauges = {}
        b.gauge_all_simple_(gauges=gauges, max_iterations=10)  # left on bonds
        b.gauge_simple_insert(gauges)  # reabsorb explicitly
        assert gauges  # not consumed (non-destructive)
        for st in a.site_types:
            assert a.shared_tensors[st][0].data == pytest.approx(
                b.shared_tensors[st][0].data
            )

    def test_normalize_simple(self):
        itn = PEPSInfinite2D.rand(geom_square_2x2, bond_dim=3)
        gauges = {}
        itn.gauge_all_simple_(gauges=gauges, max_iterations=10)
        # perturb a site_type's norm, then renormalize
        rep = ((0, 0), itn.site_types[0])
        itn.fragment[rep].modify(apply=lambda d: d * 5.0)
        itn._sync_site(rep)
        itn.normalize_simple(gauges)
        # gauges are unit norm
        for g in gauges.values():
            assert float(np.linalg.norm(g)) == pytest.approx(1.0)
        # each site_type's (0, 0) tensor has unit local norm
        for st in itn.site_types:
            t = itn.fragment[((0, 0), st)]
            tg = t.copy()
            for ix in t.inds:
                if ix in gauges:
                    tg.multiply_index_diagonal_(ix, gauges[ix])
            assert float(tg.norm()) == pytest.approx(1.0)
        # shared-tensor invariants preserved (all translates equal)
        for ts in itn.shared_tensors.values():
            for t in ts:
                assert t.data == pytest.approx(ts[0].data)

    def test_simple_update_reaches_heisenberg_ground_state(self):
        # end-to-end: imaginary-time simple update on 2D square Heisenberg.
        # QMC reference e/site ~ -0.6694; D=4 simple update lands close.
        geom = geom_square_2x2
        rng = np.random.default_rng(0)
        psi = PEPSInfinite2D.from_fill_fn(
            lambda s: rng.standard_normal(s), geom, bond_dim=2
        )
        ham = LocalHamInfinite2D(geom, qu.ham_heis(2))
        gauges = {}
        psi.gauge_all_simple_(gauges=gauges, max_iterations=100, tol=1e-10)
        ordering = ham.get_auto_ordering()
        for tau in (0.3, 0.1, 0.03):
            for _ in range(100):
                for bt in ordering:
                    psi.gate_simple_(
                        ham.get_gate_expm(bt, -tau),
                        bt,
                        gauges=gauges,
                        max_bond=4,
                    )
            psi.gauge_all_simple_(gauges=gauges, max_iterations=100, tol=1e-10)
        e = (
            psi.compute_local_expectation_cluster(ham, gauges=gauges)
            / ham.nsites
        )
        assert -0.68 < e < -0.60
