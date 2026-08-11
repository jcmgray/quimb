import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import quimb.operator as qop


def test_basic_int():
    hs = qop.HilbertSpace(10)
    print(hs)
    assert hs.nsites == 10
    assert hs.size == 1024
    assert len(hs.sites) == 10
    for i in range(10):
        assert hs.site_to_reg(i) == i
        assert hs.reg_to_site(i) == i

    fc = hs.rand_flatconfig()
    assert len(fc) == 10
    for xi, d in zip(fc, hs.sizes):
        assert 0 <= xi < d


def test_basic_sequence():
    hs = qop.HilbertSpace(["a", "b", "c"], dims=3)
    print(hs)
    assert hs.nsites == 3
    assert hs.size == 27
    assert len(hs.sites) == 3
    for i, s in enumerate(["a", "b", "c"]):
        assert hs.site_to_reg(s) == i
        assert hs.reg_to_site(i) == s

    fc = hs.rand_flatconfig()
    assert len(fc) == 3
    for xi, d in zip(fc, hs.sizes):
        assert 0 <= xi < d


def test_basic_mapping():
    hs = qop.HilbertSpace({"Z": 2, "X": 3, "Y": 4}, order=False)
    print(hs)
    assert hs.nsites == 3
    assert hs.size == 24
    assert len(hs.sites) == 3
    assert hs.site_to_reg("Z") == 0
    assert hs.site_size("Z") == 2
    assert hs.site_to_reg("X") == 1
    assert hs.site_size("X") == 3
    assert hs.site_to_reg("Y") == 2
    assert hs.site_size("Y") == 4
    assert hs.reg_to_site(0) == "Z"
    assert hs.reg_to_site(1) == "X"
    assert hs.reg_to_site(2) == "Y"
    assert hs.sites == ("Z", "X", "Y")
    assert tuple(map(int, hs.sizes)) == (2, 3, 4)
    assert tuple(map(int, hs.strides)) == (12, 4, 1)


def test_basic_mapping_sorted():
    hs = qop.HilbertSpace({"Z": 2, "X": 3, "Y": 4}, order=True)
    print(hs)
    assert hs.nsites == 3
    assert hs.size == 2 * 3 * 4
    assert len(hs.sites) == 3
    assert hs.site_to_reg("X") == 0
    assert hs.site_size("X") == 3
    assert hs.site_to_reg("Y") == 1
    assert hs.site_size("Y") == 4
    assert hs.site_to_reg("Z") == 2
    assert hs.site_size("Z") == 2
    assert hs.reg_to_site(0) == "X"
    assert hs.reg_to_site(1) == "Y"
    assert hs.reg_to_site(2) == "Z"
    assert hs.sites == ("X", "Y", "Z")
    assert tuple(map(int, hs.sizes)) == (3, 4, 2)
    assert tuple(map(int, hs.strides)) == (8, 2, 1)


def test_mixed_radix_sampling():
    hs = qop.HilbertSpace({"Z": 2, "X": 3, "Y": 4}, order=True)
    for _ in range(100):
        config = hs.rand_config()
        assert set(config.keys()) == {"X", "Y", "Z"}
        assert 0 <= config["X"] < 3
        assert 0 <= config["Y"] < 4
        assert 0 <= config["Z"] < 2
        rank = hs.config_to_rank(config)
        assert 0 <= rank < hs.size


class TestOrdering:
    def test_ordering_is_immutable(self):
        hs = qop.HilbertSpace({"Z": 2, "X": 3, "Y": 4})
        with pytest.raises(TypeError, match="immutable"):
            hs.set_ordering(True)

    def test_with_ordering(self):
        hs = qop.HilbertSpace({"Z": 2, "X": 3, "Y": 4})
        assert hs.sites == ("Z", "X", "Y")

        hs2 = hs.with_ordering(True)
        assert hs2.sites == ("X", "Y", "Z")
        assert tuple(map(int, hs2.sizes)) == (3, 4, 2)
        # original untouched
        assert hs.sites == ("Z", "X", "Y")
        assert tuple(map(int, hs.sizes)) == (2, 3, 4)

    def test_with_ordering_keeps_symmetry_and_sector(self):
        hs = qop.HilbertSpace(6, symmetry="U1", sector=3)
        hs2 = hs.with_ordering(lambda s: -s)
        assert hs2.sites == (5, 4, 3, 2, 1, 0)
        assert hs2.symmetry == "U1"
        assert hs2.sector == 3
        assert hs2.size == hs.size

    def test_order_preset_blocked(self):
        sites = [(s, i) for i in range(3) for s in "↑↓"]
        hs = qop.HilbertSpace(sites, order="blocked")
        assert hs.sites == (
            ("↑", 0),
            ("↑", 1),
            ("↑", 2),
            ("↓", 0),
            ("↓", 1),
            ("↓", 2),
        )

    def test_order_preset_interleaved(self):
        sites = [(s, i) for s in "↑↓" for i in range(3)]
        hs = qop.HilbertSpace(sites, order="interleaved")
        assert hs.sites == (
            ("↑", 0),
            ("↓", 0),
            ("↑", 1),
            ("↓", 1),
            ("↑", 2),
            ("↓", 2),
        )

    def test_order_preset_unknown(self):
        sites = [(s, i) for s in "↑↓" for i in range(3)]
        with pytest.raises(ValueError, match="blocked"):
            qop.HilbertSpace(sites, order="spin-major")

    def test_order_preset_needs_species_labels(self):
        with pytest.raises(ValueError, match="species"):
            qop.HilbertSpace(6, order="blocked")

    def test_with_ordering_configs_valid(self):
        # regression: reordering must rebuild the rank <-> config mappings,
        # else configs decode with the pre-reorder strides
        hs = qop.HilbertSpace({"a": 2, "b": 3, "c": 4}).with_ordering(
            ["c", "b", "a"]
        )
        assert hs.sites == ("c", "b", "a")
        for rank in range(hs.size):
            config = hs.rank_to_config(rank)
            for site, xi in config.items():
                assert 0 <= xi < hs.site_size(site)
            assert hs.config_to_rank(config) == rank


class TestSpecies:
    @pytest.mark.parametrize(
        "sector", [{"↑": 1, "↓": 2}, (1, 2), ((3, 1), (3, 2))]
    )
    def test_sector_forms_agree(self, sector):
        sites = [(s, i) for s in "↑↓" for i in range(3)]
        hs = qop.HilbertSpace(
            sites, order="blocked", species=lambda s: s[0], sector=sector
        )
        assert hs.symmetry == "U1U1"
        assert hs.sector == ((3, 1), (3, 2))
        assert hs.size == math.comb(3, 1) * math.comb(3, 2)

    def test_species_as_dict(self):
        sites = [(s, i) for s in "↑↓" for i in range(3)]
        hs = qop.HilbertSpace(
            sites,
            order="blocked",
            species={site: site[0] for site in sites},
            sector=(1, 2),
        )
        assert hs.sector == ((3, 1), (3, 2))

    def test_explicit_sector_needs_no_species(self):
        hs = qop.HilbertSpace(6, symmetry="U1U1", sector=((3, 1), (3, 2)))
        assert hs.sector == ((3, 1), (3, 2))

    def test_terse_sector_needs_species(self):
        with pytest.raises(ValueError, match="need `species`"):
            qop.HilbertSpace(6, symmetry="U1U1", sector=(1, 2))

    @pytest.mark.parametrize("order", ["blocked", "interleaved"])
    def test_configs_respect_species_filling(self, order):
        sites = [(s, i) for s in "↑↓" for i in range(3)]
        hs = qop.HilbertSpace(
            sites, order=order, species=lambda s: s[0], sector=(1, 2)
        )
        assert hs.size == math.comb(3, 1) * math.comb(3, 2)
        seen = set()
        for rank in range(hs.size):
            config = hs.rank_to_config(rank)
            assert sum(v for (s, _), v in config.items() if s == "↑") == 1
            assert sum(v for (s, _), v in config.items() if s == "↓") == 2
            # configs are distinct and round-trip
            assert hs.config_to_rank(config) == rank
            seen.add(tuple(sorted(config.items())))
        assert len(seen) == hs.size

    def test_interleaved_flatconfigs_in_register_order(self):
        sites = [(s, i) for s in "↑↓" for i in range(3)]
        hs = qop.HilbertSpace(
            sites, order="interleaved", species=lambda s: s[0], sector=(1, 2)
        )
        # flatconfigs follow the register ordering, not the enumeration one
        for rank in range(hs.size):
            fc = hs.rank_to_flatconfig(rank)
            config = hs.rank_to_config(rank)
            for reg, site in enumerate(hs.sites):
                assert fc[reg] == config[site]
            assert hs.flatconfig_to_rank(fc) == rank

    def test_with_ordering_carries_species(self):
        sites = [(s, i) for s in "↑↓" for i in range(3)]
        hs = qop.HilbertSpace(sites, species=lambda s: s[0], sector=(1, 2))
        hs2 = hs.with_ordering("blocked")
        assert hs2.sector == ((3, 1), (3, 2))
        assert hs2.size == hs.size


@pytest.mark.parametrize("sector", [0, 1])
def test_basic_z2_symmetry_sampling(sector):
    hs = qop.HilbertSpace(6, sector=sector, symmetry="Z2")
    print(hs)
    assert hs.size == 2 ** (hs.nsites - 1)
    for _ in range(100):
        config = hs.rand_config()
        assert sum(config.values()) % 2 == sector
        rank = hs.config_to_rank(config)
        assert 0 <= rank < hs.size


@pytest.mark.parametrize("sector", [0, 1, 2, 3, 4, 5, 6])
def test_basic_u1_symmetry_sampling(sector):
    hs = qop.HilbertSpace(6, symmetry="U1", sector=sector)
    print(hs)
    assert hs.size == math.comb(6, sector)
    for _ in range(100):
        config = hs.rand_config()
        assert sum(config.values()) == sector
        rank = hs.config_to_rank(config)
        assert 0 <= rank < hs.size


@pytest.mark.parametrize("sectora", [0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("sectorb", [0, 1, 2, 3, 4])
def test_basic_u1u1_symmetry_sampling(sectora, sectorb):
    hs = qop.HilbertSpace(
        10, symmetry="U1U1", sector=((6, sectora), (4, sectorb))
    )
    print(hs)
    assert hs.size == math.comb(6, sectora) * math.comb(4, sectorb)
    for _ in range(10):
        config = hs.rand_config()
        suma = sum(config[i] for i in range(6))
        sumb = sum(config[i] for i in range(6, 10))
        assert suma == sectora
        assert sumb == sectorb
        rank = hs.config_to_rank(config)
        assert 0 <= rank < hs.size

    def test_no_permutation_without_interleaving(self):
        # the common paths must stay permutation free
        sites = [(s, i) for s in "↑↓" for i in range(3)]
        for hs in [
            qop.HilbertSpace(6),
            qop.HilbertSpace(6, symmetry="U1", sector=3),
            qop.HilbertSpace(sites),
            qop.HilbertSpace(sites, order="blocked", species=lambda s: s[0]),
        ]:
            assert not hs.needs_blocking
            assert hs.site_to_blocked_reg(hs.sites[2]) == 2

    def test_only_u1u1_permutes_by_default(self):
        # the permutation exists whenever the species are interleaved, but is
        # only applied to the default mappings when U1U1 enumerates them
        sites = [(s, i) for s in "↑↓" for i in range(3)]
        for sector, expected in [
            (None, False),
            (3, False),
            ((1, 2), True),
        ]:
            hs = qop.HilbertSpace(
                sites,
                order="interleaved",
                species=lambda s: s[0],
                sector=sector,
            )
            assert hs.needs_blocking
            assert hs._blocked_default is expected

    def test_blocked_perm_round_trip(self):
        sites = [(s, i) for s in "↑↓" for i in range(3)]
        hs = qop.HilbertSpace(
            sites, order="interleaved", species=lambda s: s[0], sector=(1, 2)
        )
        assert hs.needs_blocking and hs._blocked_default
        # each species is contiguous in the blocked ordering
        regs = [hs.site_to_blocked_reg(s) for s in hs.sites if s[0] == "↑"]
        assert sorted(regs) == [0, 1, 2]

        fc = hs.rand_flatconfig(seed=42)
        assert_array_equal(
            hs._flatconfig_from_blocked(hs._flatconfig_to_blocked(fc)), fc
        )
        # and the permutation is applied along the last axis, for 2D too
        fcs = np.stack([hs.rand_flatconfig(seed=s) for s in range(5)])
        assert_array_equal(
            hs._flatconfig_from_blocked(hs._flatconfig_to_blocked(fcs)), fcs
        )
