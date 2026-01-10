import math

import pytest

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
