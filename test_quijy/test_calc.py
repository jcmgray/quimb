from numpy.testing import assert_allclose
from quijy import (qjf, rand_product_state, bell_state, rand_ket, rand_rho,
                   up)
from quijy.calc import (quantum_discord, one_way_classical_information,
                        mutual_information, partial_transpose,
                        entropy)


class TestPartialTranspose:
    def test_partial_transpose(self):
        a = bell_state(0, qtype='dop')
        b = partial_transpose(a)
        assert_allclose(b, [[0, 0, 0, -0.5],
                            [0, 0.5, 0, 0],
                            [0, 0, 0.5, 0],
                            [-0.5, 0, 0, 0]])


class TestEntropy:
    def test_entropy_pure(self):
        a = bell_state(1, qtype='dop')
        assert_allclose(0.0, entropy(a), atol=1e-12)

    def test_entropy_mixed(self):
        a = 0.5 * (bell_state(1, qtype='dop') +
                   bell_state(2, qtype='dop'))
        assert_allclose(1.0, entropy(a), atol=1e-12)


class TestMutualInformation:
    def test_mutual_information_pure(self):
        a = bell_state(0)
        assert_allclose(mutual_information(a), 2.)
        a = rand_product_state(2)
        assert_allclose(mutual_information(a), 0., atol=1e-12)

    def test_mutual_information_pure_sub(self):
        a = up() & bell_state(1)
        ixy = mutual_information(a, [2, 2, 2],  0, 1)
        assert_allclose(0.0, ixy, atol=1e-12)
        ixy = mutual_information(a, [2, 2, 2],  0, 2)
        assert_allclose(0.0, ixy, atol=1e-12)
        ixy = mutual_information(a, [2, 2, 2],  2, 1)
        assert_allclose(2.0, ixy, atol=1e-12)

    # test mutual information mixed

    # test mutua information mixed sub


class TestQuantumDiscord:
    def test_owci(self):
        a = qjf([1, 0], 'op')
        b = qjf([0, 1], 'op')
        for i in (0, 1, 2, 3):
            p = rand_product_state(2)
            ci = one_way_classical_information(p @ p.H, [a, b])
            assert_allclose(ci, 0., atol=1e-12)
        for i in (0, 1, 2, 3):
            p = bell_state(i)
            ci = one_way_classical_information(p @ p.H, [a, b])
            assert_allclose(ci, 1., atol=1e-12)

    def test_quantum_discord_sep(self):
        for i in range(10):
            p = rand_product_state(2)
            p = p @ p.H
            qd = quantum_discord(p)
            assert_allclose(0.0, qd, atol=1e-12)

    def test_quantum_discord_pure(self):
        for i in range(10):
            p = rand_ket(4)
            p = p @ p.H
            iab = mutual_information(p)
            qd = quantum_discord(p)
            assert_allclose(iab, qd)

    def test_quantum_discord_mixed(self):
        for i in range(10):
            p = rand_rho(4)
            p = p @ p.H
            qd = quantum_discord(p)
            assert(0 <= qd and qd <= 1)
