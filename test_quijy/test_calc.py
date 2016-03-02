from numpy.testing import assert_allclose
from quijy import (qjf, rand_product_state, bell_state, rand_ket, rand_rho)
from quijy.calc import *


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

    def  test_quantum_discord_sep(self):
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
