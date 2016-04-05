import numpy as np
from numpy.testing import assert_allclose
from quijy.core import tr, eye, chop, eyepad, inner, ptr
from quijy.solve import eigsys, groundstate
from quijy.gen import (basis_vec, up, down, plus, minus, yplus, yminus,
                       thermal_state, ham_j1j2, rand_herm, graph_state_1d,
                       sig, levi_civita, bloch_state, bell_state, singlet)


class TestBasisVec:
    def test_basis_vec(self):
        x = basis_vec(1, 2)
        assert_allclose(x, np.matrix([[0.], [1.]]))
        x = basis_vec(1, 2, qtype='b')
        assert_allclose(x, np.matrix([[0., 1.]]))

    def test_basis_vec_sparse(self):
        x = basis_vec(4, 100, sparse=True)
        assert x[4, 0] == 1.
        assert x.nnz == 1
        assert x.dtype == complex


class TestBasicStates:
    def test_up(self):
        p = up(qtype='dop')
        assert_allclose(tr(p @ sig('z')), 1.0)

    def test_down(self):
        p = down(qtype='dop')
        assert_allclose(tr(p @ sig('z')), -1.0)

    def test_plus(self):
        p = plus(qtype='dop')
        assert_allclose(tr(p @ sig('x')), 1.0)

    def test_minus(self):
        p = minus(qtype='dop')
        assert_allclose(tr(p @ sig('x')), -1.0)

    def test_yplus(self):
        p = yplus(qtype='dop')
        assert_allclose(tr(p @ sig('y')), 1.0)

    def test_yminus(self):
        p = yminus(qtype='dop')
        assert_allclose(tr(p @ sig('y')), -1.0)


class TestBlochState:
    def test_pure(self):
        for vec, op, val in zip(((1, 0, 0), (0, 1, 0), (0, 0, 1),
                                 (-1, 0, 0), (0, -1, 0), (0, 0, -1)),
                                ("x", "y", "z", "x", "y", "z"),
                                (1, 1, 1, -1, -1, -1)):
            x = tr(bloch_state(*vec) @ sig(op))
            assert_allclose(x, val)

    def test_mixed(self):
        for vec, op, val in zip(((.5, 0, 0), (0, .5, 0), (0, 0, .5),
                                 (-.5, 0, 0), (0, -.5, 0), (0, 0, -.5)),
                                ("x", "y", "z", "x", "y", "z"),
                                (.5, .5, .5, -.5, -.5, -.5)):
            x = tr(bloch_state(*vec) @ sig(op))
            assert_allclose(x, val)

    def test_purify(self):
        for vec, op, val in zip(((.5, 0, 0), (0, .5, 0), (0, 0, .5),
                                 (-.5, 0, 0), (0, -.5, 0), (0, 0, -.5)),
                                ("x", "y", "z", "x", "y", "z"),
                                (1, 1, 1, -1, -1, -1)):
            x = tr(bloch_state(*vec, purified=True) @ sig(op))
            assert_allclose(x, val)


class TestBellStates:
    def test_bell_states(self):
        for s, dic in zip(("psi-", "psi+", "phi+", "phi-"),
                          ({"qtype": 'dop'}, {}, {"sparse": True}, {})):
            p = bell_state(s, **dic)
            assert_allclose(inner(p, p), 1.0)
            pa = ptr(p, [2, 2], 0)
            assert_allclose(inner(pa, pa), 0.5)

    def test_bell_state_singlet(self):
        p = singlet(qtype="dop", sparse=True)
        assert_allclose(inner(p, p), 1.0)
        pa = ptr(p, [2, 2], 0)
        assert_allclose(inner(pa, pa), 0.5)


class TestThermalState:
    def test_thermal_state_normalization(self):
        full = rand_herm(2**4)
        for beta in (0, 0.5, 1, 10):
            rhoth = thermal_state(full, beta)
            assert_allclose(tr(rhoth), 1)

    def test_thermal_state_tuple(self):
        full = rand_herm(2**4)
        l, v = eigsys(full)
        for beta in (0, 0.5, 1, 10):
            rhoth1 = thermal_state(full, beta)
            rhoth2 = thermal_state((l, v), beta)
            assert_allclose(rhoth1, rhoth2)

    def test_thermal_state_hot(self):
        full = rand_herm(2**4)
        rhoth = chop(thermal_state(full, 0.0))
        assert_allclose(rhoth, eye(2**4) / 2**4)

    def test_thermal_state_cold(self):
        full = ham_j1j2(4, j2=0.1253)
        rhoth = thermal_state(full, 100)
        gs = groundstate(full)
        assert_allclose(tr(gs.H @ rhoth @ gs), 1.0, rtol=1e-4)

    def test_thermal_state_precomp(self):
        full = rand_herm(2**4)
        beta = 0.624
        rhoth1 = thermal_state(full, beta)
        func = thermal_state(full, None, precomp_func=True)
        rhoth2 = func(beta)
        assert_allclose(rhoth1, rhoth2)


class TestGraphState:
    def test_graph_state_1d(self):
        n = 5
        p = graph_state_1d(n, cyclic=True)
        for j in range(n):
            k = eyepad([sig('x'), sig('z'), sig('z')], [2] * n,
                       (j, (j - 1) % n, (j + 1) % n))
            o = p.H @ k @ p
            np.testing.assert_allclose(o, 1)


class TestLeviCivita:
    def test_levi_civita_pos(self):
        perm = [0, 1, 2, 3]
        assert levi_civita(perm) == 1
        perm = [2, 3, 0, 1]
        assert levi_civita(perm) == 1

    def test_levi_civita_neg(self):
        perm = [0, 2, 1, 3]
        assert levi_civita(perm) == -1
        perm = [2, 3, 1, 0]
        assert levi_civita(perm) == -1

    def test_levi_civita_nzero(self):
        perm = [2, 3, 1, 1]
        assert levi_civita(perm) == 0
        perm = [0, 0, 1, 1]
        assert levi_civita(perm) == 0
