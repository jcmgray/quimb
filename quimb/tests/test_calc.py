from pytest import fixture, mark, raises
import itertools
import numpy as np
from numpy.testing import assert_allclose
from .. import (qu, eye, rand_product_state, bell_state, up, eigvecs,
                rand_mix, rand_rho, rand_ket, sig, down, overlap,
                singlet_pairs, dop, singlet, kron)
from ..calc import (fidelity, quantum_discord, one_way_classical_information,
                    mutual_information, partial_transpose, entropy,
                    correlation, pauli_correlations, expm, sqrtm, purify,
                    concurrence, negativity, logneg, trace_distance,
                    pauli_decomp, bell_decomp, ent_cross_matrix, qid)


@fixture
def p1():
    return rand_rho(3)


@fixture
def p2():
    return rand_rho(3)


@fixture
def k1():
    return rand_ket(3)


@fixture
def k2():
    return rand_ket(3)


@fixture
def orthog_ks():
    p = rand_rho(3)
    v = eigvecs(p)
    return (v[:, 0], v[:, 1], v[:, 2])


# --------------------------------------------------------------------------- #
# TESTS                                                                       #
# --------------------------------------------------------------------------- #

class TestExpm:
    @mark.parametrize("herm", [True, False])
    def test_zeros_dense(self, herm):
        p = expm(np.zeros((2, 2), dtype=complex), herm=herm)
        assert_allclose(p, eye(2))

    @mark.parametrize("sparse", [True, False])
    @mark.parametrize("herm", [True, False])
    def test_eye(self, sparse, herm):
        p = expm(eye(2, sparse=sparse), herm=herm)
        assert_allclose((p.A if sparse else p) / np.e, eye(2))


class TestSqrtm:
    @mark.parametrize("sparse", [True, False])
    @mark.parametrize("herm", [True, False])
    def test_eye(self, herm, sparse):
        if sparse:
            with raises(NotImplementedError):
                p = sqrtm(eye(2, sparse=sparse), herm=herm)
        else:
            p = sqrtm(eye(2), herm=herm)
            assert_allclose(p, eye(2))


class TestFidelity:
    def test_both_pure(self, k1, k2):
        f = fidelity(k1, k1)
        assert_allclose(f, 1.0)
        f = fidelity(k1, k2)
        assert f > 0 and f < 1

    def test_both_mixed(self, p1, p2):
        f = fidelity(eye(3)/3, eye(3)/3)
        assert_allclose(f, 1.0)
        f = fidelity(p1, p1)
        assert_allclose(f, 1.0)
        f = fidelity(p1, p2)
        assert f > 0 and f < 1

    def test_orthog_pure(self, orthog_ks):
        k1, k2, k3 = orthog_ks
        for s1, s2, in ([k1, k2],
                        [k2, k3],
                        [k3, k1],
                        [k1 @ k1.H, k2],
                        [k1, k2 @ k2.H],
                        [k3 @ k3.H, k2],
                        [k3, k2 @ k2.H],
                        [k1 @ k1.H, k3],
                        [k1, k3 @ k3.H],
                        [k1 @ k1.H, k2 @ k2.H],
                        [k2 @ k2.H, k3 @ k3.H],
                        [k1 @ k1.H, k3 @ k3.H]):
            f = fidelity(s1, s2)
            assert_allclose(f, 0.0, atol=1e-6)


class TestPurify:
    @mark.parametrize("sparse", [True, False])
    def test_d2(self, sparse):
        rho = eye(2) / 2
        psi = purify(rho, sparse=sparse)
        assert overlap(psi, bell_state('phi+')) > 1 - 1e-14

    @mark.parametrize("sparse", [True, False])
    def test_pure(self, sparse):
        rho = up(qtype='dop')
        psi = purify(rho, sparse=sparse)
        assert abs(concurrence(psi)) < 1e-14


class TestEntropy:
    def test_entropy_pure(self):
        a = bell_state(1, qtype='dop')
        assert_allclose(0.0, entropy(a), atol=1e-12)

    def test_entropy_mixed(self):
        a = 0.5 * (bell_state(1, qtype='dop') +
                   bell_state(2, qtype='dop'))
        assert_allclose(1.0, entropy(a), atol=1e-12)

    @mark.parametrize("l, e", [([0, 1, 0, 0], 0),
                               ([0, 0.5, 0, 0.5], 1),
                               ([0.25, 0.25, 0.25, 0.25], 2)])
    def test_list(self, l, e):
        assert_allclose(entropy(l), e)

    @mark.parametrize("l, e", [([0, 1, 0, 0], 0),
                               ([0, 0.5, 0, 0.5], 1),
                               ([0.25, 0.25, 0.25, 0.25], 2)])
    def test_1darray(self, l, e):
        assert_allclose(entropy(np.asarray(l)), e)


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

    def test_mixed(self):
        # TODO ************************************************************** #
        pass

    def test_mixed_subb(self):
        # TODO ************************************************************** #
        pass


class TestPartialTranspose:
    def test_partial_transpose(self):
        a = bell_state(0, qtype='dop')
        b = partial_transpose(a)
        assert_allclose(b, [[0, 0, 0, -0.5],
                            [0, 0.5, 0, 0],
                            [0, 0, 0.5, 0],
                            [-0.5, 0, 0, 0]])


class TestNegativity:
    @mark.parametrize("bs", ['psi-', 'phi-', 'psi+', 'phi+'])
    @mark.parametrize("qtype", ['ket', 'dop'])
    def test_simple(self, qtype, bs):
        p = bell_state(bs, qtype=qtype)
        assert negativity(p) > 0.5 - 1e-14

    def test_subsystem(self):
        p = singlet_pairs(4)
        assert negativity(p, [2]*4, 0, 1) > 0.5 - 1e-14
        assert negativity(p, [2]*4, 1, 2) < 1e-14
        assert negativity(p, [2]*4, 2, 3) > 0.5 - 1e-14


class TestLogarithmicNegativity:
    @mark.parametrize("bs", ['psi-', 'phi-', 'psi+', 'phi+'])
    @mark.parametrize("qtype", ['ket', 'dop'])
    def test_bell_states(self, qtype, bs):
        p = bell_state(bs, qtype=qtype)
        assert logneg(p) > 1.0 - 1e-14

    def test_subsystem(self):
        p = singlet_pairs(4)
        assert logneg(p, [2]*4, 0, 1) > 1 - 1e-14
        assert logneg(p, [2]*4, 1, 2) < 1e-14
        assert logneg(p, [2]*4, 2, 3) > 1 - 1e-14


class TestConcurrence:
    @mark.parametrize("bs", ['psi-', 'phi-', 'psi+', 'phi+'])
    @mark.parametrize("qtype", ['ket', 'dop'])
    def test_bell_states(self, qtype, bs):
        p = bell_state(bs, qtype=qtype)
        assert concurrence(p) > 1.0 - 1e-14


class TestQuantumDiscord:
    def test_owci(self):
        a = qu([1, 0], 'op')
        b = qu([0, 1], 'op')
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
            assert_allclose(iab / 2, qd)

    def test_quantum_discord_mixed(self):
        for i in range(10):
            p = rand_mix(4)
            p = p @ p.H
            qd = quantum_discord(p)
            assert(0 <= qd and qd <= 1)


class TestTraceDistance:
    def test_types(self, k1, k2):
        td1 = trace_distance(k1, k2)
        td2 = trace_distance(dop(k1), k2)
        td3 = trace_distance(k1, dop(k2))
        td4 = trace_distance(dop(k1), dop(k2))
        assert_allclose([td1]*3, [td2, td3, td4])

    def test_same(self, p1):
        assert abs(trace_distance(p1, p1)) < 1e-14

    @mark.parametrize("uqtype", ['ket', 'dop'])
    @mark.parametrize("dqtype", ['ket', 'dop'])
    def test_distinguishable(self, uqtype, dqtype):
        assert trace_distance(up(qtype=uqtype), down(qtype=dqtype)) > 1 - 1e-10


class TestDecomp:
    @mark.parametrize("qtype", ['ket', 'dop'])
    def test_pauli_decomp_singlet(self, qtype):
        p = singlet(qtype=qtype)
        names_cffs = pauli_decomp(p, mode='cp')
        assert_allclose(names_cffs['II'], 0.25)
        assert_allclose(names_cffs['ZZ'], -0.25)
        assert_allclose(names_cffs['YY'], -0.25)
        assert_allclose(names_cffs['ZZ'], -0.25)
        for name in itertools.permutations('IXYZ', 2):
            assert_allclose(names_cffs["".join(name)], 0.0)

    def test_pauli_reconstruct(self):
        p1 = rand_rho(4)
        names_cffs = pauli_decomp(p1, mode='c')
        pr = sum(kron(*(sig(s) for s in name)) * names_cffs["".join(name)]
                 for name in itertools.product('IXYZ', repeat=2))
        assert_allclose(pr, p1)

    @mark.parametrize("state, out",
                      [(up() & down(), {0: 0.5, 1: 0.5, 2: 0, 3: 0}),
                       (down() & down(), {0: 0, 1: 0, 2: 0.5, 3: 0.5}),
                       (singlet() & singlet(), {'00': 1.0, '23': 0.0})])
    def test_bell_decomp(self, state, out):
        names_cffs = bell_decomp(state, mode='c')
        for key in out:
            assert_allclose(names_cffs[str(key)], out[key])


class TestCorrelation:
    @mark.parametrize("pre_c", [False, True])
    @mark.parametrize("p_sps", [True, False])
    @mark.parametrize("op_sps", [True, False])
    @mark.parametrize("dims", (None, [2, 2]))
    def test_types(self, dims, op_sps, p_sps, pre_c):
        p = rand_rho(4, sparse=p_sps)
        c = correlation(p, sig('x', sparse=op_sps), sig('z', sparse=op_sps),
                        0, 1, dims=dims, precomp_func=pre_c)
        c = c(p) if pre_c else c
        assert c >= -1.0
        assert c <= 1.0

    @mark.parametrize("pre_c", [False, True])
    @mark.parametrize("qtype", ["ket", "dop"])
    @mark.parametrize("dir", ['x', 'y', 'z'])
    def test_classically_no_correlated(self, dir, qtype, pre_c):
        p = up(qtype=qtype) & up(qtype=qtype)
        c = correlation(p, sig(dir), sig(dir), 0, 1, precomp_func=pre_c)
        c = c(p) if pre_c else c
        assert_allclose(c, 0.0)

    @mark.parametrize("pre_c", [False, True])
    @mark.parametrize("dir, ct", [('x', 0), ('y', 0), ('z', 1)])
    def test_classically_correlated(self, dir, ct, pre_c):
        p = 0.5 * ((up(qtype='dop') & up(qtype='dop')) +
                   (down(qtype='dop') & down(qtype='dop')))
        c = correlation(p, sig(dir), sig(dir), 0, 1, precomp_func=pre_c)
        c = c(p) if pre_c else c
        assert_allclose(c, ct)

    @mark.parametrize("pre_c", [False, True])
    @mark.parametrize("dir, ct", [('x', -1), ('y', -1), ('z', -1)])
    def test_entangled(self, dir, ct, pre_c):
        p = bell_state('psi-')
        c = correlation(p, sig(dir), sig(dir), 0, 1, precomp_func=pre_c)
        c = c(p) if pre_c else c
        assert_allclose(c, ct)

    def test_reuse_precomp(self):
        cfn = correlation(None, sig('z'), sig('z'), 0, 1, dims=[2, 2],
                          precomp_func=True)
        assert_allclose(cfn(bell_state('psi-')), -1.0)
        assert_allclose(cfn(bell_state('phi+')), 1.0)

    @mark.parametrize("pre_c", [False, True])
    def test_pauli_correlations_sum_abs(self, pre_c):
        p = bell_state('psi-')
        ct = pauli_correlations(p, sum_abs=True, precomp_func=pre_c)
        ct = ct(p) if pre_c else ct
        assert_allclose(ct, 3.0)

    @mark.parametrize("pre_c", [False, True])
    def test_pauli_correlations_no_sum_abs(self, pre_c):
        p = bell_state('psi-')
        ct = pauli_correlations(p, sum_abs=False, precomp_func=pre_c)
        assert_allclose(list(c(p) for c in ct) if pre_c else ct, (-1, -1, -1))


class TestEntCrossMatrix:
    def test_bell_state(self):
        p = bell_state('phi+')
        ecm = ent_cross_matrix(p, ent_fn=concurrence, calc_self_ent=True)
        assert_allclose(ecm, [[1, 1], [1, 1]])

    def test_bell_state_no_self_ent(self):
        p = bell_state('phi+')
        ecm = ent_cross_matrix(p, ent_fn=concurrence, calc_self_ent=False)
        assert_allclose(ecm, [[np.nan, 1], [1, np.nan]])


class TestQID:
    @mark.parametrize("bs", [0, 1, 2, 3])
    @mark.parametrize("pre_c", [False, True])
    def test_bell_state(self, bs, pre_c):
        p = bell_state(bs)
        qids = qid(p, dims=[2, 2], inds=[0, 1], precomp_func=pre_c)
        assert_allclose(qids(p) if pre_c else qids, [3, 3])

    @mark.parametrize("pre_c", [False, True])
    def test_random_product_state(self, pre_c):
        p = rand_product_state(3)
        qids = qid(p, dims=[2, 2, 2], inds=[0, 1, 2], precomp_func=pre_c)
        assert_allclose(qids(p) if pre_c else qids, [2, 2, 2])
