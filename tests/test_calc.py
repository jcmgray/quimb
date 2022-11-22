import pytest
import itertools
import math
import numpy as np
from numpy.testing import assert_allclose
import quimb as qu


@pytest.fixture
def p1():
    return qu.rand_rho(3)


@pytest.fixture
def p2():
    return qu.rand_rho(3)


@pytest.fixture
def k1():
    return qu.rand_ket(3)


@pytest.fixture
def k2():
    return qu.rand_ket(3)


@pytest.fixture
def orthog_ks():
    p = qu.rand_rho(3)
    v = qu.eigvecsh(p)
    return (v[:, [0]], v[:, [1]], v[:, [2]])


# --------------------------------------------------------------------------- #
# TESTS                                                                       #
# --------------------------------------------------------------------------- #

class TestFidelity:
    def test_both_pure(self, k1, k2):
        f = qu.fidelity(k1, k1)
        assert_allclose(f, 1.0)
        f = qu.fidelity(k1, k2)
        assert f > 0 and f < 1

    def test_both_mixed(self, p1, p2):
        f = qu.fidelity(qu.eye(3) / 3, qu.eye(3) / 3)
        assert_allclose(f, 1.0)
        f = qu.fidelity(p1, p1)
        assert_allclose(f, 1.0)
        f = qu.fidelity(p1, p2)
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
            f = qu.fidelity(s1, s2)
            assert_allclose(f, 0.0, atol=1e-6)


class TestPurify:
    def test_d2(self):
        rho = qu.eye(2) / 2
        psi = qu.purify(rho)
        assert qu.expec(psi, qu.bell_state('phi+')) > 1 - 1e-14

    def test_pure(self):
        rho = qu.up(qtype='dop')
        psi = qu.purify(rho)
        assert abs(qu.concurrence(psi)) < 1e-14


class TestDephase:
    @pytest.mark.parametrize("rand_rank", [None, 0.3, 2])
    def test_basic(self, rand_rank):
        rho = qu.rand_rho(9)
        ln = qu.logneg(rho, [3, 3])
        for p in (0.2, 0.5, 0.8, 1.0):
            rho_d = qu.dephase(rho, p, rand_rank=rand_rank)
            assert qu.logneg(rho_d, [3, 3]) <= ln
            assert rho_d.tr() == pytest.approx(1.0)


class TestKrausOp:

    @pytest.mark.parametrize("stack", [False, True])
    def test_depolarize(self, stack):
        rho = qu.rand_rho(2)
        I, X, Y, Z = (qu.pauli(s) for s in 'IXYZ')
        es = [qu.expec(rho, A) for A in (X, Y, Z)]
        p = 0.1
        Ek = [(1 - p)**0.5 * I,
              (p / 3)**0.5 * X,
              (p / 3)**0.5 * Y,
              (p / 3)**0.5 * Z]
        if stack:
            Ek = np.stack(Ek, axis=0)
        sigma = qu.kraus_op(rho, Ek, check=True)
        es2 = [qu.expec(sigma, A) for A in (X, Y, Z)]
        assert qu.tr(sigma) == pytest.approx(1.0)
        assert all(abs(e2) < abs(e) for e, e2 in zip(es, es2))
        sig_exp = sum(E @ rho @ qu.dag(E) for E in Ek)
        assert_allclose(sig_exp, sigma)

    def test_subsystem(self):
        rho = qu.rand_rho(6)
        dims = [3, 2]
        I, X, Y, Z = (qu.pauli(s) for s in 'IXYZ')
        mi_i = qu.mutual_information(rho, dims)
        p = 0.1
        Ek = [(1 - p)**0.5 * I,
              (p / 3)**0.5 * X,
              (p / 3)**0.5 * Y,
              (p / 3)**0.5 * Z]

        with pytest.raises(ValueError):
            qu.kraus_op(rho, qu.randn((3, 2, 2)), check=True,
                        dims=dims, where=1)

        sigma = qu.kraus_op(rho, Ek, check=True, dims=dims, where=1)
        mi_f = qu.mutual_information(sigma, dims)
        assert mi_f < mi_i
        assert qu.tr(sigma) == pytest.approx(1.0)
        sig_exp = sum((qu.eye(3) & E) @ rho @ qu.dag(qu.eye(3) & E)
                      for E in Ek)
        assert_allclose(sig_exp, sigma)

    def test_multisubsystem(self):
        qu.seed_rand(42)
        dims = [2, 2, 2]
        IIX = qu.ikron(qu.rand_matrix(2), dims, 2)
        dcmp = qu.pauli_decomp(IIX, mode='c')
        for p, x in dcmp.items():
            if abs(x) < 1e-12:
                assert (p[0] != 'I') or (p[1] != 'I')
            else:
                assert p[0] == p[1] == 'I'
        K = qu.rand_iso(3 * 4, 4).reshape(3, 4, 4)
        KIIXK = qu.kraus_op(IIX, K, dims=dims, where=[0, 2])
        dcmp = qu.pauli_decomp(KIIXK, mode='c')
        for p, x in dcmp.items():
             if abs(x) > 1e-12:
                assert (p == 'III') or p[0] != 'I'


class TestProjector:

    def test_simple(self):
        Z = qu.pauli('Z')
        P = qu.projector(Z & Z)
        uu = qu.dop(qu.up()) & qu.dop(qu.up())
        dd = qu.dop(qu.down()) & qu.dop(qu.down())
        assert_allclose(P, uu + dd)
        assert qu.expec(P, qu.bell_state('phi+')) == pytest.approx(1.0)
        assert qu.expec(P, qu.bell_state('psi+')) == pytest.approx(0.0)


class TestMeasure:

    def test_pure(self):
        psi = qu.bell_state('psi-')
        IZ = qu.pauli('I') & qu.pauli('Z')
        ZI = qu.pauli('Z') & qu.pauli('I')
        res, psi_after = qu.measure(psi, IZ)
        # normalized
        assert qu.expectation(psi_after, psi_after) == pytest.approx(1.0)
        # anticorrelated
        assert qu.expectation(psi_after, IZ) == pytest.approx(res)
        assert qu.expectation(psi_after, ZI) == pytest.approx(-res)
        assert isinstance(psi_after, qu.qarray)

    def test_bigger(self):
        psi = qu.rand_ket(2**5)
        assert np.sum(abs(psi) < 1e-12) == 0
        A = qu.kronpow(qu.pauli('Z'), 5)
        res, psi_after = qu.measure(psi, A, eigenvalue=-1.0)
        # should have projected to half subspace
        assert np.sum(abs(psi_after) < 1e-12) == 2**4
        assert res == -1.0

    def test_mixed(self):
        rho = qu.dop(qu.bell_state('psi-'))
        IZ = qu.pauli('I') & qu.pauli('Z')
        ZI = qu.pauli('Z') & qu.pauli('I')
        res, rho_after = qu.measure(rho, IZ)
        # normalized
        assert qu.tr(rho_after) == pytest.approx(1.0)
        # anticorrelated
        assert qu.expectation(rho_after, IZ) == pytest.approx(res)
        assert qu.expectation(rho_after, ZI) == pytest.approx(-res)
        assert isinstance(rho_after, qu.qarray)


class TestSimulateCounts:

    @pytest.mark.parametrize('qtype', ['ket', 'dop'])
    def test_ghz(self, qtype):
        psi = qu.ghz_state(3, qtype=qtype)
        results = qu.simulate_counts(psi, 1024)
        assert len(results) == 2
        assert '000' in results
        assert '111' in results


class TestCPrint:

    def test_basic(self):
        psi = qu.ghz_state(2)
        qu.cprint(psi)


class TestEntropy:
    def test_entropy_pure(self):
        a = qu.bell_state(1, qtype='dop')
        assert_allclose(0.0, qu.entropy(a), atol=1e-12)

    def test_entropy_mixed(self):
        a = 0.5 * (qu.bell_state(1, qtype='dop') +
                   qu.bell_state(2, qtype='dop'))
        assert_allclose(1.0, qu.entropy(a), atol=1e-12)

    @pytest.mark.parametrize("evals, e", [([0, 1, 0, 0], 0),
                                          ([0, 0.5, 0, 0.5], 1),
                                          ([0.25, 0.25, 0.25, 0.25], 2)])
    def test_list(self, evals, e):
        assert_allclose(qu.entropy(evals), e)

    @pytest.mark.parametrize("evals, e", [([0, 1, 0, 0], 0),
                                          ([0, 0.5, 0, 0.5], 1),
                                          ([0.25, 0.25, 0.25, 0.25], 2)])
    def test_1darray(self, evals, e):
        assert_allclose(qu.entropy(np.asarray(evals)), e)

    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_rank(self, m):
        k = qu.rand_ket(2**4)
        pab = qu.ptr(k, [2, 2, 2, 2], range(m))
        ef = qu.entropy(pab)
        er = qu.entropy(pab, rank=2**m)
        assert_allclose(ef, er)

    def test_entropy_subsystem(self):
        p = qu.rand_ket(2**9)
        # exact
        e1 = qu.entropy_subsys(p, (2**5, 2**4), 0, approx_thresh=1e30)
        # approx
        e2 = qu.entropy_subsys(p, (2**5, 2**4), 0, approx_thresh=1)
        assert e1 != e2
        assert_allclose(e1, e2, rtol=0.2)

        assert qu.entropy_subsys(p, (2**5, 2**4), [0, 1],
                                 approx_thresh=1) == 0.0


class TestMutualInformation:
    def test_mutual_information_pure(self):
        a = qu.bell_state(0)
        assert_allclose(qu.mutual_information(a), 2.)
        a = qu.rand_product_state(2)
        assert_allclose(qu.mutual_information(a), 0., atol=1e-12)

    def test_mutual_information_pure_sub(self):
        a = qu.up() & qu.bell_state(1)
        ixy = qu.mutual_information(a, [2, 2, 2], 0, 1)
        assert_allclose(0.0, ixy, atol=1e-12)
        ixy = qu.mutual_information(a, [2, 2, 2], 0, 2)
        assert_allclose(0.0, ixy, atol=1e-12)
        ixy = qu.mutual_information(a, [2, 2, 2], 2, 1)
        assert_allclose(2.0, ixy, atol=1e-12)

    @pytest.mark.parametrize('inds', [(0, 1), (1, 2), (0, 2)])
    def test_mixed_sub(self, inds):
        a = qu.rand_rho(2**3)
        rho_ab = qu.ptr(a, [2, 2, 2], inds)
        ixy = qu.mutual_information(rho_ab, (2, 2))
        assert (0 <= ixy <= 2.0)

    def test_mutinf_interleave(self):
        p = qu.dop(qu.singlet() & qu.singlet())
        ixy = qu.mutual_information(p, [2] * 4, sysa=(0, 2))
        assert_allclose(ixy, 4)

    def test_mutinf_interleave_pure(self):
        p = qu.singlet() & qu.singlet()
        ixy = qu.mutual_information(p, [2] * 4, sysa=(0, 2))
        assert_allclose(ixy, 4)

    def test_mutinf_subsys(self):
        p = qu.rand_ket(2**9)
        dims = (2**3, 2**2, 2**4)
        # exact
        rho_ab = qu.ptr(p, dims, [0, 2])
        mi0 = qu.mutual_information(rho_ab, [8, 16])
        mi1 = qu.mutinf_subsys(p, dims, sysa=0, sysb=2, approx_thresh=1e30)
        assert_allclose(mi1, mi0)
        # approx
        mi2 = qu.mutinf_subsys(p, dims, sysa=0, sysb=2, approx_thresh=1)
        assert_allclose(mi1, mi2, rtol=0.1)

    def test_mutinf_subsys_pure(self):
        p = qu.rand_ket(2**7)
        dims = (2**3, 2**4)
        # exact
        mi0 = qu.mutual_information(p, dims, sysa=0)
        mi1 = qu.mutinf_subsys(p, dims, sysa=0, sysb=1, approx_thresh=1e30)
        assert_allclose(mi1, mi0)
        # approx
        mi2 = qu.mutinf_subsys(p, dims, sysa=0, sysb=1,
                               approx_thresh=1, tol=5e-3)
        assert_allclose(mi1, mi2, rtol=0.1)


class TestSchmidtGap:
    def test_bell_state(self):
        p = qu.bell_state('psi-')
        assert_allclose(qu.schmidt_gap(p, [2, 2], 0), 0.0)
        p = qu.up() & qu.down()
        assert_allclose(qu.schmidt_gap(p, [2, 2], 0), 1.0)
        p = qu.rand_ket(2**3)
        assert 0 < qu.schmidt_gap(p, [2] * 3, sysa=[0, 1]) < 1.0


class TestPartialTranspose:
    def test_partial_transpose(self):
        a = qu.bell_state(0, qtype='dop')
        b = qu.partial_transpose(a)
        assert isinstance(b, qu.qarray)
        assert_allclose(b, np.array([[0, 0, 0, -0.5],
                                     [0, 0.5, 0, 0],
                                     [0, 0, 0.5, 0],
                                     [-0.5, 0, 0, 0]]))

    def test_tr_sqrt_rank(self):
        psi = qu.rand_ket(2**5)
        rhoa = psi.ptr([2] * 5, range(4))
        assert_allclose(qu.tr_sqrt(rhoa), qu.tr_sqrt(rhoa, rank=2))


class TestNegativity:
    @pytest.mark.parametrize("bs", ['psi-', 'phi-', 'psi+', 'phi+'])
    @pytest.mark.parametrize("qtype", ['ket', 'dop'])
    def test_simple(self, qtype, bs):
        p = qu.bell_state(bs, qtype=qtype)
        assert qu.negativity(p) > 0.5 - 1e-14

    def test_subsystem(self):
        p = qu.singlet_pairs(4)
        rhoab = p.ptr([2, 2, 2, 2], [0, 1])
        assert qu.negativity(rhoab, [2] * 2) > 0.5 - 1e-14
        rhoab = p.ptr([2, 2, 2, 2], [1, 2])
        assert qu.negativity(rhoab, [2] * 2) < 1e-14
        rhoab = p.ptr([2, 2, 2, 2], [2, 3])
        assert qu.negativity(rhoab, [2] * 2) > 0.5 - 1e-14


class TestLogarithmicNegativity:
    @pytest.mark.parametrize("bs", ['psi-', 'phi-', 'psi+', 'phi+'])
    @pytest.mark.parametrize("qtype", ['ket', 'dop'])
    def test_bell_states(self, qtype, bs):
        p = qu.bell_state(bs, qtype=qtype)
        assert qu.logneg(p) > 1.0 - 1e-14

    def test_subsystem(self):
        p = qu.singlet_pairs(4)
        rhoab = p.ptr([2, 2, 2, 2], [0, 1])
        assert qu.logneg(rhoab, [2] * 2) > 1 - 1e-14
        rhoab = p.ptr([2, 2, 2, 2], [1, 2])
        assert qu.logneg(rhoab, [2] * 2) < 1e-14
        rhoab = p.ptr([2, 2, 2, 2], [2, 3])
        assert qu.logneg(rhoab, [2] * 2) > 1 - 1e-14

    def test_interleaving(self):
        p = qu.permute(qu.singlet() & qu.singlet(),
                       [2, 2, 2, 2], [0, 2, 1, 3])
        assert qu.logneg(p, [2] * 4, sysa=[0, 3]) > 2 - 1e-13

    def test_logneg_subsys(self):
        p = qu.rand_ket(2**(2 + 3 + 1 + 2))
        dims = (2**2, 2**3, 2**1, 2**2)
        sysa = [0, 3]
        sysb = 1
        # exact 1
        ln0 = qu.logneg(qu.ptr(p, dims, [0, 1, 3]), [4, 8, 4], [0, 2])
        # exact 2
        ln1 = qu.logneg_subsys(p, dims, sysa, sysb, approx_thresh=1e30)
        assert_allclose(ln0, ln1)
        # approx
        ln2 = qu.logneg_subsys(p, dims, sysa, sysb, approx_thresh=1)
        assert ln1 != ln2
        assert_allclose(ln1, ln2, rtol=5e-2)

    def test_logneg_subsys_pure(self):
        p = qu.rand_ket(2**(3 + 4))
        dims = (2**3, 2**4)
        sysa = 0
        sysb = 1
        # exact 1
        ln0 = qu.logneg(p, dims, 0)
        # exact 2
        ln1 = qu.logneg_subsys(p, dims, sysa, sysb, approx_thresh=1e30)
        assert_allclose(ln0, ln1)
        # approx
        ln2 = qu.logneg_subsys(p, dims, sysa, sysb, approx_thresh=1, tol=5e-3)
        assert ln1 != ln2
        assert_allclose(ln1, ln2, rtol=1e-1)

    def test_logneg_subsys_pure_should_swap_subsys(self):
        p = qu.rand_ket(2**(5 + 2))
        dims = (2**5, 2**2)
        sysa = 0
        sysb = 1
        # exact 1
        ln0 = qu.logneg(p, dims, 0)
        # exact 2
        ln1 = qu.logneg_subsys(p, dims, sysa, sysb, approx_thresh=1e30)
        assert_allclose(ln0, ln1)
        # approx
        ln2 = qu.logneg_subsys(p, dims, sysa, sysb, approx_thresh=1, tol=0.005)
        assert ln1 != ln2
        assert_allclose(ln1, ln2, rtol=0.2)


class TestConcurrence:
    @pytest.mark.parametrize("bs", ['psi-', 'phi-', 'psi+', 'phi+'])
    @pytest.mark.parametrize("qtype", ['ket', 'dop'])
    def test_bell_states(self, qtype, bs):
        p = qu.bell_state(bs, qtype=qtype)
        assert qu.concurrence(p) > 1.0 - 1e-14

    def test_subsystem(self):
        p = qu.rand_rho(2**4)
        e = qu.concurrence(p, [2, 2, 2, 2], 1, 2)
        assert 0 <= e <= 1


class TestQuantumDiscord:
    def test_owci(self):
        a = qu.qu([1, 0], qtype='op')
        b = qu.qu([0, 1], qtype='op')
        for _ in (0, 1, 2, 3):
            p = qu.rand_product_state(2)
            ci = qu.one_way_classical_information(p @ p.H, [a, b])
            assert_allclose(ci, 0., atol=1e-12)
        for i in (0, 1, 2, 3):
            p = qu.bell_state(i)
            ci = qu.one_way_classical_information(p @ p.H, [a, b])
            assert_allclose(ci, 1., atol=1e-12)

    def test_quantum_discord_sep(self):
        for _ in range(10):
            p = qu.rand_product_state(2)
            p = p @ p.H
            qd = qu.quantum_discord(p)
            assert_allclose(0.0, qd, atol=1e-12)

    def test_quantum_discord_pure(self):
        for _ in range(10):
            p = qu.rand_ket(4)
            p = p @ p.H
            iab = qu.mutual_information(p)
            qd = qu.quantum_discord(p)
            assert_allclose(iab / 2, qd)

    def test_quantum_discord_mixed(self):
        for _ in range(10):
            p = qu.rand_mix(4)
            p = p @ p.H
            qd = qu.quantum_discord(p)
            assert(0 <= qd and qd <= 1)

    def test_auto_trace_out(self):
        p = qu.rand_rho(2**3)
        qd = qu.quantum_discord(p, [2, 2, 2], 0, 2)
        assert(0 <= qd and qd <= 1)


class TestTraceDistance:
    def test_types(self, k1, k2):
        td1 = qu.trace_distance(k1, k2)
        td2 = qu.trace_distance(qu.dop(k1), k2)
        td3 = qu.trace_distance(k1, qu.dop(k2))
        td4 = qu.trace_distance(qu.dop(k1), qu.dop(k2))
        assert_allclose([td1] * 3, [td2, td3, td4])

    def test_same(self, p1):
        assert abs(qu.trace_distance(p1, p1)) < 1e-14

    @pytest.mark.parametrize("uqtype", ['ket', 'dop'])
    @pytest.mark.parametrize("dqtype", ['ket', 'dop'])
    def test_distinguishable(self, uqtype, dqtype):
        assert qu.trace_distance(qu.up(qtype=uqtype),
                                 qu.down(qtype=dqtype)) > 1 - 1e-10


class TestDecomp:
    @pytest.mark.parametrize("qtype", ['ket', 'dop'])
    def test_pauli_decomp_singlet(self, qtype):
        p = qu.singlet(qtype=qtype)
        names_cffs = qu.pauli_decomp(p, mode='cp')
        assert_allclose(names_cffs['II'], 0.25)
        assert_allclose(names_cffs['ZZ'], -0.25)
        assert_allclose(names_cffs['YY'], -0.25)
        assert_allclose(names_cffs['ZZ'], -0.25)
        for name in itertools.permutations('IXYZ', 2):
            assert_allclose(names_cffs["".join(name)], 0.0)

    def test_pauli_reconstruct(self):
        p1 = qu.rand_rho(4)
        names_cffs = qu.pauli_decomp(p1, mode='c')
        pr = sum(
            qu.kron(*(qu.pauli(s) for s in name)) * names_cffs["".join(name)]
            for name in itertools.product('IXYZ', repeat=2)
        )
        assert_allclose(pr, p1)

    @pytest.mark.parametrize(
        "state, out",
        [(qu.up() & qu.down(), {0: 0.5, 1: 0.5, 2: 0, 3: 0}),
         (qu.down() & qu.down(), {0: 0, 1: 0, 2: 0.5, 3: 0.5}),
         (qu.singlet() & qu.singlet(), {'00': 1.0, '23': 0.0})])
    def test_bell_decomp(self, state, out):
        names_cffs = qu.bell_decomp(state, mode='c')
        for key in out:
            assert_allclose(names_cffs[str(key)], out[key])


class TestCorrelation:
    @pytest.mark.parametrize("pre_c", [False, True])
    @pytest.mark.parametrize("p_sps", [True, False])
    @pytest.mark.parametrize("op_sps", [True, False])
    @pytest.mark.parametrize("dims", (None, [2, 2]))
    def test_types(self, dims, op_sps, p_sps, pre_c):
        p = qu.rand_rho(4, sparse=p_sps)
        c = qu.correlation(p, qu.pauli('x', sparse=op_sps),
                           qu.pauli('z', sparse=op_sps),
                           0, 1, dims=dims, precomp_func=pre_c)
        c = c(p) if pre_c else c
        assert c >= -1.0
        assert c <= 1.0

    @pytest.mark.parametrize("pre_c", [False, True])
    @pytest.mark.parametrize("qtype", ["ket", "dop"])
    @pytest.mark.parametrize("s", ['x', 'y', 'z'])
    def test_classically_no_correlated(self, s, qtype, pre_c):
        p = qu.up(qtype=qtype) & qu.up(qtype=qtype)
        c = qu.correlation(p, qu.pauli(s), qu.pauli(s),
                           0, 1, precomp_func=pre_c)
        c = c(p) if pre_c else c
        assert_allclose(c, 0.0)

    @pytest.mark.parametrize("pre_c", [False, True])
    @pytest.mark.parametrize("s, ct", [('x', 0), ('y', 0), ('z', 1)])
    def test_classically_correlated(self, s, ct, pre_c):
        p = 0.5 * ((qu.up(qtype='dop') & qu.up(qtype='dop')) +
                   (qu.down(qtype='dop') & qu.down(qtype='dop')))
        c = qu.correlation(p, qu.pauli(s), qu.pauli(s),
                           0, 1, precomp_func=pre_c)
        c = c(p) if pre_c else c
        assert_allclose(c, ct)

    @pytest.mark.parametrize("pre_c", [False, True])
    @pytest.mark.parametrize("s, ct", [('x', -1), ('y', -1), ('z', -1)])
    def test_entangled(self, s, ct, pre_c):
        p = qu.bell_state('psi-')
        c = qu.correlation(p, qu.pauli(s), qu.pauli(s),
                           0, 1, precomp_func=pre_c)
        c = c(p) if pre_c else c
        assert_allclose(c, ct)

    def test_reuse_precomp(self):
        cfn = qu.correlation(None, qu.pauli('z'), qu.pauli('z'), 0, 1,
                             dims=[2, 2], precomp_func=True)
        assert_allclose(cfn(qu.bell_state('psi-')), -1.0)
        assert_allclose(cfn(qu.bell_state('phi+')), 1.0)

    @pytest.mark.parametrize("pre_c", [False, True])
    def test_pauli_correlations_sum_abs(self, pre_c):
        p = qu.bell_state('psi-')
        ct = qu.pauli_correlations(p, sum_abs=True, precomp_func=pre_c)
        ct = ct(p) if pre_c else ct
        assert_allclose(ct, 3.0)

    @pytest.mark.parametrize("pre_c", [False, True])
    def test_pauli_correlations_no_sum_abs(self, pre_c):
        p = qu.bell_state('psi-')
        ct = qu.pauli_correlations(p, sum_abs=False, precomp_func=pre_c)
        assert_allclose(list(c(p) for c in ct) if pre_c else ct, (-1, -1, -1))


class TestEntCrossMatrix:
    def test_bell_state(self):
        p = qu.bell_state('phi+')
        ecm = qu.ent_cross_matrix(p, ent_fn=qu.concurrence, calc_self_ent=True)
        assert_allclose(ecm, [[1, 1], [1, 1]])

    def test_bell_state_no_self_ent(self):
        p = qu.bell_state('phi+')
        ecm = qu.ent_cross_matrix(p, ent_fn=qu.concurrence,
                                  calc_self_ent=False)
        assert_allclose(ecm, [[np.nan, 1], [1, np.nan]])

    def test_block2(self):
        p = qu.bell_state('phi+') & qu.bell_state('phi+')
        ecm = qu.ent_cross_matrix(p, ent_fn=qu.logneg, sz_blc=2)
        assert_allclose(ecm[1, 1], 0)
        assert_allclose(ecm[0, 1], 0)
        assert_allclose(ecm[1, 0], 0)

    def test_block2_no_self_ent(self):
        p = qu.bell_state('phi+') & qu.bell_state('phi+')
        ecm = qu.ent_cross_matrix(p, ent_fn=qu.logneg,
                                  calc_self_ent=False, sz_blc=2)
        assert_allclose(ecm[0, 1], 0)
        assert_allclose(ecm[0, 0], np.nan)
        assert_allclose(ecm[1, 0], 0)

    def test_block2_upscale(self):
        p = qu.bell_state('phi+') & qu.bell_state('phi+')
        ecm = qu.ent_cross_matrix(p, ent_fn=qu.logneg,
                                  calc_self_ent=False, sz_blc=2)
        assert ecm.shape == (2, 2)
        ecm = qu.ent_cross_matrix(p, ent_fn=qu.logneg, calc_self_ent=False,
                                  sz_blc=2, upscale=True)
        assert ecm.shape == (4, 4)


class TestEntCrossMatrixBlocked:
    @pytest.mark.parametrize("sz_p", [2**2 for i in [2, 3, 4, 5, 6, 9, 12]])
    @pytest.mark.parametrize("sz_blc", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("calc_self_ent", [True, False])
    def test_shapes_and_blocks(self, sz_blc, sz_p, calc_self_ent):
        if sz_p // sz_blc > 0:
            p = qu.rand_rho(2**sz_p)
            n = sz_p // sz_blc
            ecm = qu.ent_cross_matrix(p, sz_blc, calc_self_ent=calc_self_ent)
            assert ecm.shape[0] == n
            if not calc_self_ent:
                assert_allclose(np.diag(ecm), [np.nan] * n, equal_nan=True)


class TestQID:
    @pytest.mark.parametrize("bs", [0, 1, 2, 3])
    @pytest.mark.parametrize("pre_c", [False, True])
    def test_bell_state(self, bs, pre_c):
        p = qu.bell_state(bs)
        qids = qu.qid(p, dims=[2, 2], inds=[0, 1], precomp_func=pre_c)
        assert_allclose(qids(p) if pre_c else qids, [3, 3])

    @pytest.mark.parametrize("pre_c", [False, True])
    def test_random_product_state(self, pre_c):
        p = qu.rand_product_state(3)
        qids = qu.qid(p, dims=[2, 2, 2], inds=[0, 1, 2], precomp_func=pre_c)
        assert_allclose(qids(p) if pre_c else qids, [2, 2, 2])


class TestIsDegenerate:
    def test_known_degenerate(self):
        h = qu.ham_heis(2)
        assert qu.is_degenerate(h) == 2

    def test_known_nondegen(self):
        h = qu.ham_heis(2, b=0.3)
        assert qu.is_degenerate(h) == 0

    def test_supply_list(self):
        evals = [0, 1, 2, 2.0, 3]
        assert qu.is_degenerate(evals)

    def test_tol(self):
        evals = [0, 1, 1.001, 3, 4, 5, 6, 7, 8, 9]
        assert not qu.is_degenerate(evals)
        assert qu.is_degenerate(evals, tol=1e-2)


class TestPageEntropy:
    def test_known_qubit_qubit(self):
        assert abs(qu.page_entropy(2, 4) - 0.4808983469629878) < 1e-12

    def test_large_m_approx(self):
        pe = qu.page_entropy(2**10, 2**20)
        ae = 0.5 * (20 - math.log2(math.e))

        assert abs(pe - ae) < 1e-5

    def test_bigger_than_half(self):
        assert_allclose(qu.page_entropy(4, 24), qu.page_entropy(6, 24))


class TestIsEigenvector:

    def test_dense_true(self):
        a = qu.rand_herm(10)
        v = qu.eigvecsh(a)
        for i in range(10):
            assert qu.is_eigenvector(v[:, [i]], a)

    def test_dense_false(self):
        a = qu.rand_herm(10)
        v = qu.rand_ket(10)
        assert not qu.is_eigenvector(v, a)

    def test_sparse(self):
        a = qu.rand_herm(10, sparse=True, density=0.9)
        vt = qu.eigvecsh(a, sigma=0, k=1)
        assert qu.is_eigenvector(vt, a)
        vf = qu.rand_ket(10)
        assert not qu.is_eigenvector(vf, a)
