import pytest
import numpy as np
from numpy.testing import assert_allclose
import quimb as qu


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("stype", ['csr', 'csc'])
@pytest.mark.parametrize("dtype", [
    "don't pass", None, np.float64, np.complex128])
def test_hamiltonian_builder(sparse, stype, dtype):
    from quimb.gen.operators import hamiltonian_builder

    @hamiltonian_builder
    def simple_ham(sparse=None, stype=None, dtype=None):
        H = qu.qu([[0.0, 1.0],
                   [1.0, 0.0]], sparse=True, stype='csr', dtype=dtype)
        return H

    @hamiltonian_builder
    def simple_ham_complex(sparse=None, stype=None, dtype=None):
        H = qu.qu([[0.0, 1.0j],
                   [-1.0j, 0.0]], sparse=True, stype='csr', dtype=dtype)
        return H

    if dtype == "don't pass":
        H = simple_ham(sparse=sparse, stype=stype)
    else:
        H = simple_ham(sparse=sparse, stype=stype, dtype=dtype)

    if dtype == "don't pass" or dtype is None:
        # check that passng no actual dtype keeps it as float
        assert H.dtype == np.float64
    else:
        # check that explicit dtypes are respected
        assert H.dtype == dtype
    assert qu.issparse(H) == sparse
    assert qu.isdense(H) != sparse
    if sparse:
        assert H.format == stype

    with pytest.raises(ValueError):  # check immutability
        H[0, 0] = 100

    if dtype == "don't pass":
        H = simple_ham_complex(sparse=sparse, stype=stype)
    elif dtype is np.float64:
        with pytest.warns(np.ComplexWarning):
            H = simple_ham_complex(sparse=sparse, stype=stype, dtype=dtype)
    else:
        H = simple_ham_complex(sparse=sparse, stype=stype, dtype=dtype)
    if dtype == "don't pass" or dtype is None:
        # check that passng no actual dtype keeps it as float
        assert H.dtype == np.complex128
    else:
        # check that explicit dtypes are respected
        assert H.dtype == dtype
    assert qu.issparse(H) == sparse
    assert qu.isdense(H) != sparse
    if sparse:
        assert H.format == stype

    with pytest.raises(ValueError):  # check immutability
        H[0, 0] = 100

    return


class TestSpinOperator:
    def test_spin_half(self):
        Sx = qu.spin_operator('x', 1 / 2)
        assert_allclose(Sx, [[0.0, 0.5], [0.5, 0.0]])

        Sy = qu.spin_operator('y', 1 / 2)
        assert_allclose(Sy, [[0.0, -0.5j], [0.5j, 0.0]])

        Sz = qu.spin_operator('z', 1 / 2)
        assert_allclose(Sz, [[0.5, 0.0], [0.0, -0.5]])

        Sp = qu.spin_operator('+', 1 / 2)
        assert_allclose(Sp, [[0.0, 1.0], [0.0, 0.0]])

        Sm = qu.spin_operator('-', 1 / 2)
        assert_allclose(Sm, [[0.0, 0.0], [1.0, 0.0]])

        SI = qu.spin_operator('I', 1 / 2)
        assert_allclose(SI, [[1.0, 0.0], [0.0, 1.0]])

    @pytest.mark.parametrize("label", ('x', 'y', 'z'))
    @pytest.mark.parametrize("S", [1, 3 / 2, 2, 5 / 2])
    def test_spin_high(self, label, S):
        D = int(2 * S + 1)
        op = qu.spin_operator(label, S)
        assert_allclose(qu.eigvalsh(op), np.linspace(-S, S, D), atol=1e-13)


class TestPauli:
    def test_pauli_dim2(self):
        for dir in (1, 'x', 'X',
                    2, 'y', 'Y',
                    3, 'z', 'Z'):
            x = qu.pauli(dir)
            assert_allclose(qu.eigvalsh(x), [-1, 1])

    def test_pauli_dim3(self):
        for dir in (1, 'x', 'X',
                    2, 'y', 'Y',
                    3, 'z', 'Z'):
            x = qu.pauli(dir, dim=3)
            assert_allclose(qu.eigvalsh(x), [-1, 0, 1],
                            atol=1e-15)

    def test_pauli_bad_dim(self):
        with pytest.raises(KeyError):
            qu.pauli('x', 4)

    def test_pauli_bad_dir(self):
        with pytest.raises(KeyError):
            qu.pauli('w', 2)


class TestControlledZ:
    def test_controlled_z_dense(self):
        cz = qu.controlled('z')
        assert_allclose(cz, np.diag([1, 1, 1, -1]))

    def test_controlled_z_sparse(self):
        cz = qu.controlled('z', sparse=True)
        assert(qu.issparse(cz))
        assert_allclose(cz.A, np.diag([1, 1, 1, -1]))


class TestGates:

    @pytest.mark.parametrize("gate", ['Rx', 'Ry', 'Rz', 'T_gate', 'S_gate',
                                      'CNOT', 'cX', 'cY', 'cZ', 'hadamard',
                                      'phase_gate', 'iswap', 'swap', 'U_gate',
                                      'fsim', 'fsimg'])
    @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
    @pytest.mark.parametrize('sparse', [False, True])
    def test_construct(self, gate, dtype, sparse):
        if gate in {'Rx', 'Ry', 'Rz', 'phase_gate'}:
            args = (0.43827,)
        elif gate in {'U_gate'}:
            args = (0.1, 0.2, 0.3)
        elif gate in {'fsim'}:
            args = (-1.3, 5.4)
        elif gate in {'fsimg'}:
            args = (-1.3, 5.4, 2., 3., 4.)
        else:
            args = ()
        G = getattr(qu, gate)(*args, dtype=dtype, sparse=sparse)
        assert G.dtype == dtype
        assert qu.issparse(G) is sparse
        psi = qu.rand_ket(G.shape[0])
        Gpsi = G @ psi
        assert qu.expec(Gpsi, Gpsi) == pytest.approx(1.0)

    def test_gates_import(self):
        from quimb.gates import Z
        assert_allclose(Z, [[1, 0], [0, -1]])

    def test_fsim(self):
        assert_allclose(qu.fsim(- qu.pi / 2, 0.0), qu.iswap(), atol=1e-12)

    def test_fsimg(self):
        assert_allclose(
            qu.fsimg(- qu.pi / 2, 0.0, 0.0, 0.0, 0.0),
            qu.iswap(), atol=1e-12
        )


class TestHamHeis:
    def test_ham_heis_2(self):
        h = qu.ham_heis(2, cyclic=False)
        evals = qu.eigvalsh(h)
        assert_allclose(evals, [-0.75, 0.25, 0.25, 0.25])
        gs = qu.groundstate(h)
        assert_allclose(qu.expec(gs, qu.singlet()), 1.)

    @pytest.mark.parametrize("parallel", [False, True])
    def test_ham_heis_sparse_cyclic_4(self, parallel):
        h = qu.ham_heis(4, sparse=True, cyclic=True, parallel=parallel)
        lk = qu.eigvalsh(h, k=4)
        assert_allclose(lk, [-2, -1, -1, -1])

    def test_ham_heis_bz(self):
        h = qu.ham_heis(2, cyclic=False, b=1)
        evals = qu.eigvalsh(h)
        assert_allclose(evals, [-3 / 4, -3 / 4, 1 / 4, 5 / 4])

    @pytest.mark.parametrize("stype", ["coo", "csr", "csc", "bsr"])
    def test_sformat_construct(self, stype):
        h = qu.ham_heis(4, sparse=True, stype=stype)
        assert h.format == stype


class TestHamJ1J2:
    def test_ham_j1j2_3_dense(self):
        h = qu.ham_j1j2(3, j2=1.0, cyclic=False)
        h2 = qu.ham_heis(3, cyclic=True)
        assert_allclose(h, h2)

    def test_ham_j1j2_6_sparse_cyc(self):
        h = qu.ham_j1j2(6, j2=0.5, sparse=True, cyclic=True)
        lk = qu.eigvalsh(h, k=5)
        assert_allclose(lk, [-9 / 4, -9 / 4, -7 / 4, -7 / 4, -7 / 4])

    def test_ham_j1j2_4_bz(self):
        h = qu.ham_j1j2(4, j2=0.5, cyclic=True, bz=0)
        lk = qu.eigvalsh(h, k=11)
        assert_allclose(lk, [-1.5, -1.5, -0.5, -0.5, -0.5, -0.5,
                             -0.5, -0.5, -0.5, -0.5, -0.5])
        h = qu.ham_j1j2(4, j2=0.5, cyclic=True, bz=0.05)
        lk = qu.eigvalsh(h, k=11)
        assert_allclose(lk, [-1.5, -1.5, -0.55, -0.55, -0.55,
                             -0.5, -0.5, -0.5, -0.45, -0.45, -0.45])


class TestHamMBL:
    @pytest.mark.parametrize("cyclic", [False, True])
    @pytest.mark.parametrize("sparse", [False, True])
    @pytest.mark.parametrize("dh_dim", [1, 2, 3, 'y', 'xz'])
    @pytest.mark.parametrize("dh_dist", ['s', 'g'])
    def test_construct(self, cyclic, sparse, dh_dim, dh_dist):
        qu.ham_mbl(n=3, dh=3, cyclic=cyclic, sparse=sparse, dh_dim=dh_dim,
                   dh_dist=dh_dist)

    @pytest.mark.parametrize("cyclic", [False, True])
    @pytest.mark.parametrize("sparse", [False, True])
    def test_construct_qp(self, cyclic, sparse):
        qu.ham_mbl(n=3, dh=3, cyclic=cyclic, sparse=sparse, dh_dist='qp')


class TestHamHeis2D:
    @pytest.mark.parametrize("cyclic", [False, True])
    @pytest.mark.parametrize("sparse", [False, True])
    @pytest.mark.parametrize("parallel", [False, True])
    @pytest.mark.parametrize("bz", [0.0, 0.7])
    def test_construct(self, cyclic, sparse, parallel, bz):
        qu.ham_heis_2D(2, 3, cyclic=cyclic, sparse=sparse,
                       parallel=parallel, bz=bz)


class TestSpinZProjector:
    @pytest.mark.parametrize("sz", [-2, -1, 0, 1, 2])
    def test_works(self, sz):
        prj = qu.zspin_projector(4, sz)
        h = qu.ham_heis(4)
        h0 = prj.T @ h @ prj
        v0s = qu.eigvecsh(h0)
        for i in range(v0s.shape[1]):
            v0 = v0s[:, [i]]
            vf = prj @ v0
            prjv = vf @ vf.H
            # Check reconstructed full eigenvectors commute with full ham
            assert_allclose(prjv @ h, h @ prjv, atol=1e-13)
        if sz == 0:
            # Groundstate must be in most symmetric subspace
            gs = qu.groundstate(h)
            gs0 = prj @ v0s[:, [0]]
            assert_allclose(qu.expec(gs, gs0), 1.0)
            assert_allclose(qu.expec(h, gs0), qu.expec(h, gs))

    def test_raises(self):
        with pytest.raises(ValueError):
            qu.zspin_projector(5, 0)
        with pytest.raises(ValueError):
            qu.zspin_projector(4, 1 / 2)

    @pytest.mark.parametrize("sz", [(-1 / 2, 1 / 2), (3 / 2, 5 / 2)])
    def test_spin_half_double_space(self, sz):
        prj = qu.zspin_projector(5, sz)
        h = qu.ham_heis(5)
        h0 = prj.T @ h @ prj
        v0s = qu.eigvecsh(h0)
        for i in range(v0s.shape[1]):
            v0 = v0s[:, [i]]
            vf = prj @ v0
            prjv = vf @ vf.H
            # Check reconstructed full eigenvectors commute with full ham
            assert_allclose(prjv @ h, h @ prjv, atol=1e-13)
        if sz == 0:
            # Groundstate must be in most symmetric subspace
            gs = qu.groundstate(h)
            gs0 = prj @ v0s[:, [0]]
            assert_allclose(qu.expec(gs, gs0), 1.0)
            assert_allclose(qu.expec(h, gs0), qu.expec(h, gs))


class TestSwap:
    @pytest.mark.parametrize("sparse", [False, True])
    def test_swap_qubits(self, sparse):
        a = qu.up() & qu.down()
        s = qu.swap(2, sparse=sparse)
        assert_allclose(s @ a, qu.down() & qu.up())

    @pytest.mark.parametrize("sparse", [False, True])
    def test_swap_higher_dim(self, sparse):
        a = qu.rand_ket(9)
        s = qu.swap(3, sparse=sparse)
        assert_allclose(s @ a, a.reshape([3, 3]).T.reshape([9, 1]))


@pytest.mark.parametrize("sparse", [False, True])
def test_3qubit_gates(sparse):
    psi = qu.rand_ket(8)
    psi = qu.toffoli(sparse=sparse) @ psi
    psi = qu.ccY(sparse=sparse) @ psi
    psi = qu.fredkin(sparse=sparse) @ psi
    psi = qu.ccZ(sparse=sparse) @ psi
    assert qu.expec(psi, psi) == pytest.approx(1.0)


class TestHubbardSpinless:

    def test_half_filling_groundstate(self):
        H = qu.ham_hubbard_hardcore(8, t=0.5, V=1.0, mu=1.0, cyclic=True)
        gs = qu.groundstate(H)
        dims = [2] * 8
        cn = qu.num(2)
        ens = [qu.expec(cn, qu.ptr(gs, dims, i)) for i in range(8)]
        for en in ens:
            assert en == pytest.approx(0.5, rel=1e-6)
