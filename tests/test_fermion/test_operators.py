import pytest
import numpy as np
import itertools
from quimb.tensor.fermion_2d import gen_mf_peps, FPEPS
from pyblock3.algebra import fermion_operators as ops
from pyblock3.algebra.symmetry import QPN
from pyblock3.algebra.core import SubTensor
from pyblock3.algebra.fermion import SparseFermionTensor

Lx = Ly = 6
np.random.seed(3)
state_array = np.random.randint(0, 4, Lx*Ly).reshape(Lx, Ly)
psi = gen_mf_peps(state_array)

class TestOperators:
    def test_hopping(self):
        t = 2
        hop = ops.hopping(t)
        blocks=[]
        states = np.ones([1,1]) * .5 ** .5
        blocks.append(SubTensor(reduced=states, q_labels=(QPN(0),QPN(1,1)))) #0+
        blocks.append(SubTensor(reduced=states, q_labels=(QPN(1,1),QPN(0)))) #+0, eigenstate of hopping
        # psi = |0+> + |+0> - |-0>, eigenstate of hopping(eigval = -t)
        ket = SparseFermionTensor(blocks=blocks, pattern="++").to_flat()
        ket1 = np.tensordot(hop, ket, axes=((2,3),(0,1)))
        bra = ket.dagger
        expec = np.tensordot(bra, ket1, axes=((1,0),(0,1))).data[0]
        assert expec == pytest.approx(-t, rel=1e-2)

    def test_exponential_hop(self):
        t = 3
        tau = 0.1
        hop = ops.hopping(t)
        hop_exp = hop.to_exponential(-tau)
        blocks=[]
        states = np.ones([1,1]) * .5
        blocks.append(SubTensor(reduced=states, q_labels=(QPN(2), QPN(0))))
        blocks.append(SubTensor(reduced=states, q_labels=(QPN(0), QPN(2))))
        blocks.append(SubTensor(reduced=-states, q_labels=(QPN(1,1), QPN(1,-1))))
        blocks.append(SubTensor(reduced=states, q_labels=(QPN(1,-1), QPN(1,1))))

        ket = SparseFermionTensor(blocks=blocks, pattern="++").to_flat()
        ket1 = np.tensordot(hop, ket, axes=((2,3),(0,1)))
        bra = ket.dagger
        expec = np.tensordot(bra, ket1, axes=((1,0),(0,1))).data[0]
        assert expec == pytest.approx(2*t, rel=1e-2)

        ket1 = np.tensordot(hop_exp, ket, axes=((2,3),(0,1)))
        expec = np.tensordot(bra, ket1, axes=((1,0),(0,1))).data[0]
        assert expec == pytest.approx(np.e**(-2*t*tau), rel=1e-2)

    def test_onsite_u(self):
        U = 4.
        uop = ops.onsite_u(U)
        terms = {coo: uop for coo in itertools.product(range(Lx), range(Ly))}
        result = psi.compute_local_expectation(terms, normalized=False, return_all=True)
        for ix, iy in itertools.product(range(Lx), range(Ly)):
            ref = U if state_array[ix,iy]==3 else 0.
            assert ref == pytest.approx(result[(ix,iy)][0], rel=1e-2)

    def test_sz(self):
        sz = ops.measure_sz()
        terms = {coo: sz for coo in itertools.product(range(Lx), range(Ly))}
        result = psi.compute_local_expectation(terms, normalized=False, return_all=True)
        ref_dic = {0:0., 1:0.5, 2:-.5, 3:0.}
        for ix, iy in itertools.product(range(Lx), range(Ly)):
            state = state_array[ix,iy]
            ref = ref_dic[state]
            assert ref == pytest.approx(result[(ix,iy)][0], rel=1e-2)

    def test_n(self):
        nop = ops.count_n()
        terms = {coo: nop for coo in itertools.product(range(Lx), range(Ly))}
        result = psi.compute_local_expectation(terms, normalized=False, return_all=True)
        ref_dic = {0:0., 1:1, 2:1, 3:2}
        for ix, iy in itertools.product(range(Lx), range(Ly)):
            state = state_array[ix,iy]
            ref = ref_dic[state]
            assert ref == pytest.approx(result[(ix,iy)][0], rel=1e-2)

    def test_hubbard(self):
        Lx = Ly = 3
        psi = FPEPS.rand(Lx, Ly, 2)
        t = 2.
        U = 6.
        mu = 0.2
        hop = ops.hopping(t)
        uop = ops.onsite_u(U)
        nop = ops.count_n()
        full_terms = {(ix, iy): uop + mu*nop for ix, iy in itertools.product(range(Lx), range(Ly))}
        hterms = {coos: hop for coos in psi.gen_horizontal_bond_coos()}
        vterms = {coos: hop for coos in psi.gen_vertical_bond_coos()}
        full_terms.update(hterms)
        full_terms.update(vterms)
        mu_terms = {(ix, iy): nop for ix, iy in itertools.product(range(Lx), range(Ly))}
        ene = psi.compute_local_expectation(full_terms, max_bond=12)

        ham = dict()
        count_neighbour = lambda i,j: (i>0) + (i<Lx-1) + (j>0) + (j<Ly-1)
        for i, j in itertools.product(range(Lx), range(Ly)):
            count_ij = count_neighbour(i,j)
            if i+1 != Lx:
                where = ((i,j), (i+1,j))
                count_b = count_neighbour(i+1,j)
                uop = ops.hubbard(t,U, mu, (1./count_ij, 1./count_b))
                ham[where] = uop
            if j+1 != Ly:
                where = ((i,j), (i,j+1))
                count_b = count_neighbour(i,j+1)
                uop = ops.hubbard(t,U, mu, (1./count_ij, 1./count_b))
                ham[where] = uop
        ene1 = psi.compute_local_expectation(ham, max_bond=12)
        assert ene == pytest.approx(ene1, rel=1e-2)

    def test_exponential_u(self):
        U = 4.
        tau = 0.02
        uop = ops.onsite_u(U)
        uop_exp = uop.to_exponential(-tau)
        terms = {coo: uop_exp for coo in itertools.product(range(Lx), range(Ly))}
        result = psi.compute_local_expectation(terms, normalized=False, return_all=True)
        for ix, iy in itertools.product(range(Lx), range(Ly)):
            ref = np.e**(-tau*U) if state_array[ix,iy]==3 else 1.
            assert ref == pytest.approx(result[(ix,iy)][0], rel=1e-2)
