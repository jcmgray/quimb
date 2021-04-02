import pytest
import numpy as np
import itertools
from quimb.tensor.fermion_2d import FPEPS
from pyblock3.algebra.core import SubTensor
from pyblock3.algebra import fermion_ops
from quimb.tensor.fermion_interface import U11, U1, Z4, Z2, SparseFermionTensor
from quimb.tensor.fermion_gen import gen_mf_peps

@pytest.fixture(scope='class')
def u11setup(request):
    request.cls.t = 2
    request.cls.U = 4
    request.cls.tau = 0.1
    request.cls.mu = 0.2
    request.cls.symmetry = U11
    states = np.ones([1,1]) * .5 ** .5
    blocks = [SubTensor(reduced=states, q_labels=(U11(0),U11(1,1))), #0+
              SubTensor(reduced=states, q_labels=(U11(1,1),U11(0)))] #+0, eigenstate of hopping
    request.cls.hop_psi = SparseFermionTensor(blocks=blocks, pattern="++").to_flat()

    blocks=[]
    states = np.ones([1,1]) * .5
    blocks = [SubTensor(reduced=states, q_labels=(U11(2), U11(0))),
              SubTensor(reduced=states, q_labels=(U11(0), U11(2))),
              SubTensor(reduced=-states, q_labels=(U11(1,1), U11(1,-1))),
              SubTensor(reduced=states, q_labels=(U11(1,-1), U11(1,1)))]
    request.cls.hop_exp_psi = SparseFermionTensor(blocks=blocks, pattern="++").to_flat()

    Lx = Ly = 4
    request.cls.Lx = Lx
    request.cls.Ly = Ly
    request.cls.state_array = state_array = np.random.randint(0, 4, Lx*Ly).reshape(Lx, Ly)
    request.cls.peps = gen_mf_peps(state_array, symmetry='u11')
    request.cls.fac = (0.5, 0.3)

@pytest.fixture(scope='class')
def u1setup(request):
    request.cls.t = 2
    request.cls.U = 4
    request.cls.tau = 0.1
    request.cls.mu = 0.2
    request.cls.symmetry = U1
    states = np.zeros([1,2])
    states[0,0] = .5 ** .5
    blocks = [SubTensor(reduced=states, q_labels=(U1(0),U1(1))), #0+
              SubTensor(reduced=states, q_labels=(U1(1),U1(0)))] #+0, eigenstate of hopping
    request.cls.hop_psi = SparseFermionTensor(blocks=blocks, pattern="++").to_flat()

    blocks=[]
    states = np.zeros([2,2])
    states[0,1] = -.5
    states[1,0] = .5
    blocks = [SubTensor(reduced=np.ones([1,1]) * .5, q_labels=(U1(2), U1(0))),
              SubTensor(reduced=np.ones([1,1]) * .5, q_labels=(U1(0), U1(2))),
              SubTensor(reduced=states, q_labels=(U1(1), U1(1)))]
    request.cls.hop_exp_psi = SparseFermionTensor(blocks=blocks, pattern="++").to_flat()

    Lx = Ly = 4
    request.cls.Lx = Lx
    request.cls.Ly = Ly
    request.cls.state_array = state_array = np.random.randint(0, 4, Lx*Ly).reshape(Lx, Ly)
    request.cls.peps = gen_mf_peps(state_array, symmetry='u1')
    request.cls.fac = (0.5, 0.3)

@pytest.fixture(scope='class')
def z4setup(request):
    request.cls.t = 2
    request.cls.U = 4
    request.cls.tau = 0.1
    request.cls.mu = 0.2
    request.cls.symmetry = Z4
    states = np.zeros([2,2])
    states[0,0] = .5 ** .5
    blocks = [SubTensor(reduced=states, q_labels=(Z4(0),Z4(1))), #0+
              SubTensor(reduced=states, q_labels=(Z4(1),Z4(0)))] #+0, eigenstate of hopping
    request.cls.hop_psi = SparseFermionTensor(blocks=blocks, pattern="++").to_flat()

    blocks=[]
    states = np.zeros([2,2])
    states[1,0] = .5
    blocks = [SubTensor(reduced=states, q_labels=(Z4(0), Z4(0))),
              SubTensor(reduced=states.T, q_labels=(Z4(0), Z4(0))),
              SubTensor(reduced=-states.T, q_labels=(Z4(1), Z4(1))),
              SubTensor(reduced=states, q_labels=(Z4(1), Z4(1)))]
    request.cls.hop_exp_psi = SparseFermionTensor(blocks=blocks, pattern="++").to_flat()

    Lx = Ly = 4
    request.cls.Lx = Lx
    request.cls.Ly = Ly
    request.cls.state_array = state_array = np.random.randint(0, 4, Lx*Ly).reshape(Lx, Ly)
    request.cls.peps = gen_mf_peps(state_array, symmetry='z4')
    request.cls.fac = (0.5, 0.3)

@pytest.fixture(scope='class')
def z2setup(request):
    request.cls.t = 2
    request.cls.U = 4
    request.cls.tau = 0.1
    request.cls.mu = 0.2
    request.cls.symmetry = Z2
    states = np.zeros([2,2])
    states[0,0] = .5 ** .5
    blocks = [SubTensor(reduced=states, q_labels=(Z2(0),Z2(1))), #0+
              SubTensor(reduced=states, q_labels=(Z2(1),Z2(0)))] #+0, eigenstate of hopping
    request.cls.hop_psi = SparseFermionTensor(blocks=blocks, pattern="++").to_flat()

    blocks=[]
    states = np.zeros([2,2])
    states[1,0] = .5
    blocks = [SubTensor(reduced=states, q_labels=(Z2(0), Z2(0))),
              SubTensor(reduced=states.T, q_labels=(Z2(0), Z2(0))),
              SubTensor(reduced=-states.T, q_labels=(Z2(1), Z2(1))),
              SubTensor(reduced=states, q_labels=(Z2(1), Z2(1)))]

    request.cls.hop_exp_psi = SparseFermionTensor(blocks=blocks, pattern="++").to_flat()

    Lx = Ly = 4
    request.cls.Lx = Lx
    request.cls.Ly = Ly
    request.cls.state_array = state_array = np.random.randint(0, 4, Lx*Ly).reshape(Lx, Ly)
    request.cls.peps = gen_mf_peps(state_array, symmetry='z2')
    request.cls.fac = (0.5, 0.3)

@pytest.mark.usefixtures('u11setup')
class TestU11:
    def test_hopping(self):
        t = self.t
        hop = fermion_ops.H1(-t, symmetry=self.symmetry)
        ket = self.hop_psi
        ket1 = np.tensordot(hop, ket, axes=((2,3),(0,1)))
        bra = ket.dagger
        expec = np.tensordot(bra, ket1, axes=((1,0),(0,1))).data[0]
        assert expec == pytest.approx(-t, rel=1e-2)

    def test_hopping_exponential(self):
        t = self.t
        tau = self.tau
        hop = fermion_ops.H1(-t, symmetry=self.symmetry)
        #hop_exp = hop.to_exponential(-tau)
        hop_exp = fermion_ops.get_flat_exponential(hop, -tau)
        ket = self.hop_exp_psi
        bra = ket.dagger
        ket1 = np.tensordot(hop, ket, axes=((2,3),(0,1)))
        expec = np.tensordot(bra, ket1, axes=((1,0),(0,1))).data[0]
        assert expec == pytest.approx(2*t, rel=1e-2)

        ket1 = np.tensordot(hop_exp, ket, axes=((2,3),(0,1)))
        expec = np.tensordot(bra, ket1, axes=((1,0),(0,1))).data[0]
        assert expec == pytest.approx(np.e**(-2*t*tau), rel=1e-2)

    def test_onsite_u(self):
        U = self.U
        uop = fermion_ops.onsite_U(U, symmetry=self.symmetry)
        terms = {coo: uop for coo in itertools.product(range(self.Lx), range(self.Ly))}
        psi = self.peps
        state_array = self.state_array
        result = psi.compute_local_expectation(terms, normalized=False, return_all=True)
        for ix, iy in itertools.product(range(self.Lx), range(self.Ly)):
            ref = U if state_array[ix,iy]==3 else 0.
            assert ref == pytest.approx(result[(ix,iy)][0], rel=1e-2)

    def test_sz(self):
        sz = fermion_ops.measure_SZ(symmetry=self.symmetry)
        terms = {coo: sz for coo in itertools.product(range(self.Lx), range(self.Ly))}
        result = self.peps.compute_local_expectation(terms, normalized=False, return_all=True)
        ref_dic = {0:0., 1:0.5, 2:-.5, 3:0.}
        for ix, iy in itertools.product(range(self.Lx), range(self.Ly)):
            state = self.state_array[ix,iy]
            ref = ref_dic[state]
            assert ref == pytest.approx(result[(ix,iy)][0], rel=1e-2)

    def test_n(self):
        nop = fermion_ops.ParticleNumber(symmetry=self.symmetry)
        terms = {coo: nop for coo in itertools.product(range(self.Lx), range(self.Ly))}
        result = self.peps.compute_local_expectation(terms, normalized=False, return_all=True)
        ref_dic = {0:0., 1:1, 2:1, 3:2}
        for ix, iy in itertools.product(range(self.Lx), range(self.Ly)):
            state = self.state_array[ix,iy]
            ref = ref_dic[state]
            assert ref == pytest.approx(result[(ix,iy)][0], rel=1e-2)

    def test_exponential_u(self):
        U = self.U
        tau = self.tau
        uop = fermion_ops.onsite_U(U, symmetry=self.symmetry)
        uop_exp = fermion_ops.get_flat_exponential(uop, -tau)
        terms = {coo: uop_exp for coo in itertools.product(range(self.Lx), range(self.Ly))}
        result = self.peps.compute_local_expectation(terms, normalized=False, return_all=True)
        for ix, iy in itertools.product(range(self.Lx), range(self.Ly)):
            ref = np.e**(-tau*U) if self.state_array[ix,iy]==3 else 1.
            assert ref == pytest.approx(result[(ix,iy)][0], rel=1e-2)

    def test_hubbard(self):
        mu = self.mu
        hop = fermion_ops.H1(-self.t, symmetry=self.symmetry)
        uop = fermion_ops.onsite_U(self.U, symmetry=self.symmetry)
        nop = fermion_ops.ParticleNumber(symmetry=self.symmetry)
        faca, facb = self.fac
        hub = fermion_ops.Hubbard(self.t, self.U, mu=mu, fac=self.fac, symmetry=self.symmetry)
        ket = self.hop_exp_psi
        bra = ket.dagger

        ket1 = np.tensordot(hop, ket, axes=((2,3),(0,1)))
        ket1 = ket1 + faca*np.tensordot(uop, ket, axes=((-1,),(0,)))
        ket1 = ket1 + facb*np.tensordot(uop, ket, axes=((-1,),(1,))).transpose([1,0])
        ket1 = ket1 + faca*mu*np.tensordot(nop, ket, axes=((-1,),(0,)))
        ket1 = ket1 + facb*mu*np.tensordot(nop, ket, axes=((-1,),(1,))).transpose([1,0])
        expec = np.tensordot(bra, ket1, axes=((1,0),(0,1))).data[0]

        ket1 = np.tensordot(hub, ket, axes=((2,3),(0,1)))
        expec1 = np.tensordot(bra, ket1, axes=((1,0),(0,1))).data[0]
        assert expec == pytest.approx(expec1, rel=1e-2)

@pytest.mark.usefixtures('u1setup')
class TestU1(TestU11):
    pass

@pytest.mark.usefixtures('z4setup')
class TestZ4(TestU11):
    pass

@pytest.mark.usefixtures('z2setup')
class TestZ2(TestU11):
    pass
