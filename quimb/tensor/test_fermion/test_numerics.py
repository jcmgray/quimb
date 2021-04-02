import pytest
import numpy as np
from quimb.tensor.fermion import (
    FermionTensor, FermionTensorNetwork, tensor_contract)
from quimb.tensor.fermion_interface import BondInfo, U11, U1, Z2, Z4
from pyblock3.algebra.fermion import SparseFermionTensor

rand = SparseFermionTensor.random

@pytest.fixture(scope='class')
def u11setup(request):
    bond1 = BondInfo({U11(0):3, U11(1,1): 3, U11(1,-1):3, U11(2):3})
    bond2 = BondInfo({U11(0):5, U11(1,1): 5, U11(1,-1):5, U11(2):5})
    request.cls.abc = abc = rand((bond2, bond1, bond1), dq=U11(1,1), pattern="+--").to_flat()
    request.cls.bcd = bcd = rand((bond1, bond1, bond1), dq=U11(1,-1), pattern="++-").to_flat()
    request.cls.ega = ega = rand((bond1, bond1, bond2), dq=U11(1,1), pattern="-++").to_flat()
    request.cls.deg = deg = rand((bond1, bond1, bond1), dq=U11(1,-1), pattern="-+-").to_flat()
    request.cls.Tabc = Tabc = FermionTensor(abc, inds=['a','b','c'], tags=["abc"])
    request.cls.Tega = Tega = FermionTensor(ega, inds=['e','g','a'], tags=["ega"])
    request.cls.Tbcd = Tbcd = FermionTensor(bcd, inds=['b','c','d'], tags=["bcd"])
    request.cls.Tdeg = Tdeg = FermionTensor(deg, inds=['d','e','g'], tags=["deg"])
    request.cls.tn = FermionTensorNetwork((Tabc, Tega, Tbcd, Tdeg))

    ab = rand((bond1, bond1), dq=U11(0), pattern="+-").to_flat()
    bc = rand((bond1, bond1), dq=U11(1,-1), pattern="++").to_flat()
    Tab = FermionTensor(ab, inds=['a','b'], tags=["ab"])
    Tbc = FermionTensor(bc, inds=['b','c'], tags=["bc"])
    Tab1 = FermionTensor(ab, inds=['a','b1'], tags=["ab1"])
    Tbc1 = FermionTensor(bc, inds=['b1','c'], tags=["bc1"])
    request.cls.norm = FermionTensorNetwork((Tab, Tbc, Tbc1, Tab1))
    yield

@pytest.fixture(scope='class')
def u1setup(request):
    bond1 = BondInfo({U1(0):3, U1(1): 3, U1(3):3, U1(2):3})
    bond2 = BondInfo({U1(0):5, U1(1): 5, U1(3):5, U1(2):5})
    request.cls.abc = abc = rand((bond2, bond1, bond1), dq=U1(-1), pattern="+--").to_flat()
    request.cls.bcd = bcd = rand((bond1, bond1, bond1), dq=U1(3), pattern="++-").to_flat()
    request.cls.ega = ega = rand((bond1, bond1, bond2), dq=U1(-1), pattern="-++").to_flat()
    request.cls.deg = deg = rand((bond1, bond1, bond1), dq=U1(3), pattern="-+-").to_flat()
    request.cls.Tabc = Tabc = FermionTensor(abc, inds=['a','b','c'], tags=["abc"])
    request.cls.Tega = Tega = FermionTensor(ega, inds=['e','g','a'], tags=["ega"])
    request.cls.Tbcd = Tbcd = FermionTensor(bcd, inds=['b','c','d'], tags=["bcd"])
    request.cls.Tdeg = Tdeg = FermionTensor(deg, inds=['d','e','g'], tags=["deg"])
    request.cls.tn = FermionTensorNetwork((Tabc, Tega, Tbcd, Tdeg))

    ab = rand((bond1, bond1), dq=U1(0), pattern="+-").to_flat()
    bc = rand((bond1, bond1), dq=U1(1), pattern="++").to_flat()
    Tab = FermionTensor(ab, inds=['a','b'], tags=["ab"])
    Tbc = FermionTensor(bc, inds=['b','c'], tags=["bc"])
    Tab1 = FermionTensor(ab, inds=['a','b1'], tags=["ab1"])
    Tbc1 = FermionTensor(bc, inds=['b1','c'], tags=["bc1"])
    request.cls.norm = FermionTensorNetwork((Tab, Tbc, Tbc1, Tab1))
    yield

@pytest.fixture(scope='class')
def z4setup(request):
    bond1 = BondInfo({Z4(0):3, Z4(1): 3, Z4(3):3, Z4(2):3})
    bond2 = BondInfo({Z4(0):5, Z4(1): 5, Z4(3):5, Z4(2):5})
    request.cls.abc = abc = rand((bond2, bond1, bond1), dq=Z4(1), pattern="+--").to_flat()
    request.cls.bcd = bcd = rand((bond1, bond1, bond1), dq=Z4(3), pattern="++-").to_flat()
    request.cls.ega = ega = rand((bond1, bond1, bond2), dq=Z4(1), pattern="-++").to_flat()
    request.cls.deg = deg = rand((bond1, bond1, bond1), dq=Z4(3), pattern="-+-").to_flat()
    request.cls.Tabc = Tabc = FermionTensor(abc, inds=['a','b','c'], tags=["abc"])
    request.cls.Tega = Tega = FermionTensor(ega, inds=['e','g','a'], tags=["ega"])
    request.cls.Tbcd = Tbcd = FermionTensor(bcd, inds=['b','c','d'], tags=["bcd"])
    request.cls.Tdeg = Tdeg = FermionTensor(deg, inds=['d','e','g'], tags=["deg"])
    request.cls.tn = FermionTensorNetwork((Tabc, Tega, Tbcd, Tdeg))

    ab = rand((bond1, bond1), dq=Z4(0), pattern="+-").to_flat()
    bc = rand((bond1, bond1), dq=Z4(1), pattern="++").to_flat()
    Tab = FermionTensor(ab, inds=['a','b'], tags=["ab"])
    Tbc = FermionTensor(bc, inds=['b','c'], tags=["bc"])
    Tab1 = FermionTensor(ab, inds=['a','b1'], tags=["ab1"])
    Tbc1 = FermionTensor(bc, inds=['b1','c'], tags=["bc1"])
    request.cls.norm = FermionTensorNetwork((Tab, Tbc, Tbc1, Tab1))
    yield

@pytest.fixture(scope='class')
def z2setup(request):
    bond1 = BondInfo({Z2(0):3, Z2(1): 3})
    bond2 = BondInfo({Z2(0):5, Z2(1): 5})
    request.cls.abc = abc = rand((bond2, bond1, bond1), dq=Z2(1), pattern="+--").to_flat()
    request.cls.bcd = bcd = rand((bond1, bond1, bond1), dq=Z2(1), pattern="++-").to_flat()
    request.cls.ega = ega = rand((bond1, bond1, bond2), dq=Z2(1), pattern="-++").to_flat()
    request.cls.deg = deg = rand((bond1, bond1, bond1), dq=Z2(1), pattern="-+-").to_flat()
    request.cls.Tabc = Tabc = FermionTensor(abc, inds=['a','b','c'], tags=["abc"])
    request.cls.Tega = Tega = FermionTensor(ega, inds=['e','g','a'], tags=["ega"])
    request.cls.Tbcd = Tbcd = FermionTensor(bcd, inds=['b','c','d'], tags=["bcd"])
    request.cls.Tdeg = Tdeg = FermionTensor(deg, inds=['d','e','g'], tags=["deg"])
    request.cls.tn = FermionTensorNetwork((Tabc, Tega, Tbcd, Tdeg))

    ab = rand((bond1, bond1), dq=Z2(0), pattern="+-").to_flat()
    bc = rand((bond1, bond1), dq=Z2(1), pattern="++").to_flat()
    Tab = FermionTensor(ab, inds=['a','b'], tags=["ab"])
    Tbc = FermionTensor(bc, inds=['b','c'], tags=["bc"])
    Tab1 = FermionTensor(ab*1.3, inds=['a','b1'], tags=["ab1"])
    Tbc1 = FermionTensor(bc*1.5, inds=['b1','c'], tags=["bc1"])
    request.cls.norm = FermionTensorNetwork((Tab, Tbc, Tbc1, Tab1))
    yield

@pytest.mark.usefixtures('u1setup')
class TestU11:
    def test_backend(self):
        Tegbc = tensor_contract(self.Tabc, self.Tega, output_inds=("e","g","b", "c"))
        egbc = np.tensordot(self.ega, self.abc, axes=[(2,),(0,)])
        err = (egbc - Tegbc.data).norm()
        assert err < 1e-10

    def test_contract_between(self):
        tn1 = self.tn.copy()
        tn1.contract_between("abc", "ega")
        Tegbc = tn1["abc"].transpose("e","g","b","c")
        egbc = np.tensordot(self.ega, self.abc, axes=[(2,),(0,)])
        err = (egbc - Tegbc.data).norm()
        assert err < 1e-10

    def test_contract_all(self):
        result = self.tn.contract(all)
        egbc = np.tensordot(self.ega, self.abc, axes=[(2,),(0,)])
        deg1 = np.tensordot(self.bcd, egbc, axes=[(0,1),(2,3)])
        ref_val = np.tensordot(self.deg, deg1, axes=[(0,1,2),]*2).data[0]
        err = abs(result - ref_val)
        assert err < 1e-10

    def test_contract_ind(self):
        tn1 = self.tn.copy()
        tn1.contract_ind("d")
        out = tn1["deg"].transpose("e","g","b","c")
        egbc = np.tensordot(self.deg, self.bcd, axes=[(0,),(2,)])
        err = (egbc - out.data).norm()
        assert err < 1e-10

    def test_balance_bonds(self):
        norm = self.norm
        exact = norm.contract(all, optimize="auto-hq")
        norm1 = norm.balance_bonds()
        exact_bb = norm1.contract(all, optimize="auto-hq")
        assert exact_bb == pytest.approx(exact, rel=1e-2)
        for tid, tsr in norm.tensor_map.items():
            tsr1 = norm1.tensor_map[tid]
            assert (tsr1-tsr).data.norm() >1e-10

    def test_equlaize_norm(self):
        norm = self.norm
        exact = norm.contract(all, optimize="auto-hq")
        norm1 = norm.equalize_norms()
        exact_en = norm1.contract(all, optimize="auto-hq")
        assert exact_en == pytest.approx(exact, rel=1e-2)
        ref1 = list(norm1.tensor_map.values())[0].norm()
        for tid, tsr in norm.tensor_map.items():
            tsr1 = norm1.tensor_map[tid]
            assert tsr1.norm() == pytest.approx(ref1, rel=1e-2)

@pytest.mark.usefixtures('u1setup')
class TestU1(TestU11):
    pass

@pytest.mark.usefixtures('z4setup')
class TestZ4(TestU11):
    pass

@pytest.mark.usefixtures('z2setup')
class TestZ2(TestU11):
    pass
