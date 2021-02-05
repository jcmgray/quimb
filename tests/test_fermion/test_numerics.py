import pytest
import numpy as np
import itertools
from quimb.tensor.fermion import (
    FermionTensor, FermionTensorNetwork, tensor_contract)

from quimb.tensor.fermion_2d import FPEPS

from pyblock3.algebra.symmetry import (SZ, BondInfo)
from pyblock3.algebra.fermion import SparseFermionTensor

def get_err(A, B):
    err = []
    nblk = A.shapes.shape[0]
    for i in range(nblk):
        dlt = np.sum(abs(A.q_labels[i] - B.q_labels), axis=1)
        j = np.where(dlt==0)[0][0]
        ist, ied = A.idxs[i], A.idxs[i+1]
        jst, jed = B.idxs[j], B.idxs[j+1]
        err.append(max(abs(A.data[ist:ied]-B.data[jst:jed])))
    return max(err)

np.random.seed(3)
bond_1 = BondInfo({SZ(0):3, SZ(1): 2})
bond_2 = BondInfo({SZ(0):5, SZ(1): 5})

abc = SparseFermionTensor.random(
    (bond_2, bond_1, bond_1), dq=SZ(1)).to_flat()

bcd = SparseFermionTensor.random(
    (bond_1, bond_1, bond_1), dq=SZ(1)).to_flat()

ega = SparseFermionTensor.random(
    (bond_1, bond_1, bond_2), dq=SZ(1)).to_flat()

deg = SparseFermionTensor.random(
    (bond_1, bond_1, bond_1), dq=SZ(1)).to_flat()

tsr_abc = FermionTensor(abc, inds=['a','b','c'], tags=["abc"])
tsr_ega = FermionTensor(ega, inds=['e','g','a'], tags=["ega"])
tsr_bcd = FermionTensor(bcd, inds=['b','c','d'], tags=["bcd"])
tsr_deg = FermionTensor(deg, inds=['d','e','g'], tags=["deg"])

tn = FermionTensorNetwork((tsr_abc, tsr_ega, tsr_bcd, tsr_deg))

# Tensor Order: deg, bcd, ega, abc
# Tensor Order: 3, 2, 1, 0

class TestContract:
    def test_backend(self):
        tsr_egbc = tensor_contract(tsr_abc, tsr_ega, output_inds=("e","g","b", "c"))

        egbc = np.tensordot(ega, abc, axes=[(2,),(0,)])
        err = get_err(tsr_egbc.data, egbc)
        assert err < 1e-10

    def test_contract_between(self):
        tn1 = tn.copy()
        tn1.contract_between("abc", "ega")
        tsr_egbc = tn1["abc"].transpose("e","g","b","c")

        egbc = np.tensordot(ega, abc, axes=[(2,),(0,)])
        err = get_err(tsr_egbc.data, egbc)
        assert err < 1e-10

    def test_contract_all(self):
        result = tn.contract(all)

        egbc = np.tensordot(ega, abc, axes=[(2,),(0,)])
        deg1 = np.tensordot(bcd, egbc, axes=[(0,1),(2,3)])
        ref_val = np.tensordot(deg, deg1, axes=[(0,1,2),]*2).data[0]

        err = abs(result - ref_val)
        assert err < 1e-10

    def test_contract_ind(self):
        tn1 = tn.copy()
        tn1.contract_ind("d")
        out = tn1["deg"].transpose("e","g","b","c")

        egbc = np.tensordot(deg, bcd, axes=[(0,),(2)])
        err = get_err(out.data, egbc)
        assert err < 1e-10

class TestCompress:
    def test_backend(self):
        pass

    def test_compress_between(self):
        pass

class TestCanonize:
    def test_backend(self):
        pass

class TestBalance:
    def test_balance_bonds(self):
        Lx = Ly = 3
        psi = FPEPS.rand(Lx, Ly, 2, seed=11)
        norm = psi.make_norm()
        exact = norm.contract(all, optimize="auto-hq")

        psi1 = psi.balance_bonds()
        norm = psi1.make_norm()
        exact_bb = norm.contract(all, optimize="auto-hq")
        assert exact_bb == pytest.approx(exact, rel=1e-2)
        for ix, iy in itertools.product(range(Lx), range(Ly)):
            assert psi[ix,iy].norm() != pytest.approx(psi1[ix,iy], rel=1e-2)

    def test_equlaize_norm(self):
        Lx = Ly = 3
        psi = FPEPS.rand(Lx, Ly, 2, seed=24)
        norm = psi.make_norm()
        exact = norm.contract(all, optimize="auto-hq")

        psi1 = psi.equalize_norms()
        norm = psi1.make_norm()
        exact_en = norm.contract(all, optimize="auto-hq")
        assert exact_en == pytest.approx(exact, rel=1e-2)
        for ix, iy in itertools.product(range(Lx), range(Ly)):
            assert psi[ix,iy].norm() != pytest.approx(psi1[ix,iy], rel=1e-2)
