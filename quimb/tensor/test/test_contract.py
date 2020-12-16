import unittest
import numpy as np
from pyblock3.algebra.symmetry import SZ, BondInfo
from pyblock3.algebra.fermion import SparseFermionTensor, FlatFermionTensor
from quimb.tensor import fermion
import copy
import quimb as qu

np.random.seed(3)
x = SZ(0,0,0)
y = SZ(1,0,0)
infox = BondInfo({x:3, y: 2})

infoy = BondInfo({x:5, y: 5})


asp = SparseFermionTensor.random((infoy,infox,infox), dq=y)
abc = FlatFermionTensor.from_sparse(asp)

bsp = SparseFermionTensor.random((infox,infox,infox), dq=y)
bcd = FlatFermionTensor.from_sparse(bsp)

csp = SparseFermionTensor.random((infox,infox,infoy), dq=y)
efa = FlatFermionTensor.from_sparse(csp)

dsp = SparseFermionTensor.random((infox,infox,infox), dq=y)
def_ = FlatFermionTensor.from_sparse(dsp)


def finger(x):
    dat = x.data.data
    return (dat*np.sin(dat.size)).sum()

bcef = np.tensordot(abc, efa, axes=[(0,),(2,)])
efd = np.tensordot(bcef, bcd, axes=[(0,1),(0,1)])
dat = np.tensordot(efd, def_, axes=[(0,1,2),(1,2,0)])

bcef2 = np.tensordot(bcd, def_, axes=[(2,),(0,)])
dat1 = np.tensordot(bcef, bcef2, axes=[(0,1,2,3),(0,1,2,3)])

x = fermion.FermionTensor(abc, inds=['a','b','c'], tags=["x"])
y = fermion.FermionTensor(efa, inds=['e','f','a'], tags=["y"])
z = fermion.FermionTensor(bcd, inds=['b','c','d'], tags=["z"])
w = fermion.FermionTensor(def_, inds=['d','e','f'], tags=["w"])

tn = fermion.FermionTensorNetwork((x, y, z, w))

tn1 = tn.copy()
tn2 = tn.copy()

tn.contract_between(["x"], ["y"])
tn.contract_between(["x", "y"], ["w"])
out = tn.contract_between(["x", "y", "w"], ["z"])

print(dat.data[0], dat1.data[0], out)

tn1.contract_between(["x"], ["z"])
tn1.contract_between(["y"], ["w"])
out = tn1.contract_between(["x", "z"], ["y","w"])
print(dat.data[0], dat1.data[0], out)


tids = tn2._get_tids_from_inds(["b","c"])
tn2.contract_ind(["b","c"])
tn2.contract_ind(["a"])
out = tn2.contract_ind(["f"])
print(dat.data[0], dat1.data[0], out)


fs = fermion.FermionSpace()
fs.add_tensor(x, virtual=True)
fs.add_tensor(y, virtual=True)
fs.add_tensor(z, virtual=True)
fs.add_tensor(w, virtual=True)
out = fermion.tensor_contract(w, y, z, x, inplace=True, direction="right")
print(dat.data[0], dat1.data[0], out)
