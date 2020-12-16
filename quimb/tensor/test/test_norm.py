import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tensor_2d import PEPS
import numpy as np
from quimb.tensor.fermion_2d import FPEPS
from quimb.tensor.fermion import _fetch_fermion_space, FermionTensor, FermionTensorNetwork
from quimb.tensor.tensor_core import oset
from pyblock3.algebra.core import SubTensor
from pyblock3.algebra.symmetry import SZ
from pyblock3.algebra.fermion import SparseFermionTensor

Lx = 2
Ly = 3
D = 2
np.random.seed(3)

def tensor_compress_bond(
    T1,
    T2,
    reduced=True,
    absorb='both',
    info=None,
    **compress_opts
):
    fs, (tid1, tid2) = _fetch_fermion_space(T1, T2, inplace=True)

    site1, site2 = fs[tid1][1], fs[tid2][1]

    if site1 < site2:
        Tl, Tr = T1, T2
        tidl, tidr = tid1, tid2
    else:
        Tl, Tr = T2, T1
        tidl, tidr = tid2, tid1

    left_inds = [ind for ind in Tl.inds if ind not in Tr.inds]
    right_inds = [ind for ind in Tr.inds if ind not in Tl.inds]

    out = fs._contract_pairs(tidl, tidr, direction="left")
    l, r = out.split(left_inds=left_inds, right_inds=right_inds, absorb=absorb, get="tensors", **compress_opts)
    return l, r

def get_err(max_bond=None):
    if max_bond is None: max_bond = 2*D**2


    psi = FPEPS.rand(Lx, Ly, bond_dim=D, seed=666)
    tsr1 = psi[0,0]
    tsr2 = psi[1,0]

    for x in range(Lx):
        psi.contract_between((0,x), (1,x))
    tsr1 = psi[0,0]
    tsr2 = psi[0,1]


    inds_contr = [i for i in tsr1.inds if i in tsr2.inds]
    outinds = [i for i in tsr1.inds if i not in tsr2.inds]
    idxa = [tsr1.inds.index(i) for i in inds_contr]
    idxb = [tsr2.inds.index(i) for i in inds_contr]

    out = np.tensordot(tsr1.data, tsr2.data, axes=(idxa, idxb))

    l, r = tensor_compress_bond(tsr1, tsr2, max_bond=max_bond)

    inds_contr = [i for i in l.inds if i in r.inds]
    outinds = [i for i in l.inds if i not in r.inds]
    idxa = [l.inds.index(i) for i in inds_contr]
    idxb = [r.inds.index(i) for i in inds_contr]
    fidx = [i for i in l.inds+r.inds if i not in inds_contr]

    out1 = np.tensordot(l.data, r.data, axes=(idxa, idxb))


    nblk = out.shapes.shape[0]

    err = []
    for i in range(nblk):
        dlt = np.sum(abs(out.q_labels[i] - out1.q_labels), axis=1)
        j = np.where(dlt==0)[0][0]
        ist, ied = out.idxs[i], out.idxs[i+1]
        jst, jed = out1.idxs[j], out1.idxs[j+1]
        err.append(max(abs(out.data[ist:ied]-out1.data[jst:jed])))
    return max(err)

dtype = "complex"
mat1 = np.zeros([2,2], dtype=dtype)
mat1[1,0] = mat1[0,1] = 0.5
blk = [SubTensor(reduced=mat1, q_labels=(SZ(0), SZ(0)))]
mat1 = np.zeros([2,2],dtype=dtype)
mat1[1,0] = 2**0.5*.5j
blk += [SubTensor(reduced=mat1, q_labels=(SZ(1), SZ(1)))]

x = FermionTensor(SparseFermionTensor(blocks=blk).to_flat(), inds=["a","b"])

y = x.H.data
out = np.tensordot(x.data, y, axes=((0,1),(1,0)))
print(out.data)

L, R = x.split(left_inds=["a"], get="tensors")

array = [[L.data,],[R.data,]]

psi = FPEPS(array, shape="rldpu") #WARNING
ket = psi.copy()
layer_tags=('KET', 'BRA')

ket.add_tag(layer_tags[0])
bra = ket.H.retag_({layer_tags[0]: layer_tags[1]}) #WARNING
bra.mangle_inner_("*")

L = ket[0,0]
R = ket[1,0]
L1 = bra[0,0]
R1 = bra[1,0]

tn = FermionTensorNetwork((R1,L1,L,R))
fs = tn.fermion_space
fs._contract_pairs(0,1) # WARNING
#fs._contract_pairs(1,2)
fs._contract_pairs(0,1)
out = fs._contract_pairs(0,1)

norm = bra & ket
def contract_all(tn):
    Lx, Ly = tn._Lx, tn._Ly
    for i in range(Lx):
        for j in range(Ly):
            x1, x2 = tn[i,j]
            tn.contract_between(x1.tags, x2.tags)
    for i in range(Lx):
        for j in range(Ly-1):
            x1 = tn[i,j]
            x2 = tn[i,j+1]
            tn.contract_between(x1.tags, x2.tags)
    for i in range(Lx-1):
        x1 = tn[i,0]
        x2 = tn[i+1,0]
        out = tn.contract_between(x1.tags, x2.tags)
    return out

def contract_left(tn):
    Lx, Ly = tn._Lx, tn._Ly
    for i in range(Lx):
        for j in range(Ly):
            x1, x2 = tn[i,j]
            tn.contract_between(x1.tags, x2.tags)
    for j in range(Ly):
        for i in range(Lx-1):
            x1 = tn[i,j]
            x2 = tn[i+1,j]
            out = tn.contract_between(x1.tags, x2.tags)
    for i in range(Ly-1):
        x1 = tn[0,i]
        x2 = tn[0,i+1]
        out = tn.contract_between(x1.tags, x2.tags)
    return out

norm1 = norm.copy()
out = contract_all(norm)
print(out)
out1 = contract_left(norm1)
print(out1)
