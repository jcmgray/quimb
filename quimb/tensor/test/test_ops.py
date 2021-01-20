import numpy as np
from quimb.tensor import fermion_ops as ops
from quimb.tensor.fermion_2d import gen_mf_peps, FermionTensorNetwork2DVector
from quimb.tensor.fermion import tensor_contract
from pyblock3.algebra.symmetry import SZ
from itertools import product


def get_state(out):
    vecmap = {(SZ(0), 0): "0,",
              (SZ(0), 1): "-+,",
              (SZ(1), 0): "+,",
              (SZ(1), 1): "-,"}
    outstring = ""
    for iblk in out.blocks:
        data = np.asarray(iblk)
        inds = np.where(abs(data)>0.)
        for ia, ib in zip(*inds):
            key1 = (iblk.q_labels[0], ia)
            key2 = (iblk.q_labels[1], ib)
            val = data[ia, ib]
            outstring += "+ %.1f|"%(val) + vecmap[key1] + vecmap[key2].replace(',','> ')

    if outstring=="":
        outstring= "|0>"
    return outstring

max_bond=4
Lx, Ly = 1,2

state_array = np.random.randint(0,4,[Lx,Ly])

def test_hopping(ix, iy):
    state_array = np.asarray([[ix,iy]])
    psi = gen_mf_peps(state_array, tags=("KET"))
    psi.view_as_(FermionTensorNetwork2DVector, like=psi)
    umat = ops.onsite_u(4)
    nmat = ops.count_n()
    zmat = ops.measure_sz()
    tmat = ops.gen_h1(1)

    instate = tensor_contract(psi[0,0], psi[0,1])

    psi1 = psi.gate(tmat.copy(), ((0,0), (0,1)), contract='split')

    outstate = tensor_contract(psi1[0,0], psi1[0,1])
    instring = get_state(instate.data.to_sparse())
    outstring = get_state(outstate.data.to_sparse())
    print("Input:", instring)
    print("Output 1:", outstring)

    state = np.tensordot(psi[0,1].data, psi[0,0].data, axes=((0,),(0,)))
    outstate = np.tensordot(tmat, state, axes=((2,3),(1,0))).transpose([1,0])
    print("Output 2:",get_state(outstate.to_sparse()))

    outstate = np.tensordot(tmat, state, axes=((2,3),(0,1)))
    print("Output 3:",get_state(outstate.to_sparse()))

    psi1 = psi.gate(tmat.copy(), ((0,1), (0,0)), contract='reduce-split')
    outstate = tensor_contract(psi1[0,0], psi1[0,1])
    outstring = get_state(outstate.data.to_sparse())

    print("Output 4:", outstring)

for ix, iy in product(range(4), repeat=2):
    if ix==iy: continue
    print("testing %i %i"%(ix, iy))
    test_hopping(ix, iy)
