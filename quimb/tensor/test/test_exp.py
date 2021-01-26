import numpy as np
from quimb.tensor import fermion_ops as ops
from pyblock3.algebra.fermion import _pack_flat_tensor, SparseFermionTensor, _unpack_flat_tensor, FlatFermionTensor
from pyblock3.algebra.core import SubTensor
from pyblock3.algebra.symmetry import SZ

t = 2
hop = ops.hopping(t)

def get_state(out):
    vecmap = {(SZ(0), 0): "0,",
              (SZ(0), 1): "-+,",
              (SZ(1), 0): "+,",
              (SZ(1), 1): "-,"}
    outstring = ""
    coeff = 0
    for iblk in out.blocks:
        data = np.asarray(iblk)
        inds = np.where(abs(data)>0.)
        for ia, ib in zip(*inds):
            key1 = (iblk.q_labels[0], ia)
            key2 = (iblk.q_labels[1], ib)
            val = data[ia, ib]
            outstring += "+ %.4f|"%(val) + vecmap[key1] + vecmap[key2].replace(',','> ')
            if vecmap[key1]+vecmap[key2] == "+,-,":
                coeff = val

    if outstring=="":
        outstring= "|0>"
    return outstring


def get_err(out, out1):
    nblk = len(out.q_labels)
    err = []
    for i in range(nblk):
        dlt = np.sum(abs(out.q_labels[i] - out1.q_labels), axis=1)
        j = np.where(dlt==0)[0][0]
        ist, ied = out.idxs[i], out.idxs[i+1]
        jst, jed = out1.idxs[j], out1.idxs[j+1]
        err.append(max(abs(out.data[ist:ied]-out1.data[jst:jed])))
    return max(err)

tau = 0.1
tsr = ops.to_exp(hop, -tau)

sx = SZ(0)
sy = SZ(1)

blocks=[]
states = np.zeros([2,2])
states[0,0] = 2**(-.5)
blocks.append(SubTensor(reduced=states, q_labels=(sx, sy)))
blocks.append(SubTensor(reduced=-states, q_labels=(sy, sx)))
# 2**.5 |0+> -2**.5|+0>, eigenstate of hopping(t)

eval = t
instate = SparseFermionTensor(blocks=blocks)
instring = get_state(instate)
print("Input: ", instring)
outstate0 = np.tensordot(hop, instate.to_flat(), axes=((2,3),(0,1))).to_sparse()
outstate = np.tensordot(tsr, instate.to_flat(), axes=((2,3),(0,1))).to_sparse()
outstring0 = get_state(outstate0)
outstring = get_state(outstate)
print("Output0:", outstring0)
print("Output:", outstring)
print("expected coeff: %.4f\n"%(np.e**(eval*-tau)*2**(-.5)))

eval = -t
blocks=[]
states = np.zeros([2,2])
states[0] = .5
blocks.append(SubTensor(reduced=states, q_labels=(sx, sy))) #0+, 0-
blocks.append(SubTensor(reduced=states.T, q_labels=(sy, sx))) #+0, -0, eigenstate of hopping

# .5 |0+> + .5 |0-> + .5 |+0> + .5 |-0>, eigenstate of hopping(-t)

instate = SparseFermionTensor(blocks=blocks)
instring = get_state(instate)
print("Input: ", instring)
outstate0 = np.tensordot(hop, instate.to_flat(), axes=((2,3),(0,1))).to_sparse()
outstate = np.tensordot(tsr, instate.to_flat(), axes=((2,3),(0,1))).to_sparse()
outstring0 = get_state(outstate0)
outstring = get_state(outstate)
print("Output0:", outstring0)
print("Output:", outstring)
print("expected coeff: %.4f\n"%(np.e**(eval*-tau)*(.5)))




eval = -2*t
blocks=[]
states = np.zeros([2,2])
states[1,0] = states[0,1] = .5
blocks.append(SubTensor(reduced=states, q_labels=(sx, sx)))
states = np.zeros([2,2])
states[1,0] = .5
states[0,1] =-.5
blocks.append(SubTensor(reduced=states, q_labels=(sy, sy)))
instate = SparseFermionTensor(blocks=blocks)
# .5 |0,-+> + .5 |-+,0> + .5 |-,+> - .5|+,->, eigenstate (-2)
instring = get_state(instate)
print("Input: ", instring)
outstate0 = np.tensordot(hop, instate.to_flat(), axes=((2,3),(0,1))).to_sparse()
outstate = np.tensordot(tsr, instate.to_flat(), axes=((2,3),(0,1))).to_sparse()
outstring0 = get_state(outstate0)
outstring = get_state(outstate)
print("Output0:", outstring0)
print("Output:", outstring)
print("expected coeff: %.4f\n"%(np.e**(eval*-tau)*(.5)))

eval= 2*t
blocks=[]
states = np.zeros([2,2])
states[1,0] = states[0,1] = .5
blocks.append(SubTensor(reduced=states, q_labels=(sx, sx)))
states = np.zeros([2,2])
states[1,0] =-.5
states[0,1] =.5
blocks.append(SubTensor(reduced=states, q_labels=(sy, sy)))
instate = SparseFermionTensor(blocks=blocks)
# .5 |0,-+> + .5 |-+,0> - .5 |-,+> + .5|+,->, eigenstate (2)
instring = get_state(instate)
print("Input: ", instring)
outstate0 = np.tensordot(hop, instate.to_flat(), axes=((2,3),(0,1))).to_sparse()
outstate = np.tensordot(tsr, instate.to_flat(), axes=((2,3),(0,1))).to_sparse()
outstring0 = get_state(outstate0)
outstring = get_state(outstate)
print("Output0:", outstring0)
print("Output:", outstring)
print("expected coeff: %.4f\n"%(np.e**(eval*-tau)*(.5)))
