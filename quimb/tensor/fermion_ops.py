import numpy as np
from itertools import product
from pyblock3.algebra.core import SubTensor
from pyblock3.algebra.fermion import SparseFermionTensor, FlatFermionTensor, _pack_flat_tensor, _unpack_flat_tensor
from pyblock3.algebra.symmetry import SZ, BondInfo
from .fermion_2d import FPEPS,FermionTensorNetwork2DVector

def to_exp(tsr, x):
    ndim = tsr.ndim
    if tsr.parity == 1:
        raise ValueError("expontial of odd parity tensor not defined")
    if np.mod(ndim, 2) !=0:
        raise ValueError("dimension of the tensor must be even (%i)"%ndim)
    ax = ndim //2
    data = []
    udata, sdata, vdata = [],[],[]
    uq,sq,vq= [],[],[]
    ushapes, vshapes, sshapes = [],[],[]
    sz_labels = ((SZ(0),SZ(0)), (SZ(1), SZ(1)))
    if ndim == 2:
        parity_axes = None
    else:
        parity_axes = list(range(ax))

    for szlab in sz_labels:
        data, row_map, col_map = _pack_flat_tensor(tsr, szlab, ax, parity_axes)
        el, ev = np.linalg.eig(data)
        s = np.diag(np.exp(el*x))
        _unpack_flat_tensor(ev, row_map, 0, udata, uq, ushapes, parity_axes)
        _unpack_flat_tensor(ev.conj().T, col_map, 1, vdata, vq, vshapes)
        sq.append([SZ.to_flat(iq) for iq in szlab])
        sshapes.append(s.shape)
        sdata.append(s.ravel())

    sq = np.asarray(sq, dtype=np.uint32)
    sshapes = np.asarray(sshapes, dtype=np.uint32)
    sdata = np.concatenate(sdata)
    s = FlatFermionTensor(sq, sshapes, sdata)

    uq = np.asarray(uq, dtype=np.uint32)
    ushapes = np.asarray(ushapes, dtype=np.uint32)
    udata = np.concatenate(udata)

    vq = np.asarray(vq, dtype=np.uint32)
    vshapes = np.asarray(vshapes, dtype=np.uint32)
    vdata = np.concatenate(vdata)
    u = FlatFermionTensor(uq, ushapes, udata)
    v = FlatFermionTensor(vq, vshapes, vdata)

    out = np.tensordot(u, s, axes=((-1,),(0,)))
    out = np.tensordot(out, v, axes=((-1,),(0,)))
    return out

eye = FlatFermionTensor.eye

def gen_h1(h=1.):
    blocks= []
    for i, j in product(range(2), repeat=2):
        qlab = (SZ(i), SZ(j), SZ(1-i), SZ(1-j))
        qlst = [q.n for q in qlab]
        iblk = np.zeros([2,2,2,2])
        blocks.append(SubTensor(reduced=np.zeros([2,2,2,2]), q_labels=(SZ(i), SZ(j), SZ(i), SZ(j))))
        if (i+j)==1:
            iblk[0,0,0,0] = iblk[i,j,j,i] = h
            iblk[1,1,1,1] = iblk[j,i,i,j] = -h
        else:
            if i == 0:
                iblk[0,1,0,1] = iblk[1,0,0,1] = h
                iblk[1,0,1,0] = iblk[0,1,1,0] = -h
            else:
                iblk[0,1,0,1] = iblk[0,1,1,0] = -h
                iblk[1,0,1,0] = iblk[1,0,0,1] = h
        blocks.append(SubTensor(reduced=iblk, q_labels=qlab))
    hop = SparseFermionTensor(blocks=blocks).to_flat()
    return hop

hopping = lambda t=1.0: gen_h1(-t)

def onsite_u(u=1):
    umat0 = np.zeros([2,2])
    umat0[1,1] = u
    umat1 = np.zeros([2,2])
    blocks = [SubTensor(reduced=umat0, q_labels=(SZ(0), SZ(0))),
              SubTensor(reduced=umat1, q_labels=(SZ(1), SZ(1)))]
    umat = SparseFermionTensor(blocks=blocks).to_flat()
    return umat

def hubbard(t, u, fac=None):
    if fac is None:
        fac = (1, 1)
    faca, facb = fac
    ham = hopping(t).to_sparse()
    for iblk in ham:
        qin, qout = iblk.q_labels[:2], iblk.q_labels[2:]
        if qin != qout: continue
        in_pair = [iq.n for iq in qin]
        if in_pair == [0,0]:
            iblk[1,0,1,0] += faca * u
            iblk[0,1,0,1] += facb * u
            iblk[1,1,1,1] += (faca + facb) * u
        elif in_pair == [0,1]:
            iblk[1,:,1,:] += faca * u * np.eye(2)
        elif in_pair == [1,0]:
            iblk[:,1,:,1] += facb * u * np.eye(2)
    return ham.to_flat()

def count_n():
    nmat0 = np.zeros([2,2])
    nmat0[1,1] = 2
    nmat1 = np.eye(2)
    blocks = [SubTensor(reduced=nmat0, q_labels=(SZ(0), SZ(0))),
              SubTensor(reduced=nmat1, q_labels=(SZ(1), SZ(1)))]
    nmat = SparseFermionTensor(blocks=blocks).to_flat()
    return nmat

def measure_sz():
    zmat0 = np.zeros([2,2])
    zmat1 = np.eye(2) * .5
    zmat1[1,1]= -.5
    blocks = [SubTensor(reduced=zmat0, q_labels=(SZ(0), SZ(0))),
              SubTensor(reduced=zmat1, q_labels=(SZ(1), SZ(1)))]
    smat = SparseFermionTensor(blocks=blocks).to_flat()
    return smat
