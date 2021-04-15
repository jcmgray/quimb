import numpy as np
from pyblock3.algebra.core import SubTensor
from pyblock3.algebra.fermion import SparseFermionTensor
from pyblock3.algebra import fermion_encoding
import quimb.tensor.block_interface as bitf
from quimb.tensor.fermion_2d import FPEPS

pattern_map = {"d": "+", "l":"+", "p":"+",
               "u": "-", "r":"-"}

def _gen_site_tsr(state, pattern=None, ndim=2, ax=0, symmetry=None):
    if symmetry is None: symmetry = bitf.DEFAULT_SYMMETRY
    state_map = fermion_encoding.get_state_map(symmetry)
    if state not in state_map:
        raise KeyError("requested state not recoginized")
    qlab, ind, dim = state_map[state]
    symmetry = qlab.__class__
    q_label = [symmetry(0),]*ax + [qlab] + [symmetry(0),] * (ndim-ax-1)
    shape = [1,] * ax + [dim,] +[1,] *(ndim-ax-1)
    dat = np.zeros(shape)
    ind = (0,)* ax + (ind,) + (0,) * (ndim-ax-1)
    dat[ind] = 1
    blocks = [SubTensor(reduced=dat, q_labels=q_label)]
    T = SparseFermionTensor(blocks=blocks, pattern=pattern)
    if bitf.USE_CPP:
        T =  T.to_flat()
    return T

def gen_mf_peps(state_array, shape='urdlp', symmetry=None, **kwargs):
    if symmetry is None: symmetry = bitf.DEFAULT_SYMMETRY
    Lx, Ly = state_array.shape
    arr = state_array.astype("int")
    cache = dict()
    def _gen_ij(i, j):
        state = arr[i, j]
        array_order = shape
        if i == Lx - 1:
            array_order = array_order.replace('u', '')
        if j == Ly - 1:
            array_order = array_order.replace('r', '')
        if i == 0:
            array_order = array_order.replace('d', '')
        if j == 0:
            array_order = array_order.replace('l', '')
        pattern = "".join([pattern_map[i] for i in array_order])
        ndim = len(array_order)
        ax = array_order.index('p')
        key = (state, ndim, ax, pattern)
        if key not in cache:
            cache[key] = _gen_site_tsr(state, pattern, ndim, ax, symmetry).copy()
        return cache[key]

    tsr_array = [[_gen_ij(i,j) for j in range(Ly)] for i in range(Lx)]
    return FPEPS(tsr_array, shape=shape, **kwargs)
