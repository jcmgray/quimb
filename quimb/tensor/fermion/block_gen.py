import numpy as np
from itertools import product

from ...gen.rand import randn
from .block_interface import dispatch_settings, get_symmetry
from pyblock3.algebra.symmetry import BondInfo

def backend_wrapper(func):
    def new_func(*args, **kwargs):
        T = func(*args, **kwargs)
        use_cpp = dispatch_settings("use_cpp")
        if use_cpp:
            T = T.to_flat()
        return T
    return new_func

def _dispatch_dq(dq, symmetry):
    '''Construct pyblock3 fermion symmetry object

    Parameters
    ----------
    dq : int or tuple of integers
        Quantum particle number(s)
    symmetry : fermion symmetry class

    Returns
    -------
    Fermion symmetry object
    '''
    if dq is None:
        dq = (0, )
    elif isinstance(dq, (int, np.integer, np.float)):
        dq = (int(dq), )
    dq = symmetry(*dq)
    return dq

@backend_wrapper
def rand_single_block(shape, dtype=float, seed=None,
                      pattern=None, dq=None, ind=None, full_shape=None):
    '''Construct random block tensor with one block

    Parameters
    ----------
    shape : tuple or list of integers
        shape for the single block
    dtype : {'float64', 'complex128', 'float32', 'complex64'}, optional
        The underlying data type.
    seed : int, optional
        A random seed.
    pattern : string consisting of ("+", "-"), optional
        The symmetry pattern for each dimension
    dq : int or tuple of integers, optional
        The net particle number(s) in this tensor, default is 0
    ind: int, optional
        The axis to dispatch the dq symmetry

    Returns
    -------
    Block tensor
    '''
    if seed is not None:
        np.random.seed(seed)
    symmetry = get_symmetry()
    dq = _dispatch_dq(dq, symmetry)
    if pattern is None:
        pattern = "-" * (len(shape)-1) + "+"
    if ind is None:
        try:
            ind = pattern.index("+")
        except:
            ind = 0
    if pattern[ind] == "-":
        dq = - dq
    q_labels = [dq if ix==ind else symmetry(0) for ix in range(len(shape))]
    array = randn(shape, dtype=dtype)
    from pyblock3.algebra import fermion_setting as setting
    use_ad = setting.dispatch_settings(ad=None)
    if use_ad:
        from pyblock3.algebra.ad.fermion import SparseFermionTensor, SubTensor
        blk = SubTensor(data=array, q_labels=q_labels)
    else:
        from pyblock3.algebra.fermion import SparseFermionTensor, SubTensor
        blk = SubTensor(reduced=array, q_labels=q_labels)
    T = SparseFermionTensor(blocks=[blk, ], pattern=pattern, shape=full_shape)
    return T

@backend_wrapper
def ones_single_block(shape, pattern=None, dq=None, ind=None, full_shape=None):
    '''Construct block tensor filled with ones with a single block

    Parameters
    ----------
    shape : tuple or list of integers
        shape for the single block
    pattern : string consisting of ("+", "-"), optional
        The symmetry pattern for each dimension
    dq : int or tuple of integers, optional
        The net particle number(s) in this tensor, default is 0
    ind: int, optional
        The axis to dispatch the dq symmetry

    Returns
    -------
    Block tensor
    '''
    symmetry = get_symmetry()
    dq = _dispatch_dq(dq, symmetry)
    if pattern is None:
        pattern = "-" * (len(shape)-1) + "+"
    if ind is None:
        try:
            ind = pattern.index("+")
        except:
            ind = 0
    if pattern[ind] == "-":
        dq = - dq
    q_labels = [dq if ix==ind else symmetry(0) for ix in range(len(shape))]
    array = np.ones(shape)
    from pyblock3.algebra import fermion_setting as setting
    use_ad = setting.dispatch_settings(ad=None)
    if use_ad:
        from pyblock3.algebra.ad.fermion import SparseFermionTensor, SubTensor
        blk = SubTensor(data=array, q_labels=q_labels)
    else:
        from pyblock3.algebra.fermion import SparseFermionTensor, SubTensor
        blk = SubTensor(reduced=array, q_labels=q_labels)
    T = SparseFermionTensor(blocks=[blk, ], pattern=pattern, shape=full_shape)
    return T

@backend_wrapper
def rand_all_blocks(shape, symmetry_info, dtype=float,
                    seed=None, pattern=None, dq=None, full_shape=None):
    '''Construct block tensor with specified blocks

    Parameters
    ----------
    shape : tuple or list of integers
        shape for all blocks
    symmetry_info: tuple / list of tuple/list of integers
        allowed quantum numbers for each dimension, eg, [(0,1),(0,1),(0,1,2)]
        means allowed quantum numbers for the three dimensions are
        (0,1), (0,1) and (0,1,2) respectively. For U1 \otimes U1 symmetry,
        this could be [((0,0), (1,1), (1,-1)), ((0,0), (1,1), (1,-1)),
        (0,0),(1,1),(1,-1),(2,0)] where each tuple denotes a particle
        number and SZ number pair
    dtype : {'float64', 'complex128', 'float32', 'complex64'}, optional
        The underlying data type.
    seed : int, optional
        A random seed.
    pattern : string consisting of ("+", "-"), optional
        The symmetry pattern for each dimension
    dq : int or tuple of integers, optional
        The net particle number(s) in this tensor, default is 0

    Returns
    -------
    Block tensor
    '''
    if seed is not None:
        np.random.seed(seed)
    if full_shape is None:
        full_shape = []
        for ish, isym in zip(shape, symmetry_info):
            full_shape.append(ish * len(isym))
        full_shape = tuple(full_shape)
    symmetry = get_symmetry()
    dq = _dispatch_dq(dq, symmetry)
    bond_infos = []
    for sh, ibonds in zip(shape, symmetry_info):
        bonds = []
        for ibond in ibonds:
            if isinstance(ibond, (int, np.integer)):
                bonds.append(symmetry(ibond))
            else:
                bonds.append(symmetry(*ibond))
        bonds = dict(zip(bonds, [sh,]*len(bonds)))
        bond_infos.append(BondInfo(bonds))
    from pyblock3.algebra import fermion_setting as setting
    use_ad = setting.dispatch_settings(ad=None)
    if use_ad:
        from pyblock3.algebra.ad.fermion import SparseFermionTensor
    else:
        from pyblock3.algebra.fermion import SparseFermionTensor
    T = SparseFermionTensor.random(bond_infos, pattern=pattern, dq=dq, dtype=dtype, shape=full_shape)
    return T

def gen_2d_bonds(*args):
    symmetry = dispatch_settings("symmetry")
    func = {"U1": gen_2d_bonds_u1,
            "Z2": gen_2d_bonds_z2,
            "Z22": gen_2d_bonds_z22,
            "U11": gen_2d_bonds_u11}[symmetry]
    return func(*args)

def gen_2d_bonds_z2(pnarray, physical_infos):
    r'''Construct Z2 symmetry informations for each leg for 2d Fermionic TensorNetwork

    Parameters
    ----------
    pnarray : array_like
        Net Z2 symmetry for each site
    physical_infos : dict[tuple[int], tuple/list of integers]
        A dictionary mapping the site coordinates to the allowed quantum particle
        number of the physical dimension

    Returns
    -------
    symmetry_infos : dict[tuple[int], list/tuple of integers]
        A dictionary mapping the site coordinates to the allowed quantum particle
        numbers in each dimension ordered by up, right, down, left and physical.
    dq_infos: dict[tuple[int], int]
        A dictionary mapping the site coordinates to the net Z2 symmetry
        on that site
    '''
    Lx, Ly = pnarray.shape
    symmetry_infos = dict()
    dq_infos = dict()
    for ix, iy in product(range(Lx), range(Ly)):
        nvir = (ix != Lx - 1) + (ix != 0) +\
               (iy != Ly - 1) + (iy != 0)
        symmetry_infos[ix,iy] = [(0,1)] * nvir + [tuple(physical_infos[ix,iy])]
        dq_infos[ix,iy]= pnarray[ix,iy]
    return symmetry_infos, dq_infos

def gen_2d_bonds_z22(n1array, n2array, physical_infos):
    r'''Construct Z2 \otimes Z2 symmetry informations for each leg for 2d Fermionic TensorNetwork

    Parameters
    ----------
    n1array : array_like
        First entry of the net Z2 symmetry pairs for each site
    n2array : array_like
        Second entry of the net Z2 symmetry pairs for each site
    physical_infos : dict[tuple[int], tuple/list of integers]
        A dictionary mapping the site coordinates to the allowed quantum particle
        number pairs of the physical dimension

    Returns
    -------
    symmetry_infos : dict[tuple[int], list/tuple]
        A dictionary mapping the site coordinates to the allowed quantum particle
        number pairs in each dimension ordered by up, right, down, left and physical.
    dq_infos: dict[tuple[int], tuple of integers]
        A dictionary mapping the site coordinates to the net quantum particle number
        pair on that site
    '''
    Lx, Ly = n1array.shape
    symmetry_infos = dict()
    dq_infos = dict()
    for ix, iy in product(range(Lx), range(Ly)):
        nvir = (ix != Lx - 1) + (ix != 0) +\
               (iy != Ly - 1) + (iy != 0)
        symmetry_infos[ix,iy] = [((0,0),(0,1),(1,0),(1,1))] * nvir + [tuple(physical_infos[ix,iy])]
        dq_infos[ix,iy]= (n1array[ix,iy], n2array[ix,iy])
    return symmetry_infos, dq_infos

def gen_2d_bonds_u1(pnarray, physical_infos):
    r'''Construct U1 symmetry informations for each leg for 2d Fermionic TensorNetwork

    Parameters
    ----------
    pnarray : array_like
        The net particle number inflow for each site
    physical_infos : dict[tuple[int], tuple/list of integers]
        A dictionary mapping the site coordinates to the allowed quantum particle
        number of the physical dimension

    Returns
    -------
    symmetry_infos : dict[tuple[int], list/tuple of integers]
        A dictionary mapping the site coordinates to the allowed quantum particle
        numbers in each dimension ordered by up, right, down, left and physical.
    dq_infos: dict[tuple[int], int]
        A dictionary mapping the site coordinates to the net quantum particle number
        on that site
    '''
    Lx, Ly = pnarray.shape
    s_type = (Lx % 2==0)
    vbonds = [[0 for _ in range(Ly)] for _ in range(Lx+1)]
    hbonds = [[0 for _ in range(Ly+1)] for _ in range(Lx)]
    def _get_bond(ix, iy, *directions):
        bond_dict = {"r": hbonds[ix][iy+1],
                     "l": hbonds[ix][iy],
                     "u": vbonds[ix+1][iy],
                     "d": vbonds[ix][iy]}
        return [bond_dict[ix] for ix in directions]

    ave = np.sum(pnarray)/pnarray.size
    for ix in range(Lx):
        sweep_left = (s_type and ix%2==0) or (not s_type and ix%2==1)
        if sweep_left:
            for iy in range(Ly-1,-1,-1):
                if iy ==0:
                    right, left, down = _get_bond(ix, iy, "r", "l", "d")
                    vbonds[ix+1][iy] = down + left + ave - right - pnarray[ix][iy]
                else:
                    right, down, up = _get_bond(ix, iy, "r", "d", "u")
                    hbonds[ix][iy] = pnarray[ix][iy] + up + right - down - ave
        else:
            for iy in range(Ly):
                if iy ==Ly-1:
                    right, left, down = _get_bond(ix, iy, "r", "l", "d")
                    vbonds[ix+1][iy] = down + left + ave - right - pnarray[ix][iy]
                else:
                    left, up, down = _get_bond(ix, iy, "l", "u", "d")
                    hbonds[ix][iy+1] = down + left + ave - up - pnarray[ix][iy]

    hbonds = np.asarray(hbonds)[:,1:-1]
    vbonds = np.asarray(vbonds)[1:-1]

    def _round_to_bond(bd):
        if bd.is_integer():
            ibond = np.rint(bd).astype(int)
            return [ibond-1, ibond, ibond+1]
        else:
            ibond = np.floor(bd).astype(int)
            return [ibond, ibond+1]

    symmetry_infos = dict()
    dq_infos = dict()
    for ix, iy in product(range(Lx), range(Ly)):
        block = []
        if ix != Lx - 1:  # bond up
            block.append(_round_to_bond(vbonds[ix,iy]))
        if iy != Ly - 1:  # bond right
            block.append(_round_to_bond(hbonds[ix,iy]))
        if ix != 0:  # bond down
            block.append(_round_to_bond(vbonds[ix-1,iy]))
        if iy != 0:  # bond left
            block.append(_round_to_bond(hbonds[ix,iy-1]))
        block.append(physical_infos[ix][iy])
        symmetry_infos[ix,iy] = block
        dq_infos[ix,iy]=pnarray[ix,iy]
    return symmetry_infos, dq_infos

def gen_2d_bonds_u11(pnarray, szarray, physical_infos):
    r'''Construct U1 \otime U1 symmetry informations for each leg for 2d Fermionic TensorNetwork

    Parameters
    ----------
    pnarray : array_like
        The net particle number inflow for each site
    szarray : array_like
        The net SZ number inflow for each site, the parity for each site must be
        consistent with pnarray
    physical_infos : dict[tuple[int], tuple/list of integers]
        A dictionary mapping the site coordinates to the allowed quantum particle
        numbers (particle number and SZ number pair) of the physical dimension

    Returns
    -------
    symmetry_infos : dict[tuple[int], list/tuple]
        A dictionary mapping the site coordinates to the allowed particle number
        and SZ number pairs in each dimension ordered by up, right, down, left and physical.
    dq_infos: dict[tuple[int], tuple of integers]
        A dictionary mapping the site coordinates to the net quantum particle number
        and SZ number pair on that site
    '''

    Lx, Ly = pnarray.shape
    if not np.allclose(pnarray % 2, szarray % 2):
        raise ValueError("parity inconsistent")
    if abs(szarray).max()>1:
        raise ValueError("net |SZ| >1 not supported yet")
    s_type = (Lx % 2==0)
    vbonds = [[0 for _ in range(Ly)] for _ in range(Lx+1)]
    hbonds = [[0 for _ in range(Ly+1)] for _ in range(Lx)]
    def _get_bond(ix, iy, *directions):
        bond_dict = {"r": hbonds[ix][iy+1],
                     "l": hbonds[ix][iy],
                     "u": vbonds[ix+1][iy],
                     "d": vbonds[ix][iy]}
        return [bond_dict[ix] for ix in directions]

    ave = np.sum(pnarray)/pnarray.size
    for ix in range(Lx):
        sweep_left = (s_type and ix%2==0) or (not s_type and ix%2==1)
        if sweep_left:
            for iy in range(Ly-1,-1,-1):
                if iy ==0:
                    right, left, down = _get_bond(ix, iy, "r", "l", "d")
                    vbonds[ix+1][iy] = down + left + ave - right - pnarray[ix][iy]
                else:
                    right, down, up = _get_bond(ix, iy, "r", "d", "u")
                    hbonds[ix][iy] = pnarray[ix][iy] + up + right - down - ave
        else:
            for iy in range(Ly):
                if iy ==Ly-1:
                    right, left, down = _get_bond(ix, iy, "r", "l", "d")
                    vbonds[ix+1][iy] = down + left + ave - right - pnarray[ix][iy]
                else:
                    left, up, down = _get_bond(ix, iy, "l", "u", "d")
                    hbonds[ix][iy+1] = down + left + ave - up - pnarray[ix][iy]
    hbonds = np.asarray(hbonds)[:,1:-1]
    vbonds = np.asarray(vbonds)[1:-1]

    def _round_to_bond(bd):
        if bd.is_integer():
            ibond = np.rint(bd).astype(int)
            if ibond % 2==0:
                return [(ibond-1,1),(ibond-1,-1), (ibond,0), (ibond+1,-1), (ibond+1,1)]
            else:
                return [(ibond-1, 0), (ibond, 1), (ibond, -1), (ibond+1, 0)]
        else:
            ibond = np.floor(bd).astype(int)
            if ibond % 2==0:
                return [(ibond,0), (ibond+1,-1), (ibond+1,1)]
            else:
                return [(ibond, 1), (ibond, -1), (ibond+1, 0)]
    symmetry_infos = dict()
    dq_infos = dict()
    for ix, iy in product(range(Lx), range(Ly)):
        block = []
        if ix != Lx - 1:  # bond up
            block.append(_round_to_bond(vbonds[ix,iy]))
        if iy != Ly - 1:  # bond right
            block.append(_round_to_bond(hbonds[ix,iy]))
        if ix != 0:  # bond down
            block.append(_round_to_bond(vbonds[ix-1,iy]))
        if iy != 0:  # bond left
            block.append(_round_to_bond(hbonds[ix,iy-1]))
        block.append(physical_infos[ix][iy])
        symmetry_infos[ix,iy] = block
        dq_infos[ix,iy]= (pnarray[ix,iy],szarray[ix,iy])
    return symmetry_infos, dq_infos
