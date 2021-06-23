"""Core tensor network tools.
"""
import os
import copy
import functools

import numpy as np

from ..utils import (oset, valmap)
from .drawing import draw_tn

from .tensor_core import TensorNetwork, rand_uuid
from .tensor_block import tensor_split as _tensor_split
from .tensor_block import _core_contract, tensor_canonize_bond, tensor_compress_bond, BlockTensor, BlockTensorNetwork, get_block_contraction_path_info, tensor_balance_bond
from .block_interface import dispatch_settings
from functools import wraps

def contract_decorator(fn):
    @wraps(fn)
    def wrapper(T1, T2, *args, **kwargs):
        tid1, site1 = T1.get_fermion_info()
        tid2, site2 = T2.get_fermion_info()
        fs = T1.fermion_owner[0]
        if site1 > site2:
            fs.move(tid1, site2+1)
            out = fn(T1, T2, *args, **kwargs)
        else:
            fs.move(tid2, site1+1)
            out = fn(T2, T1, *args, **kwargs)
        if not isinstance(out, (float, complex)):
            fs.replace_tensor(min(site1, site2), out, virtual=True)
            fs.remove_tensor(min(site1, site2)+1)
        return out
    return wrapper

_core_contract = contract_decorator(_core_contract)

def compress_decorator(fn):
    @wraps(fn)
    def wrapper(T1, T2, *args, **kwargs):
        tid1, site1 = T1.get_fermion_info()
        tid2, site2 = T2.get_fermion_info()
        fs = T1.fermion_owner[0]
        loc_dict = {tid1: site1, tid2: site2}
        if site1 > site2:
            fs.move(tid1, site2+1)
        else:
            fs.move(tid1, site2)
        fn(T1, T2, *args, **kwargs)
        fs._reorder_from_dict(loc_dict)
        return T1, T2
    return wrapper

tensor_compress_bond = compress_decorator(tensor_compress_bond)
tensor_canonize_bond = compress_decorator(tensor_canonize_bond)

class FermionSpace:
    def __init__(self, tensor_order=None, virtual=True):
        self.tensor_order = {}
        if tensor_order is not None:
            for tid, (tsr, site) in tensor_order.items():
                self.add_tensor(tsr, tid, site, virtual=virtual)

    @property
    def sites(self):
        """ return a list of all the occupied positions
        """
        if len(self.tensor_order) == 0:
            return []
        else:
            return [val[1] for val in self.tensor_order.values()]

    def copy(self):
        """ Copy the FermionSpace object. Tensor ids and positions will be
        preserved and tensors will be copied
        """
        return FermionSpace(self.tensor_order, virtual=False)

    def add_tensor(self, tsr, tid=None, site=None, virtual=False):
        """ Add a tensor to the current FermionSpace, eg
        01234            0123456
        XXXXX, (6, B) -> XXXXX-B

        Parameters
        ----------
        tsr : FermionTensor
            The fermionic tensor to operate on
        tid : string, optional
            tensor id
        site: int or None, optional
            The position to place the tensor. Tensor will be
            appended to last position if not specified
        virtual: bool, optional
            whether to add the tensor inplace

        """
        if (tid is None) or (tid in self.tensor_order.keys()):
            tid = rand_uuid(base="_T")
        if site is None:
            site = 0 if len(self.sites)==0 else max(self.sites) + 1
        elif site in self.sites:
            raise ValueError("site:%s occupied, use replace/insert_tensor method"%site)

        T = tsr if virtual else tsr.copy()
        self.tensor_order[tid] = (T, site)
        T.set_fermion_owner(self, tid)

    def replace_tensor(self, site, tsr, tid=None, virtual=False):
        """ Replace the tensor at a given site, eg
        0123456789            0123456789
        XXXXAXXXXX, (4, B) -> XXXXBXXXXX

        Parameters
        ----------
        site: int
            The position to replace the tensor
        tsr : FermionTensor
            The fermionic tensor to operate on
        tid : string, optional
            rename a new tensor id if provided
        virtual: bool, optional
            whether to replace the tensor inplace
        """
        atid = self.get_tid_from_site(site)
        atsr = self.tensor_order[atid][0]
        T = tsr if virtual else tsr.copy()
        if tid is None or (tid in self.tensor_order.keys() and tid != atid):
            tid = atid
        T.set_fermion_owner(self, tid)
        atsr.remove_fermion_owner()
        del self.tensor_order[atid]
        self.tensor_order[tid] = (T, site)

    def insert_tensor(self, site, tsr, tid=None, virtual=False):
        """ insert a tensor at a given site, all tensors afterwards
        will be shifted forward by 1, eg,
        012345678            0123456789
        ABCDEFGHI, (4, X) -> ABCDXEFGHI

        Parameters
        ----------
        site: int
            The position to insert the tensor
        tsr : FermionTensor
            The fermionic tensor to operate on
        tid : string, optional
            rename a new tensor id if provided
        virtual: bool, optional
            whether to insert the tensor inplace
        """
        if (tid is None) or (tid in self.tensor_order.keys()):
            tid = rand_uuid(base="_T")
        if site not in self.sites:
            self.add_tensor(tsr, tid, site=site, virtual=virtual)
        else:
            T = tsr if virtual else tsr.copy()
            T.set_fermion_owner(self, tid)
            for atid, (atsr, asite) in self.tensor_order.items():
                if asite >= site:
                    self.tensor_order.update({atid: (atsr, asite+1)})
            self.tensor_order.update({tid: (T, site)})

    def insert(self, site, *tsrs, virtual=False):
        """ insert a group of tensors at a given site, all tensors afterwards
        will be shifted forward accordingly, eg,
        0123456                0123456789
        ABCDEFG, (4, (X,Y,Z)) -> ABCDXYZEFG

        Parameters
        ----------
        site: int
            The position to begin inserting the tensor
        tsrs : a tuple/list of FermionTensor
            The fermionic tensors to operate on
        virtual: bool, optional
            whether to insert the tensors inplace
        """
        for T in tsrs:
            self.insert_tensor(site, T, virtual=virtual)
            site += 1

    def get_tid_from_site(self, site):
        """ Return the tensor id at given site

        Parameters
        ----------
        site: int
            The position to obtain the tensor id
        """
        if site not in self.sites:
            raise KeyError("site:%s not occupied"%site)
        idx = self.sites.index(site)
        return list(self.tensor_order.keys())[idx]

    def get_full_info(self, tid_or_site):
        if isinstance(tid_or_site, str):
            tid = tid_or_site
        else:
            tid = self.get_tid_from_site(self, tid_or_site)
        T, site = self.tensor_order[tid_or_site]
        return T, tid, site

    def get_ind_map(self):
        ind_map = dict()
        for tid, (T, _) in self.tensor_order.items():
            for ind in T.inds:
                if ind not in ind_map:
                    ind_map[ind] = oset([tid])
                else:
                    ind_map[ind] = ind_map[ind].union(oset([tid]))
        return ind_map

    def _reorder_from_dict(self, tid_map):
        """ inplace reordering of tensors from a tensor_id/position mapping.
        Pizorn algorithm will be applied during moving

        Parameters
        ----------
        tid_map: dictionary
            Mapping of tensor id to the desired location
        """
        if len(tid_map) == len(self.tensor_order):
            self.reorder_all_(tid_map)
        else:
            tid_lst = list(tid_map.keys())
            des_sites = list(tid_map.values())
            # sort the destination sites to avoid cross-overs during moving
            work_des_sites = sorted(des_sites)[::-1]
            for isite in work_des_sites:
                ind = des_sites.index(isite)
                self.move(tid_lst[ind], isite)

    def reorder_all(self, tid_map, ind_map=None, inplace=False):
        """ reordering of tensors from a tensor_id/position mapping. The tid_map
        must contains the mapping for all tensors in this FermionSpace.
        Pizorn algorithm will be applied during moving.

        Parameters
        ----------
        tid_map: dictionary
            Mapping of tensor id to the desired location
        ind_map: dictinary, optional
            Mapping of tensor index to the tensor id
        inplace: bool, optional
            Whether the orordering operation is inplace or not
        """
        fs = self if inplace else self.copy()
        # Compute Global Phase
        if len(tid_map) != len(fs.tensor_order):
            raise ValueError("tid_map must be of equal size as the FermionSpace")
        nsites = len(fs.tensor_order)
        parity_tab = []
        input_tab = []
        free_tids = []
        for isite in range(nsites):
            tid = fs.get_tid_from_site(isite)
            T = fs.tensor_order[tid][0]
            if not T.avoid_phase:
                free_tids.append(tid)
            parity_tab.append(T.parity)
            input_tab.append(tid)

        tid_lst = list(tid_map.keys())
        des_sites = list(tid_map.values())
        global_parity = 0
        for fsite in range(nsites-1, -1, -1):
            idx = des_sites.index(fsite)
            tid = tid_lst[idx]
            isite = input_tab.index(tid)
            if isite == fsite: continue
            global_parity += np.sum(parity_tab[isite+1:fsite+1]) * parity_tab[isite]
            parity_tab[isite:fsite+1] = parity_tab[isite+1:fsite+1]+[parity_tab[isite]]
            input_tab[isite:fsite+1] = input_tab[isite+1:fsite+1]+[input_tab[isite]]

        _global_flip = (int(global_parity) % 2 == 1)
        if _global_flip:
            if len(free_tids) ==0:
                raise ValueError("Global flip required on one tensor but all tensors are marked to avoid phase")
            T = fs.tensor_order[free_tids[0]][0]
            T.flip_(global_flip=_global_flip)

        # Compute Local Phase
        if ind_map is None:
            ind_map = fs.get_ind_map()
        else:
            ind_map = ind_map.copy()

        local_flip_info = dict()
        for tid1, fsite1 in tid_map.items():
            T1, isite1 = fs.tensor_order[tid1]
            for ind in T1.inds:
                tids = ind_map.pop(ind, [])
                if len(tids) <2:
                    continue
                tid2, = tids - oset([tid1])
                T2, isite2 = fs.tensor_order[tid2]
                fsite2 = tid_map[tid2]
                if (isite1-isite2) * (fsite1-fsite2) < 0:
                    if T1.avoid_phase and T2.avoid_phase:
                        raise ValueError("relative order for %s and %s changed, local phase can not be avoided"%(tid1, tid2))
                    else:
                        marked_tid = tid2 if T1.avoid_phase else tid1
                    if marked_tid not in local_flip_info:
                        local_flip_info[marked_tid] = [ind]
                    else:
                        local_flip_info[marked_tid].append(ind)

        for tid, inds in local_flip_info.items():
            T = fs.tensor_order[tid][0]
            T.flip_(local_inds=inds)

        for tid, fsite in tid_map.items():
            T = fs.tensor_order[tid][0]
            fs.tensor_order.update({tid: (T, fsite)})
        return fs

    reorder_all_ = functools.partialmethod(reorder_all, inplace=True)

    def __setitem__(self, site, tsr):
        if site in self.sites:
            self.replace_tensor(site, tsr, virtual=True)
        else:
            self.add_tensor(site, tsr, virtual=True)

    def move(self, tid_or_site, des_site):
        """ Move a tensor inside this FermionSpace to the specified position with Pizorn algorithm.
        Both local and global phase will be factorized to this single tensor

        Parameters
        ----------
        tid_or_site: string or int
            id or position of the original tensor
        des_site: int
            the position to move the tensor to
        """

        tsr, tid, site = self.get_full_info(tid_or_site)
        avoid_phase = tsr.avoid_phase
        if site == des_site: return
        move_left = (des_site < site)
        iterator = range(des_site, site) if move_left else range(site+1, des_site+1)
        shared_inds = []
        tid_lst = [self.get_tid_from_site(isite) for isite in iterator]
        parity = 0
        for itid in tid_lst:
            itsr, isite = self.tensor_order[itid]
            i_shared_inds = list(oset(itsr.inds) & oset(tsr.inds))
            if avoid_phase:
                global_flip = (tsr.parity * itsr.parity == 1)
                if len(i_shared_inds)>0 or global_flip:
                    if itsr.avoid_phase:
                        raise ValueError("Two tensors marked to avoid phase")
                    itsr.flip_(global_flip=global_flip, local_inds=i_shared_inds)
            else:
                shared_inds += i_shared_inds
                parity += itsr.parity

            if move_left:
                self.tensor_order[itid] = (itsr, isite+1)
            else:
                self.tensor_order[itid] = (itsr, isite-1)

        if not avoid_phase:
            global_parity = (parity % 2) * tsr.data.parity
            global_flip = (global_parity == 1)
            tsr.flip_(global_flip=global_flip, local_inds=shared_inds)

        self.tensor_order[tid] = (tsr, des_site)

    def move_past(self, tsr, site_range=None):
        """ Move an external tensor past the specifed site ranges with Pizorn algorithm.
        Both local and global phase will be factorized to the external tensor.
        The external tensor will not be added to this FermionSpace

        Parameters
        ----------
        tsr: FermionTensor
            the external
        site_range: a tuple of integers, optional
            the range of the tensors to move past, if not specified, will be the whole space
        """
        if site_range is None:
            sites = self.sites
            site_range = (min(sites), max(sites)+1)
        start, end = site_range
        shared_inds = []
        tid_lst = [self.get_tid_from_site(isite) for isite in range(start, end)]
        parity = 0
        for itid in tid_lst:
            itsr, isite = self.tensor_order[itid]
            parity += itsr.parity
            shared_inds += list(oset(itsr.inds) & oset(tsr.inds))
        global_parity = (parity % 2) * tsr.data.parity
        if global_parity != 0: tsr.data._global_flip()
        axes = [tsr.inds.index(i) for i in shared_inds]
        if len(axes)>0: tsr.data._local_flip(axes)
        return tsr

    def remove_tensor(self, site):
        """ remove a specified tensor at a given site, eg
        012345               01234
        ABCDEF, (3, True) -> ABCEF
        """
        tid = self.get_tid_from_site(site)
        tsr = self.tensor_order[tid][0]
        tsr.remove_fermion_owner()
        del self.tensor_order[tid]

        indent_sites = sorted([isite for isite in self.sites if isite>site])
        tid_lst = [self.get_tid_from_site(isite) for isite in indent_sites]
        for tid in tid_lst:
            tsr, site = self.tensor_order[tid]
            self.tensor_order[tid] = (tsr, site-1)

    @property
    def H(self):
        """ Construct a FermionSpace for the bra state of the tensors
        """
        max_site = max(self.sites)
        new_fs = FermionSpace()
        for tid, (tsr, site) in self.tensor_order.items():
            T = tsr.copy()
            T.modify(data=T.data.dagger, inds=T.inds[::-1])
            new_fs.add_tensor(T, tid, max_site-site, virtual=True)
        return new_fs

# --------------------------------------------------------------------------- #
#                                Tensor Funcs                                 #
# --------------------------------------------------------------------------- #

def tensor_contract(*tensors, output_inds=None, preserve_tensor=False, inplace=False, **contract_opts):
    if len(tensors) == 1:
        if inplace:
            return tensors[0]
        else:
            return tensors[0].copy()
    path_info = get_block_contraction_path_info(*tensors, **contract_opts)
    fs, tid_lst = _dispatch_fermion_space(*tensors, inplace=inplace)
    if inplace:
        tensors = list(tensors)
    else:
        tensors = [fs.tensor_order[tid][0] for tid in tid_lst]

    for conc in path_info.contraction_list:
        pos1, pos2 = sorted(conc[0])
        T2 = tensors.pop(pos2)
        T1 = tensors.pop(pos1)
        out = _core_contract(T1, T2, preserve_tensor)
        tensors.append(out)

    if not isinstance(out, (float, complex)):
        _output_inds = out.inds
        if output_inds is None:
            output_inds = _output_inds
        else:
            output_inds = tuple(output_inds)
        if output_inds!=_output_inds:
            out.transpose_(*output_inds)
    return out

def tensor_split(
    T,
    left_inds,
    method='svd',
    get=None,
    absorb='both',
    max_bond=None,
    cutoff=1e-10,
    cutoff_mode='rel',
    renorm=None,
    ltags=None,
    rtags=None,
    stags=None,
    bond_ind=None,
    right_inds=None,
    qpn_info = None,
):
    if get is not None:
        return _tensor_split(T, left_inds, method=method, get=get, absorb=absorb, max_bond=max_bond,
                            cutoff=cutoff, cutoff_mode=cutoff_mode, renorm=renorm, ltags=ltags, rtags=rtags,
                            stags=stags, bond_ind=bond_ind, right_inds=right_inds, qpn_info=qpn_info)
    else:
        tensors = _tensor_split(T, left_inds, method=method, get="tensors", absorb=absorb, max_bond=max_bond,
                            cutoff=cutoff, cutoff_mode=cutoff_mode, renorm=renorm, ltags=ltags, rtags=rtags,
                            stags=stags, bond_ind=bond_ind, right_inds=right_inds, qpn_info=qpn_info)
        return FermionTensorNetwork(tensors[::-1], check_collisions=False)

def is_mergeable(*ts_or_tsn):
    """Check if all FermionTensor or FermionTensorNetwork objects
       are part of the same FermionSpace
    """
    if len(ts_or_tsn)==1 and isinstance(ts_or_tsn, (FermionTensor, FermionTensorNetwork)):
        return True
    fs_lst = []
    site_lst = []
    for obj in ts_or_tsn:
        if isinstance(obj, FermionTensor):
            if obj.fermion_owner is None:
                return False
            fsobj, tid = obj.fermion_owner
            fs_lst.append(hash(fsobj))
            site_lst.append(fsobj.tensor_order[tid][1])
        elif isinstance(obj, FermionTensorNetwork):
            fs_lst.append(hash(obj.fermion_space))
            site_lst.extend(obj.filled_sites)
        else:
            raise TypeError("unable to find fermionspace")

    return all([fs==fs_lst[0] for fs in fs_lst]) and len(set(site_lst)) == len(site_lst)

def _dispatch_fermion_space(*tensors, inplace=True):
    """ Retrieve the FermionSpace and the associated tensor_ids for the tensors.
    If the given tensors all belong to the same FermionSpace object (fsobj),
    the underlying fsobj will be returned. Otherwise, a new FermionSpace will be created,
    and the tensors will be placed in the same order as the input tensors.

    Parameters
    ----------
    tensors : a tuple or list of FermionTensors
        input_tensors
    inplace: bool
        if not true, a new FermionSpace will be created with all tensors copied.
        so subsequent operations on the fsobj will not alter the input tensors.

    Returns
    -------
    fs : a FermionSpace object
    tid_lst: a list of strings for the tensor_ids
    """

    if is_mergeable(*tensors):
        if isinstance(tensors[0], FermionTensor):
            fs = tensors[0].fermion_owner[0]
        else:
            fs = tensors[0].fermion_space
        if not inplace:
            fs = fs.copy()
        tid_lst = []
        for tsr_or_tn in tensors:
            if isinstance(tsr_or_tn, FermionTensor):
                tid_lst.append(tsr_or_tn.get_fermion_info()[0])
            else:
                tid_lst.append(tsr_or_tn.tensor_map.keys())
    else:
        fs = FermionSpace()
        for tsr_or_tn in tensors[::-1]:
            if isinstance(tsr_or_tn, FermionTensor):
                fs.add_tensor(tsr_or_tn, virtual=inplace)
            elif isinstance(tsr_or_tn, FermionTensorNetwork):
                if not tsr_or_tn.is_continuous():
                    raise ValueError("Input Network not continous, merge not allowed")
                for itsr in tsr_or_tn:
                    fs.add_tensor(itsr, virtual=inplace)
        tid_lst = list(fs.tensor_order.keys())
    return fs, tid_lst

FERMION_FUNCS = {"tensor_contract": tensor_contract,
                 "tensor_split": tensor_split,
                 "tensor_compress_bond": tensor_compress_bond,
                 "tensor_canonize_bond": tensor_canonize_bond,
                 "tensor_balance_bond": tensor_balance_bond}

# --------------------------------------------------------------------------- #
#                                Tensor Class                                 #
# --------------------------------------------------------------------------- #

class FermionTensor(BlockTensor):

    __slots__ = ('_data', '_inds', '_tags', '_left_inds',
                '_owners', '_fermion_owner', '_avoid_phase',
                '_fermion_path')

    def __init__(self, data=1.0, inds=(), tags=None, left_inds=None):

        # a new or copied Tensor always has no owners
        self._fermion_owner = None
        BlockTensor.__init__(self, data=data, inds=inds, tags=tags, left_inds=left_inds)
        if isinstance(data, FermionTensor):
            if len(data.inds)!=0:
                self._data = data.data.copy()
            self._avoid_phase = data._avoid_phase
            self._fermion_path = data._fermion_path.copy()
        else:
            self._avoid_phase = False
            self._fermion_path = dict()

    @property
    def custom_funcs(self):
        return FERMION_FUNCS

    @property
    def avoid_phase(self):
        return self._avoid_phase

    @property
    def fermion_path(self):
        return self._fermion_path

    @avoid_phase.setter
    def avoid_phase(self, avoid_phase):
        self._avoid_phase = avoid_phase

    @fermion_path.setter
    def fermion_path(self, fermion_path):
        self._fermion_path = fermion_path

    def set_fermion_path(self, global_flip=False, local_inds=None):
        if local_inds is None:
            local_inds = []
        _global_flip = self.fermion_path.pop("global_flip", False)
        _local_inds = self.fermion_path.pop("local_inds", [])
        self._fermion_path["global_flip"] = _global_flip ^ global_flip
        all_inds = tuple(local_inds) + tuple(_local_inds)
        updated_local_inds = []
        for ind in all_inds:
            count = all_inds.count(ind)
            if count % 2 ==1:
                updated_local_inds.append(ind)
        self._fermion_path["local_inds"] = updated_local_inds

    @property
    def fermion_owner(self):
        return self._fermion_owner

    @property
    def parity(self):
        return self.data.parity

    def modify_tid(self, tid):
        if self.fermion_owner is None:
            return
        fs, old_tid = self.fermion_owner
        if old_tid == tid:
            return
        _, site = fs.tensor_order[old_tid]
        del fs.tensor_order[old_tid]
        fs.tensor_order[tid] = (self, site)
        self.set_fermion_owner(fs, tid)
        if self.owners:
            tn = list(self.owners.values())[0][0]()
            del tn.tensor_map[old_tid]
            tn.tensor_map[tid] = self

    def modify(self, **kwargs):
        if "inds" in kwargs and "data" not in kwargs:
            inds = kwargs.get("inds")
            local_inds = self.fermion_path.pop("local_inds", [])
            new_local_inds = []
            for ind in local_inds:
                if ind in self.inds:
                    new_ind = inds[self.inds.index(ind)]
                    new_local_inds.append(new_ind)
            self._fermion_path["local_inds"] = new_local_inds

        super().modify(**kwargs)

    def flip(self, global_flip=False, local_inds=None, inplace=False):
        T = self if inplace else self.copy()
        T.set_fermion_path(global_flip=global_flip, local_inds=local_inds)
        if global_flip:
            T.data._global_flip()
        if local_inds is not None and len(local_inds)>0:
            axes = [T.inds.index(ind) for ind in local_inds]
            T.data._local_flip(axes)
        return T

    flip_ = functools.partialmethod(flip, inplace=True)

    def copy(self, deep=False):
        """Copy this tensor. Note by default (``deep=False``), the underlying
        array will *not* be copied. The fermion owner will to reset to None
        """
        if deep:
            t = copy.deepcopy(self)
            t.avoid_phase = self.avoid_phase
            t.fermion_path = self.fermion_path.copy()
            t.remove_fermion_owner()
        else:
            t = self.__class__(self, None)
        return t

    def get_fermion_info(self):
        if self.fermion_owner is None:
            return None
        fs, tid = self.fermion_owner
        site = fs.tensor_order[tid][1]
        return (tid, site)

    @fermion_owner.setter
    def fermion_owner(self, fowner):
        self._fermion_owner = fowner

    def set_fermion_owner(self, fs, tid):
        self.fermion_owner = (fs, tid)

    def remove_fermion_owner(self):
        self.fermion_owner = None

    def __and__(self, other):
        """Combine with another ``Tensor`` or ``TensorNetwork`` into a new
        ``TensorNetwork``.
        """
        return FermionTensorNetwork((self, other))

    def __or__(self, other):
        """Combine virtually (no copies made) with another ``Tensor`` or
        ``TensorNetwork`` into a new ``TensorNetwork``.
        """
        return FermionTensorNetwork((self, other), virtual=True)

    def draw(self, *args, **kwargs):
        """Plot a graph of this tensor and its indices.
        """
        draw_tn(FermionTensorNetwork((self,)), *args, **kwargs)

    graph = draw

# --------------------------------------------------------------------------- #
#                            Tensor Network Class                             #
# --------------------------------------------------------------------------- #

class FermionTensorNetwork(BlockTensorNetwork):

    _EXTRA_PROPS = ()
    _CONTRACT_STRUCTURED = False

    def __init__(self, ts, *, virtual=False, check_collisions=True):

        # short-circuit for copying TensorNetworks
        if isinstance(ts, self.__class__):
            if not ts.is_continuous():
                raise TypeError("Tensors not continuously placed in the network, \
                                this maybe due to this network being part of another network")
            fs = FermionSpace()
            self.tag_map = valmap(lambda tids: tids.copy(), ts.tag_map)
            self.ind_map = valmap(lambda tids: tids.copy(), ts.ind_map)
            self.tensor_map = dict()
            for t in ts:
                tid = t.get_fermion_info()[0]
                t = t.copy()
                self.tensor_map[tid] = t
                self.tensor_map[tid].add_owner(self, tid)
                fs.add_tensor(t, tid=tid, virtual=True)
            self._inner_inds = ts._inner_inds.copy()
            self._outer_inds = ts._outer_inds.copy()
            self._tid_counter = ts._tid_counter
            self.exponent = ts.exponent
            for ep in ts.__class__._EXTRA_PROPS:
                setattr(self, ep, getattr(ts, ep))
            return
        else:
            BlockTensorNetwork.__init__(self, ts, virtual=virtual, check_collisions=True)

    @property
    def fermion_space(self):
        if len(self.tensor_map)==0:
            return FermionSpace()
        else:
            return list(self.tensor_map.values())[0].fermion_owner[0]

    @property
    def filled_sites(self):
        return [self.fermion_space.tensor_order[tid][1] for tid in self.tensor_map.keys()]

    @property
    def H(self):
        tn = self.copy(full=True)
        fs = tn.fermion_space
        max_site = max(fs.sites)
        for tid, (T, site) in fs.tensor_order.items():
            T.modify(data=T.data.dagger, inds=T.inds[::-1])
            fs.tensor_order.update({tid: (T, max_site-site)})
        return tn

    def is_continuous(self):
        """
        Check if sites in the current tensor network are contiguously occupied
        """
        filled_sites = self.filled_sites
        if len(filled_sites) ==0 : return True
        return (max(filled_sites) - min(filled_sites) + 1) == len(filled_sites)

    def copy(self, full=False):
        """ For full copy, the tensors and underlying FermionSpace(all tensors in it) will
        be copied. For partial copy, the tensors in this network must be continuously
        placed and a new FermionSpace will be created to hold this continous sector.
        """
        if full:
            fs = self.fermion_space.copy()
            tids = list(self.tensor_map.keys())
            tsr = [fs.tensor_order[tid][0] for tid in tids]
            newtn = FermionTensorNetwork(tsr, virtual=True)
        else:
            if not self.is_continuous():
                raise TypeError("Tensors not continuously placed in the network, \
                                partial copy not allowed")
            newtn = FermionTensorNetwork(self)
        newtn.view_like_(self)
        return newtn

    def __and__(self, other):
        """Combine this tensor network with more tensors, without contracting.
        Copies the tensors.
        """
        if is_mergeable(self, other):
            raise ValueError("the two networks are in the same fermionspace, use self |= other")
        return FermionTensorNetwork((self, other), virtual=False)

    def __or__(self, other):
        """Combine this tensor network with more tensors, without contracting.
        Views the constituent tensors.
        """
        return FermionTensorNetwork((self, other), virtual=True)

    def __iter__(self):
        sorted_sites = sorted(self.filled_sites)
        for isite in sorted_sites:
            tid = self.fermion_space.get_tid_from_site(isite)
            yield self.tensor_map[tid]

    def __setitem__(self, tags, tensor):
        """Set the single tensor uniquely associated with ``tags``.
        """
        tids = self._get_tids_from_tags(tags, which='all')
        if len(tids) != 1:
            raise KeyError("'TensorNetwork.__setitem__' is meant for a single "
                           "existing tensor only - found {} with tag(s) '{}'."
                           .format(len(tids), tags))

        if not isinstance(tensor, FermionTensor):
            raise TypeError("Can only set value with a new 'FermionTensor'.")

        tid, = tids
        site = self.fermion_space.tensor_order[tid][1]
        super()._pop_tensor(tid)
        super().add_tensor(tensor, tid=tid, virtual=True)
        self.fermion_space.replace_tensor(site, tensor, tid=tid, virtual=True)

    def _reorder_from_tid(self, tid_map, inplace=False):
        tn = self if inplace else self.copy(full=True)
        tn.fermion_space._reorder_from_dict(tid_map)
        return tn

    def add_tensor(self, tsr, tid=None, virtual=False):
        T = tsr if virtual else tsr.copy()
        if virtual:
            fs = T.fermion_owner
            if fs is None:
                self.fermion_space.add_tensor(T, tid, virtual=True)
            else:
                if hash(fs[0]) != hash(self.fermion_space) and len(self.tensor_map)!=0:
                    raise ValueError("the tensor is already in a different FermionSpace, inplace addition not allowed")
        else:
            self.fermion_space.add_tensor(T, tid, virtual=True)
        tid = T.get_fermion_info()[0]
        super().add_tensor(T, tid, virtual=True)


    def add_tensor_network(self, tn, virtual=False, check_collisions=True):
        if virtual:
            if min(len(self.tensor_map), len(tn.tensor_map)) == 0:
                super().add_tensor_network(tn,
                        virtual=virtual, check_collisions=check_collisions)
                return
            elif hash(tn.fermion_space) == hash(self.fermion_space):
                if is_mergeable(self, tn):
                    super().add_tensor_network(tn,
                            virtual=True, check_collisions=check_collisions)
                else:
                    raise ValueError("the two tensornetworks co-share same sites, inplace addition not allowed")
                return

        if not tn.is_continuous():
            raise ValueError("input tensor network is not contiguously ordered")

        sorted_tensors = []
        for tsr in tn:
            tid = tsr.get_fermion_info()[0]
            sorted_tensors.append([tid, tsr])
            # if inplace, fermion_owners need to be
            # removed first to avoid conflicts
            if virtual:
                tsr.remove_fermion_owner()

        if check_collisions:  # add tensors individually
            # check for matching inner_indices -> need to re-index
            clash_ix = self._inner_inds & tn._inner_inds
            reind = {ix: rand_uuid() for ix in clash_ix}
        else:
            clash_ix = False
            reind = None

        # add tensors, reindexing if necessary
        for tid, tsr in sorted_tensors:
            if clash_ix and any(i in reind for i in tsr.inds):
                tsr = tsr.reindex(reind, inplace=virtual)
            self.add_tensor(tsr, virtual=virtual, tid=tid)

        self.exponent = self.exponent + tn.exponent

    def partition(self, tags, which='any', inplace=False):
        tn = self if inplace else self.copy(full=True)
        return TensorNetwork.partition(tn, tags, which=which, inplace=True)

    def contract_between(self, tags1, tags2, **contract_opts):
        contract_opts["inplace"] = True
        super().contract_between(tags1, tags2, **contract_opts)

    def contract_ind(self, ind, **contract_opts):
        """Contract tensors connected by ``ind``.
        """
        contract_opts["inplace"] = True
        super().contract_ind(ind, **contract_opts)

    # ----------------------- contracting the network ----------------------- #
    def contract_tags(self, tags, inplace=False, which='any', **opts):
        untagged_tn, tagged_ts = self.partition_tensors(
            tags, inplace=inplace, which=which)

        contracting_all = untagged_tn is None
        if not tagged_ts:
            raise ValueError("No tags were found - nothing to contract. "
                             "(Change this to a no-op maybe?)")
        opts["inplace"] = True
        contracted = tensor_contract(
            *tagged_ts, preserve_tensor=not contracting_all, **opts
        )

        if contracting_all:
            return contracted

        untagged_tn.add_tensor(contracted, virtual=True)
        return untagged_tn

    def __matmul__(self, other):
        """Overload "@" to mean full contraction with another network.
        """
        return FermionTensorNetwork((self, other)) ^ ...
