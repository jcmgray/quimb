import numpy as np
import weakref
import functools
from .tensor_core import (Tensor, TensorNetwork,
                          rand_uuid, tags_to_oset,
                          _parse_split_opts,
                          check_opt,
                          _VALID_SPLIT_GET)
from .tensor_core import tensor_contract as _tensor_contract
from ..utils import oset, valmap
from .array_ops import asarray, ndim, transpose


def _contract_connected(tsr1, tsr2, out_inds=None):
    ainds, binds = tsr1.inds, tsr2.inds
    _output_inds = []
    ax_a, ax_b = [], []
    for kia, ia in enumerate(ainds):
        if ia not in binds:
            _output_inds.append(ia)
        else:
            ax_a.append(kia)
            ax_b.append(binds.index(ia))
    for kib, ib in enumerate(binds):
        if ib not in ainds:
            _output_inds.append(ib)
    if out_inds is None: out_inds=_output_inds
    if set(_output_inds) != set(out_inds):
        raise TypeError("specified out_inds not allowed in tensordot, \
                         make sure no summation/Hadamard product appears")
    info1 = tsr1.get_fermion_info()
    info2 = tsr2.get_fermion_info()
    reverse_contract = False

    if info1 is not None and info2 is not None:
        if info1[1] > info2[1]:
            reverse_contract=True
    if reverse_contract:
        out = np.tensordot(tsr2.data, tsr1.data, axes=[ax_b, ax_a])
    else:
        out = np.tensordot(tsr1.data, tsr2.data, axes=[ax_a, ax_b])
    if len(out_inds)==0:
        return out.data[0]

    if out_inds!=_output_inds:
        transpose_order = tuple([_output_inds.index(ia) for ia in out_inds])
        out = out.transpose(transpose_order)
    o_tags = oset.union(*(tsr1.tags, tsr2.tags))
    out = FermionTensor(out, inds=out_inds, tags=o_tags)
    return out

def _contract_pairs(fs, tid_or_site1, tid_or_site2, out_inds=None, direction='left'):
    """ Perform pairwise contraction for two tensors in a specified fermion space.
        If the two tensors are not adjacent, move one of the tensors in the given direction.
        Note this could alter the tensors that are in between the two tensors in the fermion space

    Parameters
    ----------
    fs : FermionSpace obj
        the FermionSpace obj that contains the two tensors
    tid_or_site1: a string or an integer
        The string that specifies the id for the first tensor or the site for the first tensor
    tid_or_site2: a string or an integer
        The string that specifies the id for the 2nd tensor or the site for the 2nd tensor
    out_inds: a list of strings
        The list that specifies the output indices and its order
    direction: string "left" or "right"
        The direction to move tensors if the two tensors are not adjacent

    Returns
    -------
    out : a FermionTensor object or a number
    """
    site1 = fs[tid_or_site1][1]
    site2 = fs[tid_or_site2][1]

    if not fs.is_adjacent(tid_or_site1, tid_or_site2):
        fs.make_adjacent(tid_or_site1, tid_or_site2, direction)

    if direction=="left":
        site1 = min(site1, site2)
    else:
        site1 = max(site1, site2) - 1

    site2 = site1 + 1
    tsr1 = fs[site1][2]
    tsr2 = fs[site2][2]
    return _contract_connected(tsr1, tsr2, out_inds)

def _fetch_fermion_space(*tensors, inplace=True):
    """ Retrieve the FermionSpace and the associated tensor_ids for the tensors.
        If the given tensors all belong to the same FermionSpace object (fsobj),
        the underlying fsobj will be returned. Otherwise, a new fsobj will be created,
        and the tensors will be placed in the same order as the input tensor list/tuple.

    Parameters
    ----------
    tensors : a tuple or list of FermionTensors
        input_tensors
    inplace: bool
        if not true, a new FermionSpace will be created with all tensors copied.
        so subsequent operations on the fsobj will not alter the input tensors.
    tid_or_site2: a string or an integer
        The string that specifies the id for the 2nd tensor or the site for the 2nd tensor
    out_inds: a list of strings
        The list that specifies the output indices and its order
    direction: string "left" or "right"
        The direction to move tensors if the two tensors are not adjacent

    Returns
    -------
    fs : a FermionSpace object
    tid_lst: a list of strings for the tensor_ids
    """
    if isinstance(tensors, FermionTensor):
        tensors = (tensors, )

    if is_mergeable(*tensors):
        fs = tensors[0].fermion_owner[1]()
        if not inplace:
            fs = fs.copy()
        tid_lst = [tsr.fermion_owner[2] for tsr in tensors]
    else:
        fs = FermionSpace()
        for tsr in tensors:
            fs.add_tensor(tsr, virtual=inplace)
        tid_lst = list(fs.tensor_order.keys())
    return fs, tid_lst

def tensor_contract(*tensors, output_inds=None, direction="left", inplace=False):
    """ Perform tensor contractions for all the given tensors.
        If input tensors do not belong to the same underlying fsobj,
        the position of each tensor will be the same as its order in the input tensor tuple/list.
        Note summation and Hadamard product not supported as it's not well defined for fermionic tensors

    Parameters
    ----------
    tensors : a tuple or list of FermionTensors
        input tensors
    output_inds: a list of strings
    direction: string "left" or "right"
        The direction to move tensors if the two tensors are not adjacent
    inplace: bool
        whether to move/contract tensors in place.

    Returns
    -------
    out : a FermionTensor object or a number
    """
    path_info = _tensor_contract(*tensors, get='path-info')
    fs, tid_lst = _fetch_fermion_space(*tensors, inplace=inplace)
    for conc in path_info.contraction_list:
        pos1, pos2 = conc[0]
        tid1 = tid_lst.pop(pos1)
        tid2 = tid_lst.pop(pos2)
        site1 = fs[tid1][1]
        site2 = fs[tid2][1]
        out = fs._contract_pairs(site1, site2, direction=direction, inplace=True)
        if not isinstance(out, (float, complex)):
            tid_lst.append(out.fermion_owner[2])

    if not isinstance(out, (float, complex)):
        _output_inds = out.inds
        if output_inds is None:
            output_inds = _output_inds
        else:
            output_inds = tuple(output_inds)
        if set(_output_inds) != set(output_inds):
            raise TypeError("specified out_inds not allow in tensordot, \
                         make sure not summation/Hadamard product appears")
        if output_inds!=_output_inds:
            out = out.transpose(*output_inds, inplace=True)
    return out

def tensor_split(T, left_inds, method='svd', get=None, absorb='both', max_bond=None, cutoff=1e-10,
    cutoff_mode='rel', renorm=None, ltags=None, rtags=None, stags=None, bond_ind=None, right_inds=None):
    check_opt('get', get, _VALID_SPLIT_GET)

    if left_inds is None:
        left_inds = oset(T.inds) - oset(right_inds)
    else:
        left_inds = tags_to_oset(left_inds)

    if right_inds is None:
        right_inds = oset(T.inds) - oset(left_inds)

    opts = _parse_split_opts(
           method, cutoff, absorb, max_bond, cutoff_mode, renorm)
    _left_inds = [T.inds.index(i) for i in left_inds]
    _right_inds =[T.inds.index(i) for i in right_inds]

    if method == "svd":
        left, s, right = T.data.tensor_svd(_left_inds, right_idx=_right_inds, **opts)
    else:
        raise NotImplementedError

    if get == 'arrays':
        if absorb is None:
            return left, s, right
        return left, right

    ltags = T.tags | tags_to_oset(ltags)
    rtags = T.tags | tags_to_oset(rtags)
    if bond_ind is None:
        if absorb is None:
            bond_ind = (rand_uuid(),) * 2
        else:
            bond_ind = (rand_uuid(),)
    elif isinstance(bond_ind, str):
        bond_ind = (bond_ind,) * 2

    Tl = FermionTensor(data=left, inds=(*left_inds, bond_ind[0]), tags=ltags)
    Tr = FermionTensor(data=right, inds=(bond_ind[-1], *right_inds), tags=rtags)

    if absorb is None:
        stags = T.tags | tags_to_oset(stags)
        Ts = FermionTensor(data=s, inds=bond_ind, tags=stags)
        tensors = (Tl, Ts, Tr)
    else:
        tensors = (Tl, Tr)

    if get == 'tensors':
        return tensors

    return FermionTensorNetwork(tensors, check_collisions=False, virtual=True)

def _compress_connected(Tl, Tr, absorb='both', **compress_opts):
    left_inds = [ind for ind in Tl.inds if ind not in Tr.inds]
    right_inds = [ind for ind in Tr.inds if ind not in Tl.inds]
    if Tl.get_fermion_info()[1] < Tr.get_fermion_info()[1]:
        out = _contract_connected(Tl, Tr)
        l, r = out.split(left_inds=left_inds, right_inds=right_inds, absorb=absorb, get="tensors", **compress_opts)
    else:
        out = _contract_connected(Tr, Tl)
        if absorb == "left":
            absorb = "right"
        elif absorb == "right":
            absorb = "left"
        r, l = out.split(left_inds=right_inds, right_inds=left_inds, absorb=absorb, get="tensors", **compress_opts)
    return l, r

def tensor_compress_bond(
    T1,
    T2,
    absorb='both',
    inplace=True,
    info=None,
    **compress_opts
):
    fs, (tid1, tid2) = _fetch_fermion_space(T1, T2, inplace=inplace)
    site1, site2 = fs[tid1][1], fs[tid2][1]
    fs.make_adjacent(tid1, tid2)
    l, r = _compress_connected(T1, T2, absorb, **compress_opts)
    T1.modify(data=l.data, inds=l.inds)
    T2.modify(data=r.data, inds=r.inds)
    fs.move(tid1, site1)
    fs.move(tid2, site2)
    return T1, T2

def _canonize_connected(T1, T2, absorb='right', **split_opts):
    if absorb == 'both':
        return _compress_connected(T1, T2, absorb=absorb, **split_opts)
    if absorb == "left":
        T1, T2 = T2, T1

    shared_ix, left_env_ix = T1.filter_bonds(T2)
    if not shared_ix:
        raise ValueError("The tensors specified don't share an bond.")

    if T1.get_fermion_info()[1] < T2.get_fermion_info()[1]:
        new_T1, tRfact = T1.split(left_env_ix, get='tensors', **split_opts)
        new_T2 = _contract_connected(tRfact, T2)
    else:
        tRfact, new_T1 = T1.split(shared_ix, get='tensors', **split_opts)
        new_T2 = _contract_connected(T2, tRfact)

    if absorb == "left":
        return new_T2, new_T1
    else:
        return new_T1, new_T2

def tensor_canonize_bond(T1, T2, absorb='right', **split_opts):
    check_opt('absorb', absorb, ('left', 'both', 'right'))

    if absorb == 'both':
        return tensor_compress_bond(T1, T2, absorb=absorb, **split_opts)

    fs, (tid1, tid2) = _fetch_fermion_space(T1, T2, inplace=True)
    site1, site2 = fs[tid1][1], fs[tid2][1]

    fs.make_adjacent(tid1, tid2)
    l, r = _canonize_connected(T1, T2, absorb, **split_opts)
    T1.modify(data=l.data, inds=l.inds)
    T2.modify(data=r.data, inds=r.inds)
    fs.move(tid1, site1)
    fs.move(tid2, site2)
    return T1, T2

class FermionSpace:
    """A labelled, ordered dictionary. The tensor labels point to the tensor
       and its position inside the fermion space.

    Parameters
    ----------
    tensor_order : dictionary
        tensor_order[tid] = (tensor, site)
    """

    def __init__(self, tensor_order=None, virtual=True):
        self.tensor_order = {}
        if tensor_order is not None:
            if virtual:
                self.tensor_order = tensor_order
            else:
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

    def is_continuous(self):
        """ Check whether the tensors are continously placed in the Fermion Space
        """
        sites = self.sites
        if len(sites) == 0:
            return True
        else:
            if np.unique(sites).size != len(sites):
                raise ValueError("at least one site is occupied multiple times")
            return len(sites) == (max(sites)-min(sites)+1)

    def copy(self):
        """ Copy the Fermion Space object. Tensor_ids and positions will be
            preserved and tensors will be copied
        """
        new_fs = FermionSpace(self.tensor_order, virtual=False)
        return new_fs

    def to_tensor_network(self, site_lst=None):
        """ Construct a inplace FermionTensorNetwork obj with tensors at given sites
        """
        if site_lst is None:
            tsrs = tuple([tsr for (tsr, _) in self.tensor_order.values()])
        else:
            tsrs = tuple([tsr for (tsr, site) in self.tensor_order.values() if site in site_lst])
        return FermionTensorNetwork(tsrs, virtual=True)

    def add_tensor(self, tsr, tid=None, site=None, virtual=False):
        """ Add a tensor to the current FermionSpace, eg
            01234            0123456
            XXXXX, (6, B) -> XXXXX-B

        Parameters
        ----------
        tsr : FermionTensor obj
            The desired output sequence of indices.
        tid : string, optional
            The desired tensor label
        site: int or None, optional
            The position to place the tensor. Tensor will be
            appended if not specified
        virtual: bool
            whether to add the tensor inplace

        """
        if (tid is None) or (tid in self.tensor_order.keys()):
            tid = rand_uuid(base="_T")
        if site is None:
            site = 0 if len(self.sites)==0 else max(self.sites) + 1
        if site not in self.sites:
            T = tsr if virtual else tsr.copy()
            self.tensor_order[tid] = (T, site)
            T.set_fermion_owner(self, tid)
        else:
            raise ValueError("site:%s occupied, use replace/insert_tensor method"%site)

    def replace_tensor(self, site, tsr, tid=None, virtual=False):
        """ Replace the tensor at a given site, eg
            0123456789            0123456789
            XXXXAXXXXX, (4, B) -> XXXXBXXXXX
        """
        atid, site, atsr = self[site]
        T = tsr if virtual else tsr.copy()
        if tid is None or (tid in self.tensor_order.keys() and tid != atid):
            tid = atid

        T.set_fermion_owner(self, tid)
        atsr.remove_fermion_owner()
        del self.tensor_order[atid]
        self.tensor_order[tid] = (T, site)

    def insert_tensor(self, site, tsr, tid=None, virtual=False):
        """ insert a tensor at a given site, all tensors afterwards
            will be shifted by 1 to the right, eg,
            012345678            0123456789
            ABCDEFGHI, (4, X) -> ABCDXEFGHI
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


    def insert(self, site, *tsr, virtual=False):
        for T in tsr:
            self.insert_tensor(site, T, virtual=virtual)
            site += 1

    def get_tid(self, site):
        """ Return the tensor id at given site
        """
        if site not in self.sites:
            raise KeyError("site:%s not occupied"%site)
        idx = self.sites.index(site)
        return list(self.tensor_order.keys())[idx]

    def _reorder_from_dict(self, tid_map):
        tid_lst = list(tid_map.keys())
        des_sites = list(tid_map.values())
        work_des_sites = sorted(des_sites)[::-1]
        for isite in work_des_sites:
            ind = des_sites.index(isite)
            self.move(tid_lst[ind], isite)

    def is_adjacent(self, tid_or_site1, tid_or_site2):
        """ Check whether two tensors are adjacently placed in the space
        """
        site1 = self[tid_or_site1][1]
        site2 = self[tid_or_site2][1]
        distance = abs(site1-site2)
        return distance == 1

    def __getitem__(self, tid_or_site):
        if isinstance(tid_or_site, str):
            if tid_or_site not in self.tensor_order.keys():
                raise KeyError("tid:%s not found"%tid_or_site)
            tsr, site = self.tensor_order[tid_or_site]
            return tid_or_site, site, tsr
        elif isinstance(tid_or_site, int):
            if tid_or_site not in self.sites:
                raise KeyError("site:%s not occupied"%tid_or_site)
            tid = self.get_tid(tid_or_site)
            tsr = self.tensor_order[tid][0]
            return tid, tid_or_site, tsr
        else:
            raise ValueError("not a valid key value(tid or site)")

    def __setitem__(self, site, tsr):
        if site in self.sites:
            self.replace_tensor(site, tsr)
        else:
            self.add_tensor(site, tsr)

    def move(self, tid_or_site, des_site):
        '''Both local and global phase factorized to the tensor that's being operated on
        '''
        tid, site, tsr = self[tid_or_site]
        if site == des_site: return
        move_left = (des_site < site)
        iterator = range(des_site, site) if move_left else range(site+1, des_site+1)
        shared_inds = []
        tid_lst = [self[isite][0] for isite in iterator]
        parity = 0
        for itid in tid_lst:
            itsr, isite = self.tensor_order[itid]
            parity += itsr.parity
            shared_inds += list(oset(itsr.inds) & oset(tsr.inds))
            if move_left:
                self.tensor_order[itid] = (itsr, isite+1)
            else:
                self.tensor_order[itid] = (itsr, isite-1)
        global_parity = (parity % 2) * tsr.data.parity
        if global_parity != 0: tsr.data._global_flip()
        axes = [tsr.inds.index(i) for i in shared_inds]
        if len(axes)>0: tsr.data._local_flip(axes)
        self.tensor_order[tid] = (tsr, des_site)

    def move_past(self, tsr, site_range):
        start, end = site_range
        iterator = range(start, end)
        shared_inds = []
        tid_lst = [self[isite][0] for isite in iterator]
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

    def make_adjacent(self, tid_or_site1, tid_or_site2, direction='left'):
        """ Move one tensor in the specified direction to make the two adjacent
        """
        if not self.is_adjacent(tid_or_site1, tid_or_site2):
            site1 = self[tid_or_site1][1]
            site2 = self[tid_or_site2][1]
            if site1 == site2: return
            sitemin, sitemax = min(site1, site2), max(site1, site2)
            if direction == 'left':
                self.move(sitemax, sitemin+1)
            elif direction == 'right':
                self.move(sitemin, sitemax-1)
            else:
                raise ValueError("direction %s not recognized"%direction)

    def _contract_pairs(self, tid_or_site1, tid_or_site2, out_inds=None, direction='left', inplace=True):
        """ Contract two tensors in the FermionSpace

        Parameters
        ----------
        tid_or_site1 : string or int
            Tensor_id or position for the 1st tensor
        tid_or_site2 : string or int
            Tensor_id or position for the 2nd tensor
        out_inds: list of string, optional
            The order for the desired output indices
        direction: string
            The direction to move tensors if the two are not adjacent
        inplace: bool
            Whether to contract/move tensors inplace or in a copied fermionspace

        Returns
        -------
        out : a FermionTensor object or a number
        """
        fs = self if inplace else self.copy()
        out  = _contract_pairs(fs, tid_or_site1, tid_or_site2, out_inds, direction)

        if isinstance(out, (float, complex)):
            return out

        site1 = fs[tid_or_site1][1]
        site2 = fs[tid_or_site2][1]

        if direction=="left":
            site1 = min(site1, site2)
        else:
            site1 = max(site1, site2) - 1
        site2 = site1 + 1
        # the output fermion tensor will replace the two input tensors in the space
        fs.replace_tensor(site1, out, virtual=True)
        fs.remove_tensor(site2)

        return out

    def remove_tensor(self, tid_or_site, inplace=True):
        """ remove a specified tensor at a given site, eg
            012345               01234
            ABCDEF, (3, True) -> ABCEF

            012345                012345
            ABCDEF, (3, False) -> ABC-EF
        """
        tid, site, tsr = self[tid_or_site]
        tsr.remove_fermion_owner()
        del self.tensor_order[tid]
        if inplace:
            indent_sites = []
            for isite in self.sites:
                if isite > site:
                    indent_sites.append(isite)
            indent_sites = sorted(indent_sites)
            tid_lst = [self.get_tid(isite) for isite in indent_sites]
            for tid in tid_lst:
                tsr, site = self.tensor_order[tid]
                self.tensor_order[tid] = (tsr, site-1)

    def compress_space(self):
        """ if the space is not continously occupied, compress it, eg,
            012345678    01234
            -A--B-CDE -> ABCDE
        """
        sites = self.sites
        if min(sites) ==0 and self.is_continuous():
            return
        for tid, (tsr, site) in self.tensor_order.items():
            isite = sum(sites<site)
            self.tensor_order[tid] = (tsr, isite)

    def add_fermion_space(self, other, virtual=False, compress=False):
        """ Fuse two fermion spaces sequencially

        Parameters
        ----------
        other : FermionSpace obj
            The other FermionSpace to be appended
        virtual : bool
            If true, join the tensors in two fermionspace. Otherwise, copy the
            tensors in the first fermionspace and join
        compress: bool
            Whether to re-align the joint fermionspace to make all tensors
            adjacently placed
        """
        fs = self if virtual else self.copy()
        fs.append_fermion_space(other, virtual=False, compress=compress)
        return fs

    def append_fermion_space(self, other, virtual=False, compress=False):
        """ Append a fermion space right after

        Parameters
        ----------
        other : FermionSpace obj
            The other FermionSpace to be appended
        virtual : bool
            If true, join the tensors in two fermionspace. Otherwise, copy the
            tensors in the second fermionspace and join
        compress: bool
            Whether to re-align the joint fermionspace to make all tensors
            adjacently placed
        """
        if not self.is_continuous() or not other.is_continuous():
            raise ValueError("Not all Fermion Spaces are continuously occupied")

        sites = sorted(other.sites)
        for isite in sites:
            tid, site, tsr = other[isite]
            self.add_tensor(tsr, tid, virtual=virtual)

        if compress:
            self.compress_space()

    @property
    def H(self):
        """ Construct a FermionSpace for the ket state of the tensors
        """
        max_site = max(self.sites)
        new_fs = FermionSpace()
        for tid, (tsr, site) in self.tensor_order.items():
            T = tsr.copy()
            reverse_order = list(range(tsr.ndim))[::-1]
            x = T.data.permute(reverse_order)
            new_data = T.data.permute(reverse_order).conj()
            new_inds = T.inds[::-1]
            T.modify(data=new_data, inds=new_inds)
            new_fs.add_tensor(T, tid, max_site-site, virtual=True)

        return new_fs


class FermionTensor(Tensor):
    """A labelled, tagged ndarray. The index labels are used instead of
    axis numbers to identify dimensions, and are preserved through operations.

    Parameters
    ----------
    data : numpy.ndarray
        The n-dimensional data.
    inds : sequence of str
        The index labels for each dimension. Must match the number of
        dimensions of ``data``.
    tags : sequence of str, optional
        Tags with which to identify and group this tensor. These will
        be converted into a ``oset``.
    left_inds : sequence of str, optional
        Which, if any, indices to group as 'left' indices of an effective
        matrix. This can be useful, for example, when automatically applying
        unitary constraints to impose a certain flow on a tensor network but at
        the atomistic (Tensor) level.
    fermion_owner: a tuple with mixed data type, optional
        (hash value, fsobj weak reference, tensor_id). The first one is the hash
        value (int) of the fsobj it's point to. The second is the weak reference
        to the fsobj, and the third is the tensor_id(string) for its label
    """
    def __init__(self, data=1.0, inds=(), tags=None, left_inds=None, fermion_owner=None):

        # a new or copied Tensor always has no owners
        self.owners = dict()

        # Short circuit for copying Tensors
        if isinstance(data, self.__class__):
            self._data = data.data.copy()
            self._inds = data.inds
            self._tags = data.tags.copy()
            self._left_inds = data.left_inds
            # copied Fermion Tensor points to no fermion space
            self._fermion_owner = None
            return

        self._data = data # asarray(data)
        self._inds = tuple(inds)
        self._tags = tags_to_oset(tags)
        self._left_inds = tuple(left_inds) if left_inds is not None else None

        nd = ndim(self._data)
        if nd != len(self.inds):
            raise ValueError(
                f"Wrong number of inds, {self.inds}, supplied for array"
                f" of shape {self._data.shape}.")

        if self.left_inds and any(i not in self.inds for i in self.left_inds):
            raise ValueError(f"The 'left' indices {self.left_inds} are not "
                             f"found in {self.inds}.")

        self._fermion_owner = fermion_owner

    @property
    def fermion_owner(self):
        return self._fermion_owner

    @property
    def parity(self):
        return self.data.parity

    def ind_size(self, dim_or_ind):
        if isinstance(dim_or_ind, str):
            if dim_or_ind not in self.inds:
                raise ValueError("%s indice not found in the tensor"%dim_or_ind)
            dim_or_ind = self.inds.index(dim_or_ind)

        from pyblock3.algebra.symmetry import SZ, BondInfo
        sz = [SZ.from_flat(ix) for ix in self.data.q_labels[:,dim_or_ind]]
        sp = self.data.shapes[:,dim_or_ind]
        bond_dict = dict(zip(sz, sp))
        return BondInfo(bond_dict)

    def copy(self, deep=False):
        """Copy this tensor. Note by default (``deep=False``), the underlying
        array will *not* be copied. The fermion owner will to reset to None
        """
        if deep:
            t = copy.deepcopy(self)
            t.remove_fermion_owner()
        else:
            t = self.__class__(self, None)
        return t

    def multiply_index_diagonal(self, ind, x, inplace=False, location="front"):
        """Multiply this tensor by 1D array ``x`` as if it were a diagonal
        tensor being contracted into index ``ind``.
        """
        if location not in ["front", "back"]:
            raise ValueError("invalid for the location of the diagonal")
        t = self if inplace else self.copy()
        ax = t.inds.index(ind)
        if isinstance(x, FermionTensor):
            x = x.data
        if location=="front":
            out = np.tensordot(x, t.data, axes=((1,), (ax,)))
            transpose_order = list(range(1, ax+1)) + [0] + list(range(ax+1, t.ndim))
        else:
            out = np.tensordot(t.data, x, axes=((ax,),(0,)))
            transpose_order = list(range(ax)) + [t.ndim-1] + list(range(ax, t.ndim-1))
        data = np.transpose(out, transpose_order)
        t.modify(data=data)
        return t

    multiply_index_diagonal_ = functools.partialmethod(
        multiply_index_diagonal, inplace=True)

    def get_fermion_info(self):
        if self.fermion_owner is None:
            return None
        fs, tid = self.fermion_owner[1:]
        return (tid, fs().tensor_order[tid][1])

    def contract(self, *others, output_inds=None, **opts):
        return tensor_contract(self, *others, output_inds=output_inds, **opts)

    @fermion_owner.setter
    def fermion_owner(self, fowner):
        self._fermion_owner = fowner

    def set_fermion_owner(self, fs, tid):
        self.fermion_owner = (hash(fs), weakref.ref(fs), tid)

    def remove_fermion_owner(self):
        self.fermion_owner = None

    def isel(self, selectors, inplace=False):
        raise NotImplementedError

    def expand_ind(self, ind, size):
        raise NotImplementedError

    def new_ind(self, name, size=1, axis=0):
        raise NotImplementedError

    @property
    def shapes(self):
        return self._data.shapes

    @property
    def shape(self):
        """Return the "inflated" shape composed of maximal size for each leg
        """
        shapes = self.shapes
        return tuple(np.amax(shapes, axis=0))

    @functools.wraps(tensor_split)
    def split(self, *args, **kwargs):
        return tensor_split(self, *args, **kwargs)

    def transpose(self, *output_inds, inplace=False):
        """Transpose this tensor.

        Parameters
        ----------
        output_inds : sequence of str
            The desired output sequence of indices.
        inplace : bool, optional
            Perform the tranposition inplace.

        Returns
        -------
        tt : Tensor
            The transposed tensor.

        See Also
        --------
        transpose_like
        """
        t = self if inplace else self.copy()

        output_inds = tuple(output_inds)  # need to re-use this.

        if set(t.inds) != set(output_inds):
            raise ValueError("'output_inds' must be permutation of the current"
                             f" tensor indices, but {set(t.inds)} != "
                             f"{set(output_inds)}")

        current_ind_map = {ind: i for i, ind in enumerate(t.inds)}
        out_shape = tuple(current_ind_map[i] for i in output_inds)
        t.modify(apply=lambda x: np.transpose(x, out_shape), inds=output_inds)
        return t

    transpose_ = functools.partialmethod(transpose, inplace=True)

    @property
    def H(self):
        """Return the ket of this tensor, this is different from Fermionic transposition
            U_{abc} a^{\dagger}b^{\dagger}c^{\dagger} -> U^{cba\star}cba
        """
        axes = list(range(self.ndim))[::-1]
        data = self.data.permute(axes).conj()
        inds = self.inds[::-1]
        tsr = self.copy()
        tsr.modify(data=data, inds=inds)

        return tsr

    def fuse(self, fuse_map, inplace=False):
        raise NotImplementedError

    def unfuse(self, unfuse_map, shape_map, inplace=False):
        raise NotImplementedError

    def squeeze(self, inplace=False):
        raise NotImplementedError

    def norm(self):
        """Frobenius norm of this tensor.
        """
        return np.linalg.norm(self.data.data)

    def symmetrize(self, ind1, ind2, inplace=False):
        raise NotImplementedError

    def unitize(self, left_inds=None, inplace=False, method='qr'):
        raise NotImplementedError

    def randomize(self, dtype=None, inplace=False, **randn_opts):
        raise NotImplementedError

    def flip(self, ind, inplace=False):
        raise NotImplementedError

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

    def graph(self, *args, **kwargs):
        """Plot a graph of this tensor and its indices.
        """
        FermionTensorNetwork((self,)).graph(*args, **kwargs)


def is_mergeable(*ts_or_tsn):
    """Check if all objects(FermionTensor or FermionTensorNetwork)
       are part of the same FermionSpace
    """
    if isinstance(ts_or_tsn, (FermionTensor, FermionTensorNetwork)):
        return True
    fs_lst = []
    site_lst = []
    for obj in ts_or_tsn:
        if isinstance(obj, FermionTensor):
            if obj.fermion_owner is None:
                return False
            hashval, fsobj, tid = obj.fermion_owner
            fs_lst.append(hashval)
            site_lst.append(fsobj()[tid][1])
        elif isinstance(obj, FermionTensorNetwork):
            fs_lst.append(hash(obj.fermion_space))
            site_lst.extend(obj.filled_sites)
        else:
            raise TypeError("unable to find fermionspace")

    return all([fs==fs_lst[0] for fs in fs_lst]) and len(set(site_lst)) == len(site_lst)

class FermionTensorNetwork(TensorNetwork):


    def __init__(self, ts, *,  virtual=False, check_collisions=True):

        if is_mergeable(*ts) and virtual:
            self.tensor_map = dict()
            self.tag_map = dict()
            self.ind_map = dict()
            self.fermion_space = _fetch_fermion_space(*ts)[0]
            self.assemble(ts)
        else:
            if isinstance(ts, FermionTensorNetwork):
                self.tag_map = valmap(lambda tids: tids.copy(), ts.tag_map)
                self.ind_map = valmap(lambda tids: tids.copy(), ts.ind_map)
                self.fermion_space = ts.fermion_space if virtual else ts.fermion_space.copy()
                self.tensor_map = dict()
                for tid, t in ts.tensor_map.items():
                    self.tensor_map[tid] = self.fermion_space[tid][2]
                    self.tensor_map[tid].add_owner(self, tid)
                for ep in ts.__class__._EXTRA_PROPS:
                    setattr(self, ep, getattr(ts, ep))
                return

            # internal structure
            self.fermion_space = FermionSpace()
            self.tensor_map = dict()
            self.tag_map = dict()
            self.ind_map = dict()
            self._inner_inds = oset()
            for t in ts:
                self.add(t, virtual=virtual, check_collisions=check_collisions)
            self._inner_inds = None

    def __and__(self, other):
        """Combine this tensor network with more tensors, without contracting.
        Copies the tensors.
        """
        virtual = is_mergeable(self, other)
        return FermionTensorNetwork((self, other), virtual=virtual)

    def __or__(self, other):
        """Combine this tensor network with more tensors, without contracting.
        Views the constituent tensors.
        """
        return FermionTensorNetwork((self, other), virtual=True)

    def _reorder_from_tid(self, tid_map, inplace=False):
        tn = self if inplace else self.copy()
        tn.fermion_space._reorder_from_dict(tid_map)
        return tn

    def assemble_with_tensor(self, tsr):
        if not is_mergeable(self, tsr):
            raise ValueError("tensor not same in the fermion space of the tensor network")
        tid = tsr.fermion_owner[2]
        TensorNetwork.add_tensor(self, tsr, tid, virtual=True)

    def assemble_with_tensor_network(self, tsn):
        if not is_mergeable(self, tsn):
            raise ValueError("the two tensor networks not in the fermion space")
        TensorNetwork.add_tensor_network(self, tsn, virtual=True)

    def assemble(self, t):
        if isinstance(t, (tuple, list)):
            for each_t in t:
                self.assemble(each_t)
            return

        istensor = isinstance(t, FermionTensor)
        istensornetwork = isinstance(t, FermionTensorNetwork)

        if not (istensor or istensornetwork):
            raise TypeError("TensorNetwork should be called as "
                            "`TensorNetwork(ts, ...)`, where each "
                            "object in 'ts' is a Tensor or "
                            "TensorNetwork.")
        if istensor:
            self.assemble_with_tensor(t)
        else:
            self.assemble_with_tensor_network(t)

    def add_tensor(self, tsr, tid=None, virtual=False, site=None):
        if tid is None or tid in self.fermion_space.tensor_order.keys():
            tid = rand_uuid(base="_T")
        if virtual:
            fs  = tsr.fermion_owner
            if fs is not None:
                if fs[0] != hash(self.fermion_space):
                    raise ValueError("the tensor is already is in a different Fermion Space, \
                                      inplace addition not allowed")
                else:
                    if fs[2] in self.tensor_map.keys():
                        raise ValueError("the tensor is already in this TensorNetwork, \
                                          inplace addition not allowed")
                    else:
                        self.assemble_with_tensor(tsr)
            else:
                self.fermion_space.add_tensor(tsr, tid, site, virtual=True)
                TensorNetwork.add_tensor(self, tsr, tid, virtual=True)
        else:
            T = tsr.copy()
            self.fermion_space.add_tensor(T, tid, site, virtual=True)
            TensorNetwork.add_tensor(self, T, tid, virtual=True)

    def add_tensor_network(self, tn, virtual=False, check_collisions=True):
        if virtual:
            if hash(tn.fermion_space) == hash(self.fermion_space):
                if is_mergeable(self, tn):
                    TensorNetwork.add_tensor_network(tn, virtual=virtual, check_collisions=check_collisions)
                else:
                    raise ValueError("the two tensornetworks co-share same sites, inplace addition not allow")
                return

        if not tn.is_continuous():
            raise ValueError("input tensor network is not contiguously ordered")

        filled_sites = tn.filled_sites
        sorted_sites = sorted(filled_sites)

        if check_collisions:  # add tensors individually
            if getattr(self, '_inner_inds', None) is None:
                self._inner_inds = oset(self.inner_inds())

            # check for matching inner_indices -> need to re-index
            other_inner_ix = oset(tn.inner_inds())
            clash_ix = self._inner_inds & other_inner_ix

            if clash_ix:
                can_keep_ix = other_inner_ix - self._inner_inds
                new_inds = oset(rand_uuid() for _ in range(len(clash_ix)))
                reind = dict(zip(clash_ix, new_inds))
                self._inner_inds.update(new_inds, can_keep_ix)
            else:
                self._inner_inds.update(other_inner_ix)

            # add tensors, reindexing if necessary
            for site in sorted_sites:
                tid, _, tsr = tn.fermion_space[site]
                if clash_ix and any(i in reind for i in tsr.inds):
                    tsr = tsr.reindex(reind, inplace=virtual)
                self.add_tensor(tsr, tid=tid, virtual=virtual)

        else:  # directly add tensor/tag indexes
            for site in sorted_sites:
                tid, _, tsr = tn.fermion_space[site]
                self.add_tensor(tsr, tid=tid, virtual=virtual)

    def add(self, t, virtual=False, check_collisions=True):
        """Add FermionTensor, FermionTensorNetwork or sequence thereof to self.
        """
        if isinstance(t, (tuple, list)):
            for each_t in t:
                self.add(each_t, virtual=virtual,
                         check_collisions=check_collisions)
            return

        istensor = isinstance(t, FermionTensor)
        istensornetwork = isinstance(t, FermionTensorNetwork)

        if not (istensor or istensornetwork):
            raise TypeError("TensorNetwork should be called as "
                            "`TensorNetwork(ts, ...)`, where each "
                            "object in 'ts' is a Tensor or "
                            "TensorNetwork.")

        if istensor:
            self.add_tensor(t, virtual=virtual)
        else:
            self.add_tensor_network(t, virtual=virtual,
                                    check_collisions=check_collisions)

    def select(self, tags, which='all'):

        tagged_tids = self._get_tids_from_tags(tags, which=which)
        ts = [self.tensor_map[n] for n in tagged_tids]
        tn = FermionTensorNetwork(ts, check_collisions=False, virtual=True)
        tn.view_like_(self)
        return tn

    def __iand__(self, tensor):
        """Inplace, but non-virtual, addition of a Tensor or TensorNetwork to
        this network. It should not have any conflicting indices.
        """
        if is_mergeable(self, tensor):
            self.assemble(tensor)
        else:
            self.add(tensor, virtual=False)
        return self

    def __ior__(self, tensor):
        """Inplace, virtual, addition of a Tensor or TensorNetwork to this
        network. It should not have any conflicting indices.
        """
        self.add(tensor, virtual=True)
        return self

    # ------------------------------- Methods ------------------------------- #

    @property
    def filled_sites(self):
        return [self.fermion_space[tid][1] for tid in self.tensor_map.keys()]

    def is_complete(self):
        '''
        Check if the current tensor network contains all the tensors in the fermion space
        '''
        full_tid = self.fermion_space.tensor_order.keys()
        tensor_tid = self.tensor_map.keys()
        return set(full_tid) == set(tensor_tid)

    def is_continuous(self):
        """
        Check if sites in the current tensor network are contiguously occupied
        """
        filled_sites = self.filled_sites
        if len(filled_sites) ==0 : return True
        return (max(filled_sites) - min(filled_sites) + 1) == len(filled_sites)

    def copy(self):
        """ Tensors and underlying FermionSpace(all tensors in it) will
            be copied
        """
        return self.__class__(self, virtual=False)

    def simple_copy(self):
        newtn = FermionTensorNetwork([])
        newtn.add_tensor_network(self)
        newtn.view_like_(self)
        return newtn

    def _pop_tensor(self, tid, remove_from_fs=True):
        """Remove a tensor from this network, returning said tensor.
        """
        # pop the tensor itself
        t = self.tensor_map.pop(tid)

        # remove the tid from the tag and ind maps
        self._remove_tid(t.tags, self.tag_map, tid)
        self._remove_tid(t.inds, self.ind_map, tid)

        # remove this tensornetwork as an owner
        t.remove_owner(self)
        if remove_from_fs:
            self.fermion_space.remove_tensor(tid)
            t.remove_fermion_owner()

        return t


    _pop_tensor_ = functools.partialmethod(_pop_tensor, remove_from_fs=False)

    @property
    def H(self):
        tn = self.copy()
        fs = tn.fermion_space
        max_site = max(fs.sites)

        for tid, (tsr, site) in fs.tensor_order.items():
             reverse_order = list(range(tsr.ndim))[::-1]
             new_data = tsr.data.permute(reverse_order).conj()
             new_inds = tsr.inds[::-1]
             tsr.modify(data=new_data, inds=new_inds)
             fs.tensor_order.update({tid: (tsr, max_site-site)})
        return tn

    def __mul__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError

    def __imul__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __itruediv__(self, other):
        raise NotImplementedError

    # ----------------- selecting and splitting the network ----------------- #


    def __setitem__(self, tags, tensor):
        #TODO: FIXME
        """Set the single tensor uniquely associated with ``tags``.
        """
        tids = self._get_tids_from_tags(tags, which='all')
        if len(tids) != 1:
            raise KeyError("'TensorNetwork.__setitem__' is meant for a single "
                           "existing tensor only - found {} with tag(s) '{}'."
                           .format(len(tids), tags))

        if not isinstance(tensor, Tensor):
            raise TypeError("Can only set value with a new 'Tensor'.")

        tid, = tids
        site = self.fermion_space.tensor_order[tid][1]
        TensorNetwork._pop_tensor(tid)
        TensorNetwork.add_tensor(tensor, tid=tid, virtual=True)
        self.fermion_space.replace_tensor(site, tensor, tid=tid, virtual=True)

    def partition_tensors(self, tags, inplace=False, which='any'):
        """Split this TN into a list of tensors containing any or all of
        ``tags`` and a ``TensorNetwork`` of the the rest.

        Parameters
        ----------
        tags : sequence of str
            The list of tags to filter the tensors by. Use ``...``
            (``Ellipsis``) to filter all.
        inplace : bool, optional
            If true, remove tagged tensors from self, else create a new network
            with the tensors removed.
        which : {'all', 'any'}
            Whether to require matching all or any of the tags.

        Returns
        -------
        (u_tn, t_ts) : (TensorNetwork, tuple of Tensors)
            The untagged tensor network, and the sequence of tagged Tensors.

        See Also
        --------
        partition, select, select_tensors
        """
        tagged_tids = self._get_tids_from_tags(tags, which=which)

        # check if all tensors have been tagged
        if len(tagged_tids) == self.num_tensors:
            return None, self.tensor_map.values()

        # Copy untagged to new network, and pop tagged tensors from this
        untagged_tn = self if inplace else self.copy()
        tagged_ts = tuple(map(untagged_tn._pop_tensor_, sorted(tagged_tids)))

        return untagged_tn, tagged_ts

    def partition(self, tags, which='any', inplace=False):
        """Split this TN into two, based on which tensors have any or all of
        ``tags``. Unlike ``partition_tensors``, both results are TNs which
        inherit the structure of the initial TN.

        Parameters
        ----------
        tags : sequence of str
            The tags to split the network with.
        which : {'any', 'all'}
            Whether to split based on matching any or all of the tags.
        inplace : bool
            If True, actually remove the tagged tensors from self.

        Returns
        -------
        untagged_tn, tagged_tn : (TensorNetwork, TensorNetwork)
            The untagged and tagged tensor networs.

        See Also
        --------
        partition_tensors, select, select_tensors
        """
        tagged_tids = self._get_tids_from_tags(tags, which=which)

        kws = {'check_collisions': False}

        if inplace:
            t1 = self
            t2s = [t1._pop_tensor_(tid) for tid in tagged_tids]
            t2 = FermionTensorNetwork(t2s, **kws)
            t2.view_like_(self)

        else:  # rebuild both -> quicker
            new_fs = self.fermion_space.copy()
            t1_site = []
            t2_site = []
            for tid in self.tensor_map.keys():
                (t2_site if tid in tagged_tids else t1_site).append(self.fermion_space[tid][1])
            t1 = new_fs.to_tensor_network(t1_site)
            t2 = new_fs.to_tensor_network(t2_site)
            t1.view_like_(self)
            t2.view_like_(self)

        return t1, t2

    def replace_with_svd(self, where, left_inds, eps, *, which='any',
                         right_inds=None, method='isvd', max_bond=None,
                         absorb='both', cutoff_mode='rel', renorm=None,
                         ltags=None, rtags=None, keep_tags=True,
                         start=None, stop=None, inplace=False):
        r"""Replace all tensors marked by ``where`` with an iteratively
        constructed SVD. E.g. if ``X`` denote ``where`` tensors::

                                    :__       ___:
            ---X  X--X  X---        :  \     /   :
               |  |  |  |      ==>  :   U~s~VH---:
            ---X--X--X--X---        :__/     \   :
                  |     +---        :         \__:
                  X              left_inds       :
                                             right_inds

        Parameters
        ----------
        where : tag or seq of tags
            Tags specifying the tensors to replace.
        left_inds : ind or sequence of inds
            The indices defining the left hand side of the SVD.
        eps : float
            The tolerance to perform the SVD with, affects the number of
            singular values kept. See
            :func:`quimb.linalg.rand_linalg.estimate_rank`.
        which : {'any', 'all', '!any', '!all'}, optional
            Whether to replace tensors matching any or all the tags ``where``,
            prefix with '!' to invert the selection.
        right_inds : ind or sequence of inds, optional
            The indices defining the right hand side of the SVD, these can be
            automatically worked out, but for hermitian decompositions the
            order is important and thus can be given here explicitly.
        method : str, optional
            How to perform the decomposition, if not an iterative method
            the subnetwork dense tensor will be formed first, see
            :func:`~quimb.tensor.tensor_core.tensor_split` for options.
        max_bond : int, optional
            The maximum bond to keep, defaults to no maximum (-1).
        ltags : sequence of str, optional
            Tags to add to the left tensor.
        rtags : sequence of str, optional
            Tags to add to the right tensor.
        keep_tags : bool, optional
            Whether to propagate tags found in the subnetwork to both new
            tensors or drop them, defaults to ``True``.
        start : int, optional
            If given, assume can use ``TNLinearOperator1D``.
        stop :  int, optional
            If given, assume can use ``TNLinearOperator1D``.
        inplace : bool, optional
            Perform operation in place.

        Returns
        -------

        See Also
        --------
        replace_with_identity
        """
        leave, svd_section = self.partition(where, which=which,
                                            inplace=inplace)

        tags = svd_section.tags if keep_tags else oset()
        ltags = tags_to_oset(ltags)
        rtags = tags_to_oset(rtags)

        if right_inds is None:
            # compute
            right_inds = tuple(i for i in svd_section.outer_inds()
                               if i not in left_inds)

        if (start is None) and (stop is None):
            A = svd_section.aslinearoperator(left_inds=left_inds,
                                             right_inds=right_inds)
        else:
            from .tensor_1d import TNLinearOperator1D

            # check if need to invert start stop as well
            if '!' in which:
                start, stop = stop, start + self.L
                left_inds, right_inds = right_inds, left_inds
                ltags, rtags = rtags, ltags

            A = TNLinearOperator1D(svd_section, start=start, stop=stop,
                                   left_inds=left_inds, right_inds=right_inds)

        ltags = tags | ltags
        rtags = tags | rtags

        TL, TR = tensor_split(A, left_inds=left_inds, right_inds=right_inds,
                              method=method, cutoff=eps, absorb=absorb,
                              max_bond=max_bond, cutoff_mode=cutoff_mode,
                              renorm=renorm, ltags=ltags, rtags=rtags)

        leave |= TL
        leave |= TR

        return leave

    def contract_between(self, tags1, tags2, **contract_opts):
        """Contract the two tensors specified by ``tags1`` and ``tags2``
        respectively. This is an inplace operation. No-op if the tensor
        specified by ``tags1`` and ``tags2`` is the same tensor.

        Parameters
        ----------
        tags1 :
            Tags uniquely identifying the first tensor.
        tags2 : str or sequence of str
            Tags uniquely identifying the second tensor.
        contract_opts
            Supplied to :func:`~quimb.tensor.tensor_core.tensor_contract`.
        """
        tid1, = self._get_tids_from_tags(tags1, which='all')
        tid2, = self._get_tids_from_tags(tags2, which='all')
        direction = contract_opts.pop("direction", "left")

        # allow no-op for same tensor specified twice ('already contracted')
        if tid1 == tid2:
            return

        self._pop_tensor_(tid1)
        self._pop_tensor_(tid2)

        out = self.fermion_space._contract_pairs(tid1, tid2, direction=direction, inplace=True)
        if isinstance(out, (float, complex)):
            return out
        else:
            self |= out

    def contract_ind(self, ind, **contract_opts):
        """Contract tensors connected by ``ind``.
        """
        tids = self._get_tids_from_inds(ind)
        if len(tids) <= 1: return
        ts = [self._pop_tensor_(tid) for tid in tids]
        direction = contract_opts.pop("direction", "left")
        out = tensor_contract(*ts, direction=direction, inplace=True)
        if isinstance(out, (float, complex)):
            return out
        else:
            self |= out

    def contract_tags(self, tags, inplace=False, which='any', **opts):

        tids = self._get_tids_from_tags(tags, which='any')
        if len(tids)  == 0:
            raise ValueError("No tags were found - nothing to contract. "
                             "(Change this to a no-op maybe?)")
        elif len(tids) == 1:
            return self

        untagged_tn, tagged_ts = self.partition_tensors(
            tags, inplace=inplace, which=which)


        contracted = tensor_contract(*tagged_ts, inplace=True, **opts)

        if untagged_tn is None:
            return contracted

        untagged_tn.add_tensor(contracted, virtual=True)
        return untagged_tn

    def _compress_between_tids(
        self,
        tid1,
        tid2,
        canonize_distance=None,
        canonize_opts=None,
        equalize_norms=False,
        **compress_opts
    ):

        Tl = self.tensor_map[tid1]
        Tr = self.tensor_map[tid2]
        tensor_compress_bond(Tl, Tr, inplace=True, **compress_opts)

        if equalize_norms:
            raise NotImplementedError

    def _canonize_between_tids(
        self,
        tid1,
        tid2,
        equalize_norms=False,
        **canonize_opts,
    ):
        Tl = self.tensor_map[tid1]
        Tr = self.tensor_map[tid2]
        tensor_canonize_bond(Tl, Tr, **canonize_opts)

        if equalize_norms:
            raise NotImplementedError

    def replace_section_with_svd(self, start, stop, eps,
                                 **replace_with_svd_opts):
        raise NotImplementedError

    def convert_to_zero(self):
        raise NotImplementedError

    def compress_all(self, inplace=False, **compress_opts):
        raise NotImplementedError

    def new_bond(self, tags1, tags2, **opts):
        raise NotImplementedError

    def cut_bond(self, bnd, left_ind, right_ind):
        raise NotImplementedError

    def cut_between(self, left_tags, right_tags, left_ind, right_ind):
        raise NotImplementedError


    def cut_iter(self, *inds):
        raise NotImplementedError
