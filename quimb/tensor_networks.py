import os
import functools
from functools import reduce
import operator
from operator import or_

from cytoolz import concat, frequencies
import numpy as np

from .accel import prod, make_immutable
from .gen.operators import spin_operator, eye

try:
    # opt_einsum is highly recommended as until numpy 1.14 einsum contractions
    # do not use BLAS.
    import opt_einsum
    einsum = opt_einsum.contract

    @functools.wraps(opt_einsum.contract_path)
    def einsum_path(*args, optimize='greedy', memory_limit=2**28, **kwargs):
        return opt_einsum.contract_path(
            *args, path=optimize, memory_limit=memory_limit, **kwargs)

except ImportError:

    @functools.wraps(np.einsum)
    def einsum(*args, optimize='greedy', memory_limit=2**28, **kwargs):

        if optimize is False:
            return np.einsum(*args, optimize=False, **kwargs)

        explicit_path = (isinstance(optimize, (tuple, list)) and
                         optimize[0] == 'einsum_path')

        if explicit_path:
            optimize = optimize
        else:
            optimize = (optimize, memory_limit)

        return np.einsum(*args, optimize=optimize, **kwargs)

    @functools.wraps(np.einsum_path)
    def einsum_path(*args, optimize='greedy', memory_limit=2**28, **kwargs):
        return np.einsum_path(
            *args, optimize=(optimize, memory_limit), **kwargs)


# --------------------------------------------------------------------------- #
#                                Tensor Class                                 #
# --------------------------------------------------------------------------- #

class Tensor(object):
    """A labelled, tagged ndarray.

    Parameters
    ----------
    array : numpy.ndarray
        The n-dimensions data.
    inds : sequence of hashable
        The index labels for each dimension.
    tags : sequence of hashable
        Tags with which to select and filter from multiple tensors.
    """

    def __init__(self, array, inds, tags=None):
        self.array = np.asarray(array)
        self.inds = tuple(inds)

        if self.array.ndim != len(self.inds):
            raise ValueError(
                "Wrong number of inds, {}, supplied for array"
                " of shape {}.".format(self.inds, self.array.shape))

        self.tags = (set() if tags is None else
                     {tags} if isinstance(tags, str) else
                     set(tags))

    def conj(self):
        """Conjugate this tensors data (does nothing to indices).
        """
        return Tensor(self.array.conj(), self.inds, self.tags)

    @property
    def H(self):
        """Conjugate this tensors data (does nothing to indices).
        """
        return self.conj()

    @property
    def shape(self):
        return self.array.shape

    @property
    def size(self):
        return self.array.size

    def tranpose(self, output_inds):
        """Transpose this tensor.

        Parameters
        ----------
        output_inds : sequence of hashable
            The desired output sequence of indices.

        Returns
        -------
        Tensor
        """
        return tensor_tranpose(self, output_inds)

    def contract(self, *others,
                 memory_limit=2**28,
                 optimize='greedy',
                 output_inds=None):
        """Efficiently contract multiple tensors, combining their tags.

        Parameters
        ----------
        *others : sequence of Tensor
            The other tensors to contract with.
        memory_limit : int, optional
            See :py:func:`contract`.
        optimize : str, optional
            See :py:func:`contract`.
        output_inds : sequence
            If given, the desired order of output indices, else defaults to the
            order they occur in the input indices.

        Returns
        -------
        scalar or Tensor
        """
        return tensor_contract(self, *others, memory_limit=memory_limit,
                               optimize=optimize, output_inds=output_inds)

    def fuse(self, fuse_map):
        """Combine groups of indices into single indices.

        Parameters
        ----------
        fuse_map : dict_like
            Mapping like: ``{new_ind: sequence of existing inds, ...}``.

        Returns
        -------
        Tensor
            The tranposed, reshaped and re-labeled tensor.
        """
        fuseds = tuple(fuse_map.values())  # groups of indices to be fused
        unfused_inds = tuple(i for i in self.inds if not
                             any(i in fs for fs in fuseds))

        # transpose tensor to bring groups of fused inds to the beginning
        TT = self.tranpose(output_inds=concat(fuseds + unfused_inds))

        # for each set of fused dims, group into product, then add remaining
        dims = iter(TT.shape)
        dims = [prod(next(dims) for _ in fs) for fs in fuseds] + list(dims)

        # create new tensor with new + remaining indices
        new_inds = concat((fuse_map.keys(), unfused_inds))
        return Tensor(TT.array.reshape(*dims), new_inds, self.tags)

    def __and__(self, other):
        """Combine with another ``Tensor`` or ``TensorNetwork`` into a new
        ``TensorNetwork``.
        """
        return TensorNetwork(self, other)

    def __matmul__(self, other):
        """Explicitly contract with another tensor.
        """
        return tensor_contract(self, other)

    def __repr__(self):
        return "Tensor(shape={}, inds={}, tags={})".format(
            self.array.shape,
            self.inds,
            self.tags)


# ------------------------- Add ufunc like methods -------------------------- #

def _make_promote_array_func(op, meth_name):

    @functools.wraps(getattr(np.ndarray, meth_name))
    def _promote_array_func(self, other):
        """Use standard array func, but make sure Tensor inds match.
        """
        if isinstance(other, Tensor):
            if self.inds != other.inds:
                raise ValueError(
                    "The indicies of these two tensors do not "
                    "match: {} != {}".format(self.inds, other.inds))
            return Tensor(
                array=op(self.array, other.array), inds=self.inds,
                tags=self.tags | other.tags)
        return Tensor(array=op(self.array, other),
                      inds=self.inds, tags=self.tags)

    return _promote_array_func


for meth_name, op in [('__add__', operator.__add__),
                      ('__sub__', operator.__sub__),
                      ('__mul__', operator.__mul__),
                      ('__pow__', operator.__pow__),
                      ('__truediv__', operator.__truediv__)]:
    setattr(Tensor, meth_name, _make_promote_array_func(op, meth_name))


def _make_rhand_array_promote_func(op, meth_name):

    @functools.wraps(getattr(np.ndarray, meth_name))
    def _rhand_array_promote_func(self, other):
        """Right hand operations -- no need to check ind equality first.
        """
        return Tensor(array=op(other, self.array),
                      inds=self.inds, tags=self.tags)

    return _rhand_array_promote_func


for meth_name, op in [('__radd__', operator.__add__),
                      ('__rsub__', operator.__sub__),
                      ('__rmul__', operator.__mul__),
                      ('__rpow__', operator.__pow__),
                      ('__rtruediv__', operator.__truediv__)]:
    setattr(Tensor, meth_name, _make_rhand_array_promote_func(op, meth_name))


# --------------------------------------------------------------------------- #
#                                Tensor Funcs                                 #
# --------------------------------------------------------------------------- #

def tensor_tranpose(tensor, output_inds):
    """Tranpose a tensor.

    Parameters
    ----------
    tensor : Tensor
        The tensor to tranpose.
    output_inds : sequence of hashable
        The desired output sequence of indices.

    Returns
    -------
    Tensor
    """
    output_inds = tuple(output_inds)  # need to re-use this.

    if set(tensor.inds) != set(output_inds):
        raise ValueError("'output_inds' must be permutation of the "
                         "current tensor indices.")

    current_ind_map = {ind: i for i, ind in enumerate(tensor.inds)}
    out_shape = map(current_ind_map.__getitem__, output_inds)

    return Tensor(tensor.array.transpose(*out_shape),
                  inds=output_inds, tags=tensor.tags)


def _gen_output_inds(all_inds):
    """Generate the output, i.e. unnique, indices from the set ``inds``. Raise
    if any index found more than twice.
    """
    for ind, freq in frequencies(all_inds).items():
        if freq > 2:
            raise ValueError("The index {} appears more "
                             "than twice!".format(ind))
        elif freq == 1:
            yield ind


_einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
_einsum_symbols_set = set(_einsum_symbols)


def _maybe_map_indices_to_alphabet(a_ix, i_ix, o_ix):
    """``einsum`` need characters a-z,A-Z or equivalent numbers.
    Do this early, and allow *any* index labels.

    Parameters
    ----------
    a_ix : sequence
        All of the input indices.
    i_ix : sequence of sequence
        The input indices per tensor.
    o_ix : list of int
        The output indices.

    Returns
    -------
    contract_str : str
        The string to feed to einsum/contract.
    """
    if any(i not in _einsum_symbols_set for i in a_ix):
        # need to map inds to alphabet
        amap = {i: lett for i, lett in zip(set(a_ix), _einsum_symbols)}
        in_str = map(lambda x: "".join(map(amap.__getitem__, x)), i_ix)
        out_str = "".join(map(amap.__getitem__, o_ix))
    else:
        in_str = map("".join, i_ix)
        out_str = "".join(o_ix)

    return ",".join(in_str) + "->" + out_str


def tensor_contract(*tensors,
                    memory_limit=2**28,
                    optimize='greedy',
                    output_inds=None):
    """Efficiently contract multiple tensors, combining their tags.

    Parameters
    ----------
    *tensors : sequence of Tensor
        The tensors to contract.
    memory_limit : int, optional
        See :py:func:`contract`.
    optimize : str, optional
        See :py:func:`contract`.
    output_inds : sequence
        If given, the desired order of output indices, else defaults to the
        order they occur in the input indices.

    Returns
    -------
    scalar or Tensor
    """
    i_ix = [t.inds for t in tensors]  # input indices per tensor
    a_ix = list(concat(i_ix))  # list of all input indices

    if output_inds is None:
        # sort output indices  by input order for efficiency and consistency
        o_ix = tuple(filter(lambda x: x in set(_gen_output_inds(a_ix)), a_ix))
    else:
        o_ix = output_inds

    # possibly map indices into the 0-52 range needed by einsum
    contract_str = _maybe_map_indices_to_alphabet(a_ix, i_ix, o_ix)

    # perform the contraction
    o_array = einsum(contract_str, *(t.array for t in tensors),
                     memory_limit=memory_limit, optimize=optimize)

    if not o_ix:
        return o_array

    # unison of all tags
    o_tags = reduce(or_, (t.tags for t in tensors))

    return Tensor(array=o_array, inds=o_ix, tags=o_tags)


def _gen_rand_uuids():
    """Generate shortish identifiers which are guaranteed unique.
    Will break if more than ~10**12.4 required.
    """
    import uuid
    used = set()
    while True:
        s = str(uuid.uuid4())[:8]
        while s in used:  # pragma: no cover
            s = str(uuid.uuid4())[:8]
        used.add(s)
        yield s


RAND_UUIDS = iter(_gen_rand_uuids())


def rand_uuid(base=""):
    """Return a guaranteed unique, shortish identifier, optional appended
    to ``base``.

    Examples
    --------
    >>> rand_uuid()
    '_2e1dae1b'

    >>> rand_uuid('virt-bond')
    'virt-bond_bf342e68'
    """
    return base + "_" + next(RAND_UUIDS)


# --------------------------------------------------------------------------- #
#                            Tensor Network Class                             #
# --------------------------------------------------------------------------- #

class TensorNetwork(object):
    """A collection of (as yet uncontracted) Tensors.

    Parameters
    ----------
    *tensors : sequence of Tensor or TensorNetwork
        The objects to combine.

    Members
    -------
    tensors : sequence of Tensor
        The tensors in this network.
    """

    def __init__(self, *tensors):
        self.tensors = []
        for t in tensors:
            if isinstance(t, Tensor):
                self.tensors.append(t)
            elif isinstance(t, TensorNetwork):
                self.tensors += t.tensors
            else:
                raise TypeError("TensorNetwork should be called as "
                                "``TensorNetwork(*tensors)``, where all "
                                "arguments are Tensors or TensorNetworks.")

    def conj(self):
        """Conjugate all the tensors in this network (leaves all indices).
        """
        return TensorNetwork(*[t.conj() for t in self.tensors])

    @property
    def H(self):
        """Conjugate all the tensors in this network (leaves all indices).
        """
        return self.conj()

    def split(self, tags):
        """Split this TN into a list of tensors containing any of ``tags`` and
        the rest.

        Parameters
        ----------
        tags : sequence of hashable
            The list of tags to filter the tensors by.

        Returns
        -------
        (untagged, tagged) : (list of Tensor, list of Tensor)
            The list of untagged and tagged tensors.
        """
        # contract all
        if tags is ...:
            return [], self.tensors

        if isinstance(tags, str):
            tags = {tags}

        untagged, tagged = [], []
        for t in self.tensors:
            if any(tag in t.tags for tag in tags):
                tagged.append(t)
            else:
                untagged.append(t)

        return untagged, tagged

    def contract(self, tags=...):
        """Contract some, or all, of the tensors in this network.

        Parameters
        ----------
        tags : sequence of hashable
            Any tensors with any of these tags with be contracted. Set to
            ``...`` (``Ellipsis``) to contract all tensors, the default.

        Returns
        -------
        TensorNetwork, Tensor or Scalar
            The result of the contraction, still a TensorNetwork if the
            contraction was only partial.
        """
        untagged, tagged = self.split(tags)

        if untagged:
            return TensorNetwork(*untagged) & tensor_contract(*tagged)

        return tensor_contract(*tagged)

    def cumulative_contract(self, tags_seq):
        """Cumulative contraction of tensor network. Contract the first set of
        tags, then this set with the next set, then both of these the next and
        so forth.

        Parameters
        ----------
        tags_seq : sequence of sequence of hashable
            The list of set of tags to cumulatively contract.

        Returns
        -------
        TensorNetwork, Tensor or Scalar
            The result of the contraction, still a TensorNetwork if the
            contraction was only partial.
        """
        new_tn = self
        ctags = set()

        for tags in tags_seq:
            # accumulate tags from each contractions
            ctags |= {tags} if isinstance(tags, str) else set(tags)
            # peform the next contraction
            new_tn ^= ctags

        return new_tn

    @property
    def shape(self):
        """Actual, i.e. exterior, shape of this TensorNetwork.
        """
        dims, inds = zip(*concat(zip(t.shape, t.inds) for t in self.tensors))
        ind_freqs = frequencies(inds)
        return tuple(d for d, i in zip(dims, inds) if ind_freqs[i] == 1)

    def __xor__(self, *args, **kwargs):
        """Overload of '^' for TensorNetwork.contract.
        """
        return self.contract(*args, **kwargs)

    def __ixor__(self, *args, **kwargs):
        """Overload of '^=' for inplace TensorNetwork.contract.
        """
        return self.contract(*args, **kwargs)

    def __rshift__(self, *args, **kwargs):
        """Overload of '>>' for TensorNetwork.cumulative_contract.
        """
        return self.cumulative_contract(*args, **kwargs)

    def __irshift__(self, *args, **kwargs):
        """Overload of '>>=' for inplace TensorNetwork.cumulative_contract.
        """
        return self.cumulative_contract(*args, **kwargs)

    def __and__(self, other):
        """Combine this tensor network with more tensors, without contracting.
        """
        return TensorNetwork(self, other)

    def __repr__(self):
        return "TensorNetwork({})".format(
            repr(self.tensors))

    def __str__(self):
        return "TensorNetwork([{}{}{}{}])".format(
            os.linesep,
            "".join(["    " + repr(t) + os.linesep
                     for t in self.tensors[:-1]]),
            "    " + repr(self.tensors[-1]),
            os.linesep)


# --------------------------------------------------------------------------- #
#                          Specific forms on network                          #
# --------------------------------------------------------------------------- #

def matrix_product_state(*arrays, shape='lrp', site_inds=None, bond_name=""):
    """Initialise a matrix product state, with auto labelling and tagging.

    Parameters
    ----------
    *arrays : sequence of arrays
        The tensor arrays to form into a MPS.
    site_inds : sequence of hashable, or str
        The indices to label the physical dimensions with, if a string is
        supplied, use this to format the indices thus:
        ``map(site_inds.format, range(len(arrays)))``.
        Defaults ``'k0', 'k1', 'k2'...``.
    shape : str
        String specifying layout of the tensors. E.g. 'lrp' (the default)
        indicates the shape corresponds left-bond, right-bond, physical index.
        End tensors have either 'l' or 'r' dropped from the string.
    bond_name : str, optional
        The base name of the bond indices, onto which uuids will be added.

    Returns
    -------
    TensorNetwork
    """
    arrays = tuple(arrays)
    nsites = len(arrays)

    # process site indices
    if site_inds is None:
        site_inds = 'k{}'

    if isinstance(site_inds, str):
        site_inds = tuple(map(site_inds.format, range(nsites)))
    else:
        site_inds = tuple(site_inds)

    # TODO: process tags
    # TODO: figure out cyclic or not
    # TODO: allow open ends non-cyclic

    lp_ord = tuple(map(lambda x: shape.replace('r', "").find(x), 'lp'))
    rp_ord = tuple(map(lambda x: shape.replace('l', "").find(x), 'rp'))
    lrp_ord = tuple(map(shape.find, 'lrp'))  # transpose arrays to 'lrp' order.

    tensors = []
    next_bond = rand_uuid(base=bond_name)

    # Do the first tensor seperately.
    tensors.append(Tensor(array=arrays[0].transpose(*lp_ord),
                          inds=[next_bond, site_inds[0]]))
    previous_bond = next_bond

    # Range over the middle tensors
    for array, site_ind in zip(arrays[1:-1], site_inds[1:-1]):
        next_bond = rand_uuid(base=bond_name)
        tensors.append(Tensor(array=array.transpose(*lrp_ord),
                              inds=[previous_bond, next_bond, site_ind]))
        previous_bond = next_bond

    # Do the last tensor seperately.
    tensors.append(Tensor(array=arrays[-1].transpose(*rp_ord),
                          inds=[previous_bond, site_inds[-1]]))

    return TensorNetwork(*tensors)


def matrix_product_operator(*arrays, shape='lrkb',
                            ket_site_inds=None,
                            bra_site_inds=None,
                            bond_name=""):
    """Initialise a matrix product operator, with auto labelling and tagging.

    Parameters
    ----------
    *arrays : sequence of arrays
        The tensor arrays to form into a MPO.
    ket_site_inds : sequence of hashable, or str
        The indices to label the ket physical dimensions with, if a string is
        supplied, use this to format the indices thus:
        ``map(ket_site_inds.format, range(len(arrays)))``.
        Defaults ``'k0', 'k1', 'k2'...``.
    bra_site_inds : sequence of hashable, or str
        The indices to label the ket physical dimensions with, if a string is
        supplied, use this to format the indices thus:
        ``map(bra_site_inds.format, range(len(arrays)))``.
        Defaults ``'b0', 'b1', 'b2'...``.
    shape : str
        String specifying layout of the tensors. E.g. 'lrkb' (the default)
        indicates the shape corresponds left-bond, right-bond, ket physical
        index, bra physical index.
        End tensors have either 'l' or 'r' dropped from the string.
    bond_name : str, optional
        The base name of the bond indices, onto which uuids will be added.

    Returns
    -------
    TensorNetwork
    """
    arrays = tuple(arrays)
    nsites = len(arrays)

    # process ket site indices
    if ket_site_inds is None:
        ket_site_inds = 'k{}'

    if isinstance(ket_site_inds, str):
        ket_site_inds = tuple(map(ket_site_inds.format, range(nsites)))
    else:
        ket_site_inds = tuple(ket_site_inds)

    # process bra site indices
    if bra_site_inds is None:
        bra_site_inds = 'b{}'

    if isinstance(bra_site_inds, str):
        bra_site_inds = tuple(map(bra_site_inds.format, range(nsites)))
    else:
        bra_site_inds = tuple(bra_site_inds)

    # transpose arrays to 'lrkb' order.
    lkb_ord = tuple(map(lambda x: shape.replace('r', "").find(x), 'lkb'))
    rkb_ord = tuple(map(lambda x: shape.replace('l', "").find(x), 'rkb'))
    lrkb_ord = tuple(map(shape.find, 'lrkb'))

    tensors = []
    next_bond = rand_uuid(base=bond_name)

    # Do the first tensor seperately.
    tensors += [Tensor(array=arrays[0].transpose(*lkb_ord),
                       inds=[next_bond, ket_site_inds[0], bra_site_inds[0]])]
    previous_bond = next_bond

    # Range over the middle tensors
    for array, ksi, bsi in zip(arrays[1:-1],
                               ket_site_inds[1:-1],
                               bra_site_inds[1:-1]):
        next_bond = rand_uuid(base=bond_name)
        tensors += [Tensor(array=array.transpose(*lrkb_ord),
                           inds=[previous_bond, next_bond, ksi, bsi])]
        previous_bond = next_bond

    # Do the last tensor seperately.
    tensors.append(Tensor(array=arrays[-1].transpose(*rkb_ord),
                          inds=[previous_bond,
                                ket_site_inds[-1],
                                bra_site_inds[-1]]))

    return TensorNetwork(*tensors)


# --------------------------------------------------------------------------- #
#                        Specific states and operators                        #
# --------------------------------------------------------------------------- #

@functools.lru_cache(128)
def mpo_site_ham_heis(j=1.0, bz=0.0):
    """Single site of the spin-1/2 Heisenberg Hamiltonian in MPO form.
    This is cached.

    Parameters
    ----------
    j : float
        (Isotropic) nearest neighbour coupling.
    bz : float
        Magnetic field in Z-direction.

    Returns
    -------
    H : numpy.ndarray
        The tensor, with shape (5, 5, 2, 2).
    """
    H = np.zeros((5, 5, 2, 2), dtype=complex)

    for i, s in enumerate('XYZ'):
        H[1 + i, 0, :, :] = spin_operator(s)
        H[-1, 1 + i, :, :] = j * spin_operator(s)

    H[0, 0, :, :] = eye(2)
    H[4, 4, :, :] = eye(2)
    H[4, 0, :, :] = - bz * spin_operator('Z')

    make_immutable(H)
    return H


def mpo_end_ham_heis_left(j=1.0, bz=0.0):
    """The left most site of a open boundary conditions Heisenberg
    matrix product operator.
    """
    return mpo_site_ham_heis(j=j, bz=bz)[-1, :, :, :]


def mpo_end_ham_heis_right(j=1.0, bz=0.0):
    """The right most site of a open boundary conditions Heisenberg
    matrix product operator.
    """
    return mpo_site_ham_heis(j=j, bz=bz)[:, 0, :, :]


def ham_heis_mpo(n, j=1.0, bz=0.0, ket_site_inds=None, bra_site_inds=None,
                 bond_name=""):
    """
    """
    HH_mpo = matrix_product_operator(
        mpo_end_ham_heis_left(),
        *[mpo_site_ham_heis()] * (n - 2),
        mpo_end_ham_heis_right(),
        ket_site_inds=ket_site_inds,
        bra_site_inds=bra_site_inds,
        bond_name=bond_name,
    )
    return HH_mpo
