import os
import functools
from functools import reduce
import operator
from operator import or_

from cytoolz import concat, frequencies
import numpy as np

from .accel import prod

try:
    # opt_einsum is highly recommended as until numpy 1.14 einsum contractions
    # do not use BLAS.
    import opt_einsum
    contract = opt_einsum.contract

    @functools.wraps(opt_einsum.contract_path)
    def contract_path(*args, optimize='greedy', memory_limit=2**28, **kwargs):
        return opt_einsum.contract_path(
            *args, path=optimize, memory_limit=memory_limit, **kwargs)

except ImportError:

    @functools.wraps(np.einsum)
    def contract(*args, optimize='greedy', memory_limit=2**28, **kwargs):

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
    def contract_path(*args, optimize='greedy', memory_limit=2**28, **kwargs):
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

        # tranpose tensor to bring groups of fused inds to the beginning
        transpose_inds = [*concat(fuseds + unfused_inds)]
        TT = tensor_contract(self, output_inds=transpose_inds, optimize=False)

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
        if isinstance(other, Tensor):
            return TensorNetwork(self, other)
        elif isinstance(other, TensorNetwork):
            return TensorNetwork(self, *other.tensors)
        else:
            raise TypeError("Cannot combine with object of "
                            "type {}".format(type(other)))

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
    o_array = contract(contract_str, *(t.array for t in tensors),
                       memory_limit=memory_limit, optimize=optimize)

    if not o_ix:
        return o_array

    # unison of all tags
    o_tags = reduce(or_, (t.tags for t in tensors))

    return Tensor(array=o_array, inds=o_ix, tags=o_tags)


# --------------------------------------------------------------------------- #
#                            Tensor Network Class                             #
# --------------------------------------------------------------------------- #

class TensorNetwork(object):
    """A collection of (as yet uncontracted) Tensors.

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

    def contract(self, tags=...):
        """Contract some, or all, of the tensors in this network.

        Parameters
        ----------
        tags : sequence of hashable
            Any tensors with any of these tags with be contracted.

        Returns
        -------
        TensorNetwork, Tensor or Scalar
            The result of the contraction, still a TensorNetwork if the
            contraction was only partial.
        """
        leave, lose = self.split(tags)

        if leave:
            return TensorNetwork(*leave) & tensor_contract(*lose)

        return tensor_contract(*lose)

    def split(self, tags):
        """Split this TN into a list of tensors containing any of ``tags`` and
        the rest.

        Parameters
        ----------
        tags : sequence of hashable
            The list of tags to filter the tensors by.

        Returns
        -------
        (tagged, untagged) : (list of Tensor, list of Tensor)
            The list of tagged and untagged tensors.
        """
        # contract all
        if tags is ...:
            return [], self.tensors

        if isinstance(tags, str):
            tags = {tags}

        tagged, untagged = [], []
        for t in self.tensors:
            if any(tag in t.tags for tag in tags):
                untagged.append(t)
            else:
                tagged.append(t)

        return tagged, untagged

    def conj(self):
        """Conjugate all the tensors in this network (leaves all indices).
        """
        return TensorNetwork(*[t.conj() for t in self.tensors])

    @property
    def H(self):
        """Conjugate all the tensors in this network (leaves all indices).
        """
        return self.conj()

    def __rshift__(self, *args, **kwargs):
        """ Overload of '>>' for TensorNetwork.contract
        """
        return self.contract(*args, **kwargs)

    def __and__(self, other):
        """Combine this tensor network with more tensors, without contracting.
        """
        if isinstance(other, Tensor):
            return TensorNetwork(*self.tensors, other)

        return TensorNetwork(*self.tensors, *other.tensors)

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
