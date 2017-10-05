import os
import functools
from functools import reduce
import operator
from operator import or_

from cytoolz import concat, frequencies, interleave
import numpy as np

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
        return np.einsum(
            *args, optimize=(optimize, memory_limit), **kwargs)

    @functools.wraps(np.einsum_path)
    def contract_path(*args, optimize='greedy', memory_limit=2**28, **kwargs):
        return np.einsum_path(
            *args, optimize=(optimize, memory_limit), **kwargs)


# --------------------------------------------------------------------------- #
#                                Tensor Class                                 #
# --------------------------------------------------------------------------- #

class Tensor(object):
    """
    """

    def __init__(self, array, inds, tags=None):
        self.array = np.asarray(array)
        self.inds = tuple(inds)
        self.tags = (set() if tags is None else
                     {tags} if isinstance(tags, str) else
                     set(tags))

    def conj(self):
        return Tensor(self.array.conj(), self.inds, self.tags)

    @property
    def H(self):
        """
        """
        return self.conj()

    @property
    def shape(self):
        return self.array.shape

    @property
    def size(self):
        return self.array.size

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

def gen_output_inds(inds):
    """Generate the output indices from the set ``inds``.
    """
    freqs = frequencies(concat(inds))

    for ind, freq in freqs.items():
        if freq > 2:
            raise ValueError("The index {} appears more "
                             "than twice!".format(ind))
        elif freq == 1:
            yield ind


def tensor_contract(*tensors, memory_limit=-1, optimize=True):
    """Efficiently contract some tensors.
    """
    i_inds = [t.inds for t in tensors]
    o_inds = sorted(gen_output_inds(i_inds))

    o_array = contract(*interleave(((t.array for t in tensors), i_inds)),
                       o_inds, memory_limit=memory_limit, optimize=optimize)

    if not o_inds:
        return o_array

    # unison of all tags
    o_tags = reduce(or_, (t.tags for t in tensors))

    return Tensor(array=o_array, inds=o_inds, tags=o_tags)


# --------------------------------------------------------------------------- #
#                            Tensor Network Class                             #
# --------------------------------------------------------------------------- #

class TensorNetwork(object):
    """

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

    def contract(self, tags=None):
        """
        """
        leave, lose = self.filt(tags)

        if leave:
            return TensorNetwork(*leave) & tensor_contract(*lose)

        return tensor_contract(*lose)

    def filt(self, tags):
        """
        """
        # contract all
        if tags is None:
            return [], self.tensors

        if isinstance(tags, str):
            tags = {tags}

        leave, lose = [], []
        for t in self.tensors:
            if any(tag in t.tags for tag in tags):
                lose.append(t)
            else:
                leave.append(t)

        return leave, lose

    def conj(self):
        return TensorNetwork(*[t.conj() for t in self.tensors])

    @property
    def H(self):
        return self.conj()

    def __rshift__(self, tags=None):
        return self.contract(tags)

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
