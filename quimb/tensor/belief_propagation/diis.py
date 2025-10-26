import autoray as ar
from quimb.tensor import Tensor
from quimb.utils import (
    tree_map,
    tree_unflatten,
    tree_apply,
    Leaf,
)


class ArrayInfo:
    __slots__ = ("shape", "size")

    def __init__(self, shape, size):
        self.shape = shape
        self.size = size

    def __repr__(self):
        return f"<ArrayInfo(shape={self.shape}, size={self.size})>"


class Vectorizer:
    """Object for mapping back and forth between any nested pytree of arrays
    or Tensors and a single flat vector.

    Parameters
    ----------
    tree : pytree of array, optional
        Any nested container of arrays, which will be flattened and packed into
        a single vector.
    """

    def __init__(self, tree=None, backend=None):
        self.infos = None
        self.d = None
        self.ref_tree = None
        self.backend = backend
        self._concatenate = None
        self._reshape = None
        if tree is not None:
            self.setup(tree)

    def setup(self, tree):
        self.infos = []
        self.d = 0

        def extracter(x):
            if isinstance(x, Tensor):
                array = x.data
                size = x.size
                info = x
            else:
                array = x
                shape = ar.do("shape", x, like=self.backend)
                size = ar.do("size", x, like=self.backend)
                info = ArrayInfo(shape, size)

            if self.backend is None:
                # set backend from first array encountered
                self.backend = ar.infer_backend(array)
                self._concatenate = ar.get_lib_fn(self.backend, "concatenate")
                self._reshape = ar.get_lib_fn(self.backend, "reshape")

            self.infos.append(info)
            self.d += size
            return Leaf

        self.ref_tree = tree_map(extracter, tree)

    def pack(self, tree):
        """Take ``arrays`` and pack their values into attribute `.{name}`, by
        default `.vector`.
        """
        if self.infos is None:
            self.setup(tree)

        def extractor(x):
            if isinstance(x, Tensor):
                x = x.data
            arrays.append(self._reshape(x, (-1,)))

        arrays = []
        tree_apply(extractor, tree)
        return self._concatenate(tuple(arrays))

    def unpack(self, vector):
        """Turn the single, flat ``vector`` into a sequence of arrays."""

        def _gen_arrays():
            i = 0
            for info in self.infos:
                # get the linear slice
                f = i + info.size
                array = self._reshape(vector[i:f], info.shape)
                i = f
                if isinstance(info, Tensor):
                    # insert array back into tensor, inplace
                    info.modify(data=array)
                    yield info
                else:
                    yield array

        return tree_unflatten(_gen_arrays(), self.ref_tree)

    def __repr__(self):
        return f"<Vectorizer(d={self.d})>"


class DIIS:
    """Direct Inversion in the Iterative Subspace (DIIS) method (AKA Pulay
    mixing) [1] for converging fixed-point iterations.

    [1] P. Pulay, Convergence acceleration of iterative sequences. The case of
    SCF iteration, 1980, Elsevier, https://doi.org/10.1016/0009-2614(80)80396-4.

    Parameters
    ----------
    max_history : int
        Maximum number of previous guesses to use in extrapolation.
    beta : float
        Mixing parameter, 0.0 means only use input guesses, 1.0 means only use
        extrapolated guesses (original Pulay mixing). Default is 1.0.
    rcond : float
        Cutoff for small singular values in the pseudo-inverse of the B matrix.
        Default is 1e-14.
    """

    def __init__(self, max_history=6, beta=1.0, rcond=1e-14):
        self.max_history = max_history
        self.beta = beta
        self.rcond = rcond

        # storage
        self.vectorizer = Vectorizer()
        self.guesses = [None] * max_history
        self.errors = [None] * max_history
        self.lambdas = []
        self.head = self.max_history - 1

        self.backend = None
        self.B = None
        self.y = None
        self.scalar = None

    def _extrapolate(self):
        # TODO: make this backend agnostic
        import numpy as np

        # XXX: do this all on backend? (though is very small)
        if self.B is None:
            g0 = self.guesses[0]
            self.backend = ar.infer_backend(g0)
            dtype = ar.get_dtype_name(g0)
            self.B = np.zeros((self.max_history + 1,) * 2, dtype=dtype)
            self.y = np.zeros(self.max_history + 1, dtype=dtype)
            self.B[1:, 0] = self.B[0, 1:] = self.y[0] = 1.0
            # define conversion to python scalar
            if "complex" in dtype:
                self.scalar = complex
            else:
                self.scalar = float

        # number of error estimates we have
        d = sum(e is not None for e in self.errors)
        i = self.head
        error_i_conj = self.errors[i].conj()
        for j in range(d):
            cij = self.scalar(error_i_conj @ self.errors[j])
            self.B[i + 1, j + 1] = cij
            if i != j:
                self.B[j + 1, i + 1] = cij.conjugate()

        # solve for coefficients, taking into account rank deficiency
        Binv = np.linalg.pinv(
            self.B[: d + 1, : d + 1],
            rcond=self.rcond,
            hermitian=True,
        )
        coeffs = Binv @ self.y[: d + 1]

        # first entry is -ve. lagrange multiplier -> estimated next residual
        self.lambdas.append(abs(-coeffs[0]))
        coeffs = [self.scalar(c) for c in coeffs[1:]]

        # construct linear combination of previous guesses!
        xnew = ar.do("zeros_like", self.guesses[0], like=self.backend)
        for ci, xi in zip(coeffs, self.guesses):
            xnew += ci * xi

        if self.beta != 0.0:
            # allow custom mix of x + xnew:
            # https://prefetch.eu/know/concept/pulay-mixing/
            # i.e. use not just x_i but also f(x_i) -> y_i
            # original Pulay mixing is beta=1.0 == only xnews
            for ci, ei in zip(coeffs, self.errors):
                xnew += (self.beta * ci) * ei

        return xnew

    def update(self, y):
        """Given new output `y[i]` (the result of `f(x[i])`), update the
        internal state and return the extrapolated next guess `x[i+1]`.

        Parameters
        ----------
        y : pytree of array
            The output of the function `f(x)`. Can be any arbitrary nested
            tree structure with arrays treated at leaves.

        Returns
        -------
        xnext : pytree of array
            The next guess `x[i+1]` to pass to the function `f(x)`, with the
            same tree structure as `y`.
        """
        # convert from pytree -> single real vector
        y = self.vectorizer.pack(y)
        x = self.guesses[self.head]
        if x is None:
            # first guess (no extrapolation)
            xnext = y
        else:
            self.errors[self.head] = y - x
            xnext = self._extrapolate()

        self.head = (self.head + 1) % self.max_history
        # NOTE: copy seems to be necessary here to avoid in-place modifications
        self.guesses[self.head] = ar.do("copy", xnext, like=self.backend)

        # convert new extrapolated guess back to pytree
        return self.vectorizer.unpack(xnext)


class DIISPyscf:
    """Thin wrapper around the PySCF DIIS implementation to handle arbitrary
    pytrees of arrays, for testing purposes."""

    def __init__(self, max_history=6):
        from pyscf.lib.diis import DIIS as PDIIS

        self.pdiis = PDIIS()
        self.pdiis.space = max_history
        self.vectorizer = Vectorizer()

    def update(self, y):
        y = self.vectorizer.pack(y)
        xnext = self.pdiis.update(y)
        return self.vectorizer.unpack(xnext)
