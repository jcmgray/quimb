import functools
import math
import operator

import autoray as ar

from quimb.tensor import TensorNetwork, array_contract, bonds
from quimb.utils import RollingDiffMean


def prod(xs):
    """Product of all elements in ``xs``."""
    return functools.reduce(operator.mul, xs)


class BeliefPropagationCommon:
    """Common interfaces for belief propagation algorithms.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to perform belief propagation on.
    damping : float or callable, optional
        The damping factor to apply to messages. This simply mixes some part
        of the old message into the new one, with the final message being
        ``damping * old + (1 - damping) * new``. This makes convergence more
        reliable but slower.
    update : {'sequential', 'parallel'}, optional
        Whether to update messages sequentially (newly computed messages are
        immediately used for other updates in the same iteration round) or in
        parallel (all messages are comptued using messages from the previous
        round only). Sequential generally helps convergence but parallel can
        possibly converge to differnt solutions.
    normalize : {'L1', 'L2', 'L2phased', 'Linf', callable}, optional
        How to normalize messages after each update. If None choose
        automatically. If a callable, it should take a message and return the
        normalized message. If a string, it should be one of 'L1', 'L2',
        'L2phased', 'Linf' for the corresponding norms. 'L2phased' is like 'L2'
        but also normalizes the phase of the message, by default used for
        complex dtypes.
    distance : {'L1', 'L2', 'L2phased', 'Linf', 'cosine', callable}, optional
        How to compute the distance between messages to check for convergence.
        If None choose automatically. If a callable, it should take two
        messages and return the distance. If a string, it should be one of
        'L1', 'L2', 'L2phased', 'Linf', or 'cosine' for the corresponding
        norms. 'L2phased' is like 'L2' but also normalizes the phases of the
        messages, by default used for complex dtypes if phased normalization is
        not already being used.
    contract_every : int, optional
        If not None, 'contract' (via BP) the tensor network every
        ``contract_every`` iterations. The resulting values are stored in
        ``zvals`` at corresponding points ``zval_its``.
    callback : callable, optional
        A function to call after every iteration, of signature
        ``callback(bp_instance)``.
    inplace : bool, optional
        Whether to perform any operations inplace on the input tensor network.
    """

    def __init__(
        self,
        tn: TensorNetwork,
        *,
        damping=0.0,
        update="sequential",
        normalize=None,
        distance=None,
        contract_every=None,
        callback=None,
        inplace=False,
    ):
        self.tn = tn if inplace else tn.copy()
        self.backend = self.tn.backend
        self.dtype = self.tn.dtype
        self.sign = 1.0
        self.exponent = tn.exponent
        self.damping = damping
        self.update = update
        self.callback = callback

        if normalize is None:
            if "complex" in self.dtype:
                normalize = "L2phased"
            else:
                normalize = "L2"
        self.normalize = normalize

        if distance is None:
            if ("complex" in self.dtype) and (
                callable(normalize) or ("phased" not in normalize)
            ):
                distance = "L2phased"
            else:
                distance = "L2"
        self.distance = distance

        self.contract_every = contract_every
        self.n = 0
        self.converged = False
        self.mdiffs = []
        self.rdiffs = []
        self.zval_its = []
        self.zvals = []

    @property
    def damping(self):
        return self._damping

    @damping.setter
    def damping(self, damping):
        if callable(damping):
            self._damping_fn = self._damping = damping
        else:
            self._damping = damping

            if damping == 0.0:

                def _damping_fn(old, new):
                    return new

            else:

                def _damping_fn(old, new):
                    return damping * old + (1 - damping) * new

            self._damping_fn = _damping_fn

    @property
    def normalize(self):
        return self._normalize

    @normalize.setter
    def normalize(self, normalize):
        if callable(normalize):
            # custom normalization function
            _normalize_fn = normalize

        elif normalize == "L1":
            _abs = ar.get_lib_fn(self.backend, "abs")
            _sum = ar.get_lib_fn(self.backend, "sum")

            def _normalize_fn(x):
                return x / _sum(_abs(x))

        elif normalize == "L2":
            _abs = ar.get_lib_fn(self.backend, "abs")
            _sum = ar.get_lib_fn(self.backend, "sum")

            def _normalize_fn(x):
                return x / (_sum(_abs(x) ** 2) ** 0.5)

        elif normalize == "L2phased":
            _sum = ar.get_lib_fn(self.backend, "sum")
            _abs = ar.get_lib_fn(self.backend, "abs")

            def _normalize_fn(x):
                xnrm = float(_sum(_abs(x) ** 2)) ** 0.5
                sumx = complex(_sum(x))
                if sumx != 0.0:
                    if sumx.imag == 0.0:
                        sumx = sumx.real
                    sumx /= abs(sumx)
                    xnrm *= sumx
                return x / xnrm

        elif normalize == "Linf":
            _abs = ar.get_lib_fn(self.backend, "abs")
            _max = ar.get_lib_fn(self.backend, "max")

            def _normalize_fn(x):
                return x / _max(_abs(x))

        else:
            raise ValueError(f"Unrecognized normalize={normalize}")

        self._normalize = normalize
        self._normalize_fn = _normalize_fn

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, distance):
        if callable(distance):
            _distance_fn = distance

        elif distance == "L1":
            _abs = ar.get_lib_fn(self.backend, "abs")
            _sum = ar.get_lib_fn(self.backend, "sum")

            def _distance_fn(x, y):
                return float(_sum(_abs(x - y)))

        elif distance == "L2":
            _abs = ar.get_lib_fn(self.backend, "abs")
            _sum = ar.get_lib_fn(self.backend, "sum")

            def _distance_fn(x, y):
                return float(_sum(_abs(x - y) ** 2) ** 0.5)

        elif distance == "L2phased":
            _conj = ar.get_lib_fn(self.backend, "conj")
            _sum = ar.get_lib_fn(self.backend, "sum")
            _abs = ar.get_lib_fn(self.backend, "abs")

            def _distance_fn(x, y):
                xnorm = _sum(_abs(x) ** 2) ** 0.5
                ynorm = _sum(_abs(y) ** 2) ** 0.5
                # cosine similarity with phase
                cs = _sum(_conj(x) * y)
                phase = cs / _abs(cs)
                xn = x / xnorm
                yn = y / (ynorm * phase)
                # L2 distance between normalized, phased vectors
                return float(_sum(_abs(xn - yn) ** 2) ** 0.5)

        elif distance == "Linf":
            _abs = ar.get_lib_fn(self.backend, "abs")
            _max = ar.get_lib_fn(self.backend, "max")

            def _distance_fn(x, y):
                return float(_max(_abs(x - y)))

        elif distance == "cosine":
            # this is like L2phased, but with less precision
            _conj = ar.get_lib_fn(self.backend, "conj")
            _sum = ar.get_lib_fn(self.backend, "sum")
            _abs = ar.get_lib_fn(self.backend, "abs")

            def _distance_fn(x, y):
                xnorm = float(_sum(_abs(x) ** 2) ** 0.5)
                ynorm = float(_sum(_abs(y) ** 2) ** 0.5)
                # compute cosine similarity
                cs = float(_abs(_sum(_conj(x) * y)) / (xnorm * ynorm))
                # clip to avoid numerical issues
                cs = min(max(cs, -1.0), 1.0)
                return (2 - 2 * cs) ** 0.5

        else:
            raise ValueError(f"Unrecognized distance={distance}")

        self._distance = distance
        self._distance_fn = _distance_fn

    def _maybe_contract(self):
        should_contract = (
            (self.contract_every is not None)
            and (self.n % self.contract_every == 0)
            and ((not self.zval_its) or (self.zval_its[-1] != self.n))
        )
        if should_contract:
            self.zval_its.append(self.n)
            self.zvals.append(self.contract())

    def run(
        self,
        max_iterations=1000,
        diis=False,
        tol=5e-6,
        tol_abs=None,
        tol_rolling_diff=None,
        info=None,
        progbar=False,
    ):
        """
        Parameters
        ----------
        max_iterations : int, optional
            The maximum number of iterations to perform.
        diis : bool or dict, optional
            Whether to use direct inversion in the iterative subspace to
            help converge the messages by extrapolating to low error guesses.
            If a dict, should contain options for the DIIS algorithm. The
            relevant options are {`max_history`, `beta`, `rcond`}.
        tol : float, optional
            The convergence tolerance for messages.
        tol_abs : float, optional
            The absolute convergence tolerance for maximum message update
            distance, if not given then taken as ``tol``.
        tol_rolling_diff : float, optional
            The rolling mean convergence tolerance for maximum message update
            distance, if not given then taken as ``tol``. This is used to stop
            running when the messages are just bouncing around the same level,
            without any overall upward or downward trends, roughly speaking.
        info : dict, optional
            If supplied, the following information will be added to it:
            ``converged`` (bool), ``iterations`` (int), ``max_mdiff`` (float),
            ``rolling_abs_mean_diff`` (float).
        progbar : bool, optional
            Whether to show a progress bar.
        """
        if tol_abs is None:
            tol_abs = tol
        if tol_rolling_diff is None:
            tol_rolling_diff = tol

        if progbar:
            import tqdm

            pbar = tqdm.tqdm()
        else:
            pbar = None

        if diis:
            from .diis import DIIS

            if isinstance(diis, dict):
                self._diis = DIIS(**diis)
                diis = True
            else:
                self._diis = DIIS()
        else:
            self._diis = None

        it = 0
        rdm = RollingDiffMean()
        self.converged = False
        while not self.converged and it < max_iterations:
            self._maybe_contract()

            # perform a single iteration of BP
            # we supply tol here for use with local convergence
            result = self.iterate(tol=tol)

            if diis:
                # extrapolate new guess for messages
                self.messages = self._diis.update(self.messages)

            if isinstance(result, dict):
                max_mdiff = result.get("max_mdiff", float("inf"))
            else:
                max_mdiff = result
                result = dict()

            self.mdiffs.append(max_mdiff)

            if pbar is not None:
                msg = f"max|dM|={max_mdiff:.3g}"

                nconv = result.get("nconv", None)
                if nconv is not None:
                    ncheck = result.get("ncheck", None)
                    msg += f" nconv: {nconv}/{ncheck} "

                pbar.set_description(msg, refresh=False)
                pbar.update()

            # check covergence criteria
            self.converged |= max_mdiff < tol_abs
            if tol_rolling_diff > 0.0:
                # check rolling mean convergence
                rdm.update(max_mdiff)
                amd = rdm.absmeandiff()
                self.converged |= amd < tol_rolling_diff
                self.rdiffs.append(amd)

            it += 1
            self.n += 1

            if self.callback is not None:
                self.callback(self)

        self._maybe_contract()

        # finally:
        if pbar is not None:
            pbar.close()

        if tol != 0.0 and not self.converged:
            import warnings

            warnings.warn(
                f"Belief propagation did not converge after {max_iterations} "
                f"iterations, tol={tol:.2e}, max|dM|={max_mdiff:.2e}."
            )

        if info is not None:
            info["converged"] = self.converged
            info["iterations"] = it
            info["max_mdiff"] = max_mdiff
            info["rolling_abs_mean_diff"] = rdm.absmeandiff()

    def plot(self, zvals_yscale="asinh", **kwargs):
        from quimb import plot_multi_series_zoom

        data = {
            "zvals": {
                "x": self.zval_its,
                "y": self.zvals,
                "yscale": zvals_yscale,
            },
            "mdiffs": self.mdiffs,
            "rdiffs": self.rdiffs,
        }
        if getattr(self, "_diis", None) is not None:
            data["diis.lambdas"] = self._diis.lambdas

        kwargs.setdefault("yscale", "log")
        return plot_multi_series_zoom(data, **kwargs)

    @property
    def mdiff(self):
        try:
            return self.mdiffs[-1]
        except IndexError:
            return float("nan")

    def iterate(self, tol=1e-6):
        """Perform a single iteration of belief propagation. Subclasses should
        implement this method, returning either `max_mdiff` or a dictionary
        containing `max_mdiff` and any other relevant information:

            {
                "nconv": nconv,
                "ncheck": ncheck,
                "max_mdiff": max_mdiff,
            }

        """
        raise NotImplementedError

    def contract(
        self,
        strip_exponent=False,
        check_zero=True,
        **kwargs,
    ):
        """Contract the tensor network and return the resulting value."""
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n}, mdiff={self.mdiff:.3g})"


def initialize_hyper_messages(
    tn,
    fill_fn=None,
    smudge_factor=1e-12,
):
    """Initialize messages for belief propagation, this is equivalent to doing
    a single round of belief propagation with uniform messages.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to initialize messages for.
    fill_fn : callable, optional
        A function to fill the messages with, of signature ``fill_fn(shape)``.
    smudge_factor : float, optional
        A small number to add to the messages to avoid numerical issues.

    Returns
    -------
    messages : dict
        The initial messages. For every index and tensor id pair, there will
        be a message to and from with keys ``(ix, tid)`` and ``(tid, ix)``.
    """
    backend = ar.infer_backend(next(t.data for t in tn))
    _sum = ar.get_lib_fn(backend, "sum")

    messages = {}

    # compute first messages from tensors to indices
    for tid, t in tn.tensor_map.items():
        k_inputs = tuple(range(t.ndim))
        for i, ix in enumerate(t.inds):
            if fill_fn is None:
                # sum over all other indices to get initial message
                m = array_contract(
                    arrays=(t.data,),
                    inputs=(k_inputs,),
                    output=(i,),
                )
                # normalize and insert
                messages[tid, ix] = m / _sum(m)
            else:
                d = t.ind_size(ix)
                m = fill_fn((d,))
                messages[tid, ix] = m / _sum(m)

    # compute first messages from indices to tensors
    for ix, tids in tn.ind_map.items():
        ms = [messages[tid, ix] for tid in tids]
        mp = prod(ms)
        for mi, tid in zip(ms, tids):
            m = mp / (mi + smudge_factor)
            # normalize and insert
            messages[ix, tid] = m / _sum(m)

    return messages


def combine_local_contractions(
    values,
    backend=None,
    strip_exponent=False,
    check_zero=True,
    mantissa=None,
    exponent=None,
):
    """Combine a product of local contractions into a single value, avoiding
    overflow/underflow by accumulating the mantissa and exponent separately.

    Parameters
    ----------
    values : sequence of (scalar, int)
        The values to combine, each with a power to be raised to.
    backend : str, optional
        The backend to use. Infered from the first value if not given.
    strip_exponent : bool, optional
        Whether to return the mantissa and exponent separately.
    check_zero : bool, optional
        Whether to check for zero values and return zero early.
    mantissa : float, optional
        The initial mantissa to accumulate into.
    exponent : float, optional
        The initial exponent to accumulate into.

    Returns
    -------
    result : float or (float, float)
        The combined value, or the mantissa and exponent separately.
    """
    if mantissa is None:
        mantissa = 1.0
    if exponent is None:
        exponent = 0.0

    _abs = _log10 = None
    for x, power in values:
        if _abs is None:
            # lazily get functions
            if backend is None:
                backend = ar.infer_backend(x)
            _abs = ar.get_lib_fn(backend, "abs")
            _log10 = ar.get_lib_fn(backend, "log10")

        # factor into phase and magnitude
        x_mag = _abs(x)
        x_phase = x / x_mag

        if check_zero and (x_mag == 0.0):
            # checking explicitly avoids errors from taking log(0)
            if strip_exponent:
                return 0.0, 0.0
            else:
                return 0.0

        # accumulate the mantissa and exponent separately,
        # accounting for the local power / counting factor
        mantissa = mantissa * x_phase**power
        exponent = exponent + power * _log10(x_mag)

    if strip_exponent:
        return mantissa, exponent
    else:
        return mantissa * 10**exponent


def contract_hyper_messages(
    tn,
    messages,
    backend=None,
    strip_exponent=False,
    check_zero=True,
    mantissa=None,
    exponent=None,
):
    """Estimate the contraction of ``tn`` given ``messages``, via the
    exponential of the Bethe free entropy.
    """
    zvals = []

    for tid, t in tn.tensor_map.items():
        if backend is None:
            backend = ar.infer_backend(t.data)

        arrays = [t.data]
        inputs = [range(t.ndim)]
        for i, ix in enumerate(t.inds):
            arrays.append(messages[ix, tid])
            inputs.append((i,))

            # local message overlap correction
            z = array_contract(
                (messages[tid, ix], messages[ix, tid]),
                inputs=((0,), (0,)),
                output=(),
            )
            zvals.append((z, -1))

        # local factor free entropy
        z = array_contract(arrays, inputs, output=())
        zvals.append((z, 1))

    for ix, tids in tn.ind_map.items():
        arrays = tuple(messages[tid, ix] for tid in tids)
        inputs = tuple((0,) for _ in tids)
        # local variable free entropy
        z = array_contract(arrays, inputs, output=())
        zvals.append((z, 1))

    return combine_local_contractions(
        zvals,
        backend=backend,
        strip_exponent=strip_exponent,
        check_zero=check_zero,
        mantissa=mantissa,
        exponent=exponent,
    )


def compute_index_marginal(tn, ind, messages):
    """Compute the marginal for a single index given ``messages``.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compute the marginal for.
    ind : int
        The index to compute the marginal for.
    messages : dict
        The messages to use, which should match ``tn``.

    Returns
    -------
    marginal : array_like
        The marginal probability distribution for the index ``ind``.
    """
    m = prod(messages[tid, ind] for tid in tn.ind_map[ind])
    return m / ar.do("sum", m)


def compute_tensor_marginal(tn, tid, messages):
    """Compute the marginal for the region surrounding a single tensor/factor
    given ``messages``.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compute the marginal for.
    tid : int
        The tensor id to compute the marginal for.
    messages : dict
        The messages to use, which should match ``tn``.

    Returns
    -------
    marginal : array_like
        The marginal probability distribution for the tensor/factor ``tid``.
    """
    t = tn.tensor_map[tid]

    output = tuple(range(t.ndim))
    inputs = [output]
    arrays = [t.data]

    for i, ix in enumerate(t.inds):
        mix = prod(
            messages[otid, ix] for otid in tn.ind_map[ix] if otid != tid
        )
        inputs.append((i,))
        arrays.append(mix)

    m = array_contract(
        arrays=arrays,
        inputs=inputs,
        output=output,
    )

    return m / ar.do("sum", m)


def compute_all_index_marginals_from_messages(tn, messages):
    """Compute all index marginals from belief propagation messages.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compute marginals for.
    messages : dict
        The belief propagation messages.

    Returns
    -------
    marginals : dict
        The marginals for each index.
    """
    return {ix: compute_index_marginal(tn, ix, messages) for ix in tn.ind_map}


def normalize_message_pair(mi, mj):
    """Normalize a pair of messages such that `<mi|mj> = 1` and
    `<mi|mi> = <mj|mj>` (but in general != 1).
    """
    nij = ar.do("abs", mi @ mj) ** 0.5
    nii = (mi @ mi) ** 0.25
    njj = (mj @ mj) ** 0.25
    return mi / (nij * nii / njj), mj / (nij * njj / nii)


def maybe_get_thread_pool(thread_pool):
    """Get a thread pool if requested."""
    if thread_pool is False:
        return None

    if thread_pool is True:
        import quimb as qu

        return qu.get_thread_pool()

    if isinstance(thread_pool, int):
        import quimb as qu

        return qu.get_thread_pool(thread_pool)

    return thread_pool


def create_lazy_community_edge_map(tn, site_tags=None, rank_simplify=True):
    """For lazy BP algorithms, create the data structures describing the
    effective graph of the lazily grouped 'sites' given by ``site_tags``.
    """
    if site_tags is None:
        site_tags = set(tn.site_tags)
    else:
        site_tags = set(site_tags)

    edges = {}
    neighbors = {}
    local_tns = {}
    touch_map = {}

    for ix in tn.ind_map:
        ts = tn._inds_get(ix)
        tags = {tag for t in ts for tag in t.tags if tag in site_tags}
        if len(tags) >= 2:
            i, j = tuple(sorted(tags))

            if (i, j) in edges:
                # already processed this edge
                continue

            # add to neighbor map
            neighbors.setdefault(i, []).append(j)
            neighbors.setdefault(j, []).append(i)

            # get local TNs and compute bonds between them,
            # rank simplify here also to prepare for contractions
            try:
                tn_i = local_tns[i]
            except KeyError:
                tn_i = local_tns[i] = tn.select(i, virtual=False)
                if rank_simplify:
                    tn_i.rank_simplify_()
            try:
                tn_j = local_tns[j]
            except KeyError:
                tn_j = local_tns[j] = tn.select(j, virtual=False)
                if rank_simplify:
                    tn_j.rank_simplify_()

            edges[i, j] = tuple(bonds(tn_i, tn_j))

    for i, j in edges:
        touch_map[(i, j)] = tuple((j, k) for k in neighbors[j] if k != i)
        touch_map[(j, i)] = tuple((i, k) for k in neighbors[i] if k != j)

    if len(local_tns) != len(site_tags):
        # handle potentially disconnected sites
        for i in sorted(site_tags):
            try:
                tn_i = local_tns[i] = tn.select(i, virtual=False)
                if rank_simplify:
                    tn_i.rank_simplify_()
            except KeyError:
                pass

    return edges, neighbors, local_tns, touch_map


def auto_add_indices(tn, regions):
    """Make sure all indices incident to any tensor in each region are
    included in the region.
    """
    new_regions = []
    for r in regions:
        new_r = set(r)
        tids = [x for x in new_r if isinstance(x, int)]
        for tid in tids:
            t = tn.tensor_map[tid]
            new_r.update(t.inds)
        new_regions.append(frozenset(new_r))
    return new_regions


def process_loop_series_expansion_weights(
    weights,
    mantissa=1.0,
    exponent=0.0,
    multi_excitation_correct=True,
    maxiter_correction=100,
    tol_correction=1e-14,
    strip_exponent=False,
    return_all=False,
):
    """Assuming a normalized BP fixed point, take a series of loop weights, and
    iteratively compute the free energy by requiring self-consistency with
    exponential suppression factors. See https://arxiv.org/abs/2409.03108.
    """
    # this is the single exictation approximation
    f_uncorrected = -sum(weights.values())

    if multi_excitation_correct:
        # iteratively compute a self consistent free energy
        fold = float("inf")
        f = f_uncorrected
        for _ in range(maxiter_correction):
            f = -sum(
                wl * math.exp(len(gloop) * f) for gloop, wl in weights.items()
            )
            if abs(f - fold) < tol_correction:
                break
            fold = f
    else:
        f = f_uncorrected

    if return_all:
        return {gloop: math.exp(len(gloop) * f) for gloop in weights}

    mantissa = mantissa * (1 - f)

    if strip_exponent:
        return mantissa, exponent

    return mantissa * 10**exponent
