"""Exact tensor-network circuit simulators (``Circuit``, ``CircuitDense``)."""

import functools
import itertools
import numbers
import operator
import re

import numpy as np
from autoray import (
    do,
    reshape,
)

import quimb as qu

from ...utils import (
    ensure_dict,
    partition_all,
)
from ...utils import progbar as _progbar
from .. import array_ops as ops
from ..tensor_builder import TN_from_sites_computational_state
from ..tensor_core import (
    Tensor,
    oset_union,
    rand_uuid,
)
from ..tn1d.core import Dense1D
from ..tnag.core import TensorNetworkGenOperator
from .core import CircuitBase
from .gates import (
    rehearsal_dict,
    sample_bitstring_from_prob_ndarray,
)


class Circuit(CircuitBase):
    """Class for simulating quantum circuits using tensor networks. The class
    keeps a list of :class:`Gate` objects in sync with a tensor network
    representing the current state of the circuit.

    Parameters
    ----------
    N : int, optional
        The number of qubits.
    psi0 : TensorNetwork1DVector, optional
        The initial state, assumed to be ``|00000....0>`` if not given. The
        state is always copied and the tag ``PSI0`` added.
    gate_opts : dict_like, optional
        Default keyword arguments to supply to each
        :func:`~quimb.tensor.tn1d.core.gate_TN_1D` call during the circuit.
    gate_contract : str, optional
        Shortcut for setting the default `'contract'` option in `gate_opts`.
    gate_propagate_tags : str, optional
        Shortcut for setting the default `'propagate_tags'` option in
        `gate_opts`.
    tags : str or sequence of str, optional
        Tag(s) to add to the initial wavefunction tensors (whether these are
        propagated to the rest of the circuit's tensors depends on
        ``gate_opts``).
    psi0_dtype : str, optional
        Ensure the initial state has this dtype.
    psi0_tag : str, optional
        Ensure the initial state has this tag.
    tag_gate_numbers : bool, optional
        Whether to tag each gate tensor with its number in the circuit, like
        ``"GATE_{g}"``. This is required for updating the circuit parameters.
    gate_tag_id : str, optional
        The format string for tagging each gate tensor, by default e.g.
        ``"GATE_{g}"``.
    tag_gate_rounds : bool, optional
        Whether to tag each gate tensor with its number in the circuit, like
        ``"ROUND_{r}"``.
    round_tag_id : str, optional
        The format string for tagging each round of gates, by default e.g.
        ``"ROUND_{r}"``.
    tag_gate_labels : bool, optional
        Whether to tag each gate tensor with its gate type label, e.g.
        ``{"X_1/2", "ISWAP", "CCX", ...}``..
    bra_site_ind_id : str, optional
        Use this to label 'bra' site indices when creating certain (mostly
        internal) intermediate tensor networks.
    dtype : str, optional
        A default dtype to perform calculations in. Depending on
        `convert_eager`, this is enforced *after* circuit construction
        and simplification (the default for exact simulation), or eagerly to
        the initial state and as gates are applied (the default for MPS
        simulation).
    to_backend : callable, optional
        If given, apply this function to both the initial state arrays and to
        every gate as it is applied.
    convert_eager : bool, optional
        Whether to eagerly perform dtype casting and application of
        `to_backend` as gates are supplied, or wait until after the necessary
        TNs for a particular task such as sampling are formed and simplified.
        Deferred conversion (`convert_eager=False`) is the default mode for
        full contraction.

    Attributes
    ----------
    psi : TensorNetwork1DVector
        The current circuit wavefunction as a tensor network.
    uni : TensorNetwork1DOperator
        The current circuit unitary operator as a tensor network.
    gates : tuple[Gate]
        The gates in the circuit.

    Examples
    --------

    Create 3-qubit GHZ-state:

        >>> qc = qtn.Circuit(3)
        >>> gates = [
                ('H', 0),
                ('H', 1),
                ('CNOT', 1, 2),
                ('CNOT', 0, 2),
                ('H', 0),
                ('H', 1),
                ('H', 2),
            ]
        >>> qc.apply_gates(gates)
        >>> qc.psi
        <TensorNetwork1DVector(tensors=12, indices=14, L=3, max_bond=2)>

        >>> qc.psi.to_dense().round(4)
        qarray([[ 0.7071+0.j],
                [ 0.    +0.j],
                [ 0.    +0.j],
                [-0.    +0.j],
                [-0.    +0.j],
                [ 0.    +0.j],
                [ 0.    +0.j],
                [ 0.7071+0.j]])

        >>> for b in qc.sample(10):
        ...     print(b)
        000
        000
        111
        000
        111
        111
        000
        111
        000
        000

    See Also
    --------
    Gate
    """

    def _init_state(self, N, dtype="complex128"):
        return TN_from_sites_computational_state(
            site_map={i: "0" for i in range(N)}, dtype=dtype
        )

    def get_psi(self):
        """Get a copy of the current state tensor network, with any singlet
        dimensions squeezed out.
        """
        psi = self._psi.copy()
        psi.squeeze_()
        if not self.convert_eager:
            self._maybe_convert(psi)
        return psi

    def get_uni(self, transposed=False):
        """Tensor network representation of the unitary operator (i.e. with
        the initial state removed).
        """
        U = self.psi

        if transposed:
            # rename the initial state rand_uuid bonds to 1D site inds
            ixmap = {
                self.ket_site_ind(i): self.bra_site_ind(i)
                for i in range(self.N)
            }
        else:
            ixmap = {}

        # the first `N` tensors should be the tensors of input state
        tids = tuple(U.tensor_map)[: self.N]
        for i, tid in enumerate(tids):
            t = U.pop_tensor(tid)
            (old_ix,) = t.inds

            if transposed:
                ixmap[old_ix] = f"k{i}"
            else:
                ixmap[old_ix] = f"b{i}"

        U.reindex_(ixmap)
        U.view_as_(
            TensorNetworkGenOperator,
            upper_ind_id=self._ket_site_ind_id,
            lower_ind_id=self._bra_site_ind_id,
        )

        return U

    @property
    def uni(self):
        """Tensor network representation of the unitary operator, i.e. the
        circuit with the initial state removed, such that ``circ.uni.to_dense()``
        gives ``U`` acting on a state like ``U @ psi``. For the old transposed
        convention use ``circ.get_uni(transposed=True)``.
        """
        return self.get_uni()

    def get_reverse_lightcone_tags(self, where):
        """Get the tags of gates in this circuit corresponding to the 'reverse'
        lightcone propagating backwards from registers in ``where``.

        Parameters
        ----------
        where : int or sequence of int
            The register or register to get the reverse lightcone of.

        Returns
        -------
        tuple[str]
            The sequence of gate tags (``GATE_{i}``, ...) corresponding to the
            lightcone.
        """
        if isinstance(where, numbers.Integral):
            cone = {where}
        else:
            cone = set(where)

        lightcone_tags = []

        for i, gate in reversed(tuple(enumerate(self._gates))):
            if gate.label == "IDEN":
                continue
            elif gate.controls:
                # TODO: only add if any *targets* in cone, requires changes
                # elsewhere to make sure tensors aren't then missing
                regs = {*gate.controls, *gate.qubits}
                if regs & cone:
                    lightcone_tags.append(self.gate_tag(i))
                    cone |= regs
            elif gate.label == "SWAP":
                i, j = gate.qubits
                i_in_cone = i in cone
                j_in_cone = j in cone
                if i_in_cone:
                    cone.add(j)
                else:
                    cone.discard(j)
                if j_in_cone:
                    cone.add(i)
                else:
                    cone.discard(i)
            else:
                regs = set(gate.qubits)
                if regs & cone:
                    lightcone_tags.append(self.gate_tag(i))
                    cone |= regs

        # initial state is always part of the lightcone
        lightcone_tags.append("PSI0")
        lightcone_tags.reverse()

        return tuple(lightcone_tags)

    def get_psi_reverse_lightcone(self, where, keep_psi0=False):
        """Get just the bit of the wavefunction in the reverse lightcone of
        sites in ``where`` - i.e. causally linked.

        Parameters
        ----------
        where : int, or sequence of int
            The sites to propagate the the lightcone back from, supplied to
            :meth:`~quimb.tensor.circuit.Circuit.get_reverse_lightcone_tags`.
        keep_psi0 : bool, optional
            Keep the tensors corresponding to the initial wavefunction
            regardless of whether they are outside of the lightcone.

        Returns
        -------
        psi_lc : TensorNetwork1DVector
        """
        if isinstance(where, numbers.Integral):
            where = (where,)

        psi = self.psi
        lightcone_tags = self.get_reverse_lightcone_tags(where)
        psi_lc = psi.select_any(lightcone_tags).view_like_(psi)

        if not keep_psi0:
            # these sites are in the lightcone regardless of being alone
            site_inds = set(map(psi.site_ind, where))

            for tid, t in tuple(psi_lc.tensor_map.items()):
                # get all tensors connected to this tensor (incld itself)
                neighbors = oset_union(psi_lc.ind_map[ix] for ix in t.inds)

                # lone tensor not attached to anything - drop it
                # but only if it isn't directly in the ``where`` region
                if (len(neighbors) == 1) and set(t.inds).isdisjoint(site_inds):
                    psi_lc.pop_tensor(tid)

        return psi_lc

    def get_psi_simplified(
        self, seq="ADCRS", atol=1e-12, equalize_norms=False
    ):
        """Get the full wavefunction post local tensor network simplification.

        Parameters
        ----------
        seq : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.

        Returns
        -------
        psi : TensorNetwork1DVector
        """
        self._maybe_init_storage()

        key = ("psi_simplified", seq, atol)
        if key in self._storage:
            return self._storage[key].copy()

        # we simplify and store a copy
        psi = self._psi.copy()
        psi.squeeze_()

        # make sure to keep all outer indices
        output_inds = tuple(map(psi.site_ind, range(self.N)))

        # simplify the state and cache it
        psi.full_simplify_(
            seq=seq,
            atol=atol,
            output_inds=output_inds,
            equalize_norms=equalize_norms,
        )
        self._storage[key] = psi

        # return a copy so we can modify it inplace
        return psi.copy()

    def get_rdm_lightcone_simplified(
        self,
        where,
        seq="ADCRS",
        atol=1e-12,
        equalize_norms=False,
    ):
        """Get a simplified TN of the norm of the wavefunction, with
        gates outside reverse lightcone of ``where`` cancelled, and physical
        indices within ``where`` preserved so that they can be fixed (sliced)
        or used as output indices.

        Parameters
        ----------
        where : int or sequence of int
            The region assumed to be the target density matrix essentially.
            Supplied to
            :meth:`~quimb.tensor.circuit.Circuit.get_reverse_lightcone_tags`.
        seq : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.

        Returns
        -------
        TensorNetwork
        """
        self._maybe_init_storage()

        key = ("rdm_lightcone_simplified", tuple(sorted(where)), seq, atol)
        if key in self._storage:
            return self._storage[key].copy()

        ket_lc = self.get_psi_reverse_lightcone(where)

        k_inds = tuple(map(self.ket_site_ind, where))
        b_inds = tuple(map(self.bra_site_ind, where))

        bra_lc = ket_lc.conj().reindex(dict(zip(k_inds, b_inds)))
        rho_lc = bra_lc | ket_lc

        # don't want to simplify site indices in region away
        output_inds = b_inds + k_inds

        # # simplify the norm and cache it
        rho_lc.full_simplify_(
            seq=seq,
            atol=atol,
            output_inds=output_inds,
            equalize_norms=equalize_norms,
        )
        self._storage[key] = rho_lc

        # return a copy so we can modify it inplace
        return rho_lc.copy()

    def amplitude(
        self,
        b,
        optimize="auto-hq",
        simplify_sequence="ADCRS",
        simplify_atol=1e-12,
        simplify_equalize_norms=True,
        backend=None,
        dtype=None,
        rehearse=False,
    ):
        r"""Get the amplitude coefficient of bitstring ``b``.

        .. math::

            c_b = \langle b | \psi \rangle

        Parameters
        ----------
        b : str or sequence of int
            The bitstring to compute the transition amplitude for.
        optimize : str, optional
            Contraction path optimizer to use for the amplitude, can be a
            non-reusable path optimizer as only called once (though path won't
            be cached for later use in that case).
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        backend : str, optional
            Backend to perform the contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        rehearse : bool or "tn", optional
            If ``True``, generate and cache the simplified tensor network and
            contraction tree but don't actually perform the contraction.
            Returns a dict with keys ``"tn"`` and ``'tree'`` with the tensor
            network that will be contracted and the corresponding contraction
            tree if so.
        """
        self._maybe_init_storage()

        if len(b) != self.N:
            raise ValueError(
                f"Bit-string {b} length does not "
                f"match number of qubits {self.N}."
            )

        fs_opts = {
            "seq": simplify_sequence,
            "atol": simplify_atol,
            "equalize_norms": simplify_equalize_norms,
        }

        # get the full wavefunction simplified
        psi_b = self.get_psi_simplified(**fs_opts)

        # fix the output indices to the correct bitstring
        for i, x in zip(range(self.N), b):
            psi_b.isel_({psi_b.site_ind(i): x})

        # perform a final simplification and cast
        psi_b.full_simplify_(**fs_opts)
        self._maybe_convert(psi_b, dtype)

        if rehearse == "tn":
            return psi_b

        tree = psi_b.contraction_tree(output_inds=(), optimize=optimize)

        if rehearse:
            return rehearsal_dict(psi_b, tree)

        # perform the full contraction with the tree found
        c_b = psi_b.contract(
            all, output_inds=(), optimize=tree, backend=backend
        )

        return c_b

    def amplitude_rehearse(
        self,
        b="random",
        simplify_sequence="ADCRS",
        simplify_atol=1e-12,
        simplify_equalize_norms=True,
        optimize="auto-hq",
        dtype=None,
        rehearse=True,
    ):
        """Perform just the tensor network simplifications and contraction tree
        finding associated with computing a single amplitude (caching the
        results) but don't perform the actual contraction.

        Parameters
        ----------
        b : 'random', str or sequence of int
            The bitstring to rehearse computing the transition amplitude for,
            if ``'random'`` (the default) a random bitstring will be used.
        optimize : str, optional
            Contraction path optimizer to use for the marginal, can be a
            non-reusable path optimizer as only called once (though path won't
            be cached for later use in that case).
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.

        Returns
        -------
        dict

        """
        if b == "random":
            b = "r" * self.N

        return self.amplitude(
            b=b,
            optimize=optimize,
            dtype=dtype,
            rehearse=rehearse,
            simplify_sequence=simplify_sequence,
            simplify_atol=simplify_atol,
            simplify_equalize_norms=simplify_equalize_norms,
        )

    amplitude_tn = functools.partialmethod(amplitude_rehearse, rehearse="tn")

    def partial_trace(
        self,
        keep,
        optimize="auto-hq",
        simplify_sequence="ADCRS",
        simplify_atol=1e-12,
        simplify_equalize_norms=True,
        backend=None,
        dtype=None,
        rehearse=False,
    ):
        r"""Perform the partial trace on the circuit wavefunction, retaining
        only qubits in ``keep``, and making use of reverse lightcone
        cancellation:

        .. math::

            \rho_{\bar{q}} = Tr_{\bar{p}}
            |\psi_{\bar{q}} \rangle \langle \psi_{\bar{q}}|

        Where :math:`\bar{q}` is the set of qubits to keep,
        :math:`\psi_{\bar{q}}` is the circuit wavefunction only with gates in
        the causal cone of this set, and :math:`\bar{p}` is the remaining
        qubits.

        Parameters
        ----------
        keep : int or sequence of int
            The qubit(s) to keep as we trace out the rest.
        optimize : str, optional
            Contraction path optimizer to use for the reduced density matrix,
            can be a non-reusable path optimizer as only called once (though
            path won't be cached for later use in that case).
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        rehearse : bool or "tn", optional
            If ``True``, generate and cache the simplified tensor network and
            contraction tree but don't actually perform the contraction.
            Returns a dict with keys ``"tn"`` and ``'tree'`` with the tensor
            network that will be contracted and the corresponding contraction
            tree if so.

        Returns
        -------
        array or dict
        """

        if isinstance(keep, numbers.Integral):
            keep = (keep,)

        output_inds = tuple(map(self.ket_site_ind, keep)) + tuple(
            map(self.bra_site_ind, keep)
        )

        rho = self.get_rdm_lightcone_simplified(
            where=keep,
            seq=simplify_sequence,
            atol=simplify_atol,
            equalize_norms=simplify_equalize_norms,
        )
        self._maybe_convert(rho, dtype)

        if rehearse == "tn":
            return rho

        tree = rho.contraction_tree(output_inds=output_inds, optimize=optimize)

        if rehearse:
            return rehearsal_dict(rho, tree)

        # perform the full contraction with the tree found
        rho_dense = rho.contract(
            all,
            output_inds=output_inds,
            optimize=tree,
            backend=backend,
        ).data

        return ops.reshape(rho_dense, [2 ** len(keep), 2 ** len(keep)])

    partial_trace_rehearse = functools.partialmethod(
        partial_trace, rehearse=True
    )

    partial_trace_tn = functools.partialmethod(partial_trace, rehearse="tn")

    def local_expectation(
        self,
        G,
        where,
        optimize="auto-hq",
        simplify_sequence="ADCRS",
        simplify_atol=1e-12,
        simplify_equalize_norms=True,
        backend=None,
        dtype=None,
        rehearse=False,
    ):
        r"""Compute the a single expectation value of operator ``G``, acting on
        sites ``where``, making use of reverse lightcone cancellation.

        .. math::

            \langle \psi_{\bar{q}} | G_{\bar{q}} | \psi_{\bar{q}} \rangle

        where :math:`\bar{q}` is the set of qubits :math:`G` acts one and
        :math:`\psi_{\bar{q}}` is the circuit wavefunction only with gates in
        the causal cone of this set. If you supply a tuple or list of gates
        then the expectations will be computed simultaneously.

        Parameters
        ----------
        G : array or sequence[array]
            The raw operator(s) to find the expectation of.
        where : int or sequence of int
            Which qubits the operator acts on.
        optimize : str, optional
            Contraction path optimizer to use for the local expectation,
            can be a non-reusable path optimizer as only called once (though
            path won't be cached for later use in that case).
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        gate_opts : None or dict_like
            Options to use when applying ``G`` to the wavefunction.
        rehearse : bool or "tn", optional
            If ``True``, generate and cache the simplified tensor network and
            contraction tree but don't actually perform the contraction.
            Returns a dict with keys ``'tn'`` and ``'tree'`` with the tensor
            network that will be contracted and the corresponding contraction
            tree if so.

        Returns
        -------
        scalar, tuple[scalar] or dict
        """
        if isinstance(where, numbers.Integral):
            where = (where,)

        fs_opts = {
            "seq": simplify_sequence,
            "atol": simplify_atol,
            "equalize_norms": simplify_equalize_norms,
        }

        rho = self.get_rdm_lightcone_simplified(where=where, **fs_opts)
        k_inds = tuple(self.ket_site_ind(i) for i in where)
        b_inds = tuple(self.bra_site_ind(i) for i in where)

        if isinstance(G, (list, tuple)):
            # if we have multiple expectations create an extra indexed stack
            nG = len(G)
            G_data = do("stack", G)
            G_data = reshape(G_data, (nG,) + (2,) * 2 * len(where))
            output_inds = (rand_uuid(),)
        else:
            G_data = reshape(G, (2,) * 2 * len(where))
            output_inds = ()

        TG = Tensor(data=G_data, inds=output_inds + b_inds + k_inds)

        rhoG = rho | TG

        rhoG.full_simplify_(output_inds=output_inds, **fs_opts)
        self._maybe_convert(rhoG, dtype)

        if rehearse == "tn":
            return rhoG

        tree = rhoG.contraction_tree(
            output_inds=output_inds, optimize=optimize
        )

        if rehearse:
            return rehearsal_dict(rhoG, tree)

        g_ex = rhoG.contract(
            all,
            output_inds=output_inds,
            optimize=tree,
            backend=backend,
        )

        if isinstance(g_ex, Tensor):
            g_ex = tuple(g_ex.data)

        return g_ex

    local_expectation_rehearse = functools.partialmethod(
        local_expectation, rehearse=True
    )

    local_expectation_tn = functools.partialmethod(
        local_expectation, rehearse="tn"
    )

    def compute_marginal(
        self,
        where,
        fix=None,
        optimize="auto-hq",
        backend=None,
        dtype="complex64",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
        rehearse=False,
    ):
        """Compute the probability tensor of qubits in ``where``, given
        possibly fixed qubits in ``fix`` and tracing everything else having
        removed redundant unitary gates.

        Parameters
        ----------
        where : sequence of int
            The qubits to compute the marginal probability distribution of.
        fix : None or dict[int, str], optional
            Measurement results on other qubits to fix.
        optimize : str, optional
            Contraction path optimizer to use for the marginal, can be a
            non-reusable path optimizer as only called once (though path won't
            be cached for later use in that case).
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        rehearse : bool or "tn", optional
            Whether to perform the marginal contraction or just return the
            associated TN and contraction tree.
        """
        self._maybe_init_storage()

        # index trick to contract straight to reduced density matrix diagonal
        # rho_ii -> p_i (i.e. insert a COPY tensor into the norm)
        output_inds = [self.ket_site_ind(i) for i in where]

        fs_opts = {
            "seq": simplify_sequence,
            "atol": simplify_atol,
            "equalize_norms": simplify_equalize_norms,
        }

        # lightcone region is target qubit plus fixed qubits
        region = set(where)
        if fix is not None:
            region |= set(fix)
        region = tuple(sorted(region))

        # have we fixed or are measuring all qubits?
        final_marginal = len(region) == self.N

        # these both are cached and produce TN copies
        if final_marginal:
            # won't need to partially trace anything -> just need ket
            nm_lc = self.get_psi_simplified(**fs_opts)
        else:
            # can use lightcone cancellation on partially traced qubits
            nm_lc = self.get_rdm_lightcone_simplified(region, **fs_opts)
            # re-connect the ket and bra indices as taking diagonal
            nm_lc.reindex_(
                {self.bra_site_ind(i): self.ket_site_ind(i) for i in region}
            )

        if fix:
            # project (slice) fixed tensors with bitstring
            # this severs the indices connecting bra and ket on fixed sites
            nm_lc.isel_({self.ket_site_ind(i): b for i, b in fix.items()})

        # having sliced we can do a final simplify
        nm_lc.full_simplify_(output_inds=output_inds, **fs_opts)

        # for stability with very small probabilities, scale by average prob
        if fix is not None:
            nfact = 2 ** len(fix)
            if final_marginal:
                nm_lc.multiply_(nfact**0.5, spread_over="all")
            else:
                nm_lc.multiply_(nfact, spread_over="all")

        # cast to desired data type
        self._maybe_convert(nm_lc, dtype)

        if rehearse == "tn":
            return nm_lc

        # NB. the tree isn't *neccesarily* the same each time due to the post
        #     projection full simplify, however there is also the lower level
        #     contraction path cache if the structure generated *is* the same
        #     so still pretty efficient to just overwrite
        tree = nm_lc.contraction_tree(
            output_inds=output_inds,
            optimize=optimize,
        )

        if rehearse:
            return rehearsal_dict(nm_lc, tree)

        # perform the full contraction with the tree found
        p_marginal = abs(
            nm_lc.contract(
                all,
                output_inds=output_inds,
                optimize=tree,
                backend=backend,
            ).data
        )

        if final_marginal:
            # we only did half the ket contraction so need to square
            p_marginal = p_marginal**2

        if fix is not None:
            p_marginal = p_marginal / nfact

        return p_marginal

    compute_marginal_rehearse = functools.partialmethod(
        compute_marginal, rehearse=True
    )

    compute_marginal_tn = functools.partialmethod(
        compute_marginal, rehearse="tn"
    )

    def calc_qubit_ordering(self, qubits=None, method="greedy-lightcone"):
        """Get a order to measure ``qubits`` in, by greedily choosing whichever
        has the smallest reverse lightcone followed by whichever expands this
        lightcone *least*.

        Parameters
        ----------
        qubits : None or sequence of int
            The qubits to generate a lightcone ordering for, if ``None``,
            assume all qubits.

        Returns
        -------
        tuple[int]
            The order to 'measure' qubits in.
        """
        self._maybe_init_storage()

        if qubits is None:
            qubits = tuple(range(self.N))
        else:
            qubits = tuple(sorted(qubits))

        key = ("lightcone_ordering", method, qubits)

        # check the cache first
        if key in self._storage:
            return self._storage[key]

        if method == "greedy-lightcone":
            cone = set()
            lctgs = {
                i: set(self.get_reverse_lightcone_tags(i)) for i in qubits
            }

            order = []
            while lctgs:
                # get the next qubit which adds least num gates to lightcone
                next_qubit = min(lctgs, key=lambda i: len(lctgs[i] - cone))
                cone |= lctgs.pop(next_qubit)
                order.append(next_qubit)

        else:
            # use graph distance based hierachical clustering
            psi = self.get_psi_simplified("R")
            qubit_inds = tuple(map(psi.site_ind, qubits))
            tids = psi._get_tids_from_inds(qubit_inds, "any")
            matcher = re.compile(psi.site_ind_id.format(r"(\d+)"))
            order = []
            for tid in psi.compute_hierarchical_ordering(tids, method=method):
                t = psi.tensor_map[tid]
                for ind in t.inds:
                    for sq in matcher.findall(ind):
                        order.append(int(sq))

        order = self._storage[key] = tuple(order)
        return order

    def _parse_qubits_order(self, qubits=None, order=None):
        """Simply initializes the default of measuring all qubits, and the
        default order, or checks that ``order`` is a permutation of ``qubits``.
        """
        if qubits is None:
            qubits = range(self.N)
        if order is None:
            order = self.calc_qubit_ordering(qubits)
        elif set(qubits) != set(order):
            raise ValueError("``order`` must be a permutation of ``qubits``.")

        return qubits, order

    def _group_order(self, order, group_size=1):
        """Take the qubit ordering ``order`` and batch it in groups of size
        ``group_size``, sorting the qubits (for caching reasons) within each
        group.
        """
        return tuple(
            tuple(sorted(g)) for g in partition_all(group_size, order)
        )

    def get_qubit_distances(self, method="dijkstra", alpha=2):
        """Get a nested dictionary of qubit distances. This is computed from a
        graph representing qubit interactions. The graph has an edge between
        qubits if they are acted on by the same gate, and the distance-weight
        of the edge is exponentially small in the number of gates between them.

        Parameters
        ----------
        method : {'dijkstra', 'resistance'}, optional
            The method to use to compute the qubit distances. See
            :func:`networkx.all_pairs_dijkstra_path_length` and
            :func:`networkx.resistance_distance`.
        alpha : float, optional
            The distance weight between qubits is ``alpha**(num_gates - 1 )``.

        Returns
        -------
        dict[int, dict[int, float]]
            The distance between each pair of qubits, accessed like
            ``distances[q1][q2]``. If two qubits are not connected, the
            distance is missing.
        """
        import networkx as nx

        G = nx.Graph()
        for g in self.gates:
            for q1, q2 in itertools.combinations(g.qubits, 2):
                if G.has_edge(q1, q2):
                    G[q1][q2]["weight"] /= alpha
                else:
                    G.add_edge(q1, q2, weight=1)

        if method == "dijkstra":
            distances = dict(
                nx.all_pairs_dijkstra_path_length(G, weight="weight")
            )
        elif method == "resistance":
            distances = nx.resistance_distance(G, weight="weight")
        else:
            raise ValueError(f"Unknown method {method}.")

        return distances

    def reordered_gates_dfs_clustered(self):
        """Get the gates reordered by a depth first search traversal of the
        multi-qubit gate graph that greedily selects successive gates which
        are 'close' in graph distance, and shifts single qubit gates to be
        adjacent to multi-qubit gates where possible.
        """
        # first we make a directed graph of the multi-qubit gates
        successors = {}
        predecessors = {}
        single_qubit_stacks = {}
        single_qubit_predecessors = {}
        last_gates = {}
        queue = []

        for i, g in enumerate(self.gates):
            if g.total_qubit_count == 1:
                # lazily accumulate single qubit gates
                (q,) = g.qubits
                single_qubit_stacks.setdefault(q, []).append(i)

            else:
                pi = predecessors[i] = []
                sqpi = single_qubit_predecessors[i] = []

                for q in g.qubits:
                    # collect any single qubit gates acting on this qubit
                    sqpi.extend(single_qubit_stacks.pop(q, []))

                    if q in last_gates:
                        # qubit has already been acted on -> have an edge
                        h = last_gates[q]
                        # mark h as a predecessor of i
                        pi.append(h)
                        # mark i as a successor of h
                        successors.setdefault(h, []).append(i)

                    # mark qubit as acted on
                    last_gates[q] = i

                if len(pi) == 0:
                    # no predecessors -> is possible starting multiqubit gate
                    queue.append(i)

        # then we traverse the multi-qubit gates in a depth first, topological
        # order, breaking ties by minimizing the distance between active qubits
        distances = self.get_qubit_distances()

        def gate_distance(i, j):
            qis = self.gates[i].qubits
            qjs = self.gates[j].qubits
            return min(
                distances[q1].get(q2, float("inf")) for q1 in qis for q2 in qjs
            )

        # sort initial queue by qubit with smallest index
        queue.sort(key=lambda i: min(self.gates[i].qubits))
        new_gates = []

        while queue:
            i = queue.pop(0)

            # first flush any single qubit gates acting on the qubits of gate i
            new_gates.extend(
                self.gates[j] for j in single_qubit_predecessors.pop(i, [])
            )
            # then add the gate itself
            new_gates.append(self.gates[i])

            # then remove i as a predecessor of its successors
            for j in successors.pop(i, []):
                pj = predecessors[j]
                pj.remove(i)
                if not pj:
                    # j has no more predecessors -> can be added to queue
                    queue.append(j)

            # check if this is the last time q is acted on,
            # if so flush any remaining single qubit gates
            for q in self.gates[i].qubits:
                if last_gates[q] == i:
                    # qubit has been acted on for the last time
                    new_gates.extend(
                        self.gates[j] for j in single_qubit_stacks.pop(q, [])
                    )

            # sort the queue of possible next gates
            queue.sort(key=lambda k: gate_distance(i, k))

        # flush any remaining single qubit gates
        for q in sorted(single_qubit_stacks):
            new_gates.extend(self.gates[j] for j in single_qubit_stacks.pop(q))

        return new_gates

    def sample(
        self,
        C,
        qubits=None,
        order=None,
        group_size=10,
        max_marginal_storage=2**20,
        seed=None,
        optimize="auto-hq",
        backend=None,
        dtype="complex64",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
    ):
        r"""Sample the circuit given by ``gates``, ``C`` times, using lightcone
        cancelling and caching marginal distribution results. This is a
        generator. This proceeds as a chain of marginal computations.

        Assuming we have ``group_size=1``, and some ordering of the qubits,
        :math:`\{q_0, q_1, q_2, q_3, \ldots\}` we first compute:

        .. math::

            p(q_0) = \mathrm{diag} \mathrm{Tr}_{1, 2, 3,\ldots}
            | \psi_{0} \rangle \langle \psi_{0} |

        I.e. simply the probability distribution on a single qubit, conditioned
        on nothing. The subscript on :math:`\psi` refers to the fact that we
        only need gates from the causal cone of qubit 0.
        From this we can sample an outcome, either 0 or 1, if we
        call this :math:`r_0` we can then move on to the next marginal:

        .. math::

            p(q_1 | r_0) = \mathrm{diag} \mathrm{Tr}_{2, 3,\ldots}
            \langle r_0
            | \psi_{0, 1} \rangle \langle \psi_{0, 1} |
            r_0 \rangle

        I.e. the probability distribution of the next qubit, given our prior
        result. We can sample from this to get :math:`r_1`. Then we compute:

        .. math::

            p(q_2 | r_0 r_1) = \mathrm{diag} \mathrm{Tr}_{3,\ldots}
            \langle r_0 r_1
            | \psi_{0, 1, 2} \rangle \langle \psi_{0, 1, 2} |
            r_0 r_1 \rangle

        Eventually we will reach the 'final marginal', which we can compute as

        .. math::

            |\langle r_0 r_1 r_2 r_3 \ldots | \psi \rangle|^2

        since there is nothing left to trace out.

        Parameters
        ----------
        C : int
            The number of times to sample.
        qubits : None or sequence of int, optional
            Which qubits to measure, defaults (``None``) to all qubits.
        order : None or sequence of int, optional
            Which order to measure the qubits in, defaults (``None``) to an
            order based on greedily expanding the smallest reverse lightcone.
            If specified it should be a permutation of ``qubits``.
        group_size : int, optional
            How many qubits to group together into marginals, the larger this
            is the fewer marginals need to be computed, which can be faster at
            the cost of higher memory. The marginal themselves will each be
            of size ``2**group_size``.
        max_marginal_storage : int, optional
            The total cumulative number of marginal probabilites to cache, once
            this is exceeded caching will be turned off.
        seed : None or int, optional
            A random seed, passed to ``numpy.random.seed`` if given.
        optimize : str, optional
            Contraction path optimizer to use for the marginals, shouldn't be
            a non-reusable path optimizer as called on many different TNs.
            Passed to :func:`cotengra.array_contract_tree`.
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.

        Yields
        ------
        bitstrings : sequence of str
        """
        # init TN norms, contraction trees, and marginals
        self._maybe_init_storage()

        rng = np.random.default_rng(seed)

        # which qubits and an ordering e.g. (2, 3, 4, 5), (5, 3, 4, 2)
        qubits, order = self._parse_qubits_order(qubits, order)

        # group the ordering e.g. ((5, 3), (4, 2))
        groups = self._group_order(order, group_size)

        result = dict()
        for _ in range(C):
            for where in groups:
                # key - (tuple[int] where, tuple[tuple[int q, str b])
                # value  - marginal probability distribution of `where` given
                #     prior results, as an ndarray
                # e.g. ((2,), ((0, '0'), (1, '0'))): array([1., 0.]), means
                #     prob(qubit2='0')=1 given qubit0='0' and qubit1='0'
                #     prob(qubit2='1')=0 given qubit0='0' and qubit1='0'
                key = (where, tuple(sorted(result.items())))
                if key not in self._sampled_conditionals:
                    # compute p(qs=x | current bitstring)
                    p = self.compute_marginal(
                        where=where,
                        fix=result,
                        optimize=optimize,
                        backend=backend,
                        dtype=dtype,
                        simplify_sequence=simplify_sequence,
                        simplify_atol=simplify_atol,
                        simplify_equalize_norms=simplify_equalize_norms,
                    )
                    p = do("to_numpy", p).astype("float64")
                    p /= p.sum()

                    if self._marginal_storage_size <= max_marginal_storage:
                        self._sampled_conditionals[key] = p
                        self._marginal_storage_size += p.size
                else:
                    p = self._sampled_conditionals[key]

                # the sampled bitstring e.g. '1' or '001010101'
                b_where = sample_bitstring_from_prob_ndarray(p, seed=rng)

                # split back into individual qubit results
                for q, b in zip(where, b_where):
                    result[q] = b

            yield "".join(result[i] for i in qubits)
            result.clear()

    def sample_rehearse(
        self,
        qubits=None,
        order=None,
        group_size=10,
        result=None,
        optimize="auto-hq",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
        rehearse=True,
        progbar=False,
    ):
        """Perform the preparations and contraction tree findings for
        :meth:`~quimb.tensor.circuit.Circuit.sample`, caching various
        intermedidate objects, but don't perform the main contractions.

        Parameters
        ----------
        qubits : None or sequence of int, optional
            Which qubits to measure, defaults (``None``) to all qubits.
        order : None or sequence of int, optional
            Which order to measure the qubits in, defaults (``None``) to an
            order based on greedily expanding the smallest reverse lightcone.
        group_size : int, optional
            How many qubits to group together into marginals, the larger this
            is the fewer marginals need to be computed, which can be faster at
            the cost of higher memory. The marginal's size itself is
            exponential in ``group_size``.
        result : None or dict[int, str], optional
            Explicitly check the computational cost of this result, assumed to
            be all zeros if not given.
        optimize : str, optional
            Contraction path optimizer to use for the marginals, shouldn't be
            a non-reusable path optimizer as called on many different TNs.
            Passed to :func:`cotengra.array_contract_tree`.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        progbar : bool, optional
            Whether to show the progress of finding each contraction tree.

        Returns
        -------
        dict[tuple[int], dict]
            One contraction tree object per grouped marginal computation.
            The keys of the dict are the qubits the marginal is computed for,
            the values are a dict containing a representative simplified tensor
            network (key: 'tn') and the main contraction tree (key: 'tree').
        """
        # init TN norms, contraction trees, and marginals
        self._maybe_init_storage()
        qubits, order = self._parse_qubits_order(qubits, order)
        groups = self._group_order(order, group_size)

        if result is None:
            result = {q: "r" for q in qubits}

        fix = {}
        tns_and_trees = {}

        for where in _progbar(groups, disable=not progbar):
            tns_and_trees[where] = self.compute_marginal(
                where=where,
                fix=fix,
                optimize=optimize,
                simplify_sequence=simplify_sequence,
                simplify_atol=simplify_atol,
                simplify_equalize_norms=simplify_equalize_norms,
                rehearse=rehearse,
            )

            # set the result of qubit ``q`` arbitrarily
            for q in where:
                fix[q] = result[q]

        return tns_and_trees

    sample_tns = functools.partialmethod(sample_rehearse, rehearse="tn")

    def sample_chaotic(
        self,
        C,
        marginal_qubits,
        fix=None,
        max_marginal_storage=2**20,
        seed=None,
        optimize="auto-hq",
        backend=None,
        dtype="complex64",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
    ):
        r"""Sample from this circuit, *assuming* it to be chaotic. Which is to
        say, only compute and sample correctly from the final marginal,
        assuming that the distribution on the other qubits is uniform.
        Given ``marginal_qubits=5`` for instance, for each sample a random
        bit-string :math:`r_0 r_1 r_2 \ldots r_{N - 6}` for the remaining
        :math:`N - 5` qubits will be chosen, then the final marginal will be
        computed as

        .. math::

            p(q_{N-5}q_{N-4}q_{N-3}q_{N-2}q_{N-1}
            | r_0 r_1 r_2 \ldots r_{N-6})
            =
            |\langle r_0 r_1 r_2 \ldots r_{N - 6} | \psi \rangle|^2

        and then sampled from. Note the expression on the right hand side has
        5 open indices here and so is a tensor, however if ``marginal_qubits``
        is not too big then the cost of contracting this is very similar to
        a single amplitude.

        .. note::

            This method *assumes* the circuit is chaotic, if its not, then the
            samples produced will not be an accurate representation of the
            probability distribution.

        Parameters
        ----------
        C : int
            The number of times to sample.
        marginal_qubits : int or sequence of int
            The number of qubits to treat as marginal, or the actual qubits. If
            an int is given then the qubits treated as marginal will be
            ``circuit.calc_qubit_ordering()[:marginal_qubits]``.
        fix : None or dict[int, str], optional
            Measurement results on other qubits to fix. These will be randomly
            sampled if ``fix`` is not given or a qubit is missing.
        seed : None or int, optional
            A random seed, passed to ``numpy.random.seed`` if given.
        optimize : str, optional
            Contraction path optimizer to use for the marginal, can be a
            non-reusable path optimizer as only called once (though path won't
            be cached for later use in that case).
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.

        Yields
        ------
        str
        """
        # init TN norms, contraction trees, and marginals
        self._maybe_init_storage()
        qubits = tuple(range(self.N))

        rng = np.random.default_rng(seed)

        # choose which qubits to treat as marginal - ideally 'towards one side'
        #     to increase contraction efficiency
        if isinstance(marginal_qubits, numbers.Integral):
            marginal_qubits = self.calc_qubit_ordering()[:marginal_qubits]
        where = tuple(sorted(marginal_qubits))

        # we will uniformly sample, and post-select on, the remaining qubits
        fix_qubits = tuple(q for q in qubits if q not in where)

        result = dict()
        for _ in range(C):
            # generate a random bit-string for the fixed qubits
            for q in fix_qubits:
                if (fix is None) or (q not in fix):
                    result[q] = rng.choice(("0", "1"))
                else:
                    result[q] = fix[q]

            # compute the remaining marginal
            key = (where, tuple(sorted(result.items())))
            if key not in self._sampled_conditionals:
                p = self.compute_marginal(
                    where=where,
                    fix=result,
                    optimize=optimize,
                    backend=backend,
                    dtype=dtype,
                    simplify_sequence=simplify_sequence,
                    simplify_atol=simplify_atol,
                    simplify_equalize_norms=simplify_equalize_norms,
                )
                p = do("to_numpy", p).astype("float64")
                p /= p.sum()

                if self._marginal_storage_size <= max_marginal_storage:
                    self._sampled_conditionals[key] = p
                    self._marginal_storage_size += p.size
            else:
                p = self._sampled_conditionals[key]

            # sample a bit-string for the marginal qubits
            b_where = sample_bitstring_from_prob_ndarray(p)

            # split back into individual qubit results
            for q, b in zip(where, b_where):
                result[q] = b

            yield "".join(result[i] for i in qubits)
            result.clear()

    def sample_chaotic_rehearse(
        self,
        marginal_qubits,
        result=None,
        optimize="auto-hq",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
        dtype="complex64",
        rehearse=True,
    ):
        """Rehearse chaotic sampling (perform just the TN simplifications and
        contraction tree finding).

        Parameters
        ----------
        marginal_qubits : int or sequence of int
            The number of qubits to treat as marginal, or the actual qubits. If
            an int is given then the qubits treated as marginal will be
            ``circuit.calc_qubit_ordering()[:marginal_qubits]``.
        result : None or dict[int, str], optional
            Explicitly check the computational cost of this result, assumed to
            be all zeros if not given.
        optimize : str, optional
            Contraction path optimizer to use for the marginal, can be a
            non-reusable path optimizer as only called once (though path won't
            be cached for later use in that case).
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        dtype : str, optional
            Data type to cast the TN to before contraction.

        Returns
        -------
        dict[tuple[int], dict]
            The contraction path information for the main computation, the key
            is the qubits that formed the final marginal. The value is itself a
            dict with keys ``'tn'`` - a representative tensor network - and
            ``'tree'`` - the contraction tree.
        """

        # init TN norms, contraction trees, and marginals
        self._maybe_init_storage()
        qubits = tuple(range(self.N))

        if isinstance(marginal_qubits, numbers.Integral):
            marginal_qubits = self.calc_qubit_ordering()[:marginal_qubits]
        where = tuple(sorted(marginal_qubits))

        fix_qubits = tuple(q for q in qubits if q not in where)

        if result is None:
            fix = {q: "0" for q in fix_qubits}
        else:
            fix = {q: result[q] for q in fix_qubits}

        rehs = self.compute_marginal(
            where=where,
            fix=fix,
            optimize=optimize,
            simplify_sequence=simplify_sequence,
            simplify_atol=simplify_atol,
            simplify_equalize_norms=simplify_equalize_norms,
            dtype=dtype,
            rehearse=rehearse,
        )

        if rehearse == "tn":
            return rehs

        return {where: rehs}

    sample_chaotic_tn = functools.partialmethod(
        sample_chaotic_rehearse, rehearse="tn"
    )

    def get_gate_by_gate_circuits(self, group_size=10):
        """Get a sequence of circuits by partitioning the gates into groups
        such circuit `i + 1` acts on at most ``group_size`` new qubits compared
        to circuit `i`.

        Parameters
        ----------
        group_size : int, optional
            The maximum number of new qubits that can be acted on by a circuit
            compared to its predecessor.

        Returns
        -------
        Sequence[dict]
            A sequence of dicts, each with keys ``'circuit'`` and ``'where'``,
            where the former is a :class:`~quimb.tensor.circuit.Circuit` and
            the latter the tuple of new qubits that it acts on comparaed to
            the previous circuit.
        """
        circs = [self.__class__(self.N)]
        groups = []
        current_group = set()

        # this ensures that single qubit gates are always adjacent to
        # multi-qubit gates and will thus always be included in the same group
        gates = self.reordered_gates_dfs_clustered()

        for gate in gates:
            # if we were to add next gate, how many new qubits would we have?
            next_group = current_group.union(gate.qubits)
            if len(next_group) > group_size:
                # over the limit: flush a copy of the current circuit and group
                groups.append(tuple(sorted(current_group)))
                circs.append(circs[-1].copy())
                # start a new group
                current_group = set(gate.qubits)
            else:
                # add the gate to the current group
                current_group = next_group
            circs[-1].apply_gate(gate)

        # add the final group corresponding to circs[-1]
        groups.append(tuple(sorted(current_group)))

        return tuple({"circuit": c, "where": g} for c, g in zip(circs, groups))

    def sample_gate_by_gate(
        self,
        C,
        group_size=10,
        seed=None,
        max_marginal_storage=2**20,
        optimize="auto-hq",
        backend=None,
        dtype="complex64",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
    ):
        """Sample this circuit using the gate-by-gate method, where we 'evolve'
        a result bitstring by sequentially including more and more gates, at
        each step updating the result by computing a full conditional marginal.
        See "How to simulate quantum measurement without computing marginals"
        by Sergey Bravyi, David Gosset, Yinchen Liu
        (https://arxiv.org/abs/2112.08499). The overall complexity of this is
        guaranteed to be similar to that of computing a single amplitude which
        can be much better than the naive "qubit-by-qubit" (`.sample`) method.
        However, it requires evaluting a number of tensor networks that scales
        linearly with the number of gates which can offset any practical
        advantages for shallow circuits for example.

        Parameters
        ----------
        C : int
            The number of samples to generate.
        group_size : int, optional
            The maximum number of qubits that can be acted on by a circuit
            compared to its predecessor. This will be the dimension of the
            marginal computed at each step.
        seed : None or int, optional
            A random seed, passed to ``numpy.random.seed`` if given.
        max_marginal_storage : int, optional
            The total cumulative number of marginal probabilites to cache, once
            this is exceeded caching will be turned off.
        optimize : str, optional
            Contraction path optimizer to use for the marginals, shouldn't be
            a non-reusable path optimizer as called on many different TNs.
            Passed to :func:`cotengra.array_contract_tree`.
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        rehearse : bool, optional
            If ``True``, generate and cache the simplified tensor network and
            contraction tree but don't actually perform the contraction.
            Returns a dict with keys ``'tn'`` and ``'tree'`` with the tensor
            network that will be contracted and the corresponding contraction
            tree if so.

        Yields
        ------
        str
        """
        self._maybe_init_storage()

        rng = np.random.default_rng(seed)

        key = ("gate_by_gate_circuits", group_size)
        try:
            circs_wheres = self._storage[key]
        except KeyError:
            circs_wheres = self.get_gate_by_gate_circuits(group_size)
            self._storage[key] = circs_wheres

        for _ in range(C):
            # start with all qubits in the |0> state
            result = {q: "0" for q in range(self.N)}

            for circ_where in circs_wheres:
                # get the next circuit and the new group of qubits
                circ_g = circ_where["circuit"]
                where = circ_where["where"]

                # remove the new group of qubits from our current result
                for q in where:
                    result.pop(q)

                # check if we have already computed the conditional
                key = (where, tuple(sorted(result.items())))

                if key not in circ_g._sampled_conditionals:
                    p = circ_g.compute_marginal(
                        where,
                        fix=result,
                        optimize=optimize,
                        backend=backend,
                        dtype=dtype,
                        simplify_sequence=simplify_sequence,
                        simplify_atol=simplify_atol,
                        simplify_equalize_norms=simplify_equalize_norms,
                    )
                    p /= p.sum()

                    if circ_g._marginal_storage_size <= max_marginal_storage:
                        circ_g._sampled_conditionals[key] = p
                        circ_g._marginal_storage_size += p.size
                else:
                    p = circ_g._sampled_conditionals[key]

                # sample a configuration for our new group
                b_where = sample_bitstring_from_prob_ndarray(p, seed=rng)

                # update the fixed qubits given new group result
                for q, qx in zip(where, b_where):
                    result[q] = qx

            yield "".join(result[i] for i in range(self.N))

    def sample_gate_by_gate_rehearse(
        self,
        group_size=10,
        optimize="auto-hq",
        dtype="complex64",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
        rehearse=True,
        progbar=False,
    ):
        """Perform the preparations and contraction tree findings for
        :meth:`~quimb.tensor.circuit.Circuit.sample_gate_by_gate`, caching
        various intermedidate objects, but don't perform the main contractions.

        Parameters
        ----------
        group_size : int, optional
            The maximum number of qubits that can be acted on by a circuit
            compared to its predecessor. This will be the dimension of the
            marginal computed at each step.
        optimize : str, optional
            Contraction path optimizer to use for the marginals, shouldn't be
            a non-reusable path optimizer as called on many different TNs.
            Passed to :func:`cotengra.array_contract_tree`.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        rehearse : True or "tn", optional
            If ``True``, generate and cache the simplified tensor network and
            contraction tree but don't actually perform the contraction. If
            "tn", only generate the simplified tensor networks.

        Returns
        -------
        Sequence[dict] or Sequence[TensorNetwork]
        """
        self._maybe_init_storage()

        key = ("gate_by_gate_circuits", group_size)
        try:
            circs_wheres = self._storage[key]
        except KeyError:
            circs_wheres = self.get_gate_by_gate_circuits(group_size)
            self._storage[key] = circs_wheres

        rehs = []
        result = {q: "0" for q in range(self.N)}

        for circs_wheres in _progbar(circs_wheres, disable=not progbar):
            # get the next circuit and the new group of qubits
            circ_g = circs_wheres["circuit"]
            where = circs_wheres["where"]

            # remove the new group of qubits from our current result
            for q in where:
                result.pop(q)

            r = circ_g.compute_marginal(
                where,
                fix=result,
                optimize=optimize,
                dtype=dtype,
                simplify_sequence=simplify_sequence,
                simplify_atol=simplify_atol,
                simplify_equalize_norms=simplify_equalize_norms,
                rehearse=rehearse,
            )

            if rehearse != "tn":
                r["where"] = where
                r["circuit"] = circ_g

            rehs.append(r)

            # update the fixed qubits with randomly rotated results so we
            # don't get zero probability networks when simplifying
            for q in where:
                result[q] = "r"

        return rehs

    sample_gate_by_gate_tns = functools.partialmethod(
        sample_gate_by_gate_rehearse, rehearse="tn"
    )

    def to_dense(
        self,
        reverse=False,
        optimize="auto-hq",
        simplify_sequence="R",
        simplify_atol=1e-12,
        simplify_equalize_norms=True,
        backend=None,
        dtype=None,
        rehearse=False,
    ):
        """Generate the dense representation of the final wavefunction.

        Parameters
        ----------
        reverse : bool, optional
            Whether to reverse the order of the subsystems, to match the
            convention of qiskit for example.
        optimize : str, optional
            Contraction path optimizer to use for the contraction, can be a
            non-reusable path optimizer as only called once (though path won't
            be cached for later use in that case).
        dtype : dtype or str, optional
            If given, convert the tensors to this dtype prior to contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        backend : str, optional
            Backend to perform the contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        rehearse : bool, optional
            If ``True``, generate and cache the simplified tensor network and
            contraction tree but don't actually perform the contraction.
            Returns a dict with keys ``'tn'`` and ``'tree'`` with the tensor
            network that will be contracted and the corresponding contraction
            tree if so.

        Returns
        -------
        psi : qarray
            The densely represented wavefunction with ``dtype`` data.
        """
        psi = self.get_psi_simplified(
            seq=simplify_sequence,
            atol=simplify_atol,
            equalize_norms=simplify_equalize_norms,
        )
        self._maybe_convert(psi, dtype)

        if rehearse == "tn":
            return psi

        output_inds = tuple(map(psi.site_ind, range(self.N)))
        if reverse:
            output_inds = output_inds[::-1]

        tree = psi.contraction_tree(output_inds=output_inds, optimize=optimize)

        if rehearse:
            return rehearsal_dict(psi, tree)

        # perform the full contraction with the path found
        psi_tensor = psi.contract(
            all,
            output_inds=output_inds,
            optimize=tree,
            backend=backend,
        ).data

        k = ops.reshape(psi_tensor, (-1, 1))

        if isinstance(k, np.ndarray):
            k = qu.qarray(k)

        return k

    to_dense_rehearse = functools.partialmethod(to_dense, rehearse=True)

    to_dense_tn = functools.partialmethod(to_dense, rehearse="tn")

    def schrodinger_contract(self, *args, **contract_opts):
        ntensor = self._psi.num_tensors
        path = [(0, 1)] + [(0, i) for i in reversed(range(1, ntensor - 1))]
        return self.psi.contract(*args, optimize=path, **contract_opts)

    def xeb_ex(
        self,
        optimize="auto-hq",
        simplify_sequence="R",
        simplify_atol=1e-12,
        simplify_equalize_norms=True,
        dtype=None,
        backend=None,
        autojit=False,
        progbar=False,
        **contract_opts,
    ):
        """Compute the exactly expected XEB for this circuit. The main feature
        here is that if you supply a cotengra optimizer that searches for
        sliced indices then the XEB will be computed without constructing the
        full wavefunction.

        Parameters
        ----------
        optimize : str or PathOptimizer, optional
            Contraction path optimizer.
        simplify_sequence : str, optional
            Simplifications to apply to tensor network prior to contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        backend : str, optional
            Convert tensors to, and then use contractions from, this library.
        autojit : bool, optional
            Apply ``autoray.autojit`` to the contraciton and map-reduce.
        progbar : bool, optional
            Show progress in terms of number of wavefunction chunks processed.
        """
        # get potentially simplified TN of full wavefunction
        psi = self.to_dense_tn(
            simplify_sequence=simplify_sequence,
            simplify_atol=simplify_atol,
            simplify_equalize_norms=simplify_equalize_norms,
            dtype=dtype,
        )

        # find a possibly sliced contraction tree
        output_inds = tuple(map(psi.site_ind, range(self.N)))
        tree = psi.contraction_tree(optimize=optimize, output_inds=output_inds)

        arrays = psi.arrays
        if backend is not None:
            arrays = [do("array", x, like=backend) for x in arrays]

        # perform map-reduce style computation over output wavefunction chunks
        # so we don't need entire wavefunction in memory at same time
        chunks = tree.gen_output_chunks(
            arrays, autojit=autojit, **contract_opts
        )
        if progbar:
            chunks = _progbar(chunks, total=tree.nchunks)

        def f(chunk):
            return do("sum", do("abs", chunk) ** 4)

        if autojit:
            # since we convert the arrays above, the jit backend is
            # automatically inferred
            from autoray import autojit

            f = autojit(f)

        p2sum = functools.reduce(operator.add, map(f, chunks))
        return 2**self.N * p2sum - 1

    def apply_gates(self, gates, progbar=False, **gate_opts):
        # exact/dense representation drops trivial dimensions after a batch
        super().apply_gates(gates, progbar=progbar, **gate_opts)
        self._psi.squeeze_()


class CircuitDense(Circuit):
    """Quantum circuit simulation keeping the state in full dense form."""

    def __init__(
        self,
        N=None,
        psi0=None,
        gate_opts=None,
        gate_contract=True,
        tags=None,
        convert_eager=True,
        **circuit_opts,
    ):
        gate_opts = ensure_dict(gate_opts)
        gate_opts.setdefault("contract", gate_contract)
        gate_opts.setdefault("convert_eager", convert_eager)
        super().__init__(N, psi0, gate_opts, tags, **circuit_opts)

    def get_psi(self):
        """Get the dense wavefunction as a length one tensor network, with a
        ``Dense1D`` view.
        """
        t = self._psi ^ ...
        psi = t.as_network()
        psi.view_as_(Dense1D, like=self._psi, L=self.N)
        return psi

    def get_uni(self, transposed=False):
        raise NotImplementedError(
            "You can't extract the circuit unitary TN from a "
            "``CircuitDense``, which contracts the state as it goes."
        )

    def calc_qubit_ordering(self, qubits=None):
        """Qubit ordering doesn't matter for a dense wavefunction."""
        if qubits is None:
            return tuple(range(self.N))
        else:
            return tuple(sorted(qubits))

    def get_psi_reverse_lightcone(self, where, keep_psi0=False):
        """Override ``get_psi_reverse_lightcone`` as for a dense wavefunction
        the lightcone is not meaningful.
        """
        return self.psi
