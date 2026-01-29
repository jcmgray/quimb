import autoray as ar

import quimb.tensor as qtn
from quimb.tensor.belief_propagation.bp_common import (
    BeliefPropagationCommon,
    combine_local_contractions,
    create_lazy_community_edge_map,
)
from quimb.utils import oset


class MPS1BP(BeliefPropagationCommon):
    """Matrix product state 1-norm lazy belief propagation.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to perform belief propagation on. It should have a
        'graph + 1' structure, i.e. some geometry (given by `site_tags`)
        repeated in layers (given by `layer_tags`).
    max_bond : int
        The maximum bond dimension to use when compressing the message update
        tensor network.
    layer_tags : sequence[str, ...]
        The tags which identify the layers of the tensor network. Each tensor
        should have exactly one of these tags.
    site_tags : sequence[str, ...], optional
        The tags which identify the sites of the tensor network. If None, these
        are assumed to be default `tn.site_tags`.
    cutoff : float, optional
        The cutoff to use when compressing the message update tensor network.
        Default is 0.0.
    method : str, optional
        The method to use when compressing the message update tensor network.
        See `qtn.tensor_network_1d_compress` for options.
    fit_messages : bool, optional
        Whether to use the current message as a fit target when using 'fit'
        or 'srcmps' compression methods. Default is True.
    compress_opts : dict, optional
        Additional options to pass to `qtn.tensor_network_1d_compress`.
    optimize : str, optional
        The optimization strategy to use when contracting tensor networks.
    kwargs
        Additional keyword arguments to pass to the base
        `BeliefPropagationCommon` class.
    """

    def __init__(
        self,
        tn,
        max_bond,
        layer_tags,
        site_tags=None,
        *,
        cutoff=0.0,
        method="srcmps",
        fit_messages=True,
        compress_opts=None,
        damping=0.0,
        update="sequential",
        normalize="L2",
        distance="L2",
        local_convergence=True,
        optimize="auto-hq",
        **kwargs,
    ):
        if damping != 0.0:
            raise NotImplementedError("MPS1BP does not yet support damping.")
        if normalize not in (None, "L2"):
            raise NotImplementedError(
                "MPS1BP only supports L2 normalization of messages."
            )
        if distance not in (None, "L2"):
            raise NotImplementedError(
                "MPS1BP only supports L2 distance for messages."
            )

        super().__init__(
            tn,
            normalize=normalize,
            distance=distance,
            update=update,
            **kwargs,
        )
        self.local_convergence = local_convergence

        # create the lazy community network structure
        if site_tags is None:
            self.site_tags = tuple(tn.site_tags)
        else:
            self.site_tags = tuple(site_tags)
        (
            self.edges,
            self.neighbors,
            self.local_tns,
            self.touch_map,
        ) = create_lazy_community_edge_map(
            tn,
            site_tags=site_tags,
            rank_simplify=False,
        )
        self.touched = oset()
        # the tags describing the '+1' dimension
        self.layer_tags = tuple(layer_tags)

        # compress and contraction options
        self.max_bond = max_bond
        self.compress_opts = compress_opts or {}
        self.compress_opts.setdefault("method", method)
        self.compress_opts.setdefault("max_bond", max_bond)
        self.compress_opts.setdefault("cutoff", cutoff)
        self.fit_messages = fit_messages
        self.backend = ar.infer_backend(next(t.data for t in tn))
        self.optimize = optimize

        # initialize messages
        self.messages = {}
        for pair, bix in self.edges.items():
            for a, b in [sorted(pair), sorted(pair, reverse=True)]:
                tn_a_to_b = self.local_tns[a].copy()

                # for initial messages we sum over dangling indices connected
                # to other neighbors, i.e. no connecting a to b
                kix = [ix for ix in tn_a_to_b.outer_inds() if ix not in bix]
                for t in tn_a_to_b:
                    for ix in t.inds:
                        if ix in kix:
                            t.sum_reduce_(ix)

                # then contract into MPS like form
                for ltag in layer_tags:
                    try:
                        # make sure there is a single (MPS) tensor per layer
                        tn_a_to_b ^= ltag
                        # also drop all tags which aren't layer tags
                        t = tn_a_to_b[ltag]
                        t.modify(tags=(ltag,))
                    except KeyError:
                        pass

                # XXX: need to possibly expand bond here?
                tn_a_to_b /= tn_a_to_b.contract(output_inds=())
                self.messages[a, b] = tn_a_to_b

    def get_message_tn(self, a, b) -> qtn.TensorNetwork:
        """Get the (uncompressed) tensor network describing the current message
        from site `a` to site `b`.
        """
        message_tns = (
            self.messages[c, a] for c in self.neighbors[a] if c != b
        )
        return qtn.TensorNetwork((self.local_tns[a], *message_tns))

    def compute_message(self, a, b) -> qtn.TensorNetwork:
        """Compute the (compressed) tensor network message from site `a` to
        site `b`.
        """
        # form the message update tensor network
        tn_a_to_b = self.get_message_tn(a, b)

        kwargs = {}
        if self.compress_opts["method"] in ("fit", "srcmps"):
            if self.fit_messages:
                kwargs["tn_fit"] = self.messages[a, b]

        # compress it to MPS form
        qtn.tensor_network_1d_compress(
            tn_a_to_b,
            site_tags=self.layer_tags,
            normalize=True,
            inplace=True,
            **self.compress_opts,
            **kwargs,
        )

        # remove all but layer tags so they don't propagate
        for lt in self.layer_tags:
            t = tn_a_to_b[lt]
            t.modify(tags=(lt,))

        return tn_a_to_b

    def iterate(self, tol=5e-6):
        """Perform one round of message passing."""
        if (not self.local_convergence) or (not self.touched):
            # assume if asked to iterate that we want to check all messages
            self.touched.update(
                pair for edge in self.edges for pair in (edge, edge[::-1])
            )

        ncheck = len(self.touched)
        nconv = 0
        max_mdiff = -1.0
        new_touched = oset()

        def _update_m(key, new):
            nonlocal nconv, max_mdiff

            old = self.messages[key]
            mdiff = new.distance(old)

            if mdiff > tol:
                # mark touching messages for update
                new_touched.update(self.touch_map[key])
            else:
                nconv += 1

            max_mdiff = max(max_mdiff, mdiff)
            self.messages[key] = new

        if self.update == "parallel":
            new_messages = {}
            # compute all new messages
            while self.touched:
                a, b = self.touched.pop()
                new_messages[a, b] = self.compute_message(a, b)
            # insert all new messages
            for key, data in new_messages.items():
                _update_m(key, data)

        elif self.update == "sequential":
            while self.touched:
                a, b = self.touched.pop()
                new_message = self.compute_message(a, b)
                _update_m((a, b), new_message)

        self.touched = new_touched
        return {
            "nconv": nconv,
            "ncheck": ncheck,
            "max_mdiff": max_mdiff,
        }

    def contract(self, strip_exponent=False, check_zero=True, **kwargs):
        """Contract the target tensor network, via the belief propagation
        approximation, using the current messages.

        Parameters
        ----------
        strip_exponent : bool, optional
            Whether to return the mantissa and exponent separately.
        check_zero : bool, optional
            Whether to check for zero values and return zero early.
        """
        zvals = []

        for a, tn_a in self.local_tns.items():
            # form the 'tensor' cluster network
            tn_am = qtn.TensorNetwork(
                (tn_a, *(self.messages[b, a] for b in self.neighbors[a]))
            )
            # contract it!
            za = tn_am.contract(optimize=self.optimize)

            # power / counting factor is +1 for local tensors, i.e. multiply
            zvals.append((za, 1))

        for a, b in self.edges:
            # form the 'message' cluster network
            zab = (self.messages[a, b] & self.messages[b, a]).contract(
                optimize=self.optimize,
            )

            # power / counting factor is -1 for messages, i.e. divide
            zvals.append((zab, -1))

        return combine_local_contractions(
            zvals,
            self.backend,
            strip_exponent=strip_exponent,
            check_zero=check_zero,
            mantissa=self.sign,
            exponent=self.exponent,
            **kwargs,
        )
