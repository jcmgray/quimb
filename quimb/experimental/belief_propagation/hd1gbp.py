import numpy as np

from quimb.tensor import Tensor, TensorNetwork, tensor_contract
from quimb.tensor.belief_propagation import (
    RegionGraph,
    combine_local_contractions,
)
from quimb.tensor.belief_propagation.bp_common import BeliefPropagationCommon


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


class HD1GBP(BeliefPropagationCommon):
    """Generalized belief propagation for hyper tensor networks.

    Parameters
    ----------
    tn : TensorNetwork
        The hyper tensor network to run GBP on.
    regions : sequence[sequence[int | str]]
        The regions to use for GBP. Each region can be a set of tids and
        indices. If a tid is present in a region, all its indices are
        automatically included in the region when ``autocomplete=True``.
    autocomplete : bool, optional
        Whether to automatically compute all intersection subregions for the
        RegionGraph.
    autoprune : bool, optional
        Whether to automatically remove all regions with a count of zero.
    damping : float, optional
        The damping factor to use for the messages.
    optimize : str, optional
        The contraction path optimization strategy to use.
    """

    def __init__(
        self,
        tn: TensorNetwork,
        regions,
        *,
        messages=None,
        autocomplete=True,
        autoprune=True,
        damping=1 / 2,
        optimize="auto-hq",
        **kwargs,
    ):
        super().__init__(
            tn,
            damping=damping,
            **kwargs,
        )

        if autocomplete:
            regions = auto_add_indices(tn, regions)

        self.rg = RegionGraph(
            regions,
            autocomplete=autocomplete,
            autoprune=autoprune,
        )

        if callable(messages):
            self._message_init_function = messages
            self.messages = {}
        elif messages is not None:
            self._message_init_function = None
            self.messages = messages
        else:
            self._message_init_function = None
            self.messages = {}

        self.new_messages = {}
        self.contract_opts = dict(
            optimize=optimize,
            drop_tags=True,
        )

    def get_message_tensors(self, source, target):
        """Get all message tensors needed to compute the message from
        ``source`` to ``target``.
        """
        # get the nodes and edge keys for the message
        r_a_without_b, pairs_mul, pairs_div = self.rg.get_message_parts(
            (source, target)
        )

        ts = []
        # first add factors not in target region
        for x in r_a_without_b:
            if isinstance(x, int):
                ts.append(self.tn.tensor_map[x])

        # then messages only appearing in source belief
        for pair in pairs_mul:
            try:
                ts.append(self.messages[pair])
            except KeyError:
                pass

        # then messages only appearing in target belief
        # note we use the *new* messages here, as per GBP stability
        for pair in pairs_div:
            try:
                ts.append(1 / self.new_messages[pair])
            except KeyError:
                try:
                    ts.append(1 / self.messages[pair])
                except KeyError:
                    pass

        return ts

    def compute_message(
        self,
        source,
        target,
        **contract_opts,
    ):
        """Compute the message from source to target region.

        Parameters
        ----------
        source : Region
            The source region.
        target : Region
            The target region.
        contract_opts
            Supplied to :func:`~quimb.tensor.tensor_contract`.
        """
        contract_opts = {**self.contract_opts, **contract_opts}

        ts = self.get_message_tensors(source, target)

        if ts:
            # can only output indices which are present
            output_inds = sorted(
                {ind for t in ts for ind in t.inds}.intersection(target)
            )
            # perform the message contraction!
            m = tensor_contract(
                *ts,
                output_inds=output_inds,
                preserve_tensor=True,
                **contract_opts,
            )
        else:
            # output uniform distribution
            m = Tensor()

        # normalize
        m.modify(apply=self._normalize_fn)

        return m

    def iterate(self, tol=5e-6):
        max_mdiff = 0.0

        ncheck = 0
        nconv = 0

        # compute messages into smaller regions first
        for child in sorted(self.rg.regions, key=len):
            for parent in self.rg.get_parents(child):
                # contract the message!
                m = self.compute_message(parent, child)

                if self._message_init_function is not None:
                    # the messages change size during the first few iterations
                    # so we perform a delayed initialization upon seeing
                    # each fresh shape
                    mprev = self.new_messages.get((parent, child), None)
                    if (mprev is None) or (mprev.shape != m.shape):
                        m.modify(data=self._message_init_function(m.shape))

                # immediately update the new messages for stability in GBP
                # (they are used in the 'denominator' of higher messages)
                self.new_messages[parent, child] = m

                # check for convergence
                try:
                    m_old = self.messages[parent, child]
                    # XXX: need to handle index alignment here to compare
                    # using _distance_fn:
                    # mdiff = self._distance_fn(m_old.data, m.data)
                    mdiff = (m_old - m).norm()
                except KeyError:
                    mdiff = 1.0
                max_mdiff = max(mdiff, max_mdiff)

                ncheck += 1
                if mdiff < tol:
                    nconv += 1

        # damped update of messages
        #     note that the raw, undamped `new_messages` are used in the
        #     denominator of the message computations, and so kept 'as is'
        for pair in self.new_messages:
            if pair not in self.messages:
                # no old message yet
                self.messages[pair] = self.new_messages[pair]
            else:
                self.messages[pair] = self._damping_fn(
                    self.messages[pair],
                    self.new_messages[pair],
                )

        # self.new_messages.clear()

        return {
            "nconv": nconv,
            "ncheck": ncheck,
            "max_mdiff": max_mdiff,
        }

    def get_belief_tensors(self, region):
        """Get the (uncontracted) tensors for the belief of ``region``."""
        ts = []

        # add factors
        for x in region:
            if isinstance(x, int):
                ts.append(self.tn.tensor_map[x])

        # add messages
        for pair in self.rg.get_coparent_pairs(region):
            try:
                ts.append(self.messages[pair])
            except KeyError:
                pass

        return ts

    def contract(self, strip_exponent=False, check_zero=True):
        """Contract this tensor network given the current GBP messages.

        Parameters
        ----------
        sstrip_exponent : bool, optional
            Whether to strip the exponent from the final result. If ``True``
            then the returned result is ``(mantissa, exponent)``.

        Returns
        -------
        result : float or (float, float)
            The combined value, or the mantissa and exponent separately.
        """
        zvals = []

        for r in self.rg.regions:
            c = self.rg.get_count(r)
            ts = self.get_belief_tensors(r)
            if ts:
                zr = tensor_contract(*ts, output_inds=(), **self.contract_opts)
                zvals.append((zr, c))

        return combine_local_contractions(
            zvals,
            backend=self.backend,
            strip_exponent=strip_exponent,
            check_zero=check_zero,
        )

    def draw(self, rhighlight=None, zfactor=2):
        from quimb.schematic import Drawing

        tid2site = {}
        for site in self.tn.sites:
            (tid,) = self.tn._get_tids_from_tags(site)
            tid2site[tid] = site

        def region_to_site(region):
            z = self.rg.get_level(region) * zfactor + np.random.uniform(0, 0.2)
            tids = []
            for x in region:
                if isinstance(x, int):
                    tids.append(x)
                else:
                    tids.extend(self.tn.ind_map[x])

            sites = [tid2site[tid] for tid in tids]
            xs, ys = zip(*sites)
            xmean = sum(xs) / len(sites)
            ymean = sum(ys) / len(sites)
            return xmean, ymean, z

        def ind_to_pos(ind):
            tids = self.tn.ind_map[ind]
            sites = [tid2site[tid] for tid in tids]
            xs, ys = zip(*sites)
            xmean = sum(xs) / len(sites)
            ymean = sum(ys) / len(sites)
            return xmean, ymean

        def region_to_pos(region):
            z = self.rg.get_level(region) * zfactor + np.random.uniform(0, 0.2)
            poss = []
            for x in region:
                if isinstance(x, int):
                    poss.append(tid2site[x])
                else:
                    poss.append(ind_to_pos(x))
            return tuple((*pos, z) for pos in poss)

        d = Drawing(figsize=(10, 10))

        if rhighlight == "random":
            import random

            rhighlight = random.choice(self.rg.regions)

        if rhighlight is not None:
            rchildren = self.rg.get_children(rhighlight)
            rdescendents = self.rg.get_descendents(rhighlight)
            rparents = self.rg.get_parents(rhighlight)
            rcoparents = [x[0] for x in self.rg.get_coparent_pairs(rhighlight)]
            rancestors = self.rg.get_ancestors(rhighlight)
        else:
            rchildren = rdescendents = rparents = rcoparents = rancestors = []

        for r in self.rg.regions:
            # color = hash_to_color(str(r))
            # color = "grey"

            if r == rhighlight:
                color = (1.0, 0.0, 0.0, 0.3)
            elif r in rchildren:
                color = (1.0, 0.5, 0.0, 0.3)
            elif r in rdescendents:
                color = (1.0, 1.0, 0.0, 0.3)
            elif r in rparents:
                color = (0.2, 0.5, 0.8, 0.3)
            elif r in rcoparents:
                color = (0.3, 0.7, 0.5, 0.3)
            elif r in rancestors:
                color = (0.3, 0.5, 0.2, 0.3)
            else:
                color = (0.5, 0.5, 0.5, 0.1)

            pos = region_to_site(r)
            d.circle(pos, radius=0.05, color=color)
            tids = [x for x in r if isinstance(x, int)]
            if tids:
                d.patch_around(
                    region_to_pos(r),
                    radius=0.05,
                    facecolor=color,
                )

            for rc in self.rg.get_children(r):
                posc = region_to_site(rc)
                d.line(
                    pos,
                    posc,
                    color=color,
                    # arrowhead=False,
                )

        return d.fig, d.ax
