"""Tools for generic VMC optimization of tensor networks.
"""

import array
import random

import numpy as np
import autoray as ar

from quimb.utils import default_to_neutral_style
from quimb import format_number_with_error


# --------------------------------------------------------------------------- #


def sample_bitstring_from_prob_ndarray(p, rng):
    flat_idx = rng.choice(np.arange(p.size), p=p.flat)
    return np.unravel_index(flat_idx, p.shape)


def shuffled(it):
    """Return a copy of ``it`` in random order.
    """
    it = list(it)
    random.shuffle(it)
    return it


class NoContext:
    """A convenience context manager that does nothing.
    """

    def __enter__(self):
        pass

    def __exit__(self, *_, **__):
        pass


class MovingStatistics:
    """Keep track of the windowed mean and estimated variance of a stream of
    values on the fly.
    """

    def __init__(self, window_size):
        self.window_size = window_size
        self.xs = []
        self.vs = []
        self._xsum = 0.0
        self._vsum = 0.0

    def update(self, x):

        # update mean
        self.xs.append(x)
        if len(self.xs) > self.window_size:
            xr = self.xs.pop(0)
        else:
            xr = 0.0
        self._xsum += (x - xr)

        # update approx variance
        v = (x - self.mean)**2
        self.vs.append(v)
        if len(self.vs) > self.window_size:
            vr = self.vs.pop(0)
        else:
            vr = 0.0
        self._vsum += (v - vr)

    @property
    def mean(self) :
        N = len(self.xs)
        if N == 0:
            return 0.0
        return self._xsum / N

    @property
    def var(self):
        N = len(self.xs)
        if N == 0:
            return 0.0
        return self._vsum / N

    @property
    def std(self):
        return self.var**0.5

    @property
    def err(self):
        N = len(self.xs)
        if N == 0:
            return 0.0
        return self.std / N**0.5


# --------------------------------------------------------------------------- #

class DenseSampler:
    """Sampler that explicitly constructs the full probability distribution.
    Useful for debugging small problems.
    """

    def __init__(self, psi=None, seed=None, **contract_opts):
        if psi is not None:
            self._set_psi(psi)
        contract_opts.setdefault('optimize', 'auto-hq')
        self.contract_opts = contract_opts
        self.rng = np.random.default_rng(seed)

    def _set_psi(self, psi):
        psi_dense = psi.contract(
            ..., output_inds=psi.site_inds, **self.contract_opts,
        ).data
        self.p = (abs(psi_dense.ravel())**2)
        self.p /= self.p.sum()
        self.sites = psi.sites
        self.shape = tuple(psi.ind_size(ix) for ix in psi.site_inds)
        self.flat_indexes = np.arange(self.p.size)

    def sample(self):
        flat_idx = self.rng.choice(self.flat_indexes, p=self.p)
        omega = self.p[flat_idx]
        config = np.unravel_index(flat_idx, self.shape)
        return dict(zip(self.sites, config)), omega

    def update(self, **kwargs):
        self._set_psi(kwargs['psi'])


class DirectTNSampler:
    """

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to sample from.
    sweeps : int, optional
        The number of sweeps to perform.
    max_group_size : int, optional
        The maximum number of sites to include in a single marginal.
    chi : int, optional
        The maximum bond dimension to use for compressed contraction.
    optimize : PathOptimizer, optional
        The path optimizer to use.
    optimize_share_path : bool, optional
        If ``True``, a single path will be used for all contractions regardless
        of which marginal (i.e. which indices are open) is begin computed.
    """

    def __init__(
        self,
        tn,
        sweeps=1,
        max_group_size=8,
        chi=None,
        optimize=None,
        optimize_share_path=False,
        seed=None,
        track=False,
    ):
        self.tn = tn.copy()

        self.ind2site = {}
        self.tid2ind = {}
        for site in self.tn.sites:
            ix = self.tn.site_ind(site)
            tid, = self.tn._get_tids_from_inds(ix)
            self.tid2ind[tid] = ix
            self.ind2site[ix] = site

        self.chi = chi
        self.sweeps = sweeps
        self.max_group_size = max_group_size

        self.optimize = optimize
        self.optimize_share_path = optimize_share_path
        self.groups = None
        self.tree = None
        self.path = None

        self.rng = np.random.default_rng(seed)

        self.track = track
        if self.track:
            self.omegas = []
            self.probs = []
        else:
            self.omegas = self.probs = None

    def plot(self,):
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        mins = min(self.omegas)
        maxs = max(self.omegas)
        ax.plot([mins, maxs], [mins, maxs], color='red')
        ax.scatter(self.probs, self.omegas, marker='.', alpha=0.5)
        ax.set_xlabel('$\pi(x)$')
        ax.set_ylabel('$\omega(x)$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, c=(0.97 ,0.97, 0.97), which='major')
        ax.set_axisbelow(True)

    def calc_groups(self, **kwargs):
        """Calculate how to group the sites into marginals.
        """
        self.groups = self.tn.compute_hierarchical_grouping(
            max_group_size=self.max_group_size,
            tids=tuple(self.tid2ind),
            **kwargs,
        )

    def get_groups(self):
        if self.groups is None:
            self.calc_groups()
        return self.groups

    def calc_path(self):
        tn0 = self.tn.isel({ix: 0 for ix in self.ind2site})
        self.tree = tn0.contraction_tree(self.optimize)
        self.path = self.tree.get_path()

    def get_path(self):
        if self.path is None:
            self.calc_path()
        return self.path

    def get_optimize(self):
        if self.optimize_share_path:
            return self.get_path()
        else:
            return self.optimize

    def contract(self, tn, output_inds):
        if self.chi is None:
            return tn.contract(
                optimize=self.get_optimize(),
                output_inds=output_inds,
            )
        else:
            return tn.contract_compressed(
                max_bond=self.chi,
                optimize=self.get_optimize(),
                output_inds=output_inds,
                cutoff=0.0,
                compress_opts=dict(absorb='both'),
            )

    def sample(self):

        config = {}

        tnm = self.tn.copy()

        for tid, ix in self.tid2ind.items():
            t = tnm.tensor_map[tid]
            t.rand_reduce_(
                ix, rand_fn=lambda d: self.rng.choice([-1.0, 1.0], size=d)
            )

        tnm.apply_to_arrays(ar.lazy.array)
        with ar.lazy.shared_intermediates():
        # with NoContext():

            for _ in range(self.sweeps):

                # random.shuffle(self.groups)
                omega = 1.0

                for group in self.get_groups():
                    # get corresponding indices
                    inds = [self.tid2ind[tid] for tid in group]

                    # insert the orig tensors with output index
                    for tid in group:
                        t_full = self.tn.tensor_map[tid]
                        tnm.tensor_map[tid].modify(
                            data=ar.lazy.array(t_full.data),
                            inds=t_full.inds,
                        )

                    # contract the current conditional marginal
                    tg = self.contract(tnm, inds)

                    # convert into normalized prob and sample a config
                    prob_g = ar.do('abs', tg.data.compute())**2
                    prob_g /= ar.do('sum', prob_g)
                    config_g = sample_bitstring_from_prob_ndarray(
                        prob_g, self.rng
                    )
                    omega *= prob_g[config_g]

                    # re-project the tensors according to the sampled config
                    for tid, ix, bi in zip(group, inds, config_g):

                        # the 'measurement' for this tensor
                        # bi = int(bi)

                        # project tensor from full wavefunction
                        t_full = self.tn.tensor_map[tid]
                        tm = t_full.isel({ix: bi})
                        tnm.tensor_map[tid].modify(
                            data=ar.lazy.array(tm.data),
                            inds=tm.inds,
                        )

                        # update the bitstring
                        config[self.ind2site[ix]] = bi

        if self.track:
            self.omegas.append(omega)
            self.probs.append(abs(tg.data[config_g].compute())**2)

        # final chosen marginal is prob of whole config
        return config, omega # tg.data[config_g].compute()


def compute_amplitude(tn, config, chi, optimize):
    tni = tn.isel({tn.site_ind(site): v for site, v in config.items()})
    return tni.contract_compressed_(
        optimize=optimize,
        max_bond=chi,
        cutoff=0.0,
        compress_opts={'absorb': 'both'},
        inplace=True,
    )


def compute_amplitudes(tn, configs, chi, optimize):
    with ar.lazy.shared_intermediates():
        tnlz = tn.copy()
        tnlz.apply_to_arrays(ar.lazy.array)

        amps = []
        for config in configs:
            amps.append(compute_amplitude(tnlz, config, chi, optimize))

    amps = ar.do('stack', amps)
    return amps.compute()


def compute_local_energy(ham, tn, config, chi, optimize):
    """
    """
    c_configs, c_coeffs = ham.config_coupling(config)
    amps = compute_amplitudes(tn, [config] + c_configs, chi, optimize)
    c_coeffs = ar.do('array', c_coeffs, like=amps)
    return ar.do('sum', amps[1:] * c_coeffs) / amps[0]


def draw_config(edges, config):
    import networkx as nx
    G = nx.Graph(edges)
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, node_color=[config[node] for node in G.nodes], pos=pos)


class ClusterSampler:

    def __init__(
        self,
        psi=None,
        max_distance=1,
        use_gauges=True,
        seed=None,
        contract_opts=None,
    ):
        self.rng = np.random.default_rng(seed)
        self.use_gauges = use_gauges
        self.max_distance = max_distance
        self.contract_opts = (
            {} if contract_opts is None else dict(contract_opts)
        )
        self.contract_opts.setdefault('optimize', 'auto-hq')
        if psi is not None:
            self._set_psi(psi)

    def _set_psi(self, psi):
        self._psi = psi.copy()
        if self.use_gauges:
            self.gauges0 = {}
            self._psi.gauge_all_simple_(gauges=self.gauges0)
        else:
            self.gauges0 = None

        self.tid2site = {}
        for site in self._psi.sites:
            tid, = self._psi._get_tids_from_tags(site)
            self.tid2site[tid] = site
        self.ordering = self._psi.compute_hierarchical_ordering()

    def sample(self):
        """
        """
        config = {}
        psi = self._psi.copy()

        if self.use_gauges:
            gauges = self.gauges0.copy()
        else:
            gauges = None

        omega = 1.0

        for tid in self.ordering:
            site = self.tid2site[tid]
            ind = psi.site_ind(site)

            # select a local patch
            k = psi._select_local_tids(
                [tid],
                max_distance=self.max_distance,
                fillin=0,
                virtual=False,
            )

            if self.use_gauges:
                # gauge it including dangling bonds
                k.gauge_simple_insert(gauges)

            # contract the approx reduced density matrix diagonal
            pk = (k.H & k).contract(
                ...,
                output_inds=[ind],  # directly extract diagonal
                **self.contract_opts,
            ).data

            # normalize and sample a state for this site
            pk /= pk.sum()
            idx = self.rng.choice(np.arange(2), p=pk)
            config[site] = idx

            # track the probability chain
            omega *= pk[idx]

            # fix the site to measurement
            psi.tensor_map[tid].isel_({ind: idx})

            if self.use_gauges:
                # update local gauges to take measurement into account
                psi._gauge_local_tids(
                    [tid],
                    max_distance=(self.max_distance + 1),
                    method='simple', gauges=gauges
                )

        return config, omega

    candidate = sample

    def accept(self, config):
        pass

    def update(self, **kwargs):
        self._set_psi(kwargs['psi'])


class ExchangeSampler:

    def __init__(self, edges, seed=None):
        self.edges = tuple(sorted(edges))
        self.Ne = len(self.edges)
        self.sites = sorted(set(site for edge in edges for site in edge))
        self.N = len(self.sites)
        self.rng = np.random.default_rng(seed)
        values0 = [0] * (self.N // 2) + [1] * (self.N // 2)
        if self.N % 2 == 1:
            values0.append(0)
        values0 = self.rng.permutation(values0)
        self.config = dict(zip(self.sites, values0))

    def candidate(self):
        nconfig = self.config.copy()
        for i in self.rng.permutation(np.arange(self.Ne)):
            cooa, coob = self.edges[i]
            xa, xb = nconfig[cooa], nconfig[coob]
            if xa == xb:
                continue
            nconfig[cooa], nconfig[coob] = xb, xa
            return nconfig, 1.0

    def accept(self, config):
        self.config = config

    def sample(self):
        config, omega = self.candidate()
        self.accept(config)
        return config, omega

    def update(self, **_):
        pass


class HamiltonianSampler:

    def __init__(self, ham, seed=None):
        self.ham = ham
        self.rng = np.random.default_rng(seed)

        self.N = len(self.ham.sites)
        values0 = [0] * (self.N // 2) + [1] * (self.N // 2)
        if self.N % 2 == 1:
            values0.append(0)
        values0 = self.rng.permutation(values0)
        self.config = dict(zip(self.ham.sites, values0))

    def candidate(self):
        generate = True
        while generate:
            # XXX: could do this much more efficiently with a single random
            # term
            configs, _ = self.ham.config_coupling(self.config)
            i = self.rng.integers(len(configs))
            new_config = configs[i]
            generate = (new_config == self.config)
        return new_config, 1.0

    def accept(self, config):
        self.config = config

    def sample(self):
        config, omega = self.candidate()
        self.accept(config)
        return config, omega

    def update(self, **_):
        pass


class MetropolisHastingsSampler:
    """
    """

    def __init__(
        self,
        sub_sampler,
        amplitude_factory=None,
        initial=None,
        burn_in=0,
        seed=None,
        track=False,
    ):
        self.sub_sampler = sub_sampler

        if amplitude_factory is not None:
            self.prob_fn = amplitude_factory.prob
        else:
            # will initialize later
            self.prob_fn = None

        if initial is not None:
            self.config, self.omega, self.prob = initial
        else:
            self.config = self.omega = self.prob = None

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.accepted = 0
        self.total = 0
        self.burn_in = burn_in

        # should we record the history?
        self.track = track
        if self.track:
            self.omegas = array.array('d')
            self.probs = array.array('d')
            self.acceptances = array.array('d')
        else:
            self.omegas = self.probs = self.acceptances = None

    @property
    def acceptance_ratio(self):
        if self.total == 0:
            return 0.0
        return self.accepted / self.total

    def sample(self):
        if self.config is None:
            # check if we are starting from scratch
            self.config, self.omega = self.sub_sampler.sample()
            self.prob = self.prob_fn(self.config)

        while True:
            self.total += 1

            # generate candidate configuration
            nconfig, nomega = self.sub_sampler.candidate()
            nprob = self.prob_fn(nconfig)

            # compute acceptance probability
            acceptance = (nprob * self.omega) / (self.prob * nomega)

            if self.track:
                self.omegas.append(nomega)
                self.probs.append(nprob)
                self.acceptances.append(acceptance)

            if (self.rng.uniform() < acceptance):
                self.config = nconfig
                self.omega = nomega
                self.prob = nprob
                self.accepted += 1
                self.sub_sampler.accept(nconfig)

                if (self.total > self.burn_in):
                    return self.config, self.omega

    def update(self, **kwargs):
        self.prob_fn = kwargs['amplitude_factory'].prob
        self.sub_sampler.update(**kwargs)

    @default_to_neutral_style
    def plot(self):
        from matplotlib import pyplot as plt

        fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
        fig.suptitle(f"acceptance ratio = {100 * self.acceptance_ratio:.2f} %")

        mins = min(self.omegas)
        maxs = max(self.omegas)

        axs[0].plot([mins, maxs], [mins, maxs], color="red")
        axs[0].scatter(
            self.probs, self.omegas, marker=".", alpha=0.5, zorder=-10
        )
        axs[0].set_rasterization_zorder(0)
        axs[0].set_xlabel("$\pi(x)$")
        axs[0].set_ylabel("$\omega(x)$")
        axs[0].set_xscale("log")
        axs[0].set_yscale("log")
        axs[0].grid(True, c=(0.97, 0.97, 0.97), which="major")
        axs[0].set_axisbelow(True)

        minh = np.log10(min(self.acceptances))
        maxh = np.log10(max(self.acceptances))
        axs[1].hist(
            self.acceptances, bins=np.logspace(minh, maxh), color="green"
        )
        axs[1].set_xlabel("$A = \dfrac{\pi(x)\omega(y)}{\pi(y)\omega(x)}$")
        axs[1].axvline(1.0, color="orange")
        axs[1].set_xscale("log")
        axs[1].grid(True, c=(0.97, 0.97, 0.97), which="major")
        axs[1].set_axisbelow(True)

        return fig, axs


# --------------------------------------------------------------------------- #


def auto_share_multicall(func, arrays, configs):
    """Call the function ``func``, which should be an array
    function making use of autoray dispatched calls, multiple
    times, automatically reusing shared intermediates.
    """
    with ar.lazy.shared_intermediates():
        lzarrays_all = [
            # different variants provided as first dimension
            ar.lazy.array(x) if hasattr(x, 'shape') else
            # different variants provided as a sequence
            list(map(ar.lazy.array, x))
            for x in arrays
        ]
        lzarrays_config = lzarrays_all.copy()

        outs = []
        for config in configs:
            # for each config, insert the correct inputs
            for k, v in config.items():
                lzarrays_config[k] = lzarrays_all[k][v]
            # evaluate the function
            outs.append(func(lzarrays_config))

        # combine into single output object
        final = ar.lazy.stack(tuple(outs))

    # evaluate all configs simulteneously
    return final.compute()


class ComposePartial:

    __slots__ = (
        "f",
        "f_args",
        "f_kwargs",
        "g",
    )

    def __init__(self, f, f_args, f_kwargs, g):
        self.f = f
        self.f_args = f_args
        self.f_kwargs = f_kwargs
        self.g = g

    def __call__(self, *args, **kwargs):
        y = self.g(*args, **kwargs)
        f_args = (
            y if isinstance(v, ar.lazy.LazyArray) else v
            for v in self.f_args
        )
        return self.f(*f_args, **self.f_kwargs)


_partial_compose_cache = {}


def get_compose_partial(f, f_args, f_kwargs, g):

    key = (
        f,
        tuple(
            '__placeholder__' if isinstance(v, ar.lazy.LazyArray)
            else v
            for v in f_args
        ),
        tuple(sorted(f_kwargs.items())),
        g,
    )

    try:
        fg = _partial_compose_cache[key]
    except KeyError:
        fg = _partial_compose_cache[key] = ComposePartial(f, f_args, f_kwargs, g)
    except TypeError:
        fg = ComposePartial(f, f_args, f_kwargs, g)

    return fg


def fuse_unary_ops_(Z):
    queue = [Z]
    seen = set()
    while queue:
        node = queue.pop()
        if (
            len(node._deps) == 1 and
            any(isinstance(v, ar.lazy.LazyArray) for v in node.args)
        ):
            dep, = node._deps
            if dep._nchild == 1 and dep._fn:
                node._fn = get_compose_partial(node._fn, node._args, node._kwargs, dep._fn)
                node._args = dep._args
                node._kwargs = dep._kwargs
                node._deps = dep._deps
                queue.append(node)
                continue

        for dep in node._deps:
            if dep not in seen:
                queue.append(dep)
                seen.add(dep)



class AmplitudeFactory:

    def __init__(
        self,
        psi=None,
        contract_fn=None,
        maxsize=2**20,
        autojit_opts=(),
        **contract_opts,
    ):
        from quimb.utils import LRU

        self.contract_fn = contract_fn
        self.contract_opts = contract_opts
        if self.contract_opts.get('max_bond', None) is not None:
            self.contract_opts.setdefault('cutoff', 0.0)

        self.autojit_opts = dict(autojit_opts)

        if psi is not None:
            self._set_psi(psi)

        self.store = LRU(maxsize=maxsize)
        self.hits = 0
        self.queries = 0

    def _set_psi(self, psi):
        psi0 = psi.copy()

        self.arrays = []
        self.sitemap = {}
        variables = []

        for site in psi0.sites:
            ix = psi0.site_ind(site)
            t, = psi0._inds_get(ix)

            # want variable index first
            t.moveindex_(ix, 0)
            self.sitemap[site] = len(self.arrays)
            self.arrays.append(t.data)

            # insert lazy variable for sliced tensor
            variable = ar.lazy.Variable(t.shape[1:], backend='autoray.lazy')
            variables.append(variable)
            t.modify(data=variable, inds=t.inds[1:])

        # trace the function lazily
        if self.contract_fn is None:
            Z = psi0.contract(..., output_inds=(), **self.contract_opts)
        else:
            Z = self.contract_fn(psi0, **self.contract_opts)

        # get the functional form of this traced contraction
        self.f_lazy = Z.get_function(variables)

        # this can then itself be traced with concrete arrays
        self.f = ar.autojit(self.f_lazy, **self.autojit_opts)

    def compute_single(self, config):
        """Compute the amplitude of ``config``, making use of autojit.
        """
        arrays = self.arrays.copy()
        for site, v in config.items():
            i = self.sitemap[site]
            arrays[i] = self.arrays[i][v]
        return self.f(arrays)

    def compute_multi(self, configs):
        """Compute the amplitudes corresponding to the sequence ``configs``,
        making use of shared intermediates.
        """
        # translate index config to position configs
        iconfigs = [
            {self.sitemap[site]: v for site, v in config.items()}
            for config in configs
        ]
        return auto_share_multicall(self.f_lazy, self.arrays, iconfigs)

    # def update(self, config, coeff):
    #     """Update the amplitude cache with a new configuration.
    #     """
    #     self.store[tuple(sorted(config.items()))] = coeff

    def amplitude(self, config):
        """Get the amplitude of ``config``, either from the cache or by
        computing it.
        """
        key = tuple(sorted(config.items()))
        self.queries += 1
        if key in self.store:
            self.hits += 1
            return self.store[key]

        coeff = self.compute_single(self.psi, config)

        self.store[key] = coeff
        return coeff

    def amplitudes(self, configs):
        """
        """
        # first parse out the configurations we need to compute
        all_keys = []
        new_keys = []
        new_configs = []
        for config in configs:
            key = tuple(sorted(config.items()))
            all_keys.append(key)
            self.queries += 1
            if key in self.store:
                self.hits += 1
            else:
                new_keys.append(key)
                new_configs.append(config)

        # compute the new configurations
        if new_configs:
            new_coeffs = self.compute_multi(new_configs)
            for key, coeff in zip(new_keys, new_coeffs):
                self.store[key] = coeff

        # return the full set of old and new coefficients
        return [self.store[key] for key in all_keys]

    def prob(self, config):
        """Calculate the probability of a configuration.
        """
        coeff = self.amplitude(config)
        return ar.do("abs", coeff)**2

    def clear(self):
        self.store.clear()

    def __contains__(self, config):
        return tuple(sorted(config.items())) in self.store

    def __setitem__(self, config, c):
        self.store[tuple(sorted(config.items()))] = c

    def __getitem__(self, config):
        return self.amplitude(config)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}(hits={self.hits}, "
            f"queries={self.queries})>"
        )


# class AmplitudeStore:

#     def __init__(self, psi, amp_fn, maxsize=2**20):
#         from quimb.utils import LRU
#         self.psi = psi
#         self.amp_fn = amp_fn
#         self.store = LRU(maxsize=maxsize)
#         self.hits = 0
#         self.queries = 0

#     def update(self, config, coeff):
#         """Update the amplitude cache with a new configuration.
#         """
#         self.store[tuple(sorted(config.items()))] = coeff

#     def amplitude(self, config):
#         """Calculate the amplitude of a configuration.
#         """
#         self.queries += 1

#         key = tuple(sorted(config.items()))
#         if key in self.store:
#             self.hits += 1
#             return self.store[key]

#         amp = self.amp_fn(self.psi, config)

#         self.store[key] = amp
#         return amp

#     def prob(self, config):
#         """Calculate the probability of a configuration.
#         """
#         amp = self.amplitude(config)
#         return ar.do("abs", amp)**2

#     def clear(self):
#         self.store.clear()

#     def __contains__(self, config):
#         return tuple(sorted(config.items())) in self.store

#     def __setitem__(self, config, c):
#         self.store[tuple(sorted(config.items()))] = c

#     def __getitem__(self, config):
#         return self.amplitude(config)

#     def __repr__(self):
#         return (
#             f"<{self.__class__.__name__}(hits={self.hits}, "
#             f"queries={self.queries})>"
#         )


# @autojit_tn(backend='torch', check_inputs=False)
# def contract_amplitude_tn(tni, **contract_opts):
#     if 'max_bond' in contract_opts:
#         contract_opts.setdefault('cutoff', 0.0)
#     return tni.contract(..., output_inds=(), **contract_opts)


# def compute_amplitude_aj(tn, config, **contract_opts):
#     tni = tn.isel({tn.site_ind(site): v for site, v in config.items()})
#     return contract_amplitude_tn(tni, **contract_opts)


# def compute_local_energy_aj(ham, config, amp):
#     en = 0.0
#     c_configs, c_coeffs = ham.config_coupling(config)
#     cx = amp.amplitude(config)
#     for hxy, config_y in zip(c_coeffs, c_configs):
#         cy = amp.amplitude(config_y)
#         en += hxy * cy / cx
#     return en


# def compute_amp_and_gradients(psi, config, **contract_opts):
#     import torch
#     psi_t = psi.copy()
#     psi_t.apply_to_arrays(lambda x: torch.tensor(x).requires_grad_())
#     c = compute_amplitude_aj(psi_t, config, **contract_opts)
#     c.backward()
#     c = c.item()
#     return [t.data.grad.numpy() / c for t in psi_t], c


# --------------------------------------------------------------------------- #


class GradientAccumulator:

    def __init__(self):
        self._grads_logpsi = None
        self._grads_energy = None
        self._batch_energy = None
        self._num_samples = 0

    def _init_storage(self, grads):
        self._batch_energy = 0.0
        self._grads_logpsi = [np.zeros_like(g) for g in grads]
        self._grads_energy = [np.zeros_like(g) for g in grads]

    def update(self, grads_logpsi_sample, local_energy):
        if self._batch_energy is None:
            self._init_storage(grads_logpsi_sample)

        self._batch_energy += local_energy
        for g, ge, g_i in zip(
            self._grads_logpsi, self._grads_energy, grads_logpsi_sample,
        ):
            g += g_i
            ge += g_i * local_energy
        self._num_samples += 1

    def extract_grads_energy(self):
        e = self._batch_energy / self._num_samples
        grads_energy_batch = []
        for g, ge in zip(self._grads_logpsi, self._grads_energy):
            g /= self._num_samples
            ge /= self._num_samples
            grads_energy_batch.append(ge - g * e)
            # reset storage
            g.fill(0.0)
            ge.fill(0.0)
        self._batch_energy = 0.0
        self._num_samples = 0
        return grads_energy_batch


class SGD(GradientAccumulator):

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        super().__init__()

    def transform_gradients(self):
        return [
            self.learning_rate * g
            for g in self.extract_grads_energy()
        ]


class SignDescent(GradientAccumulator):

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        super().__init__()

    def transform_gradients(self):
        return [
            self.learning_rate * np.sign(g)
            for g in self.extract_grads_energy()
        ]


class RandomSign(GradientAccumulator):

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        super().__init__()

    def transform_gradients(self):
        return [
            self.learning_rate * np.sign(g) * np.random.uniform(size=g.shape)
            for g in self.extract_grads_energy()
        ]


class Adam(GradientAccumulator):

    def __init__(
        self,
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._num_its = 0
        self._ms = None
        self._vs = None
        super().__init__()

    def transform_gradients(self):
        # get the standard SGD gradients
        grads = self.extract_grads_energy()

        self._num_its += 1
        if self._num_its == 1:
            # first iteration, initialize storage
            self._ms = [np.zeros_like(g) for g in grads]
            self._vs = [np.zeros_like(g) for g in grads]

        deltas = []
        for i, g in enumerate(grads):
            # first  moment estimate
            m = (1 - self.beta1) * g + self.beta1 * self._ms[i]
            # second moment estimate
            v = (1 - self.beta2) * (g**2) + self.beta2 * self._vs[i]
            # bias correction
            mhat = m / (1 - self.beta1**(self._num_its))
            vhat = v / (1 - self.beta2**(self._num_its))
            deltas.append(
                self.learning_rate * mhat / (np.sqrt(vhat) + self.eps)
            )
        return deltas


from quimb.tensor.optimize import Vectorizer


class StochasticReconfigureGradients:

    def __init__(self, delta=1e-5):
        self.delta = delta
        self.vectorizer = None
        self.gs = []

    def update(self, grads_logpsi_sample, local_energy):
        if self.vectorizer is None:
            # first call, initialize storage
            self.vectorizer = Vectorizer(grads_logpsi_sample)
        self.gs.append(self.vectorizer.pack(grads_logpsi_sample).copy())
        super().update(grads_logpsi_sample, local_energy)

    def extract_grads_energy(self):
        # number of samples
        num_samples = len(self.gs)

        gs = np.stack(self.gs)
        self.gs.clear()
        # <g_i g_j>
        S = (gs.T / num_samples) @ gs
        # minus <g_i><g_j> to get S
        g = gs.sum(axis=0) / num_samples
        S -= np.outer(g, g)

        # condition by adding to diagonal
        S.flat[::S.shape[0] + 1] += self.delta

        # the uncorrected energy gradient / 'force' vector
        y = self.vectorizer.pack(super().extract_grads_energy())

        # the corrected energy gradient, which we then unvectorize
        x = np.linalg.solve(S, y)
        return self.vectorizer.unpack(x)



class SR(SGD, StochasticReconfigureGradients):

    def __init__(self, learning_rate=0.05, delta=1e-5):
        StochasticReconfigureGradients.__init__(self, delta=delta)
        SGD().__init__(self, learning_rate=learning_rate)



class SRADAM(Adam, StochasticReconfigureGradients):

    def __init__(
        self,
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        delta=1e-5,
    ):
        StochasticReconfigureGradients.__init__(self, delta=delta)
        Adam.__init__(
            self, learning_rate=learning_rate,
            beta1=beta1, beta2=beta2, eps=eps,
        )


# --------------------------------------------------------------------------- #

class TNVMC:

    def __init__(
        self,
        psi,
        ham,
        sampler,
        conditioner='auto',
        learning_rate=1e-2,
        optimizer='adam',
        optimizer_opts=None,
        track_window_size=1000,
        **contract_opts
    ):
        from quimb.utils import ensure_dict

        self.psi = psi.copy()
        self.ham = ham
        self.sampler = sampler

        if conditioner == 'auto':

            def conditioner(psi):
                psi.equalize_norms_(1.0)

        else:
            self.conditioner = conditioner

        if self.conditioner is not None:
            # want initial arrays to be in conditioned form so that gradients
            # are approximately consistent across runs (e.g. for momentum)
            self.conditioner(self.psi)

        optimizer_opts = ensure_dict(optimizer_opts)
        self.optimizer = {
            'adam': Adam,
            'sgd': SGD,
            'sign': SignDescent,
            'signu': RandomSign,
            'sr': SR,
            'sradam': SRADAM,
        }[optimizer.lower()](learning_rate=learning_rate, **optimizer_opts)
        self.contract_opts = contract_opts

        self.amplitude_factory = AmplitudeFactory(self.psi, **contract_opts)
        self.sampler.update(psi=self.psi, amplitude_factory=self.amplitude_factory)

        # tracking information
        self.moving_stats = MovingStatistics(track_window_size)
        self.local_energies = array.array('d')
        self.energies = array.array('d')
        self.energy_errors = array.array('d')
        self.num_tensors = self.psi.num_tensors
        self.nsites = self.psi.nsites
        self._progbar = None

    def _compute_log_gradients_torch(self, config):
        import torch
        psi_t = self.psi.copy()
        psi_t.apply_to_arrays(lambda x: torch.tensor(x).requires_grad_())
        c = self.amplitude_factory(psi_t, config)
        c.backward()
        c = c.item()
        self.amplitude_factory[config] = c
        return [t.data.grad.numpy() / c for t in psi_t]

    def _compute_local_energy(self, config):
        en = 0.0
        c_configs, c_coeffs = self.ham.config_coupling(config)
        cx = self.amplitude_factory.amplitude(config)
        for hxy, config_y in zip(c_coeffs, c_configs):
            cy = self.amplitude_factory.amplitude(config_y)
            en += hxy * cy / cx
        return en / self.nsites

    def _run(self, steps, batchsize):
        for _ in range(steps):
            for _ in range(batchsize):
                config, omega = self.sampler.sample()

                # compute and track local energy
                local_energy = self._compute_local_energy(config)
                self.local_energies.append(local_energy)
                self.moving_stats.update(local_energy)
                self.energies.append(self.moving_stats.mean)
                self.energy_errors.append(self.moving_stats.err)

                # compute the sample log amplitude gradients
                grads_logpsi_sample = self._compute_log_gradients_torch(config)

                self.optimizer.update(grads_logpsi_sample, local_energy)

                if self._progbar is not None:
                    self._progbar.update()
                    self._progbar.set_description(
                        format_number_with_error(
                            self.moving_stats.mean,
                            self.moving_stats.err))

            # apply learning rate and other transforms to gradients
            deltas = self.optimizer.transform_gradients()

            # update the actual tensors
            for t, delta in zip(self.psi.tensors, deltas):
                t.modify(data=t.data - delta)

            # reset having just performed a gradient step
            if self.conditioner is not None:
                self.conditioner(self.psi)

            self.amplitude_factory.clear()
            self.sampler.update(psi=self.psi, amplitude_factory=self.amplitude_factory)

    def run(
        self,
        total=10_000,
        batchsize=100,
        progbar=True,
    ):
        steps = total // batchsize
        total = steps * batchsize

        if progbar:
            from quimb.utils import progbar as Progbar
            self._progbar = Progbar(total=total)

        try:
            self._run(steps, batchsize)
        except KeyboardInterrupt:
            pass
        finally:
            if self._progbar is not None:
                self._progbar.close()

    def measure(
        self,
        max_samples=10_000,
        rtol=1e-4,
        progbar=True,
    ):
        from xyzpy import RunningStatistics

        rs = RunningStatistics()
        energies = array.array('d')

        if progbar:
            from quimb.utils import progbar as Progbar
            pb = Progbar(total=max_samples)
        else:
            pb = None

        try:
            for _ in range(max_samples):
                config, _ = self.sampler.sample()
                local_energy = self._compute_local_energy(config)
                rs.update(local_energy)
                energies.append(local_energy)

                if pb is not None:
                    pb.update()
                    err = rs.err
                    if err != 0.0:
                        pb.set_description(format_number_with_error(rs.mean, err))

                if 0.0 < rs.rel_err < rtol:
                    break

        except KeyboardInterrupt:
            pass
        finally:
            if pb is not None:
                pb.close()

        return rs, energies

    @default_to_neutral_style
    def plot(
        self,
        figsize=(12, 6),
        yrange_quantile=(0.01, 0.99),
        zoom="auto",
        hlines=(),
    ):
        from matplotlib import pyplot as plt

        x = np.arange(len(self.local_energies))
        # these are all views
        y = np.array(self.local_energies)
        ym = np.array(self.energies)
        yerr = np.array(self.energy_errors)
        yplus = ym + yerr
        yminus = ym - yerr
        yv = np.array(self.energy_variances[10:])

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows=2, ncols=3)

        ax = fig.add_subplot(gs[:, :2])
        ax.plot(
            x,
            y,
            ".",
            alpha=0.5,
            markersize=1.0,
            zorder=-10,
            color=(0.1, 0.5, 0.7),
        )
        ax.fill_between(
            x, yminus, yplus,
            alpha=0.45,
            color=(0.6, 0.8, 0.6),
            zorder=-11,
        )
        ax.plot(
            x,
            ym,
            "-",
            alpha=0.9,
            zorder=-10,
            linewidth=2,
            color=(0.6, 0.8, 0.6),
        )
        ax.set_ylim(
            np.quantile(y, yrange_quantile[0]),
            np.quantile(y, yrange_quantile[1]),
        )
        ax.set_xlabel("Number of local energy evaluations")
        ax.set_ylabel("Energy per site", color=(0.6, 0.8, 0.6))

        if hlines:
            from matplotlib.colors import hsv_to_rgb

            hlines = dict(hlines)
            for i, (label, value) in enumerate(hlines.items()):
                color = hsv_to_rgb([(0.1 * i) % 1.0, 0.9, 0.9])
                ax.axhline(value, color=color, ls="--", label=label)
                ax.text(1, value, label, color=color, va="bottom", ha="left")

        ax.set_rasterization_zorder(0)

        ax_var = fig.add_subplot(gs[1, 2])
        ax_var.plot(
            x[10:],
            yv,
            "-",
            alpha=0.9,
            zorder=-10,
            linewidth=2,
            color=(1.0, 0.7, 0.4),
        )
        ax_var.set_yscale('log')
        ax_var.text(
            0.9,
            0.9,
            "Energy variance",
            color=(1.0, 0.7, 0.4),
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax_var.transAxes,
        )
        ax_var.set_rasterization_zorder(0)

        if zoom is not None:
            if zoom == "auto":
                zoom = min(10_000, y.size // 2)

            ax_zoom = fig.add_subplot(gs[0, 2])
            ax_zoom.fill_between(
                x[-zoom:],
                yminus[-zoom:],
                yplus[-zoom:],
                alpha=0.45,
                color=(0.6, 0.8, 0.6),
                zorder=-11,
            )
            ax_zoom.plot(
                x[-zoom:],
                ym[-zoom:],
                "-",
                alpha=0.9,
                zorder=-10,
                linewidth=2,
                color=(0.6, 0.8, 0.6),
            )
            ax_zoom.text(
                0.9,
                0.9,
                "Zoom",
                color=(0.6, 0.8, 0.6),
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax_zoom.transAxes,
            )
            ax_zoom.set_rasterization_zorder(0)

        return fig, [ax, ax_zoom, ax_var]
