"""Helpers for planning reusable tensor network environments."""

from typing import NamedTuple

EnvironmentKey = tuple[int, int]


class EnvironmentMove(NamedTuple):
    """A single environment construction or cache operation.

    Parameters
    ----------
    kind : {'init', 'contract', 'output', 'delete'}
        ``init`` builds ``output_env`` from ``input_sites``. ``contract``
        builds it from cached ``input_envs`` and any ``input_sites``.
        ``output`` returns ``input_envs`` as the environment of
        ``output_block``. ``delete`` removes them after their last use.
    output_env : tuple[int, int], optional
        Cache key ``(lo, hi)`` for the result. The environment contains sites
        ``[i % L for i in range(lo, hi)]``. The bounds can exceed ``L``.
    input_envs : tuple[tuple[int, int], ...], optional
        Keys of cached input environments. A 'tree' contraction uses one
        environment and one site. A 'cut' contraction can also combine two
        environments without adding a site.
    input_sites : tuple[int, ...], optional
        Original sites for an ``init`` or ``contract``, each reduced modulo
        the periodic length. The current schedules use at most one.
    output_block : tuple[int, int], optional
        The ``(start, size)`` target block of an ``output`` move.
    """

    kind: str
    output_env: EnvironmentKey | None = None
    input_envs: tuple[EnvironmentKey, ...] = ()
    input_sites: tuple[int, ...] = ()
    output_block: tuple[int, int] | None = None


class EnvironmentPlan:
    """Plan reusable environments for consecutive blocks in one dimension.

    The default 'tree' schedule adds one site at a time. It starts near the
    opposite side of each target block. It suits approximate contraction,
    where combining two large environments can be costly or less accurate.

    The 'cut' schedule can combine two cached environments. It suits exact
    contraction and uses linear work for a full sweep.

    Parameters
    ----------
    L : int
        The number of sites.
    cyclic : bool, optional
        Whether the sites have periodic boundary conditions.
    schedule : {'tree', 'cut'}, optional
        How to share work between target blocks. For a full periodic sweep:

        - 'tree': add one site at a time, using ``O(L log L)`` contractions
          and ``O(log L)`` cached environments.
        - 'cut': build left and right environments from a fixed cut, then
          combine them for each target. Uses about ``3L`` contractions and
          ``O(L)`` cached environments.

        Ignored for open boundaries, where environments grow from each end.

    Examples
    --------
    Plan the environment of the one-site block starting at site three in a
    six-site ring:

    >>> plan = EnvironmentPlan(6)
    >>> plan.show([(3, 1)])
     0: init      [I] .  .  .  .  .
     1: contract  [E++I] .  .  .  .
     2: delete    [D] .  .  .  .  .
     3: contract  +E--E] .  .  . [I+
     4: delete    [D--D] .  .  .  .
     5: contract  -E--E++I] .  . [E-
     6: delete    -D--D] .  .  . [D-
     7: contract  -E--E--E] . [I++E-
     8: delete    -D--D--D] .  . [D-
     9: output    -E--E--E] B [E--E-    => (3, 1)
    10: delete    -D--D--D] . [D--D-

    Plan every two-site block:

    >>> plan.show([(i, 2) for i in range(6)])
     0: init       .  .  .  . [I] .
     1: contract   .  .  .  . [E++I]
     2: delete     .  .  .  . [D] .
     3: contract   .  .  . [I++E--E]
     4: contract   .  . [I++E--E--E]
     5: delete     .  .  . [D--D--D]
     6: output     B  B [E--E--E--E]    => (0, 2)
     7: delete     .  . [D--D--D--D]
     8: contract  +I] .  .  . [E--E+
     9: delete     .  .  .  . [D--D]
    10: contract  -E] .  . [I++E--E-
    11: output    -E] B  B [E--E--E-    => (1, 2)
    12: delete    -D] .  . [D--D--D-
    13: contract  -E++I] .  . [E--E-
    14: delete    -D] .  .  . [D--D-
    15: output    -E--E] B  B [E--E-    => (2, 2)
    16: delete    -D--D] .  . [D--D-
    17: init       . [I] .  .  .  .
    18: contract   . [E++I] .  .  .
    19: delete     . [D] .  .  .  .
    20: contract  [I++E--E] .  .  .
    21: contract  +E--E--E] .  . [I+
    22: delete    [D--D--D] .  .  .
    23: output    -E--E--E] B  B [E-    => (3, 2)
    24: delete    -D--D--D] .  . [D-
    25: contract   . [E--E++I] .  .
    26: delete     . [D--D] .  .  .
    27: contract  [I++E--E--E] .  .
    28: output    [E--E--E--E] B  B     => (4, 2)
    29: delete    [D--D--D--D] .  .
    30: contract   . [E--E--E++I] .
    31: delete     . [D--D--D] .  .
    32: output     B [E--E--E--E] B     => (5, 2)
    33: delete     . [D--D--D--D] .
    """

    def __init__(self, L, cyclic=True, schedule="tree"):
        if L < 1:
            raise ValueError("L must be at least 1")
        if schedule not in ("tree", "cut"):
            raise ValueError("schedule must be 'tree' or 'cut'")

        self.L = L
        self.cyclic = cyclic
        self.schedule = schedule

    def _init_tree_interval(self, lo, hi, moves):
        """Build an environment for ``[lo, hi)``, starting at its midpoint."""
        mid = (lo + hi - 1) // 2
        env_key = (mid, mid + 1)
        moves.append(
            EnvironmentMove(
                "init",
                output_env=env_key,
                input_sites=(mid % self.L,),
            )
        )
        return self._extend_interval(env_key, (lo, hi), moves)

    def _extend_interval(self, env_key, target, moves):
        """Extend ``env_key`` to ``target``, adding sites on alternate sides."""
        clo, chi = env_key
        tlo, thi = target
        take_right = True

        while (clo > tlo) or (chi < thi):
            if take_right and (chi < thi):
                site = chi
                chi += 1
            elif clo > tlo:
                clo -= 1
                site = clo
            else:
                site = chi
                chi += 1

            output = (clo, chi)
            moves.append(
                EnvironmentMove(
                    "contract",
                    output_env=output,
                    input_envs=(env_key,),
                    input_sites=(site % self.L,),
                )
            )
            env_key = output
            take_right = not take_right

        return env_key

    def _build_cyclic_tree(self, blocks, moves):
        L = self.L

        def build_group(group, parent_key=None):
            a = group[0][0]
            stop = max(start + size for start, size in group)
            # the common environment runs from the last block end to the
            # first block start on the next turn of the ring
            #
            # .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
            #         |+++++++++++++++++++++++++++++++)
            #         a                              stop
            # --------)                               |----------------
            #       a + L                            stop
            interval = (stop, a + L)

            if interval[0] >= interval[1]:
                # no sites lie outside all blocks in this group
                env_key = None
            elif parent_key is None:
                # start at the middle of the common environment
                env_key = self._init_tree_interval(*interval, moves)
            else:
                # reuse the parent environment
                env_key = self._extend_interval(parent_key, interval, moves)

            if len(group) == 1:
                moves.append(
                    EnvironmentMove(
                        "output",
                        input_envs=() if env_key is None else (env_key,),
                        output_block=group[0],
                    )
                )
                return

            # visit each half, extending the shared environment as needed
            mid = len(group) // 2
            build_group(group[:mid], env_key)
            build_group(group[mid:], env_key)

        build_group(blocks)

    def _build_cut(self, blocks, moves):
        L = self.L
        # each block has a left and right env
        #
        # .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
        # |---- left -----|++++ block ++++|---------- right ----------)
        # 0             start            stop                         L
        #
        # on a ring, left envs go after cut at [L, L + start), so that
        # right + left = [stop, L + start)
        offset = L if self.cyclic else 0

        right_lo = min(start + size for start, size in blocks)
        if right_lo < L:
            # build all right envs, growing leftwards from L - 1
            env_key = (L - 1, L)
            moves.append(
                EnvironmentMove(
                    "init", output_env=env_key, input_sites=(L - 1,)
                )
            )
            self._extend_interval(env_key, (right_lo, L), moves)

        # sweep blocks by start, growing one left env rightwards
        left_key = None
        for start, size in blocks:
            stop = start + size
            input_envs = ()
            if stop < L:
                input_envs += ((stop, L),)
            if start > 0:
                if left_key is None:
                    left_key = (offset, offset + 1)
                    moves.append(
                        EnvironmentMove(
                            "init", output_env=left_key, input_sites=(0,)
                        )
                    )
                left_key = self._extend_interval(
                    left_key, (offset, offset + start), moves
                )
                input_envs += (left_key,)

            if self.cyclic and len(input_envs) == 2:
                # on a ring they share bonds across the cut, so merge
                output_env = (stop, L + start)
                moves.append(
                    EnvironmentMove(
                        "contract",
                        output_env=output_env,
                        input_envs=input_envs,
                    )
                )
                input_envs = (output_env,)

            moves.append(
                EnvironmentMove(
                    "output",
                    input_envs=input_envs,
                    output_block=(start, size),
                )
            )

    def _build_cyclic_cut(self, blocks, moves):
        # blocks crossing the cut at L have no left + right split, use tree
        plain = [block for block in blocks if sum(block) <= self.L]
        wrapped = [block for block in blocks if sum(block) > self.L]

        if plain:
            self._build_cut(plain, moves)
        if wrapped:
            self._build_cyclic_tree(wrapped, moves)

    @staticmethod
    def _remove_redundant(moves):
        """Remove any environment moves that produce the same interval."""
        # the key is an interval of sites, so reuse its first construction
        built = set()
        unique = []
        for move in moves:
            if move.output_env is not None:
                if move.output_env in built:
                    continue
                built.add(move.output_env)
            unique.append(move)

        # then drop constructions whose result is never read
        needed = set()
        kept = []
        for move in reversed(unique):
            if (move.output_env is None) or (move.output_env in needed):
                needed.update(move.input_envs)
                kept.append(move)
        return kept[::-1]

    @staticmethod
    def _add_deletes(moves):
        """Add a ``delete`` move after each environment's last use."""
        last_use = {}
        for i, move in enumerate(moves):
            for env_key in move.input_envs:
                last_use[env_key] = i

        deletes = {}
        for env_key, i_last in last_use.items():
            deletes.setdefault(i_last, []).append(env_key)

        with_deletes = []
        for i, move in enumerate(moves):
            with_deletes.append(move)
            for env_key in deletes.get(i, ()):
                with_deletes.append(
                    EnvironmentMove("delete", input_envs=(env_key,))
                )
        return with_deletes

    def get_moves(self, blocks, include_deletes=True):
        """Get the sequence of operations needed to construct all environments
        for ``blocks``.

        Parameters
        ----------
        blocks : sequence of tuple[int, int]
            The ``(start, size)`` target blocks, which can have different
            sizes. Use :func:`all_blocks` to get every block of one size.
        include_deletes : bool, optional
            Whether to add ``delete`` moves after the last use of each cached
            environment.

        Returns
        -------
        tuple[EnvironmentMove, ...]
            Operations in execution order.

        Examples
        --------
        Select only the one-site environments at sites one and three:

        >>> plan = EnvironmentPlan(4)
        >>> moves = plan.get_moves([(1, 1), (3, 1)])
        >>> [move.output_block for move in moves if move.kind == "output"]
        [(1, 1), (3, 1)]
        """
        L = self.L
        blocks = tuple(sorted({(start, size) for start, size in blocks}))
        if not blocks:
            return ()
        for start, size in blocks:
            if not 1 <= size <= L:
                raise ValueError(f"block sizes must lie in [1, {L}]")
            # a periodic block can wrap, an open one must fit
            end = L - 1 if self.cyclic else L - size
            if not 0 <= start <= end:
                raise ValueError(f"block ({start}, {size}) is out of range")

        moves = []
        if self.cyclic:
            if self.schedule == "tree":
                self._build_cyclic_tree(blocks, moves)
            else:
                self._build_cyclic_cut(blocks, moves)
        else:
            self._build_cut(blocks, moves)

        moves = self._remove_redundant(moves)
        if include_deletes:
            moves = self._add_deletes(moves)

        return tuple(moves)

    def show(self, blocks, include_deletes=True):
        """Print a visual representation of the environment moves.

        ``E`` marks an input environment, ``I`` an original input site,
        ``B`` a target block, and ``D`` an environment being deleted.

        Parameters
        ----------
        blocks : sequence of tuple[int, int]
            The ``(start, size)`` target blocks to show.
        include_deletes : bool, optional
            Whether to show cache deletion moves.
        """
        L = self.L

        def mark_interval(row, env_key, marker):
            lo, hi = env_key
            for site in range(lo, hi):
                if site == lo == hi - 1:
                    row[site % L] = f"[{marker}]"
                elif site == lo:
                    row[site % L] = f"[{marker}-"
                elif site == hi - 1:
                    row[site % L] = f"-{marker}]"
                else:
                    row[site % L] = f"-{marker}-"

        for m, move in enumerate(self.get_moves(blocks, include_deletes)):
            s = [" . "] * L
            if move.kind in ("init", "contract"):
                for env_key in move.input_envs:
                    mark_interval(s, env_key, "E")
                for site in move.input_sites:
                    s[site] = "[I]"
            elif move.kind == "delete":
                for env_key in move.input_envs:
                    mark_interval(s, env_key, "D")
            else:
                for env_key in move.input_envs:
                    mark_interval(s, env_key, "E")
                start, size = move.output_block
                for i in range(start, start + size):
                    s[i % L] = " B "

            s = "".join(s)
            s = s.replace("][", "++")
            if self.cyclic and s[0] == "[" and s[-1] == "]":
                s = s[1:-1]
                s = f"+{s}+"

            s = f"{m:>2}: {move.kind:<9} {s}"
            if move.kind == "output":
                s += f"    => {move.output_block}"
            # no trailing spaces, so the output can be used in docstrings
            print(s.rstrip())


def execute_environment_plan(plan, blocks, init, contract):
    """Execute an environment plan with supplied numerical operations,
    yielding each block's environment as soon as it is ready. Every cached
    environment is dropped after its last use.

    Parameters
    ----------
    plan : EnvironmentPlan
        The symbolic environment schedule to execute.
    blocks : sequence of tuple[int, int]
        The ``(start, size)`` target blocks to compute.
    init : callable
        Called as ``init(input_sites)`` for each ``'init'`` move. It should
        construct an environment from the specified original sites.
    contract : callable
        Called as ``contract(input_envs, input_sites)`` for each
        ``'contract'`` move. ``input_envs`` is a tuple of one or two cached
        values. ``input_sites`` is a tuple of original sites, one for a tree
        contraction and none for an environment-environment merge.

    Yields
    ------
    block : tuple[int, int]
        The ``(start, size)`` block.
    pieces : tuple
        The environment pieces for ``block``.
    """
    cache = {}

    for move in plan.get_moves(blocks):
        if move.kind == "init":
            cache[move.output_env] = init(move.input_sites)
        elif move.kind == "contract":
            input_envs = tuple(cache[env_key] for env_key in move.input_envs)
            cache[move.output_env] = contract(input_envs, move.input_sites)
        elif move.kind == "output":
            yield (
                move.output_block,
                tuple(cache[env_key] for env_key in move.input_envs),
            )
        else:
            del cache[move.input_envs[0]]


def all_blocks(L, size, cyclic=True):
    """Get every valid ``(start, size)`` block of one size.

    Parameters
    ----------
    L : int
        Total number of sites.
    size : int
        Number of consecutive sites in each block.
    cyclic : bool, optional
        Whether blocks can wrap around the periodic boundary.

    Returns
    -------
    tuple[tuple[int, int], ...]
    """
    if not 1 <= size <= L:
        raise ValueError(f"block size must lie in [1, {L}]")
    nstarts = L if cyclic else L - size + 1
    return tuple((start, size) for start in range(nstarts))


def find_1d_block(sites, L, cyclic):
    """Find the shortest consecutive interval containing specified sites.

    Parameters
    ----------
    sites : sequence of int
        Sites to cover. Periodic sites are reduced modulo ``L`` and duplicate
        sites are ignored.
    L : int
        Total number of sites.
    cyclic : bool
        Whether the interval is allowed to cross the periodic boundary.

    Returns
    -------
    start : int
        Start of the shortest covering interval.
    size : int
        Number of consecutive sites in the interval.

    Notes
    -----
    In the cyclic case this finds the largest gap between requested sites and
    returns its complement. Ties between equal-size intervals resolve to the
    smallest start.
    """
    sites = tuple(sorted({site % L if cyclic else site for site in sites}))
    if not sites:
        raise ValueError("at least one site is required")
    if not cyclic:
        return sites[0], sites[-1] - sites[0] + 1

    candidates = []
    next_sites = sites[1:] + (sites[0] + L,)
    for site, next_site in zip(sites, next_sites):
        gap = next_site - site - 1
        start = next_site % L
        candidates.append((-gap, start, L - gap))

    _, start, size = min(candidates)
    return start, size


def gen_exact_environments(
    tn,
    plane_tags,
    blocks,
    *,
    cyclic=True,
    schedule="auto",
    contract_opts=None,
):
    """Yield exact environments for blocks of planes. Each plane is a group
    of tensors selected by one tag, e.g. a site in 1D or a row in 2D.
    Yield each environment as soon as it is ready.

    Parameters
    ----------
    tn : TensorNetwork
        Tensor network containing the planes to contract.
    plane_tags : sequence of str
        One tag per plane, in order along the contraction direction.
        The number of tags defines ``L``. Plane ``i`` is site ``i`` in the
        :class:`EnvironmentPlan`.
    blocks : sequence of tuple[int, int]
        The ``(start, size)`` target blocks of planes, which can have
        different sizes.
    cyclic : bool, optional
        Whether the planes wrap around periodically.
    schedule : {'auto', 'tree', 'cut'}, optional
        Environment construction schedule, only relevant if ``cyclic``. By
        default use 'cut', which uses linear work by permitting exact
        environment-environment contractions.
    contract_opts : dict, optional
        Supplied to :meth:`TensorNetwork.contract`. Always uses
        ``preserve_tensor=True`` so even scalar environments remain tensors.

    Yields
    ------
    block : tuple[int, int]
        The ``(start, size)`` block.
    environment : TensorNetwork
        The exact environment of ``block``.

    Notes
    -----
    An empty complement is represented by an empty tensor network.
    """
    from .tensor_core import TensorNetwork

    contract_opts = dict(contract_opts or ())
    contract_opts["preserve_tensor"] = True

    if schedule == "auto":
        schedule = "cut"
    plan = EnvironmentPlan(len(plane_tags), cyclic=cyclic, schedule=schedule)

    def get_planes(planes):
        tags = [plane_tags[plane] for plane in planes]
        return tn.select_any(tags, virtual=False)

    def init(input_sites):
        return get_planes(input_sites).contract(all, **contract_opts)

    def contract(input_envs, input_sites):
        tnc = input_envs[0]
        for input_env in input_envs[1:]:
            tnc = tnc | input_env
        if input_sites:
            tnc = tnc | get_planes(input_sites)
        return tnc.contract(all, **contract_opts)

    for block, pieces in execute_environment_plan(
        plan, blocks, init, contract
    ):
        yield block, TensorNetwork(pieces)


def gen_compressed_environments(
    tn,
    plane_tags,
    transverse_tags,
    blocks,
    max_bond=None,
    *,
    cyclic=True,
    compress_fn="ag",
    schedule="auto",
    method=None,
    layer_tags=None,
    cutoff=None,
    canonize=True,
    optimize="auto-hq",
    equalize_norms=False,
    compress_opts=None,
    **compress_method_opts,
):
    """Yield compressed environments for blocks of planes. Each plane is a
    group of tensors selected by one tag, e.g. a row in 2D. Add planes along
    ``plane_tags`` and compress along ``transverse_tags``, e.g. the columns.
    Yield each environment as soon as it is ready.

    With rows ``R0 ... R5`` and columns ``C0 ... C4``, the environment of
    block ``(2, 2)`` is::

        R0, R1  ●━━━━●━━━━●━━━━●━━━━●   compressed planes before the block
                │    │    │    │    │
        R2      o────o────o────o────o   ┬
                │    │    │    │    │   ┊ the block, not included
        R3      o────o────o────o────o   ┴
                │    │    │    │    │
        R4, R5  ●━━━━●━━━━●━━━━●━━━━●   compressed planes after the block
                C0   C1   C2   C3   C4

    If ``cyclic``, the planes before and after the block form a single
    environment, connected to both the first and last plane of the block.

    Parameters
    ----------
    tn : TensorNetwork
        Tensor network containing the planes to contract.
    plane_tags : sequence of str
        One tag per plane, in order along the contraction direction.
        The number of tags defines ``L``. Plane ``i`` is site ``i`` in the
        :class:`EnvironmentPlan`.
    transverse_tags : sequence of str
        Tags for sites within each plane, in compression order.
        Each tensor must have exactly one of these tags.
    blocks : sequence of tuple[int, int]
        The ``(start, size)`` target blocks of planes, which can have
        different sizes.
    max_bond : int, optional
        Maximum compressed bond dimension.
    cyclic : bool, optional
        Whether the planes wrap around periodically.
    compress_fn : {'ag', '1d', '2d'}, optional
        Which compression function to use, for the geometry of each
        environment along the transverse sites:

        - 'ag': :func:`~quimb.tensor.tnag.compress.tensor_network_ag_compress`,
          for any geometry, such as a ring.
        - '1d': :func:`~quimb.tensor.tn1d.compress.tensor_network_1d_compress`,
          for an open chain.
        - '2d': :func:`~quimb.tensor.tn2d.compress.tensor_network_2d_compress`,
          for a 2D lattice.

    schedule : {'auto', 'tree', 'cut'}, optional
        Environment construction schedule, only relevant if ``cyclic``. By
        default use 'tree', which never compresses two environments together.
        The 'cut' schedule uses fewer compressions but can degrade the
        approximation by combining two already compressed environments.
    method : str or callable, optional
        The compression method, supplied to ``compress_fn``. By default use
        its own default method.
    layer_tags : None or sequence[str], optional
        Add each plane's layers in this order, compressing after each.
        Assign each tensor to its first matching tag. A compressed tensor
        with all layer tags is thus added with the first layer.
    cutoff : float, optional
        Compression cutoff, supplied to ``compress_fn``. By default use its
        own default cutoff.
    canonize : bool or str, optional
        Canonicalization option supplied to the compressor.
    optimize : str, optional
        Contraction path optimizer supplied to the compressor.
    equalize_norms : bool or float, optional
        Whether to equalize tensor norms after each compression.
    compress_opts : dict, optional
        Additional options supplied to the compressor.
    compress_method_opts
        Additional options supplied to the compression method.

    Yields
    ------
    block : tuple[int, int]
        The ``(start, size)`` block.
    environment : TensorNetwork
        The compressed environment of ``block``.
    """
    from .tensor_core import TensorNetwork

    if compress_fn == "ag":
        from .tnag.compress import tensor_network_ag_compress as compressor
    elif compress_fn == "1d":
        from .tn1d.compress import tensor_network_1d_compress as compressor
    elif compress_fn == "2d":
        from .tn2d.compress import tensor_network_2d_compress as compressor
    else:
        raise ValueError(
            f"Unrecognized compress_fn: {compress_fn}, should be one of: "
            "'ag', '1d', '2d'."
        )

    if schedule == "auto":
        schedule = "tree"
    if method is not None:
        compress_method_opts["method"] = method
    if cutoff is not None:
        compress_method_opts["cutoff"] = cutoff
    if isinstance(layer_tags, str):
        layer_tags = (layer_tags,)

    plan = EnvironmentPlan(len(plane_tags), cyclic=cyclic, schedule=schedule)

    def get_plane_layers(plane):
        # copy before combining networks, which can rename shared indices
        plane_tn = tn.select(plane_tags[plane], virtual=False)
        if layer_tags is None:
            return (plane_tn,)

        # assign tensors with several layer tags to the first matching layer
        layers = []
        for tag in layer_tags:
            if tag in plane_tn.tag_map:
                plane_tn, layer = plane_tn.partition(tag)
                layers.append(layer)
        if plane_tn.num_tensors:
            raise ValueError(
                f"each tensor must have one of layer_tags {layer_tags}, "
                f"but got {plane_tn.tags}."
            )
        return tuple(layers)

    def compress(env):
        return compressor(
            env,
            max_bond=max_bond,
            site_tags=transverse_tags,
            canonize=canonize,
            optimize=optimize,
            equalize_norms=equalize_norms,
            compress_opts=compress_opts,
            **compress_method_opts,
        )

    def init(input_sites):
        layers = [
            layer for plane in input_sites for layer in get_plane_layers(plane)
        ]
        # a single layer needs no compression on its own
        env = layers[0]
        for layer in layers[1:]:
            env = compress(env | layer)
        return env

    def contract(input_envs, input_sites):
        env = input_envs[0]
        for input_env in input_envs[1:]:
            env = env | input_env
        if not input_sites:
            return compress(env)
        for plane in input_sites:
            for layer in get_plane_layers(plane):
                env = compress(env | layer)
        return env

    for block, pieces in execute_environment_plan(
        plan, blocks, init, contract
    ):
        yield block, TensorNetwork(pieces)
