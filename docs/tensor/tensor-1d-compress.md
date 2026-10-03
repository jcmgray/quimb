(tensor-network-1d-compress)=

# 1D Compression

This page describes functionality in `quimb` for compressing 1D-like tensor
networks, the core routine being:
  - [`tensor_network_1d_compress`](#tensor_network_1d_compress).

Which is also often called as a *subroutine* when contracting 2D tensor
networks, e.g. via:

- [`TensorNetwork2D.contract_boundary`](#TensorNetwork2D.contract_boundary) or
- [`PEPS.compute_local_expectation`](#TensorNetwork2DVector.compute_local_expectation)

The core function makes use of the tag and named index architecture of `quimb`
to accept ***any 1D-like*** tensor network, regardless of number of indices and
internal structure, and produce a ***compressed, fully 1D*** tensor network.

1D-like here means a 'factored' tensor network with multiple tensors per
'site', usually with multiple bonds between sites, and arbitrary outer indices
per site. The output is a tensor network with a single tensor per site, single
bond between sites, and the same outer indices per site as the input. The
classic and simplest example is an MPO applied to an MPS, which we want to
compress back to an MPS with `max_bond` bond dimension:

```{image} figs/tn1dcompress-example-mpo-mps.png
:width: 500px
```

The `site_tags`, here `("I0", "I1", ...)`, specify the chain geometry and each
tensor should have **exactly one of them**, designating which site it is
grouped into. `site_tags` can be specified manually, or if present as an
attribute, they will be picked up automatically. The MPO-MPS example is such a
case:

```python
import quimb.tensor as qtn

mps = qtn.MPS_rand_state(8, bond_dim=7)
mpo = qtn.MPO_rand(8, bond_dim=5)

# apply the mpo to the mps lazily, i.e. without contracting
tn = mps.gate_with_op_lazy(mpo)
print(tn.site_tags)
# ('I0', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7')

# compress!
tnc = qtn.tensor_network_1d_compress(tn, max_bond=7)
tnc
# MatrixProductState(tensors=8, indices=15, L=8, max_bond=7)
```

A typical example of a case with multiple tensors per site and an inhomogeneous
structure is applying gates, $UAU^\dagger$ sandwich-style, to an MPO to perform
Heisenberg evolution, but where we only compress *lazily* after applying many
gates:

```{image} figs/tn1dcompress-example-mpo-heis-gates.png
:width: 500px
```

```{hint}
Note by default, as depicted by the arrows in these figures, all 1D compression
methods produce a right canonical (canonical center at the first site) output
TN by default.
```

Long range bonds are also supported. Here's a schematic of applying some long
range gates (suitably split spatially) to an MPS, and then compressing back to
an MPS (here demonstrating custom `site_tags=("A", "B", "C", ...)` also):

```{image} figs/tn1dcompress-example-mps-long-range-gates.png
:width: 500px
```

All of these compressions share the same interface and methods.
Of course, the structure of the tensor network to be compressed, including
internal to each site and any long-range bonds, still affects the computational
cost and scaling of compression!

```{warning}
The ability to compress long range bonds means that it will happily try and
compress, for example, a **periodic** 1D-like tensor network into a
**non-periodic** one, which is usually not what you want. Instead consider
using [`tensor_network_ag_compress`](#tensor_network_ag_compress) which retains
whatever the effective geometry of `site_tags` is.
```

Note if there is no 'factored' structure (i.e. it's already a MPS or MPO like
object), there the methods below will still work, but won't offer much in the
way of speedup over a direct SVD based compression (see the `"direct"` method
below), or for example
[`TensorNetwork1DFlat.compress`](#TensorNetwork1DFlat.compress) or
[`TensorNetwork.compress_all_1d`](#TensorNetwork.compress_all_1d).


## Specifying `site_tags`

In the simplest case, 1D tensor networks carry their `site_tags` (when you
combine a MPO and MPS these are preserved for instance) and you don't need to
specify `site_tags`.

If you are supplying `site_tags` manually, there are three ways to do it:

1. a sequence of individual tags, e.g. `site_tags=['I0', 'I1', 'I2', ...]`: each
  tensor must have exactly one of these tags, and they are ordered in the
  order of the sequence.

2. a sequence of *groups* of tags, e.g.
  `site_tags=[('A', 'B'), ('C', 'D'), ...]`: now each tensor can just have
  *any one* of the tags in a group. I.e. (C or D).

3. finally, if supplying the grouped form, each element in the group can itself
  be a sequence, in which case to match that element a tensor must have *all*
  of the tags in the sub-group. For example, with:
  `site_tags=[("X", ("Y", "Z")), ...]` to match the first site group a tensor
  must have (X or (Y and Z)).

## Basic compress options

[`tensor_network_1d_compress`](#tensor_network_1d_compress) takes the following
main options:

- `method` - the compression algorithm to use, see below for details.
- `max_bond` - the maximum bond dimension of the compressed TN.
- `cutoff` - a dynamic truncation threshold to compress with.
- `cutoff_mode` - how to interpret `cutoff`
- `normalize` - whether to normalize the final compressed TN.
- `equalize_norms` - whether to keep norms of tensors during the compression at
  a similar scale, and accrue any overall scaling into the `tnc.exponent`
  attribute or distribute at the end.

Other more specialized options that not all methods support include:

- `canonize` - whether to perform a pre-compression canonicalization or
  pseudo-canonicalization of the TN.
- `permute_arrays` - whether to permute the internal storage of the arrays
  after compression to some standard, if the TN class supports it.
- `optimize` - the contraction strategy to use for any contractions.
- `sweep_reverse` - whether to reverse the direction of the compression sweep,
  resulting in a left canonical output TN (canonical center at the last site).
- `inplace` - whether to perform the compression in-place, modifying the input
  TN.
- `contract_opts` - more specialized contraction options.
- `canonize_opts` - specialized options for any `canonize` step.
- `project_opts` - specialized options for any projection step.
- `compress_opts` - specialized options for the compression step.

## Method comparison

The table below gives a rough comparison of the methods available. The columns
are:

*Scaling:*

- **MPO-MPS**: compress MPS with bond dimension $\chi$ with MPO of bond
  dimension $D$, back to $\chi$, with physical dimension assumed
  $p=\mathcal{O}(1)$
- **TN2D**: contract a 2D TN with bond dimension $D$ using boundary bond
  dimension $\chi$. This is the cost of contracting a classical partition
  function, or an amplitude of a 2D PEPS.
- **PEPS Norm**: boundary contract the norm of a 2D PEPS, using the layered
  approach to compress ket and bra one-by-one. This is the cost of computing
  local observables.

*Other features:*

- **Sum of TNS**: whether the method can directly compress a sum of tensor networks given as a sequence.
- **Full Spectrum**: whether the method considers the full spectrum of singular values before truncating.
- **Precision**: the precision floor of the compression, either $\epsilon$ or $\sqrt{\epsilon}$.
- **Random Input**: whether the method requires an initial random tensor network (or arrays)

More specific features are noted in the method subsections below.

|Method    |MPO-MPS        |TN2D           |PEPS Norm     |Sum of TNS|Full Spectrum|Precision      |Random Input|
|----------|---------------|---------------|--------------|----------|-------------|---------------|------------|
|`"direct"`|🟥 $D^3 \chi^3$|🟥 $D^4 \chi^3$|🟥 $D^5 \chi^3$|🟥 No     |🟢 Yes    |🟢 $\epsilon$      |🟢 No       |
|`"dm"`    |🔶 $D^2 \chi^3$|🔶 $D^3 \chi^3$|🔶 $D^4 \chi^3$|🟥 No     |🟢 Yes    |🟥 $\sqrt{\epsilon}$|🟢 No      |
|`"zipup"` |🟢 $D \chi^3$  |🔶 $D^3 \chi^3$|🔶 $D^4 \chi^3$|🟥 No     |🟥 No     |🟢 $\epsilon$      |🟢 No       |
|`"fit"`-1 |🟢 $D \chi^3$  |🟢 $D^2 \chi^3$|🟢 $D^3 \chi^3$|🟢 Yes    |🟥 No     |🟢 $\epsilon$      |🔶 guess    |
|`"fit"`-2 |🟢 $D \chi^3$  |🔶 $D^3 \chi^3$|🟥 $D^6 \chi^3$|🟢 Yes    |🔶 Approx |🟢 $\epsilon$      |🔶 guess    |
|`"src"`   |🟢 $D \chi^3$  |🟢 $D^2 \chi^3$|🟢 $D^3 \chi^3$|🔶 Not yet$^1$ |🟥 No|🟢 $\epsilon$      |🟥 Yes      |
|`"srcmps"`|🟢 $D \chi^3$  |🟢 $D^2 \chi^3$|🟢 $D^3 \chi^3$|🔶 Not yet$^1$ |🟥 No|🟢 $\epsilon$      |🔶 guess    |
|`"sdc"`   |🟢 $D \chi^3$  |🔶 $D^3 \chi^3$|🔶 $D^4 \chi^3$|🟥 No     |🟥 No     |🟢 $\epsilon$      |🟢 No       |
|`"sdcr"`  |🟢 $D \chi^3$  |🟢 $D^2 \chi^3$|🟢 $D^3 \chi^3$|🟥 No     |🟥 No     |🟢 $\epsilon$      |🔶 noise$^2$|

1. `"src"` and `"srcmps"` in principle can be easily adapted to directly
   compress a sum of TNs, by sampling each TN with the same random sketch TN.
2. random noise is needed at the array level for individual RSVDs, but not at
   the tensor network level, meaning symmetric tensors don't have to guess a
   charge distribution.
- `"fit"`-1 and `"fit"`-2 refer here to the `bsz=1` and `bsz=2` modes of the
  `"fit"` method, which have very different costs when bonds *between* layers
  (i.e. $p$) are large.

## Benchmarks

The following benchmarks are for a single compression of a random MPO-MPS like
object, showing accuracy vs. time for a sweep of `max_bond` = $\chi$. They
give a very rough idea of the relative performance of the methods in different
regimes.


### MPO-MPS compression,

Here we take $D_\mathrm{MPS}=D_\mathrm{MPO}$ large, but $p$ is still small.

*Accuracy vs time:*

```{image} figs/bigbigsmall_error_vs_time.png
:width: 600px
```

*Accuracy at fixed bond dimension:*

```{image} figs/bigbigsmall_error_vs_chi.png
:width: 500px
```

Clearly `"src"`, `"srcmps"`, `"sdc"` and `"sdcr"` all perform similarly well.
The default settings for `"fit"` do 10 iterations, which results in a high
overhead, but the best accuracy for a given `max_bond`. The precision limit
of `"dm"` is evident, as is the lesser accuracy of `"zipup"`.



### TN2D boundary like

Here we take both $p$ and $D_\mathrm{MPS}=D_\mathrm{MPO}$ large, like
contracting two rows of a 2D TN or PEPS amplitude.

*Accuracy vs time:*

```{image} figs/bigbigbig_error_vs_time.png
:width: 600px
```

*Accuracy at fixed bond dimension:*

```{image} figs/bigbigbig_error_vs_chi.png
:width: 500px
```

Here `"src"`, `"srcmps"` and `"sdcr"` all perform similarly well.

### small MPO-MPS compression

For the final benchmark we take $D_\mathrm{MPO} = p = 2$, with
$D_\mathrm{MPS} = \chi$. Similar to applying a layer of short range gates for
example. Note the input MPS bond dimension scales with $\chi$ here - the
decrease in error comes from the compression becoming easier.

*Accuracy vs time:*

```{image} figs/bigsmallsmall_error_vs_time.png
:width: 600px
```

*Accuracy at fixed bond dimension:*

```{image} figs/bigsmallsmall_error_vs_chi.png
:width: 500px
```

Here, since the 'inner' bond dimensions are small, all methods perform more
similarly. Nonetheless, `"src"`, followed closely by `"srcmps"` and `"sdcr"`,
still give the best accuracy/cost tradeoff.

As you can see `"sdcr"` is generally a solid choice for most tasks, offering
both a good accuracy/cost tradeoff and also good absolute accuracy for a given
`max_bond`. Its variant `"sdcr-oversample"` is recommended if `cutoff` is
expected to be important along with `max_bond`, or for block sparse abelian
symmetric or fermionic TNs, where the charge distribution needs adjusting.

## Methods

Finally we list each method in more detail, with their use cases and
references.

```{hint}
The following methods are the *1D specific routines*, found in
[`quimb.tensor.tn1d.compress`](quimb.tensor.tn1d.compress).
Other `method` names fall through to [`tensor_network_ag_compress`](#tensor_network_ag_compress), the arbitrary
geometry backend in `quimb.tensor.tnag.compress`. This means methods such as
`"projector"`, `"l2bp"`, `"superorthogonal"`, `"local-early"` and `"local-late"`
can also be used through the same interface. However, note if called through
the *1D interface* they coerce an exactly 1D output by possibly **inserting
identity** tensors.
```

---

### `'direct'`

[`tensor_network_1d_compress_direct`](#tensor_network_1d_compress_direct)

The `"direct"` method is the simplest 'contract + canonicalize + compress'
approach. Each site is contracted and then canonicalized (by default using QR)
to shift the orthogonality center to one side, and then a truncation sweep
using SVD is performed in the reverse direction. It is almost optimal (each
truncation is optimal with respect to the current *truncated* state, meaning
they have some slight non-optimal inter-dependence).

*Use cases:*
- if the effective $D$ is small.
- you need the full spectrum and need very high precision.
- you want a deterministic algorithm based only on the input TN.

*References:*
- 'The density-matrix renormalization group in the age of matrix product
  states', Ulrich Schollwöck - <https://arxiv.org/abs/1008.3477> (§4.5.1).

---

### `'dm'`

[`tensor_network_1d_compress_dm`](#tensor_network_1d_compress_dm)

The `"dm"` method is the 'density matrix' approach, which is a more efficient
version of the `"direct"` method. It contracts the full environment overlap
rather than shifting an orthogonality center, and then performs a hermitian
eigen-decomposition in the reverse truncating sweep. It has the same *accuracy*
as `"direct"`, i.e. *almost* optimal, but contracting the full environment
squares the effective singular values, limiting the *precision* to
$\sqrt{\epsilon}$.

*Use cases:*
- if the effective $D$ is moderate.
- you need the full spectrum and but not very high precision.
- you want to avoid QR and SVD in favour of contractions and EIGH (for example
  these can be much more efficient on GPU depending on library).

*References:*
- <https://tensornetwork.org/mps/algorithms/denmat_mpo_mps/>
- 'Approximate Contraction of Arbitrary Tensor Networks with a Flexible and
  Efficient Density Matrix Algorithm', Linjian Ma, Matthew Fishman,
  E. M. Stoudenmire & Edgar Solomonik - <https://arxiv.org/abs/2406.09769>
  (§3.2.2 for MPO-MPS compression; §6.2 for QR-SVD).

---

### `'zipup'`

[`tensor_network_1d_compress_zipup`](#tensor_network_1d_compress_zipup)

The `"zipup"` method is a fast method that 'pseudo-canonicalizes' the
*factored* TN before performing a single truncated sweep. Its accuracy is not
as good as most methods because the truncations are not optimal. It also scales
worse with 'inter-layer' bond dimension. Although `"zipup"` takes a `cutoff`,
it is not applied in the full canonical gauge, and thus is best used
conservatively with the default `cutoff_mode="rel"` (or see oversampling
section below).

*Use cases:*
- Generally the methods below are faster and more accurate, but for certain
  easy contractions it can be a useful, fast, deterministic reference.

*References:*
- 'Minimally Entangled Typical Thermal State Algorithms', E.M. Stoudenmire &
  Steven R. White - <https://arxiv.org/abs/1002.1305>

---

### `'fit'`

[`tensor_network_1d_compress_fit`](#tensor_network_1d_compress_fit)

The `"fit"` method is an iterative least squares approach, similar to DMRG.
It starts from a guess for the compressed TN and sweeps back and forth,
optimizing one or two tensors at a time using the overlap with the target TN.
It is the most flexible and potentially most accurate method at a fixed
`max_bond`. It also has the joint best scaling per sweep (with `bsz=1`), but
the overhead of iterating can outweigh this advantage.

With `bsz=1`, each update optimizes a single tensor, which is cheap but does
not consider the full singular value spectrum. Use `cutoff=0.0` for this mode.
With `bsz=2`, each update optimizes two tensors and splits them with a
truncated SVD, allowing both `max_bond` and `cutoff` to control the bond
size. This is more expensive but can help avoid getting stuck in local minima.

The initial guess is random by default, and can give good results even after
a single sweep. You can supply a guess with `tn_fit`, or specify another
compression method, e.g. `tn_fit="zipup"`. The `"fit-zipup"` and
`"fit-projector"` variants generate a guess then perform one-site fitting.
Use `max_iterations` to control the number of sweeps and `tol` to stop when
local tensor changes become small.

This is also the only method that can directly compress a *sum* of TNs,
supplied as a sequence, without first constructing their explicitly summed
representation.

*Use cases:*
- you want the best accuracy at a fixed bond dimension.
- you want to refine a guess from another method.
- you want to compress a sum of tensor networks directly.

*References:*
- 'The density-matrix renormalization group in the age of matrix product
  states', Ulrich Schollwöck - <https://arxiv.org/abs/1008.3477> (§4.5.2).

---

### `'src'`

[`tensor_network_1d_compress_src`](#tensor_network_1d_compress_src)

The `"src"` method is 'successive randomized compression'. It sketches the
whole TN using random Khatri-Rao (diagonal MPS) tensors, forming low rank
left environments in a first sweep. A sweep in the opposite direction then
uses these to form orthogonal projectors, by default via QR, and
builds the compressed TN one site at a time. Only `max_bond` is relevant for
SRC, not `cutoff` (or see oversampling section below).

This is equivalent to a single one-site `"fit"` sweep starting from a random
diagonal MPS, and has the same scaling, but is more efficient. It has the joint
best scaling and often gives fairly close to optimal accuracy, giving it one of
the best cost/accuracy tradeoffs.

Use `seed` to control the random sketch. `noise_dist` defaults to `"normal"`,
with `"rademacher"` also available. For sites with multiple outer indices,
`noise_mode="joint"` (the default) sketches them together, while `"separable"`
uses independent noise for each index. The Khatri-Rao sketch has an effective
hyper index shared across sites, which is currently not supported for block
sparse abelian symmetric or fermionic TNs.

*Use cases:*
- you have dense arrays and want fast and accurate compression at a fixed bond
  dimension, and don't mind a small amount of randomness in the result.

*References:*
- 'Successive randomized compression: A randomized algorithm for the
  compressed MPO-MPS product', Chris Camaño, Ethan N. Epperly &
  Joel A. Tropp - <https://arxiv.org/abs/2504.06475>.

---

### `'srcmps'`

[`tensor_network_1d_compress_srcmps`](#tensor_network_1d_compress_srcmps)

The `"srcmps"` method is SRC but using a random (or supplied) MPS as the
sketch. It has the same scaling as `"src"`, but with a slight overhead and
very slight accuracy increase. It is equivalent to a single one-site `"fit"`
sweep starting from the sampling MPS, but more efficient. As with `"src"`, only
`max_bond` is relevant, not `cutoff` (or see oversampling section below).

By default, a random MPS with bond dimension `max_bond` is generated. Use
`seed` to control this guess and `noise_dist` to choose its distribution,
either `"normal"` (the default) or `"rademacher"`. Alternatively, supply an
MPS with `tn_fit`, or generate one with another compression method, e.g.
`tn_fit="zipup"` or `tn_fit={"method": "zipup", "cutoff": 1e-8}`. The sketch
must have matching outer indices and site grouping. The sampling MPS's bond
dimensions override `max_bond`.

Abelian symmetric and fermionic arrays are supported, but you will need an
initial guess that specifies the bond charge distribution.

*Use cases:*
- you have dense arrays and want fast and accurate compression at a fixed bond
  dimension, and don't mind a small amount of randomness in the result.
- you want a single compression sweep using a supplied or informed guess.
- you have symmetric or fermionic arrays and a compatible sampling MPS.

*References:*
- 'Successive randomized compression: A randomized algorithm for the
  compressed MPO-MPS product', Chris Camaño, Ethan N. Epperly &
  Joel A. Tropp - <https://arxiv.org/abs/2504.06475>
  (§2.2 discusses MPS sketches).
- 'Randomized Algorithms for Rounding in the Tensor-Train Format',
  Hussam Al Daas et al. - <https://doi.org/10.1137/21M1451191>
  (related tensor train sketching methods that don't utilize the factorized
  structure).

---

### `'sdc'`

[`tensor_network_1d_compress_sdc`](#tensor_network_1d_compress_sdc)

Successive deterministic compression (SDC) is a deterministic analog of SRC,
where the 'sketch' is formed by successive truncated SVDs of the TN's left
environments. Compared with `"src"`, this base version scales worse in the
effective physical dimension $p$, which is effectively $D$ in TN2D or PEPS
norm contractions. Although `"sdc"` takes a `cutoff`, it is not applied in the
full canonical gauge, and thus is best used conservatively with the default `cutoff_mode="rel"` (or see oversampling section below).

It is equivalent to a single one-site `"fit"` sweep starting with the
`"zipup"` guess, but more efficient. Since the sketch is generated from the TN
itself, abelian symmetric and fermionic arrays are supported, though note the
accuracy of the charge distribution is set by the initial lower accuracy sweep,
(again see oversampling section below).

*Use cases:*
- you want a deterministic and accurate compression that is fast when the
  'inter-layer' bond dimension is not too large.

*References:*
- 'Efficient Application of Tensor Network Operators to Tensor Network States
  Through Successive Deterministic Compression', Richard M. Milbradt et al. -
  <https://arxiv.org/abs/2601.19650> (§II.B).

---

### `'sdcr'`
[`tensor_network_1d_compress_sdcr`](#tensor_network_1d_compress_sdcr)

The `"sdcr"` method is SDC with a 'randomized QB' for each low rank left
environment. These only need to be useful sketches of the current state,
so no oversampling or power iterations are used by default. It has the joint
best scaling, with somewhat more overhead than `"src"` or `"srcmps"`, but
typically somewhat better accuracy. As with `"sdc"`, `cutoff` is only applied
in the sketching sweep, and is best left at the default `0.0`.

No initial guess is needed, and the random sketching is done at the array
level (i.e. within blocks), making it suitable for abelian symmetric and
fermionic arrays. The the charge distribution is set by the low accuracy
sketching pass however, so oversampling (see below) may be desired.

*Use cases:*
- you want fast and accurate compression of dense, symmetric or fermionic
  tensor networks, and don't need `cutoff` driven truncation or mind a small
  amount of noise.

*References:*
- 'Efficient Application of Tensor Network Operators to Tensor Network States
  Through Successive Deterministic Compression', Richard M. Milbradt et al. -
  <https://arxiv.org/abs/2601.19650> (SDC).
- 'Finding structure with randomness: Probabilistic algorithms for
  constructing approximate matrix decompositions', Nathan Halko,
  Per-Gunnar Martinsson & Joel A. Tropp - <https://arxiv.org/abs/0909.4061>
  (randomized SVD).

---

### `"{}-oversample"` methods

The following methods perform an initial compression sweep with the base
`method` at some constant factor times the target `max_bond` (typically
`1.5 * max_bond`, controlled by `max_bond_oversample`), and then a final direct
sweep to the target `max_bond` on the resulting TN (which now has a single
tensor per site):

- [`tensor_network_1d_compress_zipup_oversample`](#tensor_network_1d_compress_zipup_oversample)
- [`tensor_network_1d_compress_fit_oversample`](#tensor_network_1d_compress_fit_oversample)
- [`tensor_network_1d_compress_src_oversample`](#tensor_network_1d_compress_src_oversample)
- [`tensor_network_1d_compress_srcmps_oversample`](#tensor_network_1d_compress_srcmps_oversample)
- [`tensor_network_1d_compress_sdc_oversample`](#tensor_network_1d_compress_sdc_oversample)
- [`tensor_network_1d_compress_sdcr_oversample`](#tensor_network_1d_compress_sdcr_oversample)

In most cases simply raising `max_bond` achieves better accuracy for a given
computational cost vs. oversampling. Nonetheless it can be useful if:

- For **block sparse, abelian symmetric or fermionic** tensor networks, the
  initial method might not distribute the charge sectors well enough, in which
  case oversampling allows them to be dynamically adjusted during truncation.
- You want better accuracy at a *fixed* `max_bond` for some reason other than
  immediate cost.
- You want to rely on `cutoff` to control truncation, which cannot be reliably
  done in the initial, 'non-canonical' sweep of these base methods.

The options for the direct sweep can be controlled via `compress_opts_final`.
