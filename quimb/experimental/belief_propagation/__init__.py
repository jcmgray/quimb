"""Belief propagation (BP) routines. There are three potential categorizations
of BP and each combination of them is potentially valid specific algorithm.

1-norm vs 2-norm BP
-------------------

- 1-norm (normal): BP runs directly on the tensor network, messages have size
  ``d`` where ``d`` is the size of the bond(s) connecting two tensors or
  regions.
- 2-norm (quantum): BP runs on the squared tensor network, messages have size
  ``d^2`` where ``d`` is the size of the bond(s) connecting two tensors or
  regions. Each local tensor or region is partially traced (over dangling
  indices) with its conjugate to create a single node.


Graph vs Hypergraph BP
----------------------

- Graph (simple): the tensor network lives on a graph, where indices either
  appear on two tensors (a bond), or appear on a single tensor (are outputs).
  In this case, messages are exchanged directly between tensors.
- Hypergraph: the tensor network lives on a hypergraph, where indices can
  appear on any number of tensors. In this case, the update procedure is two
  parts, first all 'tensor' messages are computed, these are then used in the
  second step to compute all the 'index' messages, which are then fed back into
  the 'tensor' message update and so forth. For 2-norm BP one likely needs to
  specify which indices are outputs and should be traced over.

The hypergraph case of course includes the graph case, but since the 'index'
message update is simply the identity, it is convenient to have a separate
simpler implementation, where the standard TN bond vs physical index
definitions hold.


Dense vs Vectorized vs Lazy BP
------------------------------

- Dense: each node is a single tensor, or pair of tensors for 2-norm BP. If all
  multibonds have been fused, then each message is a vector (1-norm case) or
  matrix (2-norm case).
- Vectorized: the same as the above, but all matching tensor update and message
  updates are stacked and performed simultaneously. This can be enormously more
  efficient for large numbers of small tensors.
- Lazy: each node is potentially a tensor network itself with arbitrary inner
  structure and number of bonds connecting to other nodes. The message are
  generally tensors and each update is a lazy contraction, which is potentially
  much cheaper / requires less memory than forming the 'dense' node for large
  tensors.

(There is also the MPS flavor where each node has a 1D structure and the
messages are matrix product states, with updates involving compression.)


Overall that gives 12 possible BP flavors, some implemented here:

- [x] (HD1BP) hyper, dense, 1-norm - this is the standard BP algorithm
- [x] (HD2BP) hyper, dense, 2-norm
- [x] (HV1BP) hyper, vectorized, 1-norm
- [ ] (HV2BP) hyper, vectorized, 2-norm
- [ ] (HL1BP) hyper, lazy, 1-norm
- [ ] (HL2BP) hyper, lazy, 2-norm
- [x] (D1BP) simple, dense, 1-norm - simple BP for simple tensor networks
- [x] (D2BP) simple, dense, 2-norm - this is the standard PEPS BP algorithm
- [ ] (V1BP) simple, vectorized, 1-norm
- [ ] (V2BP) simple, vectorized, 2-norm
- [x] (L1BP) simple, lazy, 1-norm
- [x] (L2BP) simple, lazy, 2-norm

The 2-norm methods can be used to compress bonds or estimate the 2-norm.
The 1-norm methods can be used to estimate the 1-norm, i.e. contracted value.
Both methods can be used to compute index marginals and thus perform sampling.

The vectorized methods can be extremely fast for large numbers of small
tensors, but do currently require all dimensions to match.

The dense and lazy methods can can converge messages *locally*, i.e. only
update messages adjacent to messages which have changed.
"""

from .bp_common import initialize_hyper_messages
from .d1bp import D1BP, contract_d1bp
from .d2bp import D2BP, compress_d2bp, contract_d2bp, sample_d2bp
from .hd1bp import HD1BP, contract_hd1bp, sample_hd1bp
from .hv1bp import HV1BP, contract_hv1bp, sample_hv1bp
from .l1bp import L1BP, contract_l1bp
from .l2bp import L2BP, compress_l2bp, contract_l2bp
from .regions import RegionGraph

__all__ = (
    "compress_d2bp",
    "compress_l2bp",
    "contract_d1bp",
    "contract_d2bp",
    "contract_hd1bp",
    "contract_hv1bp",
    "contract_l1bp",
    "contract_l2bp",
    "D1BP",
    "D2BP",
    "HD1BP",
    "HV1BP",
    "initialize_hyper_messages",
    "L1BP",
    "L2BP",
    "RegionGraph",
    "sample_d2bp",
    "sample_hd1bp",
    "sample_hv1bp",
)
