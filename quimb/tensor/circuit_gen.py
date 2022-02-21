import math
import random
import itertools

from .. import rand, seed_rand
from . import Circuit


def inject_u3s(
    ent_gates,
    gate2='cz',
    avoid_doubling=False,
    seed=None,
):
    r"""Take a sequence of pairs denoting qubits to entangle and interleave one
    single qubit gate inbetween every leg. For example:

        ent_gates = [(0, 1), (2, 3), (1, 2)]

    Would go get made into a circuit like::

        |  |  |  |        |  |  |  |
        |  |  |  |        |  u  u  |
        |  |  |  |        |  |  |  |
        |  o++o  |        u  o++o  u
        |  |  |  |        |  |  |  |
        |  |  |  |  -->   |  u  u  |
        |  |  |  |        |  |  |  |
        o++o  o++o        o++o  o++o
        |  |  |  |        |  |  |  |
        |  |  |  |        u  u  u  u
        |  |  |  |        |  |  |  |

    Technically, this generates a bipartite graph between single qubit and two
    qubit tensors, and should be the most expressive circuit possible for that
    'budget' of entangling gates.

    Parameters
    ----------
    ent_gates : sequence[tuple[int]]
        A 'stack' of entangling gate pairs to apply.
    gate2 : {'cx', 'cy', 'cz', 'iswap', ..., str}, optional
        The gate to use for the entanling pairs.
    avoid_doubling : bool, optional
        Whether to avoid placing an entangling gate directly above the same
        entangling gate (there will still be single qubit gates interleaved).

    Returns
    -------
    Circuit
    """
    if seed is not None:
        seed_rand(seed)

    # keep track of where not to apply another entangling gate
    just_entangled = set()

    # keep track of where its worth putting a U3
    n = max(itertools.chain.from_iterable(ent_gates)) + 1
    needs_u3 = [True] * n

    # create the circuit!
    gates = []
    # consume list of pairs to entangle
    while ent_gates:

        # break up entanling gates with U3s where necesary
        for i in range(n):
            if needs_u3[i]:
                gates.append(('U3', *rand(3, scale=2 * math.pi), i))
                needs_u3[i] = False

        # try and get the next entanling gate which is not 'doubled'
        for k, pair in enumerate(ent_gates):
            # (just_entangled will never be populated if avoid_doubling=False)
            if pair not in just_entangled:
                break

        i, j = ent_gates.pop(k)
        gates.append((gate2, i, j))

        #  1  2  3  4
        #  ^  ^  ^  ^
        #  |  |  |  |
        #  o++o  o++o
        #  |  |  |  |
        #  |  o++o  |     <- if we have just placed (2, 3), disable it in next
        #  |  |  |  |        round but enable (1, 2) and (3, 4) etc
        #  ^  ^  ^  ^
        if avoid_doubling:
            just_entangled = {
                ij for ij in just_entangled
                if (i not in ij) and (j not in ij)
            }
            just_entangled.add((i, j))

        # update the register of where to place U3s
        needs_u3[i] = needs_u3[j] = True

    # place the final layer of U3s
    for i in range(n):
        if needs_u3[i]:
            gates.append(('U3', *rand(3, scale=2 * math.pi), i))

    return gates


def gates_to_param_circuit(gates, n, parametrize='U3', **circuit_opts):
    """Turn the sequence ``gates`` into a ``Circuit`` of ``n`` qubits, with any
    gates that appear in ``parametrize`` being... parametrized.

    Parameters
    ----------
    gates : sequence[tuple[str, float, int]]
        The gates describing the circuit.
    n : int
        The number of qubits to make the circuit act one.
    parametrize : str or sequence[str], optional
        Which gates to parametrize.
    circuit_opts
        Supplied to :class:`~quimb.tensor.circuit.Circuit`.

    Returns
    -------
    Circuit
    """
    if isinstance(parametrize, str):
        parametrize = (parametrize,)

    circ = Circuit(n, **circuit_opts)
    for g in gates:
        circ.apply_gate(*g, parametrize=g[0] in parametrize)

    return circ


def circ_ansatz_1D_zigzag(
    n,
    depth,
    gate2='cz',
    seed=None,
    **circuit_opts
):
    r"""A 1D circuit ansatz with forward and backward layers of entangling
    gates interleaved with U3 single qubit unitaries::

        |  |  |  |
        u  u  |  |
        o++o  u  |
        |  |  |  u
        |  o++o  |
        |  |  u  |
        |  |  o++o
        u  u  u  u
        |  |  o++o
        |  |  u  |
        |  o++o  |
        |  u  |  u
        o++o  u  |
        u  u  |  |
        |  |  |  |

    Parameters
    ----------
    n : int
        The number of qubits.
    depth : int
        The number of entangling gates per pair.
    gate2 : {'cx', 'cy', 'cz', 'iswap', ..., str}, optional
        The gate to use for the entanling pairs.
    seed : int, optional
        Random seed for parameters.
    opts
        Supplied to :func:`~quimb.tensor.circuit_gen.gates_to_param_circuit`.

    Returns
    -------
    Circuit

    See Also
    --------
    circ_ansatz_1D_rand, circ_ansatz_1D_brickwork
    """
    ent_gates = []
    forward_layer = [(i, i + 1) for i in range(n - 1)]
    backward_layer = [(i + 1, i) for i in range(n - 2, -1, -1)]

    for d in range(depth):
        if d % 2 == 0:
            ent_gates.extend(forward_layer)
        else:
            ent_gates.extend(backward_layer)

    # inject U3 gates!
    gates = inject_u3s(ent_gates, gate2=gate2, seed=seed)
    circ = gates_to_param_circuit(gates, n, **circuit_opts)

    return circ


def circ_ansatz_1D_brickwork(
    n, depth,
    cyclic=False,
    gate2='cz',
    seed=None,
    **circuit_opts
):
    r"""A 1D circuit ansatz with odd and even layers of entangling
    gates interleaved with U3 single qubit unitaries::

        |  |  |  |  |
        |  u  u  u  u
        u  o++o  o++o
        |  u  u  u  |
        o++o  o++o  u
        |  u  u  u  |
        u  o++o  o++o
        |  u  u  u  |
        o++o  o++o  u
        |  u  u  u  u
        u  o++o  o++o
        |  u  u  u  |
        o++o  o++o  u
        u  u  u  u  |
        |  |  |  |  |

    Parameters
    ----------
    n : int
        The number of qubits.
    depth : int
        The number of entangling gates per pair.
    cyclic : bool, optional
        Whether to add entangling gates between qubits 0 and n - 1.
    gate2 : {'cx', 'cy', 'cz', 'iswap', ..., str}, optional
        The gate to use for the entanling pairs.
    seed : int, optional
        Random seed for parameters.
    opts
        Supplied to :func:`~quimb.tensor.circuit_gen.gates_to_param_circuit`.


    Returns
    -------
    Circuit

    See Also
    --------
    circ_ansatz_1D_zigzag, circ_ansatz_1D_rand
    """
    ent_gates = []
    for d in range(depth):

        # the even pairs layer
        ent_gates.extend((i, i + 1) for i in range(0, n - 1, 2))
        if cyclic and (n % 2 == 1):
            ent_gates.append((n - 1, 0))

        # the odd pairs layer
        ent_gates.extend((i, i + 1) for i in range(1, n - 1, 2))
        if cyclic and (n % 2 == 0):
            ent_gates.append((n - 1, 0))

    # inject U3 gates!
    gates = inject_u3s(ent_gates, gate2=gate2, seed=seed)
    circ = gates_to_param_circuit(gates, n, **circuit_opts)

    return circ


def circ_ansatz_1D_rand(
    n,
    depth,
    seed=None,
    cyclic=False,
    gate2='cz',
    avoid_doubling=True,
    **circuit_opts
):
    """A 1D circuit ansatz with randomly place entangling gates interleaved
    with U3 single qubit unitaries.

    Parameters
    ----------
    n : int
        The number of qubits.
    depth : int
        The number of entangling gates per pair.
    seed : int, optional
        Random seed.
    cyclic : bool, optional
        Whether to add entangling gates between qubits 0 and n - 1.
    gate2 : {'cx', 'cy', 'cz', 'iswap', ..., str}, optional
        The gate to use for the entanling pairs.
    avoid_doubling : bool, optional
        Whether to avoid placing an entangling gate directly above the same
        entangling gate (there will still be single qubit gates interleaved).
    opts
        Supplied to :func:`~quimb.tensor.circuit_gen.gates_to_param_circuit`.

    Returns
    -------
    Circuit

    See Also
    --------
    circ_ansatz_1D_zigzag, circ_ansatz_1D_brickwork
    """
    if seed is not None:
        random.seed(seed)

    # the set number of entangling pairs to distribute randomly
    ent_gates = [(i, i + 1) for i in range(n - 1) for _ in range(depth)]
    if cyclic:
        ent_gates.extend((n - 1, 0) for _ in range(depth))

    # randomly permute the order
    random.shuffle(ent_gates)

    # inject U3 gates!
    gates = inject_u3s(ent_gates, avoid_doubling=avoid_doubling,
                       gate2=gate2, seed=seed)
    circ = gates_to_param_circuit(gates, n, **circuit_opts)

    return circ


def circ_qaoa(
    terms,
    depth,
    gammas,
    betas,
    **circuit_opts,
):
    r"""Generate the QAOA circuit for weighted graph described by ``terms``.

    .. math::

        |{\bar{\gamma}, \bar{\beta}}\rangle = U_B (\beta _p)
        U_C (\gamma _p) \cdots U_B (\beta _1) U_C (\gamma _1) |{+}\rangle

    with

    .. math::

        U_C (\gamma) = e^{-i \gamma \mathcal{C}} = \prod \limits_{i, j
        \in E(G)} e^{-i \gamma w_{i j} Z_i Z_j}

    and

    .. math::

        U_B (\beta) = \prod \limits_{i \in G} e^{-i \beta X_i}


    Parameters
    ----------
    terms : dict[tuple[int], float]
        The mapping of integer pair keys ``(i, j)`` to the edge weight values,
        ``wij``. The integers should be a contiguous range enumerated from
        zero, with the total number of qubits being inferred from this.
    depth : int
        The number of layers of gates to apply, ``p`` above.
    gammas : iterable of float
        The interaction angles for each layer.
    betas : iterable of float
        The rotation angles for each layer.
    circuit_opts
        Supplied to :class:`~quimb.tensor.circuit.Circuit`. Note
        ``gate_opts={'contract': False}`` is set by default (it can be
        overridden) since the RZZ gate, even though it has a rank-2
        decomposition, is also diagonal.
    """
    circuit_opts.setdefault('gate_opts', {})
    circuit_opts['gate_opts'].setdefault('contract', False)

    n = max(itertools.chain.from_iterable(terms)) + 1

    gates = []

    # layer of hadamards to get into plus state
    for i in range(n):
        gates.append((0, 'h', i))

    for d in range(depth):
        for (i, j), wij in terms.items():
            gates.append((d, 'rzz', wij * gammas[d], i, j))

        for i in range(n):
            gates.append((d, 'rx', -betas[d] * 2, i))

    circ = Circuit(n, **circuit_opts)
    circ.apply_gates(gates)

    return circ
