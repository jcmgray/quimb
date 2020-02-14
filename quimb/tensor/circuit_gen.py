import math
import itertools

from .. import rand
from . import Circuit


def inject_u3s(
    ent_gates,
    gate2='cz',
    avoid_doubling=False,
):
    r"""Take a sequence of pairs denoting qubits to entangle and interleave one
    single qubit gate inbetween every leg. For example:

        ent_gates = [(0, 1), (2, 3), (1, 2)]

    Would go get made into a circuit like:

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
    gates : sequence[tuple[str, *float, *int]]
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
    gates = inject_u3s(ent_gates, gate2=gate2)
    circ = gates_to_param_circuit(gates, n, **circuit_opts)

    return circ


def circ_ansatz_1D_brickwork(
    n, depth,
    cyclic=False,
    gate2='cz',
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
    gates = inject_u3s(ent_gates, gate2=gate2)
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
    import random

    if seed is not None:
        random.seed(seed)

    # the set number of entangling pairs to distribute randomly
    ent_gates = [(i, i + 1) for i in range(n - 1) for _ in range(depth)]
    if cyclic:
        ent_gates.extend((n - 1, 0) for _ in range(depth))

    # randomly permute the order
    random.shuffle(ent_gates)

    # inject U3 gates!
    gates = inject_u3s(ent_gates, avoid_doubling=avoid_doubling, gate2=gate2)
    circ = gates_to_param_circuit(gates, n, **circuit_opts)

    return circ
