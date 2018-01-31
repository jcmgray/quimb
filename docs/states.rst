######
States
######

Basic Representation
====================

States and operators in :py:mod:`quimb` are simply dense numpy or sparse scipy complex matrices.
The :py:func:`~quimb.core.quimbify` function (aliased to :py:func:`~quimb.core.qu`) can convert between the various representations.

.. code-block:: python

    >>> data = [1, 2j, -3]

Kets are column vectors, i.e. with shape ``(d, 1)``:

.. code-block:: python

    >>> qu(data, qtype='ket')
    matrix([[ 1.+0.j],
            [ 0.+2.j],
            [-3.+0.j]])

The ``normalized=True`` option can be used to ensure a normalized output.

Bras are row vectors, i.e. with shape ``(1, d)``:

.. code-block:: python

    >>> qu(data, qtype='bra')  # also conjugates the data
    matrix([[ 1.-0.j,  0.-2.j, -3.-0.j]])

And operators are square matrices, i.e. have shape ``(d, d)``:

.. code-block:: python

    >>> qu(data, qtype='dop')
    matrix([[ 1.+0.j,  0.-2.j, -3.-0.j],
            [ 0.+2.j,  4.+0.j,  0.-6.j],
            [-3.+0.j,  0.+6.j,  9.+0.j]])

Which can also be sparse:

.. code-block:: python

    >>> qu(data, qtype='dop', sparse=True)
    <3x3 sparse matrix of type '<class 'numpy.complex128'>'
            with 9 stored elements in Compressed Sparse Row format>

The sparse format can be specified with the ``stype`` keyword. The partial function versions of each of the above are also available:

* :py:func:`~quimb.core.ket()`
* :py:func:`~quimb.core.bra()`
* :py:func:`~quimb.core.dop()`
* :py:func:`~quimb.core.sparse()`


.. note::

    If a simple 1d-list is supplied and no ``qtype`` is given, ``'ket'`` is assumed.


Basic Operations
================

The 'dagger', or hermitian conjugate, operation is performed with the ``.H`` attribute:

.. code-block:: python

    >>> psi = 1.0j * bell_state('psi-')
    >>> psi
    matrix([[ 0.+0.j        ],
            [ 0.+0.70710678j],
            [ 0.-0.70710678j],
            [ 0.+0.j        ]])

    >>> psi.H
    matrix([[ 0.-0.j        ,  0.-0.70710678j,  0.+0.70710678j,  0.-0.j        ]])

This is just the combination of ``.conj()`` and ``.T``, but only available for :mod:`scipy.sparse` matrices  and :class:`numpy.matrix` s (not :class:`numpy.ndarray` s).

The product of two quantum objects is the dot or matrix product, which, since python 3.5, has been overloaded with the ``@`` symbol. Using it is recommended:

.. code:: python

    >>> psi = up()
    >>> psi
    matrix([[ 1.+0.j],
            [ 0.+0.j]])
    >>> psi.H @ psi  # inner product
    matrix([[ 1.+0.j]])
    >>> X = sig('X')
    >>> X @ psi  # act as gate
    matrix([[ 0.+0.j],
            [ 1.+0.j]])
    >>> psi.H @ X @ psi  # operator expectation
    matrix([[ 0.+0.j]])


Combining Objects - Tensoring
=============================

There are a number of ways to combine states and operators, i.e. tensoring them together.

Functional form using :py:func:`~quimb.accel.kron`:

.. code-block:: python

    >>> kron(psi1, psi2, psi3, ...)
    ...

This can also be done using the ``&`` overload on numpy and scipy matrices:

.. code-block:: python

    >>> psi1 & psi2 & psi3
    ...

.. warning::

    When :mod:`quimb` is imported, it overloads the ``&``/``__and__`` of :class:`numpy.matrix` which replaces the overload of :func:`numpy.bitwise_and`.

Often one wants to sandwich an operator with many identities, :py:func:`~quimb.core.eyepad` can be used for this:

.. code-block:: python

    >>> dims = [2] * 10  # overall space of 10 qubits
    >>> X = qu([[0, 1], [1, 0]])  # pauli-X
    >>> IIIXXIIIII = eyepad(X, dims, inds=[3, 4])  # act on 4th and 5th spin only
    >>> IIIXXIIIII.shape
    (1024, 1024)

For more advanced tensor constructions, such as reversing and interleaving identities within operators :py:func:`~quimb.core.perm_eyepad` can be used:

.. code-block:: python

    >>> dims = [2] * 3
    >>> XZ = pauli('X') & pauli('Z')
    >>> ZIX = perm_eyepad(op, dims, inds=[2, 0])  # now acts with Z on first spin, and X on 3rd


Removing Objects - Partial Trace
================================

To remove, or ignore, certain parts of a quantum state the partial trace function :func:`~quimb.core.partial_trace` (aliased to :func:`~quimb.core.ptr`) is used.
Here, the internal dimensions of a state must be supplied as well as the indicies of which of these subsystems to *keep*.

For example, if we have a random system of 10 qubits (hilbert space of dimension ``2**10``), and we want just the reduced density matrix describing the first and last spins:

.. code-block:: python

    >>> dims = [2] * 10
    >>> D = prod(dims)
    >>> psi = rand_ket(D)
    >>> rho_ab = ptr(psi, dims, [0, 9])
    >>> rho_ab.round(3)  # probably pretty close to identity
    matrix([[ 0.252+0.j   , -0.012+0.011j, -0.004-0.017j,  0.008+0.005j],
            [-0.012-0.011j,  0.254+0.j   , -0.017+0.006j,  0.014-0.006j],
            [-0.004+0.017j, -0.017-0.006j,  0.251+0.j   , -0.017-0.011j],
            [ 0.008-0.005j,  0.014+0.006j, -0.017+0.011j,  0.244+0.j   ]])

:func:`~quimb.core.partial_trace` accepts dense or sparse, operators or vectors.
