######
States
######

Basic Representation
====================

States and operators in :py:mod:`quimb` are simply dense numpy or sparse scipy complex matrices.
The :py:func:`~quimb.core.quimbify` function (aliased to :py:func:`~quimb.core.qu`) can convert between the various representations.

.. code:: python

    >>> data = [1, 2j, -3]

Kets are column vectors, i.e. with shape ``(d, 1)``:

.. code:: python

    >>> qu(data, qtype='ket')
    matrix([[ 1.+0.j],
            [ 0.+2.j],
            [-3.+0.j]])

The ``normalized=True`` option can be used to ensure a normalized output.

Bras are row vectors, i.e. with shape ``(1, d)``:

.. code:: python

    >>> qu(data, qtype='bra')  # also conjugates the data
    matrix([[ 1.-0.j,  0.-2.j, -3.-0.j]])

And operators are square matrices, i.e. have shape ``(d, d)``:

.. code:: python

    >>> qu(data, qtype='dop')
    matrix([[ 1.+0.j,  0.-2.j, -3.-0.j],
            [ 0.+2.j,  4.+0.j,  0.-6.j],
            [-3.+0.j,  0.+6.j,  9.+0.j]])

Which can also be sparse:

.. code:: python

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

dagger = .H
conj
trans
@ and


Composing States
================

There are a number of ways to combine states and operators, i.e. tensoring them together.

Functional form using :py:func:`~quimb.accel.kron`:

.. code:: python

    >>> kron(psi1, psi2, psi3, ...)
    ...

This can also be done using the ``&`` overload on numpy and scipy matrices:

 .. code:: python

    >>> psi1 & psi2 & psi3
    ...

Often one wants to sandwich an operator with many identities, :py:func:`~quimb.core.eyepad` can be used for this:

.. code:: python

    >>> dims = [2] * 10  # overall space of 10 qubits
    >>> X = qu([[0, 1], [1, 0]])  # pauli-X
    >>> IIIXXIIIII = eyepad(X, dims, inds=[3, 4])  # act on 4th and 5th spin only
    >>> IIIXXIIIII.shape
    (1024, 1024)

For more advanced tensor constructions, such as reversing and interleaving identities within operators :py:func:`~quimb.core.perm_eyepad` can be used:

.. code:: python

    >>> dims = [2] * 3
    >>> XZ = pauli('X') & pauli('Z')
    >>> ZIX = perm_eyepad(op, dims, inds=[2, 0])  # now acts with Z on first spin, and X on 3rd
