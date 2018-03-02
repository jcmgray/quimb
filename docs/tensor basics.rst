#############
Tensor Basics
#############

Basic Manipulations
-------------------

Creating a :class:`~quimb.tensor.Tensor`:

.. code-block:: python

    >>> ket = Tensor(bell_state('psi-').reshape(2, 2), inds=('k0', 'k1'), tags={'ket'})
    >>> X = Tensor(pauli('X'), inds=('k0', 'b0'), tags={'pauli', 'X', '0'})
    >>> Y = Tensor(pauli('Y'), inds=('k1', 'b1'), tags={'pauli', 'Y', '1'})
    >>> bra = Tensor(rand_ket(4).reshape(2, 2), inds=('b0', 'b1'), tags={'bra'})

Can now combine these into a network (``.H`` conjugates the data).

.. code-block:: python

    >>> TN = ket.H X & Y & bra

Plot your creation:

.. code-block:: python

    >>> TN.graph(color=['ket', 'X', 'Y', 'bra'])

Contract everything (with optimized contraction order):

.. code-block:: python

    >>> TN ^ ...
    (0.7212531527120138-0.03982265659016575j)

Or just the paulis:

.. code-block:: python

    >>> TN ^ 'pauli'
    TensorNetwork([
        Tensor(shape=(2, 2), inds=('k0', 'k1'), tags={'ket'}),
        Tensor(shape=(2, 2), inds=('b0', 'b1'), tags={'bra'}),
        Tensor(shape=(2, 2, 2, 2), inds=('k0', 'b0', 'k1', 'b1'), tags={'Y', 'X', '0', 'pauli', '1'}),
    ])

Get the ket, split it in half and replace the original:

.. code-block:: python

    >>> Tk_s = TN['ket'].split(left_inds=['k0'])
    >>> Tk_s  # note new index created
    TensorNetwork([
        Tensor(shape=(2, 2), inds=('k0', '_89dcdf0000016'), tags={'ket'}),
        Tensor(shape=(2, 2), inds=('_89dcdf0000016', 'k1'), tags={'ket'}),
    ])

    >>> del TN['ket']
    >>> TN &= Tk_s
    >>> TN ^ ...
    (0.7212531527120138-0.03982265659016575j)


Other overloads
---------------

You can also add tensors/networks together using ``|`` or the inplace ``|=``, which act like ``&`` and ``&=`` respectively, but are virtual, meaning that changes to the tensors propogate across all networks viewing it (see :class:`~quimb.tensor.TensorNetwork`).

The ``@`` symbol is overloaded to combine the objects into a network and then contract them all, and so mimics dense dot product. E.g.

.. code-block:: python

    >>> ket.H @ ket
    >>> 1.0

In this case, the conjugated copy ``ket.H`` has the same outer indices as ``ket`` and so the inner product is naturally formed.
