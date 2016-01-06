"""
Contains functions for generating hamiltonians.
"""

from quijy.core import kron, eyepad, eye
from quijy.gen import sig


def ham_heis(n, jx=1, jy=1, jz=1, bz=0, periodic=False, sparse=False):
    """ Constructs the heisenberg spin 1/2 hamiltonian
    Input:
        n: number of spins
        jx, jy, jz: coupling constants, with convention that positive =
        antiferromagnetic
        bz: z-direction magnetic field
        periodic: whether to couple the first and last spins
        sparse: whether to return the hamiltonian in sparse form
    Returns:
        ham: hamiltonian as matrix
    """
    dims = [2] * n
    sds = (jx * kron(sig('x', sparse=True), sig('x', sparse=True)) +
           jy * kron(sig('y', sparse=True), sig('y', sparse=True)) +
           jz * kron(sig('z', sparse=True), sig('z', sparse=True)) -
           bz * kron(sig('z', sparse=True), eye(2)))
    # Begin with last spin, not covered by loop
    ham = eyepad(-bz * sig('z', sparse=True), dims, n - 1)
    for i in range(n - 1):
        ham = ham + eyepad(sds, dims[:-1], i)
    if periodic:
        ham = ham + eyepad(sig('x', sparse=True), dims, [0, n - 1])  \
                  + eyepad(sig('y', sparse=True), dims, [0, n - 1])  \
                  + eyepad(sig('z', sparse=True), dims, [0, n - 1])
    if not sparse:
        ham = ham.todense()  # always construct sparse though
    return ham
