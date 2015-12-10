from quijy import (kron, sig, eyepad, eye)


def ham_heis(n, jx=1, jy=1, jz=1, bz=0, periodic=False, sparse=False):
    """
    Constructs the heisenberg spin 1/2 hamiltonian, defaulting to isotropic
    coupling with no magnetic field and open boundary conditions.
    """
    dims = [2] * n

    sds = (jx * kron(sig('x', True), sig('x', True)) +
           jy * kron(sig('y', True), sig('y', True)) +
           jz * kron(sig('z', True), sig('z', True)) -
           bz * kron(sig('z', True), eye(2)))
    # Begin with last spin, not covered by loop
    ham = eyepad(-bz * sig('z', True), dims, n - 1)
    for i in range(n - 1):
        ham = ham + eyepad(sds, dims[:-1], i)
    if periodic:
        ham = ham + eyepad(sig('x', True), dims[:], [0, n - 1])  \
                  + eyepad(sig('y', True), dims[:], [0, n - 1])  \
                  + eyepad(sig('z', True), dims[:], [0, n - 1])
    if not sparse:
        ham = ham.todense()  # always construct sparse though
    return ham
