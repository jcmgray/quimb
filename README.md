#QUIJY

 * A python library for many-body quantum calcuations, focused on flexibility and efficiency using ```numpy```, ```numba``` and ```numexpr```.
 * ```quijy``` uses ```numpy``` matrices as its basic object and is otherwise function based to reduce overhead.


## Example Usage
```python
>>> from quijy import *
```
First we construct a hamiltonian with next-nearest-neighbour interactions and find its groundstate.
```python
>>> n = 10
>>> ham = ham_j1j2(n, j2=0.24, sparse=True, cyclic=False)
>>> gs = groundstate(ham)
>>> gs.H @ ham @ gs  # energy expectation
matrix([[-15.7496449 -4.44089210e-16j]])
```
Now we construct an operator that projects all but the first and last spins into singlet states.
```python
>>> prj = qjf(bell_state('psi-'), qtype='dop')  # qjf (quijify) ensures things are complex matrices
>>> prj  # qtype='dop' ensures its a density operator
matrix([[ 0.0+0.j,  0.0+0.j,  0.0-0.j,  0.0+0.j],
        [ 0.0+0.j,  0.5+0.j, -0.5-0.j,  0.0+0.j],
        [ 0.0+0.j, -0.5+0.j,  0.5+0.j,  0.0+0.j],
        [ 0.0+0.j,  0.0+0.j,  0.0-0.j,  0.0+0.j]])
>>> # eyepad allows complex tensor constructions
>>> full_prj = eyepad(prj, dims=[2] * n, inds=range(1, n-1), sparse=True)
>>> # kron(eye(2), prj, ..., prj, eye(2)) # is equivalent
>>> # eye(2) & prj & ... & prj & eye(2)  # is too
```
Finally, measure the groundstate, trace out its middle, and calculate a few properties of the reduced density matrix.
```python
>>> gs_prj = (full_prj @ gs).nmlz()
>>> rho_ab = ptr(gs_prj, dims=[2] * n, keep=[0, n-1])
>>> tr(rho_ab)
1.0
>>> quantum_discord(rho_ab)
2.0
>>> pauli_decomp(rho_ab)  # final two should now be singlet as well
II  0.250
ZZ -0.250
XX -0.250
YY -0.250
```
