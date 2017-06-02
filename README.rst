# QUIMB

[![Build Status](https://travis-ci.org/jcmgray/quimb.svg?branch=master)](https://travis-ci.org/jcmgray/quimb)
[![Test Coverage](https://codeclimate.com/github/jcmgray/quimb/badges/coverage.svg)](https://codeclimate.com/github/jcmgray/quimb/coverage)
[![Code Climate](https://codeclimate.com/github/jcmgray/quimb/badges/gpa.svg)](https://codeclimate.com/github/jcmgray/quimb)
[![Issue Count](https://codeclimate.com/github/jcmgray/quimb/badges/issue_count.svg)](https://codeclimate.com/github/jcmgray/quimb)

 * Python library for quantum information and many-body calcuations.
 * Uses `numpy` and `scipy.sparse` matrices as quantum objects.
 * Function orientated aside from a few very convenient monkey-patches.
 * Many operations are accelerated using `numba` and `numexpr`.
 * Optional `slepc4py` interface for advanced eigenvalue problems.


## Installation
`quimb` requires python 3.5+, `numpy`, `scipy`, `numba` and `numexpr`, which are obtained most easily using `conda` (http://conda.pydata.org/miniconda.html):
```bash
$ conda install numpy scipy numba numexpr
```
The optional advanced solvers require `slepc4py` (https://bitbucket.org/slepc/slepc4py). Instructions for its installation can be inferred from the `quimb/deps/travis-install-slepc4py.sh` script.

`quimb` can then be installed directly from github:
```bash
$ pip install git+https://github.com/jcmgray/quimb.git
```
or via a local editable repo:
```bash
$ git clone https://github.com/jcmgray/quimb.git
$ cd quimb
$ pip install -e .
```

## Example Usage
```python
>>> from quimb import *
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
>>> prj = qu(bell_state('psi-'), qtype='dop')  # `qu` (`quimbify`) ensures things are complex matrices
>>> prj                                        # qtype='dop' ensures its a density operator
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
1.0
>>> pauli_decomp(rho_ab)  # final two should now be singlet as well
II  0.250
ZZ -0.250
XX -0.250
YY -0.250
```
