def slepc4py_found():
    import importlib
    slepc4py_spec = importlib.util.find_spec("slepc4py")
    return slepc4py_spec is not None

# Standard Numpy and Scipy (LAPACK/ARPACK backend)
from .base_solver import *
from .scipy_solver import *

# slepc4py Interface (Many backends)
if slepc4py_found():
    from .slepc_solver import *
