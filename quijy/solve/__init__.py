# Standard Numpy and Scipy (LAPACK/ARPACK backend)
from .basic_solve import *

def slepc4py_found():
    import importlib
    slepc4py_spec = importlib.util.find_spec("slepc4py")
    return slepc4py_spec is not None

# # slepc4py Interface (Many backends)
# if slepc4py_found():
#     from .advanced_solve import *
