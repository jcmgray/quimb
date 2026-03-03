import importlib

import pytest

found_torch = importlib.util.find_spec("torch") is not None
found_autograd = importlib.util.find_spec("autograd") is not None
found_jax = importlib.util.find_spec("jax") is not None
found_tensorflow = importlib.util.find_spec("tensorflow") is not None

if found_tensorflow:
    import tensorflow.experimental.numpy as tnp

    tnp.experimental_enable_numpy_behavior()

jax_case = pytest.param(
    "jax", marks=pytest.mark.skipif(not found_jax, reason="jax not installed")
)
autograd_case = pytest.param(
    "autograd",
    marks=pytest.mark.skipif(
        not found_autograd, reason="autograd not installed"
    ),
)
tensorflow_case = pytest.param(
    "tensorflow",
    marks=pytest.mark.skipif(
        not found_tensorflow, reason="tensorflow not installed"
    ),
)
pytorch_case = pytest.param(
    "torch",
    marks=pytest.mark.skipif(not found_torch, reason="pytorch not installed"),
)
