import warnings

warnings.warn(
    "The module 'quimb.tensor.tensor_1d_tebd' is deprecated and will be "
    "removed in a future release. Most functionality can be still be accessed "
    "directly from 'quimb.tensor' instead. The actual implementations have "
    "moved to `quimb.tensor.tn1d.tebd`.",
    category=FutureWarning,
    stacklevel=2,
)

from ..tensor.tn1d.tebd import *  # noqa: F401,F403
