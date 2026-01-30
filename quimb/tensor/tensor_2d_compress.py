import warnings

warnings.warn(
    "The module 'quimb.tensor.tensor_2d_compress' is deprecated and will be "
    "removed in a future release. Most functionality can be still be accessed "
    "directly from 'quimb.tensor' instead. The actual implementations have "
    "moved to `quimb.tensor.tn2d.compress`.",
    category=FutureWarning,
    stacklevel=2,
)

from ..tensor.tn2d.compress import *  # noqa: F401,F403
