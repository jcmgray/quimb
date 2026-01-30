import warnings

warnings.warn(
    "The module 'quimb.tensor.tensor_arbgeom_compress' is deprecated and will be "
    "removed in a future release. Most functionality can be still be accessed "
    "directly from 'quimb.tensor' instead. The actual implementations have "
    "moved to `quimb.tensor.tnag.compress`.",
    category=FutureWarning,
    stacklevel=2,
)

from ..tensor.tnag.compress import *  # noqa: F401,F403
