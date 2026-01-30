import warnings

warnings.warn(
    "The module 'quimb.tensor.tensor_arbgeom' is deprecated and will be "
    "removed in a future release. Most functionality can be still be accessed "
    "directly from 'quimb.tensor' instead. The actual implementations have "
    "moved to `quimb.tensor.tnag.core`.",
    category=FutureWarning,
    stacklevel=2,
)

from ..tensor.tnag.core import *  # noqa: F401,F403
