import warnings

warnings.warn(
    "The module 'quimb.tensor.tensor_arbgeom_tebd' is deprecated and will be "
    "removed in a future release. Most functionality can be still be accessed "
    "directly from 'quimb.tensor' instead. The actual implementations have "
    "moved to `quimb.tensor.tnag.tebd`.",
    category=FutureWarning,
    stacklevel=2,
)

from ..tensor.tnag.tebd import *  # noqa: F401,F403
