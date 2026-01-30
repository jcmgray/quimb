import warnings

warnings.warn(
    "The module 'quimb.tensor.tensor_mera' is deprecated and will be "
    "removed in a future release. Most functionality can be still be accessed "
    "directly from 'quimb.tensor' instead. The actual implementations have "
    "moved to `quimb.tensor.tn1d.mera`.",
    category=FutureWarning,
    stacklevel=2,
)

from ..tensor.tn1d.mera import *  # noqa: F401,F403
