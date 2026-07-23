"""TacThru UMI real-robot deployment bridge for LingBot-VLA v2.

The package is intentionally split at the network boundary:

* :mod:`http_server` runs with the LingBot GPU environment.
* :mod:`realman_client` runs with TacThru's Python 3.11 Realman environment.

Neither module imports the other side's hardware/model dependencies at import
time, which keeps protocol and safety tests lightweight.
"""

from .protocol import (
    CAMERA_KEY,
    POSE_FRAME,
    POSE_SEMANTICS,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
)
from .transforms import ACTION_DIM, STATE_DIM

__all__ = [
    "ACTION_DIM",
    "CAMERA_KEY",
    "POSE_FRAME",
    "POSE_SEMANTICS",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "STATE_DIM",
]
