"""Always-listening local call harness for the omni adapter.

Microphone in, speakers out, state in the desktop's top bar. It drives the same
endpoint and the same request shape as the portal's browser call mode, so a
spoken turn is answered here exactly as it is there.
"""

from harness.call import CallConfig, CallSession, TurnResult, run_call_loop
from harness.vad import Vad, VadConfig

__all__ = [
    "CallConfig",
    "CallSession",
    "TurnResult",
    "Vad",
    "VadConfig",
    "run_call_loop",
]
