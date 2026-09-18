"""ReSpeaker 4 Mic Array support: LED ring and direction of arrival.

Optional by design. When the array is present the harness drives its ring to
show what it is doing and reads which way a voice came from; when it is not,
everything here answers "no" and the harness runs against whatever microphone
the desktop has selected. Nothing else in the harness needs to know which.

The USB tuning protocol is the vendor's own: a control transfer per parameter,
addressed by (command, offset). The values here are the XVF3000 firmware's.
"""

from __future__ import annotations

import logging
import struct
import threading
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

VENDOR_ID = 0x2886
PRODUCT_ID = 0x0018

# Ring animations the firmware provides. The harness maps its own states onto
# these so the room can see what it is doing without looking at a screen.
LED_COMMANDS: dict[str, int] = {
    "trace": 0,
    "listen": 2,
    "speak": 3,
    "think": 4,
    "spin": 5,
}

# (command, offset, type) triples from the XVF3000 tuning interface.
PARAMETERS: dict[str, tuple[int, int, str]] = {
    "doa_angle": (21, 0, "int"),
    "voice_activity": (19, 32, "int"),
    "speech_detected": (19, 22, "int"),
    "aec_far_end_silence": (18, 31, "int"),
}

_usb_lock = threading.RLock()


def _device(vendor_id: int = VENDOR_ID, product_id: int = PRODUCT_ID):
    try:
        import usb.core
    except ImportError:
        return None
    try:
        return usb.core.find(idVendor=vendor_id, idProduct=product_id)
    except Exception as error:  # noqa: BLE001 - absence is the common case
        logger.debug("ReSpeaker lookup failed: %s", error)
        return None


def available() -> bool:
    """Whether a ReSpeaker array is attached and reachable."""

    return _device() is not None


def _read_parameter(device, parameter: tuple[int, int, str]) -> float:
    import usb.util

    parameter_id, offset, kind = parameter
    command = 0x80 | offset | (0x40 if kind == "int" else 0)
    response = device.ctrl_transfer(
        usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
        0,
        command,
        parameter_id,
        8,
        100000,
    )
    mantissa, exponent = struct.unpack("ii", bytes(response))
    return float(mantissa if kind == "int" else mantissa * (2.0**exponent))


@dataclass
class ReSpeaker:
    """The array's ring and direction, or a no-op when it is not there."""

    brightness: int = 8
    vendor_id: int = VENDOR_ID
    product_id: int = PRODUCT_ID
    _angles: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _present: bool = False
    _last_state: str = ""
    _speech_detected: bool | None = None
    _voice_activity: bool | None = None
    _aec_far_end_silence: bool | None = None

    def __post_init__(self) -> None:
        self._present = _device(self.vendor_id, self.product_id) is not None
        if self._present:
            logger.info("ReSpeaker array detected: ring and direction are live")
        else:
            logger.info("no ReSpeaker array; using the default microphone")

    @property
    def present(self) -> bool:
        return self._present

    @property
    def direction(self) -> float | None:
        """Most recent direction of arrival in degrees, if known."""

        return self._angles[-1] if self._angles else None

    @property
    def speech_detected(self) -> bool | None:
        """Native post-AEC speech decision, or unknown before the first poll."""

        return self._speech_detected

    @property
    def voice_activity(self) -> bool | None:
        return self._voice_activity

    @property
    def aec_far_end_silence(self) -> bool | None:
        return self._aec_far_end_silence

    def set_state(self, state: str) -> bool:
        """Show a harness state on the ring. Never raises; the ring is cosmetic."""

        if not self._present or state == self._last_state:
            return False
        command = LED_COMMANDS.get(state)
        if command is None:
            return False
        try:
            import usb.util

            device = _device(self.vendor_id, self.product_id)
            if device is None:
                self._present = False
                return False
            with _usb_lock:
                device.ctrl_transfer(
                    usb.util.CTRL_OUT
                    | usb.util.CTRL_TYPE_VENDOR
                    | usb.util.CTRL_RECIPIENT_DEVICE,
                    0,
                    0x20,
                    0x1C,
                    [self.brightness],
                    8000,
                )
                device.ctrl_transfer(
                    usb.util.CTRL_OUT
                    | usb.util.CTRL_TYPE_VENDOR
                    | usb.util.CTRL_RECIPIENT_DEVICE,
                    0,
                    command,
                    0x1C,
                    [0],
                    8000,
                )
        except Exception as error:  # noqa: BLE001 - a dark ring is not a failure
            logger.debug("ReSpeaker ring write failed: %s", error)
            return False
        self._last_state = state
        return True

    def start(self) -> None:
        """Begin following the direction of arrival in the background."""

        if not self._present or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._follow, name="respeaker-doa", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2)
        # Leave the ring in its resting animation rather than mid-think.
        self.set_state("trace")

    def _follow(self) -> None:
        while not self._stop.is_set():
            try:
                device = _device(self.vendor_id, self.product_id)
                if device is None:
                    self._present = False
                    return
                with _usb_lock:
                    angle = _read_parameter(device, PARAMETERS["doa_angle"]) % 360
                    speech = _read_parameter(device, PARAMETERS["speech_detected"])
                    activity = _read_parameter(device, PARAMETERS["voice_activity"])
                    far_end_silence = _read_parameter(
                        device, PARAMETERS["aec_far_end_silence"]
                    )
                self._angles.append(float(angle))
                self._speech_detected = bool(speech)
                self._voice_activity = bool(activity)
                self._aec_far_end_silence = bool(far_end_silence)
            except Exception as error:  # noqa: BLE001
                logger.debug("ReSpeaker DSP status read failed: %s", error)
            self._stop.wait(0.1)


# The harness's states, named as the ring's animations. Kept separate from the
# top-bar labels: one is an animation vocabulary, the other is English.
STATE_TO_RING: dict[str, str] = {
    "starting": "trace",
    "listening": "listen",
    "hearing": "listen",
    "thinking": "think",
    "speaking": "speak",
    "muted": "trace",
    "offline": "trace",
}


def describe_direction(angle: float | None) -> str:
    """A spoken-language bearing, for evidence attached to a turn."""

    if angle is None:
        return ""
    points = [
        (0, "ahead"),
        (45, "ahead and to the right"),
        (90, "to the right"),
        (135, "behind and to the right"),
        (180, "behind"),
        (225, "behind and to the left"),
        (270, "to the left"),
        (315, "ahead and to the left"),
    ]
    nearest = min(points, key=lambda point: min(
        abs(angle - point[0]), 360 - abs(angle - point[0])
    ))
    return f"{nearest[1]} ({angle:.0f}°)"


# -- choosing the microphone ----------------------------------------------

# How the array's processed capture source names itself to PulseAudio. Channel
# 0 of that source is the AEC/beamformed/denoised stream, which is the one
# worth listening to; the rest are the raw capsules.
SOURCE_HINT = "ReSpeaker"
PROCESSED_CHANNELS = 6
PROCESSED_CHANNEL = 0


def find_source() -> tuple[str | None, int, int]:
    """Pick the capture source, and how many channels to ask it for.

    Returns ``(device, channels, channel)``. With the array attached this is
    its multichannel input; without it, the desktop's own default and a single
    channel. Asking a two-channel laptop microphone for six channels is how a
    harness that hard-codes the array fails on every other machine.
    """

    import shutil
    import subprocess

    if shutil.which("pactl") is None:
        return None, 1, 0
    try:
        listing = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, 1, 0
    if listing.returncode != 0:
        return None, 1, 0

    for line in listing.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        name = fields[1]
        if SOURCE_HINT.lower() not in name.lower() or ".monitor" in name:
            continue
        # "s16le 6ch 16000Hz" -> 6
        channels = PROCESSED_CHANNELS
        for field_value in fields:
            for token in field_value.split():
                if token.endswith("ch") and token[:-2].isdigit():
                    channels = int(token[:-2])
        logger.info("using the ReSpeaker capture source (%d channels)", channels)
        return name, channels, PROCESSED_CHANNEL if channels > 1 else 0

    return None, 1, 0
