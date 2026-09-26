from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ManagedModel:
    key: str
    label: str
    tag: str
    size_gib: float
    generation: str
    max_context_tokens: int
    language_disable_thinking: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# This is intentionally an exact allowlist. Indicator actions must never accept
# an arbitrary tag, path, or shell fragment. The rejected Ornith obliterated
# release is intentionally absent.
MANAGED_MODELS: tuple[ManagedModel, ...] = (
    ManagedModel(
        key="ornith15",
        label="Ornith 1.5 compact audio bridge",
        tag="robit/ornith-1.5-omni-audio-bridge:q4km",
        size_gib=8.15,
        generation="compact",
        # The virtual-context layer treats this resident window as L0.  The
        # GGUF declares a 262K native range.  The 32 GB Orin profile uses q8 KV
        # and independently sheds TTS/pointing during action work.  Admission
        # is still based on the model-derived KV slope and measured live
        # residency, so 64K is a ceiling rather than an unconditional load.
        max_context_tokens=65_536,
        language_disable_thinking=True,
    ),
    ManagedModel(
        key="qwen38",
        label="Qwen3.8 E03 compact audio bridge",
        tag="robit/qwen3.8-27b-e03-obliterated-omni-audio-bridge:q4km",
        size_gib=18.33,
        generation="compact",
        max_context_tokens=16_384,
        language_disable_thinking=False,
    ),
    ManagedModel(
        key="ornith15_full",
        label="Ornith 1.5 full Omni",
        tag="robit/ornith-1.5-omni:q4km",
        size_gib=32.1,
        generation="legacy-full",
        max_context_tokens=65_536,
        language_disable_thinking=True,
    ),
    ManagedModel(
        key="qwen38_full",
        label="Qwen3.8 E03 full Omni",
        tag="robit/qwen3.8-27b-e03-obliterated-omni:q4km",
        size_gib=52.5,
        generation="legacy-full",
        max_context_tokens=65_536,
        language_disable_thinking=False,
    ),
)

MODEL_BY_TAG = {model.tag: model for model in MANAGED_MODELS}
