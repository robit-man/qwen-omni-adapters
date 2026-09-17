"""The live comprehension window follows available unified memory safely."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from adapter_server import _active_context_tokens  # noqa: E402
from comprehension_launcher import (  # noqa: E402
    available_memory_gib,
    candidate_windows,
    choose_context_tokens,
)


def test_largest_context_that_leaves_the_reserve_is_selected() -> None:
    # Four llama.cpp slots make the aggregate estimates 16.7 GiB at 4K,
    # 17.1 at 8K, and 17.9 at 16K. With 19.1 GiB available and 2 GiB held
    # back, 8K is the largest safe choice.
    assert choose_context_tokens(19.1, reserve_gib=2.0) == 8192


def test_context_falls_back_instead_of_loading_past_available_memory() -> None:
    assert choose_context_tokens(18.75, reserve_gib=2.0) == 4096
    assert choose_context_tokens(18.6, reserve_gib=2.0) is None


def test_configured_non_power_of_two_ceiling_is_considered() -> None:
    assert candidate_windows(4096, 24_000)[-1] == 24_000
    assert choose_context_tokens(20.8, maximum=24_000, reserve_gib=2.0) == 24_000


def test_memavailable_is_read_in_gib(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemFree: 1 kB\nMemAvailable: 2097152 kB\n", encoding="utf-8")
    assert available_memory_gib(meminfo) == 2.0


def test_adapter_uses_the_window_published_by_the_launcher(tmp_path: Path) -> None:
    state = tmp_path / "context"
    state.write_text("16384\n", encoding="utf-8")
    config = SimpleNamespace(
        comprehension_context_tokens=65_536,
        comprehension_context_file=str(state),
    )
    assert _active_context_tokens(config) == 16_384


def test_published_window_cannot_exceed_configured_ceiling(tmp_path: Path) -> None:
    state = tmp_path / "context"
    state.write_text("65536\n", encoding="utf-8")
    config = SimpleNamespace(
        comprehension_context_tokens=32_768,
        comprehension_context_file=str(state),
    )
    assert _active_context_tokens(config) == 32_768
