"""The live comprehension window follows available unified memory safely."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from adapter_server import _active_context_tokens  # noqa: E402
from comprehension_launcher import (  # noqa: E402
    _component_window_fits,
    _effective_context_maximum,
    _live_calibrated_base,
    _probe_backed_off,
    _record_failed_context,
    _record_live_sample,
    available_memory_gib,
    candidate_windows,
    choose_context_tokens,
    context_headroom_gib,
    estimated_resident_gib,
)


def test_largest_context_that_fits_live_capacity_is_selected() -> None:
    # These values are supplied by live calibration and GGUF metadata; the
    # selector itself has no machine-specific footprint constants.
    assert choose_context_tokens(
        25.2,
        base_gib=16.2,
        kv_gib_per_token=0.375 / 4096,
    ) == 65_536


def test_context_falls_back_instead_of_loading_past_available_memory() -> None:
    arguments = {
        "base_gib": 16.2,
        "kv_gib_per_token": 0.375 / 4096,
    }
    assert choose_context_tokens(20.2, **arguments) == 16_384
    assert choose_context_tokens(19.8, **arguments) == 16_384
    assert choose_context_tokens(16.95, **arguments) == 4096
    assert choose_context_tokens(16.5, **arguments) is None


def test_context_selection_retains_the_generic_runtime_reserve() -> None:
    arguments = {
        "base_gib": 16.2,
        "kv_gib_per_token": 0.375 / 4096,
        "runtime_reserve_gib": 4.0,
    }

    assert choose_context_tokens(22.0, **arguments) == 16_384
    assert choose_context_tokens(21.5, **arguments) == 8192
    assert choose_context_tokens(20.4, **arguments) is None


def test_context_reserve_is_derived_from_the_adjacent_kv_tier() -> None:
    windows = candidate_windows(4096, 65_536)
    kv = 0.375 / 4096

    assert context_headroom_gib(
        16_384, windows=windows, kv_gib_per_token=kv
    ) == 1.5
    assert context_headroom_gib(
        32_768, windows=windows, kv_gib_per_token=kv
    ) == 3.0
    # The configured ceiling still retains the previous adjacent increment.
    assert context_headroom_gib(
        65_536, windows=windows, kv_gib_per_token=kv
    ) == 3.0


def test_configured_non_power_of_two_ceiling_is_considered() -> None:
    assert candidate_windows(4096, 24_000)[-1] == 24_000
    assert choose_context_tokens(
        22.0,
        maximum=24_000,
        base_gib=16.2,
        kv_gib_per_token=0.375 / 4096,
    ) == 24_000


def test_each_parallel_slot_is_charged_for_its_own_kv_cache() -> None:
    one = estimated_resident_gib(
        8192, base_gib=16.2, kv_gib_per_token=0.375 / 4096, parallel_slots=1
    )
    four = estimated_resident_gib(
        8192, base_gib=16.2, kv_gib_per_token=0.375 / 4096, parallel_slots=4
    )

    assert one == 16.95
    assert four == 19.2


def test_successful_load_calibrates_base_from_live_memory(
    tmp_path: Path,
) -> None:
    state = tmp_path / "memory.json"
    calibration = {
        "components": [],
        "kv_gib_per_token": 0.375 / 4096,
    }

    _record_live_sample(
        state,
        calibration,
        before_gib=20.2,
        after_gib=3.25,
        context_tokens=8192,
        parallel_slots=1,
        floor_gib=0.0,
    )

    assert calibration["base_gib"] == pytest.approx(16.2)
    assert calibration["last_sample"]["context_tokens"] == 8192
    assert state.is_file()


def test_live_samples_override_a_conservative_component_byte_floor() -> None:
    calibration = {
        "base_gib": 18.52,
        "base_samples": [16.1, 16.3, 16.2, 22.0],
    }

    assert _live_calibrated_base(calibration) == pytest.approx(16.3)


def test_base_can_never_sit_below_installed_component_bytes(
    tmp_path: Path,
) -> None:
    state = tmp_path / "memory.json"
    calibration = {
        "base_samples": [16.2],
        "base_gib": 16.2,
        "kv_gib_per_token": 0.375 / 4096,
    }

    for _ in range(3):
        _record_live_sample(
            state,
            calibration,
            before_gib=17.0,
            after_gib=2.425,
            context_tokens=4096,
            parallel_slots=1,
            floor_gib=15.0,
        )

    assert calibration["base_gib"] == pytest.approx(15.0)
    assert calibration["base_samples"] == [16.2, 14.2, 14.2, 14.2]


def test_rebaselining_tracks_healthier_loads_instead_of_staying_anchored_high(
    tmp_path: Path,
) -> None:
    state = tmp_path / "memory.json"
    calibration = {
        "base_samples": [16.2],
        "base_gib": 16.2,
        "kv_gib_per_token": 0.375 / 4096,
    }

    for _ in range(3):
        _record_live_sample(
            state,
            calibration,
            before_gib=17.0,
            after_gib=2.425,
            context_tokens=4096,
            parallel_slots=1,
            floor_gib=10.0,
        )

    assert calibration["base_gib"] == pytest.approx(14.2)


def test_minimum_window_component_probe_fits_with_headroom() -> None:
    kv = 0.375 / 4096
    assert _component_window_fits(
        component_gib=16.2,
        context_tokens=4096,
        minimum=4096,
        maximum=65_536,
        kv_gib_per_token=kv,
        parallel_slots=1,
        available_gib=17.0,
    )
    assert not _component_window_fits(
        component_gib=16.2,
        context_tokens=4096,
        minimum=4096,
        maximum=65_536,
        kv_gib_per_token=kv,
        parallel_slots=1,
        available_gib=16.5,
    )


def test_probe_recovery_is_backed_off_after_a_failed_minimum_window() -> None:
    assert _probe_backed_off(
        {"probe_backoff_until": time.time() + 100}
    )
    assert not _probe_backed_off(
        {"probe_backoff_until": time.time() - 1}
    )
    assert not _probe_backed_off({})


def test_abnormal_exit_caps_the_next_load_at_the_next_standard_window(
    tmp_path: Path,
) -> None:
    state = tmp_path / "memory.json"
    calibration = {"kv_gib_per_token": 0.375 / 4096}

    _record_failed_context(
        state,
        calibration,
        context_tokens=32_768,
        minimum=4096,
        maximum=65_536,
        available_gib=20.0,
    )

    assert calibration["context_cap"] == 16_384
    assert calibration["last_failure"]["context_tokens"] == 32_768


def test_crash_cap_lifts_only_when_live_memory_can_fund_the_failed_tier() -> None:
    calibration = {
        "context_cap": 16_384,
        "last_failure": {
            "context_tokens": 32_768,
            "available_before_gib": 20.0,
        },
    }
    kv = 0.375 / 4096

    assert _effective_context_maximum(
        calibration,
        configured_maximum=65_536,
        available_gib=21.49,
        kv_gib_per_token=kv,
        parallel_slots=1,
    ) == 16_384
    assert _effective_context_maximum(
        calibration,
        configured_maximum=65_536,
        available_gib=21.5,
        kv_gib_per_token=kv,
        parallel_slots=1,
    ) == 65_536
    assert "context_cap" not in calibration
    assert "last_failure" not in calibration


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
