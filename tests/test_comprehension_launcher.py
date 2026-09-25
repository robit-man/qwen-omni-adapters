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
    _expansion_backed_off,
    _expansion_required_gib,
    _launcher_restart_command,
    _live_calibrated_base,
    _next_context_tier,
    _pressure_started_at,
    _probe_backed_off,
    _record_failed_context,
    _record_live_sample,
    _runtime_resize_enabled,
    _runtime_resize_ready,
    _safe_context_tokens,
    available_memory_gib,
    candidate_windows,
    choose_calibrated_context,
    choose_context_tokens,
    choose_context_tokens_with_recovery,
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


def test_recovery_window_uses_shared_governor_floor_without_deadlocking() -> None:
    selected, recovery = choose_context_tokens_with_recovery(
        22.53,
        minimum=4096,
        maximum=32_768,
        base_gib=18.52,
        kv_gib_per_token=0.375 / 4096,
        parallel_slots=1,
        startup_reserve_gib=4.0,
        recovery_reserve_gib=3.0,
    )

    assert selected == 8192
    assert recovery is True


def test_recovery_floor_is_not_used_when_normal_admission_fits() -> None:
    selected, recovery = choose_context_tokens_with_recovery(
        25.0,
        minimum=4096,
        maximum=32_768,
        base_gib=18.52,
        kv_gib_per_token=0.375 / 4096,
        parallel_slots=1,
        startup_reserve_gib=4.0,
        recovery_reserve_gib=3.0,
    )

    assert selected == 16_384
    assert recovery is False


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


def test_runtime_pressure_requires_one_continuous_low_memory_interval() -> None:
    started = _pressure_started_at(2.5, 4.0, None, now=10.0)
    assert started == 10.0
    assert _pressure_started_at(2.0, 4.0, started, now=14.0) == 10.0
    assert _pressure_started_at(4.1, 4.0, started, now=15.0) is None
    assert _pressure_started_at(3.9, 4.0, None, now=16.0) == 16.0


def test_runtime_resize_waits_for_idle_above_the_emergency_floor() -> None:
    assert not _runtime_resize_ready(3.5, hard_floor_gib=2.0, server_idle=False)
    assert _runtime_resize_ready(3.5, hard_floor_gib=2.0, server_idle=True)
    assert _runtime_resize_ready(1.9, hard_floor_gib=2.0, server_idle=False)


def test_runtime_process_resize_is_pinned_by_default_on_tegra() -> None:
    assert _runtime_resize_enabled(tegra=True, configured=None) is False
    assert _runtime_resize_enabled(tegra=False, configured=None) is True
    assert _runtime_resize_enabled(tegra=True, configured="1") is True
    assert _runtime_resize_enabled(tegra=False, configured="off") is False


def test_runtime_process_resize_rejects_an_invalid_override() -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        _runtime_resize_enabled(tegra=True, configured="sometimes")


def test_planned_resize_reexecs_the_same_launcher_contract() -> None:
    command = _launcher_restart_command(["--max-context", "65536", "--", "llama"])

    assert command[0] == sys.executable
    assert command[1].endswith("runtime/comprehension_launcher.py")
    assert command[2:] == ["--max-context", "65536", "--", "llama"]


def test_live_context_expansion_charges_growth_and_target_headroom() -> None:
    kv = 0.375 / 4096

    assert _next_context_tier(16_384, 4096, 65_536) == 32_768
    assert _next_context_tier(65_536, 4096, 65_536) is None
    assert _expansion_required_gib(
        16_384,
        32_768,
        minimum=4096,
        maximum=65_536,
        kv_gib_per_token=kv,
        parallel_slots=1,
        runtime_reserve_gib=3.0,
    ) == 4.5


def test_live_context_expansion_observes_failure_cooldown() -> None:
    calibration = {"last_failure": {"failed_at": 100.0}}

    assert _expansion_backed_off(calibration, now=999.0, cooldown_s=900.0)
    assert not _expansion_backed_off(calibration, now=1000.0, cooldown_s=900.0)
    assert not _expansion_backed_off({}, now=100.0, cooldown_s=900.0)


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
        required_headroom_gib=3.0,
    )

    assert calibration["base_gib"] == pytest.approx(16.2)
    assert calibration["last_sample"]["context_tokens"] == 8192
    assert calibration["safe_context_tokens"] == 8192
    assert state.is_file()


def test_live_samples_override_the_conservative_component_byte_estimate() -> None:
    calibration = {
        "base_gib": 18.52,
        "base_samples": [16.1, 16.3, 16.2, 22.0],
    }

    assert _live_calibrated_base(calibration) == pytest.approx(16.3)


def test_calibrated_selection_uses_the_largest_component_safe_unproven_tier() -> None:
    selected, recovery, probing = choose_calibrated_context(
        28.19,
        minimum=4096,
        maximum=65_536,
        live_base_gib=16.2,
        component_gib=18.52,
        kv_gib_per_token=0.375 / 4096,
        parallel_slots=1,
        startup_reserve_gib=3.0,
        recovery_reserve_gib=3.0,
        safe_context_tokens=16_384,
    )

    assert selected == 65_536
    assert recovery is False
    assert probing is True


def test_live_base_tracks_observed_residency_below_mapped_component_bytes(
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
        )

    assert calibration["base_gib"] == pytest.approx(14.2)
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
        )

    assert calibration["base_gib"] == pytest.approx(14.2)


def test_pressure_sample_does_not_become_a_proven_safe_tier(tmp_path: Path) -> None:
    state = tmp_path / "memory.json"
    calibration = {
        "base_samples": [16.2],
        "base_gib": 16.2,
        "kv_gib_per_token": 0.375 / 4096,
        "safe_context_tokens": 8192,
    }

    _record_live_sample(
        state,
        calibration,
        before_gib=28.0,
        after_gib=0.5,
        context_tokens=65_536,
        parallel_slots=1,
        required_headroom_gib=3.0,
    )

    assert calibration["safe_context_tokens"] == 8192


def test_healthy_legacy_sample_migrates_to_a_proven_tier() -> None:
    safe = _safe_context_tokens(
        {
            "last_sample": {
                "context_tokens": 16_384,
                "available_after_gib": 5.2,
            }
        },
        minimum=4096,
        maximum=65_536,
        runtime_reserve_gib=3.0,
        kv_gib_per_token=0.375 / 4096,
        parallel_slots=1,
    )

    assert safe == 16_384


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
