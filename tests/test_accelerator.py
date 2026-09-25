from __future__ import annotations

import subprocess

import pytest

from qwen_omni_adapters import accelerator


@pytest.fixture(autouse=True)
def _clear_caches():
    for cached in (
        accelerator.is_tegra,
        accelerator.tegra_soc,
        accelerator.l4t_release,
        accelerator.compute_apps_supported,
    ):
        cached.cache_clear()
    yield
    for cached in (
        accelerator.is_tegra,
        accelerator.tegra_soc,
        accelerator.l4t_release,
        accelerator.compute_apps_supported,
    ):
        cached.cache_clear()


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=returncode, stdout=stdout, stderr=""
    )


def test_discrete_residency_still_requires_compute_app_accounting(monkeypatch) -> None:
    monkeypatch.setattr(accelerator, "is_tegra", lambda: False)
    monkeypatch.setattr(
        accelerator.subprocess,
        "run",
        lambda *args, **kwargs: _completed("4242, GPU-abc, 8192\n"),
    )

    assert accelerator.process_is_gpu_resident(4242, "GPU-abc") is True
    assert accelerator.process_is_gpu_resident(4242, "GPU-other") is False
    assert accelerator.process_is_gpu_resident(9999, "GPU-abc") is False


def test_discrete_residency_rejects_a_process_holding_no_device_memory(monkeypatch) -> None:
    monkeypatch.setattr(accelerator, "is_tegra", lambda: False)
    monkeypatch.setattr(
        accelerator.subprocess, "run", lambda *args, **kwargs: _completed("4242, GPU-abc, 0\n")
    )

    assert accelerator.process_is_gpu_resident(4242) is False


def test_tegra_residency_requires_both_a_gpu_node_and_the_nvmap_allocator(monkeypatch) -> None:
    monkeypatch.setattr(accelerator, "is_tegra", lambda: True)

    def handles(value: set[str]):
        return lambda _pid: value

    monkeypatch.setattr(
        accelerator,
        "_process_device_handles",
        handles({"/dev/nvgpu/igpu0/ctrl", "/dev/nvmap"}),
    )
    assert accelerator.process_is_gpu_resident(4242) is True

    # A CPU-only worker opens neither node.
    monkeypatch.setattr(accelerator, "_process_device_handles", handles({"/dev/null"}))
    assert accelerator.process_is_gpu_resident(4242) is False

    # nvmap alone is the multimedia allocator, not a CUDA context.
    monkeypatch.setattr(accelerator, "_process_device_handles", handles({"/dev/nvmap"}))
    assert accelerator.process_is_gpu_resident(4242) is False

    # A GPU handle without allocator-backed memory is not residency either.
    monkeypatch.setattr(
        accelerator, "_process_device_handles", handles({"/dev/nvgpu/igpu0/ctrl"})
    )
    assert accelerator.process_is_gpu_resident(4242) is False


def test_tegra_residency_accepts_the_jetpack_4_and_5_device_layout(monkeypatch) -> None:
    monkeypatch.setattr(accelerator, "is_tegra", lambda: True)
    monkeypatch.setattr(
        accelerator,
        "_process_device_handles",
        lambda _pid: {"/dev/nvhost-gpu", "/dev/nvhost-ctrl-gpu", "/dev/nvmap"},
    )

    assert accelerator.process_is_gpu_resident(4242) is True


def test_tegra_never_consults_compute_app_accounting(monkeypatch) -> None:
    monkeypatch.setattr(accelerator, "is_tegra", lambda: True)

    def refuse(*args, **kwargs):
        raise AssertionError("nvidia-smi has no compute-app accounting on Tegra")

    monkeypatch.setattr(accelerator.subprocess, "run", refuse)
    monkeypatch.setattr(accelerator, "_process_device_handles", lambda _pid: set())

    assert accelerator.process_is_gpu_resident(4242) is False
    accelerator.compute_apps_supported.cache_clear()
    assert accelerator.compute_apps_supported() is False


def test_cuda_architecture_is_pinned_per_tegra_soc(monkeypatch) -> None:
    monkeypatch.delenv("OMNI_CUDA_ARCHITECTURES", raising=False)
    for soc, expected in (
        ("tegra210", "53"),
        ("tegra186", "62"),
        ("tegra194", "72"),
        ("tegra234", "87"),
    ):
        monkeypatch.setattr(accelerator, "tegra_soc", lambda soc=soc: soc)
        assert accelerator.cuda_architectures() == expected


def test_discrete_hosts_keep_llama_cpp_architecture_detection(monkeypatch) -> None:
    monkeypatch.delenv("OMNI_CUDA_ARCHITECTURES", raising=False)
    monkeypatch.setattr(accelerator, "tegra_soc", lambda: None)
    monkeypatch.setattr(accelerator, "is_tegra", lambda: False)

    assert accelerator.cuda_architectures() is None


def test_explicit_architecture_override_wins(monkeypatch) -> None:
    monkeypatch.setenv("OMNI_CUDA_ARCHITECTURES", "90")
    monkeypatch.setattr(accelerator, "tegra_soc", lambda: "tegra234")

    assert accelerator.cuda_architectures() == "90"


def test_residency_backend_names_the_evidence(monkeypatch) -> None:
    monkeypatch.setattr(accelerator.platform, "system", lambda: "Linux")
    monkeypatch.setattr(accelerator, "is_tegra", lambda: True)
    assert accelerator.residency_backend() == "tegra-device-handles"

    monkeypatch.setattr(accelerator, "is_tegra", lambda: False)
    monkeypatch.setattr(accelerator, "compute_apps_supported", lambda: True)
    assert accelerator.residency_backend() == "nvidia-smi-compute-apps"

    monkeypatch.setattr(accelerator, "compute_apps_supported", lambda: False)
    assert accelerator.residency_backend() == "unavailable"

    monkeypatch.setattr(accelerator.platform, "system", lambda: "Darwin")
    assert accelerator.residency_backend() == "metal-build"


def test_tegra_gpu_facts_are_empty_off_tegra(monkeypatch) -> None:
    monkeypatch.setattr(accelerator, "is_tegra", lambda: False)

    assert accelerator.tegra_gpu_facts() == []
    assert accelerator.gpu_facts() is None


def test_accelerator_profile_describes_the_host(monkeypatch) -> None:
    monkeypatch.setattr(accelerator, "is_tegra", lambda: True)
    monkeypatch.setattr(accelerator, "tegra_soc", lambda: "tegra234")
    monkeypatch.setattr(accelerator, "l4t_release", lambda: "R36.3.0")
    monkeypatch.delenv("OMNI_CUDA_ARCHITECTURES", raising=False)

    profile = accelerator.accelerator_profile()

    assert profile["tegra"] is True
    assert profile["tegra_soc"] == "tegra234"
    assert profile["l4t_release"] == "R36.3.0"
    assert profile["cuda_architectures"] == "87"
    assert profile["gpu_memory_model"] == "unified"
    assert profile["residency_backend"] == "tegra-device-handles"


def test_tegra_runs_the_language_stage_on_the_logical_tag(monkeypatch) -> None:
    """The logical tag carries the language model; a second name = a second copy."""

    from qwen_omni_adapters import daemon

    monkeypatch.setattr(daemon, "is_tegra", lambda: True)
    monkeypatch.setenv("OMNI_MODEL", "robit/ornith-1.5-omni:q4km")
    monkeypatch.delenv("OMNI_LANGUAGE_MODEL", raising=False)
    monkeypatch.setattr(daemon, "_load_env_file", lambda _root: None)

    config = daemon.DaemonConfig.from_environment(cloudflare=False)

    assert config.language_model == "robit/ornith-1.5-omni:q4km"
    assert config.model == config.language_model


def test_discrete_hosts_keep_the_separate_core_language_tag(monkeypatch) -> None:
    from qwen_omni_adapters import daemon

    monkeypatch.setattr(daemon, "is_tegra", lambda: False)
    monkeypatch.setenv("OMNI_MODEL", "robit/ornith-1.5-omni:q4km")
    monkeypatch.delenv("OMNI_LANGUAGE_MODEL", raising=False)
    monkeypatch.setattr(daemon, "_load_env_file", lambda _root: None)

    config = daemon.DaemonConfig.from_environment(cloudflare=False)

    assert config.language_model == "robit/qwen3.8-27b-obliterated-e03:27b"


def test_an_explicit_language_model_always_wins(monkeypatch) -> None:
    from qwen_omni_adapters import daemon

    monkeypatch.setattr(daemon, "is_tegra", lambda: True)
    monkeypatch.setenv("OMNI_MODEL", "robit/ornith-1.5-omni:q4km")
    monkeypatch.setenv("OMNI_LANGUAGE_MODEL", "robit/ornith-1.5:9b")
    monkeypatch.setattr(daemon, "_load_env_file", lambda _root: None)

    config = daemon.DaemonConfig.from_environment(cloudflare=False)

    assert config.language_model == "robit/ornith-1.5:9b"


def test_tegra_bridge_uses_a_bounded_virtual_memory_working_set(monkeypatch) -> None:
    from qwen_omni_adapters import daemon

    monkeypatch.setattr(daemon, "_load_env_file", lambda _root: None)
    monkeypatch.delenv("OMNI_COMPREHENSION_CONTEXT_TOKENS", raising=False)
    monkeypatch.setenv(
        "OMNI_MODEL", "robit/ornith-1.5-omni-audio-bridge:q4km"
    )

    monkeypatch.setattr(daemon, "is_tegra", lambda: True)
    assert daemon.DaemonConfig.from_environment(cloudflare=False).context_tokens == 16384

    monkeypatch.setattr(daemon, "is_tegra", lambda: False)
    assert daemon.DaemonConfig.from_environment(cloudflare=False).context_tokens == 65536
