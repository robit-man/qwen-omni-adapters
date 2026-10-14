"""The dedicated point worker accepts only bounded local image evidence."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

import pointing_server  # noqa: E402
from pointing_server import (  # noqa: E402
    PointingModel,
    _decode_observation_request,
    _decode_request,
)


def _request(target: str = "blue triangle") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (320, 180), "blue").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return ('{"image":"' + encoded + '","target":"' + target + '"}').encode()


def test_pointing_request_decodes_bounded_pixels_and_referring_expression() -> None:
    image, target = _decode_request(_request("  blue   triangle "))

    assert image.size == (320, 180)
    assert image.mode == "RGB"
    assert target == "blue triangle"


def test_observation_request_accepts_bounded_pixels_without_a_prompt() -> None:
    image = _decode_observation_request(_request())

    assert image.size == (320, 180)
    assert image.mode == "RGB"


@pytest.mark.parametrize(
    "payload",
    [b"{}", b'{"image":"not base64","target":"shape"}', _request(241 * "x")],
)
def test_pointing_request_rejects_missing_or_unbounded_input(payload: bytes) -> None:
    with pytest.raises(ValueError):
        _decode_request(payload)


def test_pointing_control_plane_starts_without_loading_weights(monkeypatch) -> None:
    loads: list[str] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeFactory:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            loads.append("load")
            return SimpleNamespace(eval=lambda: None)

    monkeypatch.setattr(
        pointing_server, "_install_torch_24_gqa_compatibility", lambda: None
    )
    monkeypatch.delenv("OLLAMA_UNIFY_GPU_LEASE", raising=False)
    monkeypatch.setattr(pointing_server, "_broker_present", lambda: False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda()))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=FakeFactory),
    )

    model = PointingModel("test/point-head", "revision-1")

    assert model.resident is False
    assert model._torch is None
    assert loads == []


@pytest.fixture
def cuda_runtime(monkeypatch):
    events: list[str] = []
    faults: dict[str, str] = {}

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            events.append("cuda_available")
            return faults.get("cuda") != "unavailable"

        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def synchronize() -> None:
            events.append("synchronize")

        @staticmethod
        def empty_cache() -> None:
            events.append("empty_cache")

    class FakeTorch:
        bfloat16 = "bf16"
        cuda = FakeCuda()

        @staticmethod
        def inference_mode():
            return nullcontext()

    class FakeLoadedModel:
        def eval(self):
            return self

        def point(self, _image: Image.Image, _target: str):
            events.append("point")
            if faults.get("inference"):
                raise RuntimeError("point inference failed")
            return {"points": [{"x": 0.25, "y": 0.75}]}

        def query(self, _image: Image.Image, _prompt: str):
            events.append("observe")
            return {"answer": "Visible completion marker: done-42"}

    class FakeFactory:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            events.append(f"load:{args[0]}:{kwargs['revision']}")
            assert kwargs["device_map"] == {"": "cuda"}
            if faults.get("load"):
                raise RuntimeError("model load failed")
            return FakeLoadedModel()

    def transition(action, token):
        assert token == "private-lease-token"
        events.append(action)
        if faults.get("transition") == action:
            faults.pop("transition")
            raise RuntimeError(f"{action} failed")

    def resident(pid, uuid):
        assert pid == os.getpid()
        assert uuid in {"GPU-test", ""}
        events.append("resident")
        return not faults.get("residency")

    monkeypatch.setattr(pointing_server, "_install_torch_24_gqa_compatibility", lambda: None)
    monkeypatch.setattr(pointing_server, "_broker_present", lambda: False)
    monkeypatch.setattr(pointing_server, "_broker_transition", transition)
    monkeypatch.setattr(pointing_server, "process_is_gpu_resident", resident)
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForCausalLM=FakeFactory))
    monkeypatch.setenv("OLLAMA_UNIFY_GPU_LEASE", "private-lease-token")
    monkeypatch.setenv("OMNI_POINTING_GPU_UUID", "GPU-test")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    return events, faults


@pytest.mark.parametrize("operation", ["point", "observe"])
def test_pointing_weights_load_for_one_grounded_action_then_shed(cuda_runtime, operation) -> None:
    events, _faults = cuda_runtime
    model = PointingModel("test/point-head", "revision-1")
    assert events == []  # construction does not import Torch or initialize CUDA

    image = Image.new("RGB", (16, 16))
    if operation == "point":
        assert model.point(image, "target") == [{"x": 0.25, "y": 0.75}]
    else:
        assert model.observe(image) == "Visible completion marker: done-42"

    assert model.resident is False
    assert events == [
        "prepare", "cuda_available", "load:test/point-head:revision-1", operation,
        "synchronize", "resident", "ready", "synchronize", "empty_cache",
    ]


@pytest.mark.parametrize("fault", ["load", "inference"])
def test_failed_pointing_frees_cuda_before_restoring_shared_lease(cuda_runtime, fault) -> None:
    events, faults = cuda_runtime
    faults[fault] = "fail"
    model = PointingModel("test/point-head", "revision-1")

    with pytest.raises(RuntimeError):
        model.point(Image.new("RGB", (16, 16)), "target")

    assert events[0] == "prepare"
    assert events[-3:] == ["synchronize", "empty_cache", "ready"]
    assert model.resident is False


def test_failed_prepare_never_initializes_cuda(cuda_runtime) -> None:
    events, faults = cuda_runtime
    faults["transition"] = "prepare"
    model = PointingModel("test/point-head", "revision-1")

    with pytest.raises(RuntimeError, match="prepare failed"):
        model.point(Image.new("RGB", (16, 16)), "target")

    assert events == ["prepare", "ready"]
    assert model._torch is None


def test_failed_ready_frees_cuda_before_rollback(cuda_runtime) -> None:
    events, faults = cuda_runtime
    faults["transition"] = "ready"
    model = PointingModel("test/point-head", "revision-1")

    with pytest.raises(RuntimeError, match="ready failed"):
        model.point(Image.new("RGB", (16, 16)), "target")

    assert events[-4:] == ["ready", "synchronize", "empty_cache", "ready"]
    assert model.resident is False


def test_pointing_requires_own_pid_residency_before_ready(cuda_runtime, monkeypatch) -> None:
    events, faults = cuda_runtime
    faults["residency"] = "absent"
    clock = iter([0, 121])
    monkeypatch.setattr(pointing_server.time, "monotonic", lambda: next(clock))
    model = PointingModel("test/point-head", "revision-1")

    with pytest.raises(RuntimeError, match="did not prove CUDA residency"):
        model.point(Image.new("RGB", (16, 16)), "target")

    assert events.count("ready") == 1  # rollback only, after all CUDA allocations are freed
    assert events[-2:] == ["empty_cache", "ready"]


def test_warmed_pointing_prepares_again_before_inference_growth(cuda_runtime) -> None:
    events, _faults = cuda_runtime
    model = PointingModel("test/point-head", "revision-1")
    model.warm()
    assert model.resident is True
    assert events[-3:] == ["synchronize", "resident", "ready"]
    events.clear()

    model.point(Image.new("RGB", (16, 16)), "target")

    assert events[:2] == ["prepare", "point"]
    assert events[-3:] == ["ready", "synchronize", "empty_cache"]


def test_direct_tegra_path_keeps_accelerator_residency_without_broker(cuda_runtime, monkeypatch) -> None:
    events, _faults = cuda_runtime
    monkeypatch.delenv("OLLAMA_UNIFY_GPU_LEASE")
    monkeypatch.delenv("OMNI_POINTING_GPU_UUID")
    model = PointingModel("test/point-head", "revision-1")

    assert model.point(Image.new("RGB", (16, 16)), "target")
    assert "resident" in events
    assert "prepare" not in events and "ready" not in events


def test_broker_pointing_refuses_missing_lease_before_importing_torch(cuda_runtime, monkeypatch) -> None:
    events, _faults = cuda_runtime
    monkeypatch.delenv("OLLAMA_UNIFY_GPU_LEASE")
    monkeypatch.setattr(pointing_server, "_broker_present", lambda: True)

    with pytest.raises(RuntimeError, match="requires a scoped GPU lease"):
        PointingModel("test/point-head", "revision-1")
    assert events == []


@pytest.mark.parametrize("visible", ["", "GPU-other", "GPU-test,GPU-other"])
def test_broker_pointing_rejects_unreserved_cuda_visibility(cuda_runtime, monkeypatch, visible) -> None:
    events, _faults = cuda_runtime
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)

    with pytest.raises(RuntimeError, match="exactly its leased GPU UUID"):
        PointingModel("test/point-head", "revision-1")
    assert events == []


def test_pointing_refuses_cpu_fallback(cuda_runtime) -> None:
    events, faults = cuda_runtime
    faults["cuda"] = "unavailable"
    model = PointingModel("test/point-head", "revision-1")

    with pytest.raises(RuntimeError, match="refusing a CPU pointing fallback"):
        model.point(Image.new("RGB", (16, 16)), "target")
    assert not any(event.startswith("load:") for event in events)
    assert events[-1] == "ready"


@pytest.fixture
def photon_runtime(cuda_runtime, monkeypatch):
    """Offline lifecycle doubles; no model quality or CUDA acceptance claim."""
    events, faults = cuda_runtime

    class Pinned:
        def __init__(self, name):
            self.name = name
            self.path = "/owned/checkpoint.safetensors" if name == "weights" else "/proc/self/fd/" + name

        def assert_unchanged(self):
            events.append("check:" + self.name)
            if faults.get("changed") == self.name:
                raise RuntimeError("pinned file changed")

    def artifacts():
        events.append("hash_files")
        return {name: Pinned(name) for name in ("weights", "tokenizer", "cuda12")}

    class Client:
        def point(self, image, target, *, settings):
            assert image.size == (16, 16) and target == "target"
            assert settings == {"max_tokens": 64, "temperature": 0.0}
            events.append("photon_point")
            if faults.get("inference"):
                raise RuntimeError("native inference failed")
            return {"points": [{"x": 0.25, "y": 0.75}]}

        def query(self, image, prompt, *, settings):
            assert image.size == (16, 16) and prompt == pointing_server.OBSERVATION_PROMPT
            assert settings == {"max_tokens": 512, "temperature": 0.0}
            events.append("photon_observe")
            if faults.get("inference"):
                raise RuntimeError("native inference failed")
            return {"answer": "Visible current application state"}

        def close(self):
            events.append("engine_close")
            if faults.get("close"):
                raise RuntimeError("native shutdown failed")

    def photon(model, **kwargs):
        events.append("engine_warmup")
        assert model == "moondream3.1-9B-A2B"
        assert kwargs == {
            "api_key": "", "device": "cuda:0", "model_path": "/owned/checkpoint.safetensors",
            "tokenizer_path": "/proc/self/fd/tokenizer", "max_batch_size": 1,
            "kv_cache_pages": 8192, "enable_cuda_graphs": False,
            "enable_prefix_cache": True, "decode_path": "native", "service_name": "qwen_omni_pointing",
        }
        if faults.get("constructor"):
            raise RuntimeError("partially constructed native engine")
        return Client()

    def library(path, *, mode):
        assert path == "/proc/self/fd/cuda12" and mode == pointing_server.ctypes.RTLD_GLOBAL
        events.append("cuda12_library")
        return object()

    monkeypatch.setattr(pointing_server, "_photon_artifacts", artifacts)
    monkeypatch.setattr(pointing_server.ctypes, "CDLL", library)
    monkeypatch.setitem(sys.modules, "moondream", SimpleNamespace(photon=photon))
    return events, faults


def test_photon_control_plane_defers_all_imports_hashing_and_cuda_until_prepare(photon_runtime) -> None:
    events, _faults = photon_runtime
    model = PointingModel(pointing_server.PHOTON_MODEL, pointing_server.PHOTON_REVISION)
    assert events == [] and model._torch is None and model._photon_files is None
    assert not model.resident


@pytest.mark.parametrize("operation", ["point", "observe"])
def test_photon_uses_native_structured_api_and_closes_engine_before_shedding(photon_runtime, operation) -> None:
    events, _faults = photon_runtime
    model = PointingModel(pointing_server.PHOTON_MODEL, pointing_server.PHOTON_REVISION)
    image = Image.new("RGB", (16, 16))
    result = model.point(image, "target") if operation == "point" else model.observe(image)
    assert result == ([{"x": 0.25, "y": 0.75}] if operation == "point" else "Visible current application state")
    assert events[0] == "prepare"
    assert events.index("prepare") < events.index("cuda_available") < events.index("engine_warmup")
    assert events.index("engine_warmup") < events.index("photon_" + operation) < events.index("resident") < events.index("ready")
    assert events[-3:] == ["synchronize", "engine_close", "empty_cache"]
    assert not model.resident
    assert not any(item.startswith("load:") for item in events)  # No Transformers substitution.


def test_photon_reloads_exact_engine_for_next_request_without_rehashing_unchanged_inodes(photon_runtime) -> None:
    events, _faults = photon_runtime
    model = PointingModel(pointing_server.PHOTON_MODEL, pointing_server.PHOTON_REVISION)
    for _ in range(2):
        model.point(Image.new("RGB", (16, 16)), "target")
    assert events.count("hash_files") == events.count("cuda12_library") == 1
    for event in ("prepare", "engine_warmup", "photon_point", "resident", "ready", "engine_close", "empty_cache", "check:weights"):
        assert events.count(event) == 2


def test_photon_warm_keeps_engine_then_prepares_for_request_growth(photon_runtime) -> None:
    events, _faults = photon_runtime
    model = PointingModel(pointing_server.PHOTON_MODEL, pointing_server.PHOTON_REVISION)
    model.warm()
    assert model.resident and "engine_close" not in events
    events.clear()
    model.point(Image.new("RGB", (16, 16)), "target")
    assert events[:2] == ["prepare", "photon_point"]
    assert "engine_warmup" not in events
    assert events.index("resident") < events.index("ready") < events.index("engine_close")
    assert not model.resident


@pytest.mark.parametrize("operation", ["point", "observe"])
def test_photon_inference_error_closes_native_allocations_before_rollback_ready(photon_runtime, operation) -> None:
    events, faults = photon_runtime
    faults["inference"] = True
    model = PointingModel(pointing_server.PHOTON_MODEL, pointing_server.PHOTON_REVISION)
    with pytest.raises(RuntimeError, match="native inference failed"):
        if operation == "point":
            model.point(Image.new("RGB", (16, 16)), "target")
        else:
            model.observe(Image.new("RGB", (16, 16)))
    assert "resident" not in events
    assert events[-4:] == ["synchronize", "engine_close", "empty_cache", "ready"]
    assert not model.resident


def test_photon_changed_pinned_checkpoint_rejects_before_engine_initialization(photon_runtime) -> None:
    events, faults = photon_runtime
    faults["changed"] = "weights"
    model = PointingModel(pointing_server.PHOTON_MODEL, pointing_server.PHOTON_REVISION)
    with pytest.raises(RuntimeError, match="pinned file changed"):
        model.point(Image.new("RGB", (16, 16)), "target")
    assert "engine_warmup" not in events and "cuda_available" not in events
    assert events[0] == "prepare" and events[-1] == "ready"


@pytest.mark.parametrize("fault", ["constructor", "close"])
def test_uncertain_native_engine_lifecycle_requests_process_teardown_not_ready(photon_runtime, monkeypatch, fault) -> None:
    events, faults = photon_runtime

    class Fatal(BaseException):
        pass

    def terminate(*_args, **_kwargs):
        events.append("terminate_owned_worker")
        raise Fatal

    monkeypatch.setattr(pointing_server, "_fatal_photon_cleanup", terminate)
    model = PointingModel(pointing_server.PHOTON_MODEL, pointing_server.PHOTON_REVISION)
    if fault == "close":
        model.warm()
        events.clear()
    faults[fault] = True
    with pytest.raises(Fatal):
        # Actual os._exit never executes a caller's finally block. Call the
        # lifecycle boundary directly so the double retains that guarantee.
        model._load_locked() if fault == "constructor" else model._shed_locked()
    assert events[-1] == "terminate_owned_worker"
    assert "ready" not in events


def test_photon_revision_cannot_silently_use_latest_alias(photon_runtime) -> None:
    events, _faults = photon_runtime
    with pytest.raises(RuntimeError, match="exact pinned checkpoint revision"):
        PointingModel(pointing_server.PHOTON_MODEL, "main")
    assert events == []


def test_pinned_photon_file_verifies_bytes_and_detects_changed_path_inode(tmp_path) -> None:
    path = tmp_path / "weights"
    raw = b"checked bytes, not model weights"
    path.write_bytes(raw)
    with pytest.raises(RuntimeError, match="byte length"):
        pointing_server._PinnedPhotonFile(path, hashlib.sha256(raw).hexdigest(), len(raw) + 1)
    with pytest.raises(RuntimeError, match="SHA-256"):
        pointing_server._PinnedPhotonFile(path, "0" * 64, len(raw))
    pinned = pointing_server._PinnedPhotonFile(path, hashlib.sha256(raw).hexdigest(), len(raw))
    try:
        assert Path(pinned.path).read_bytes() == raw
        replacement = tmp_path / "replacement"
        replacement.write_bytes(raw)
        replacement.replace(path)
        with pytest.raises(RuntimeError, match="changed"):
            pinned.assert_unchanged()
    finally:
        pinned.stream.close()


def test_photon_runtime_version_mismatch_rejects_before_any_gpu_import(monkeypatch) -> None:
    monkeypatch.setattr(pointing_server.importlib.metadata, "version", lambda name: "unreviewed-release")
    with pytest.raises(RuntimeError, match="requires pinned moondream==2.6.1"):
        pointing_server._photon_artifacts()


def test_photon_safetensors_alias_preserves_suffix_and_exact_checked_descriptor(tmp_path) -> None:
    raw = b"checked model bytes"
    source = tmp_path / "model.safetensors"
    source.write_bytes(raw)
    pinned = pointing_server._PinnedPhotonFile(source, hashlib.sha256(raw).hexdigest(), len(raw), suffix=".safetensors")
    try:
        assert pinned.path.endswith(".safetensors")  # Pinned Kestrel0.9.1 dispatches on this suffix.
        assert Path(pinned.path).is_symlink()
        assert os.readlink(pinned.path) == f"/proc/self/fd/{pinned.stream.fileno()}"
        assert Path(pinned.path).stat().st_ino == os.fstat(pinned.stream.fileno()).st_ino
        assert Path(pinned.path).parent.stat().st_mode & 0o777 == 0o700
        assert Path(pinned.path).read_bytes() == raw
        pinned.assert_unchanged()
        Path(pinned.path).unlink()
        Path(pinned.path).symlink_to(source)
        with pytest.raises(RuntimeError, match="alias changed"):
            pinned.assert_unchanged()
    finally:
        pinned.stream.close()
        pinned._alias_directory.cleanup()


def test_real_artifact_verifier_passes_suffix_preserving_weight_path_to_sdk(tmp_path, monkeypatch) -> None:
    raw = b"small immutable file"
    sources = {}
    for name in ("weights", "tokenizer", "cuda12"):
        sources[name] = tmp_path / name
        sources[name].write_bytes(raw)
        monkeypatch.setattr(pointing_server, "PHOTON_" + ("TOKENIZER" if name == "tokenizer" else name.upper()) + "_SHA256", hashlib.sha256(raw).hexdigest())
        monkeypatch.setattr(pointing_server, "PHOTON_" + ("TOKENIZER" if name == "tokenizer" else name.upper()) + "_BYTES", len(raw))
    monkeypatch.setenv("OMNI_PHOTON_WEIGHTS", str(sources["weights"]))
    monkeypatch.setenv("OMNI_PHOTON_TOKENIZER", str(sources["tokenizer"]))
    monkeypatch.setattr(pointing_server.importlib.metadata, "version", pointing_server.PHOTON_PACKAGES.__getitem__)
    monkeypatch.setattr(pointing_server.importlib.metadata, "distribution", lambda _name: SimpleNamespace(locate_file=lambda _path: sources["cuda12"]))
    files = pointing_server._photon_artifacts()
    try:
        assert files["weights"].path.endswith(".safetensors")
        assert Path(files["weights"].path).read_bytes() == raw
        assert Path(files["weights"].path).stat().st_ino == sources["weights"].stat().st_ino
        assert files["tokenizer"].path.startswith("/proc/self/fd/")
        assert files["cuda12"].path.startswith("/proc/self/fd/")
    finally:
        for artifact in files.values():
            artifact.stream.close()
            if artifact._alias_directory is not None:
                artifact._alias_directory.cleanup()


@pytest.mark.parametrize("stage", ["constructor", "cleanup"])
def test_fail_closed_photon_diagnostics_are_bounded_and_keep_paths_credentials_and_request_private(monkeypatch, capsys, stage) -> None:
    class Exited(BaseException):
        pass

    monkeypatch.setenv("OLLAMA_UNIFY_GPU_LEASE", "private-broker-token")
    monkeypatch.setattr(pointing_server.os, "_exit", lambda _status: (_ for _ in ()).throw(Exited()))
    error = RuntimeError("format mismatch /home/private/model.safetensors https://host.test/#access=credential "
                         "private-broker-token Bearer other-credential user_request_private " + "x" * 5000)
    with pytest.raises(Exited):
        pointing_server._fatal_photon_cleanup(error, stage=stage)
    output = capsys.readouterr().err
    assert '"type": "RuntimeError"' in output
    assert "private-broker-token" not in output and "other-credential" not in output
    assert "/home/private" not in output and "host.test" not in output
    assert len(output) < 1300
    if stage == "constructor":
        assert "format mismatch" in output
    else:
        assert "user_request_private" not in output and "format mismatch" not in output
