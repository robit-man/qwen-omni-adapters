"""Portable contract and lease-ordering gates; live A100 acceptance is separate."""

from __future__ import annotations

import copy
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
import kev_server as kev  # noqa: E402

GPU = "GPU-3538758b-6ba7-fa1b-a80c-e315a606c3f0"


def request():
    return {"model": kev.MODEL, "state": "Current UI contains a Search button.", "questions": {
        "action": {"type": "choice", "instructions": "Choose the requested legal action.",
                   "criteria": {"click": "Click Search", "abstain": "Do not act"}},
    }}


@pytest.mark.parametrize("mutation", [
    lambda value: value.pop("model"),
    lambda value: value.update(model="kev-latest"),
    lambda value: value.update(state="x" * 20_001),
    lambda value: value["questions"]["action"].update(type="noul"),
    lambda value: value["questions"]["action"].update(criteria={"one": "Only one"}),
    lambda value: value["questions"]["action"]["criteria"].update({"run code": "bad ID"}),
])
def test_kev_rejects_unbounded_or_wrong_model_requests(mutation) -> None:
    value = request()
    mutation(value)
    with pytest.raises(ValueError):
        kev.validate_request(value)


@pytest.mark.parametrize("answer", [
    {"type": "choice", "choice": "click", "confidence": 0.99},
    {"type": "choice", "choice": "unknown", "probabilities": {"click": 1, "abstain": 0}},
    {"type": "choice", "choice": "click", "probabilities": {"click": float("nan"), "abstain": 0}},
    {"type": "choice", "choice": "click", "probabilities": {"click": True, "abstain": 0}},
    {"type": "choice", "choice": "click", "probabilities": {"click": 0.5, "abstain": 0.1}},
])
def test_kev_requires_a_complete_native_distribution(answer) -> None:
    with pytest.raises(RuntimeError):
        kev.validate_answer(answer, request()["questions"]["action"]["criteria"])


@pytest.fixture
def native_runtime(monkeypatch, tmp_path):
    events = []
    faults = {}
    paths = {"adapter": tmp_path / kev.WEIGHTS_REVISION, "base": tmp_path / kev.BASE_REVISION}

    class Model:
        backend = "torch"
        graphs = None

        def parameters(self):
            return [SimpleNamespace(device=faults.get("weight_device", "cuda:0"))]

        def buffers(self):
            return []

        def encode(self, _tokenizer, _record, **kwargs):
            assert kwargs == {"max_state": 2048, "max_branch": 4096, "strict": True}
            events.append("encode")
            return {"ids": [1] * faults.get("tokens", 10)}

        def probs(self, _encoded):
            events.append("native_probs")
            if faults.get("inference"):
                raise RuntimeError("inference failed")
            return [SimpleNamespace(tolist=lambda: [0.8, 0.2])]

    class Checkpoint:
        def load(self, device, options):
            assert device == "cuda:0"
            assert self.meta.base == str(paths["base"]) and self.meta.base_revision is None
            assert vars(options) == {"dtype": "float32", "merge": False, "attn": "sdpa",
                                     "backend": "torch", "cuda_graphs": False, "fused": False}
            events.append("load_weights")
            return "tokenizer", Model()

    class Cuda:
        @staticmethod
        def is_available():
            events.append("cuda_available")
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def synchronize(_device):
            events.append("synchronize")

        @staticmethod
        def empty_cache():
            events.append("empty_cache")

    def load(_path, **kwargs):
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return {"base": kev.BASE_REPOSITORY, "base_revision": kev.BASE_REVISION,
                "weights": "lora", "weights_dtype": "fp32"}

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=Cuda(), float32="float32", load=load, inference_mode=nullcontext,
    ))
    monkeypatch.setitem(sys.modules, "kev.checkpoint", SimpleNamespace(
        Checkpoint=Checkpoint, Meta=SimpleNamespace(from_dict=lambda value: SimpleNamespace(**value)),
        LoadOptions=lambda **kwargs: SimpleNamespace(**kwargs),
    ))
    monkeypatch.setitem(sys.modules, "kev.api", SimpleNamespace(
        SystemOneRequest=lambda **kwargs: kwargs,
        to_record=lambda value: (value, None),
        to_answers=lambda probabilities, _metadata: {"action": {
            "type": "choice", "choice": "click",
            "probabilities": dict(zip(("click", "abstain"), probabilities[0], strict=True)),
        }},
    ))

    def resident(pid, gpu_uuid):
        assert pid == os.getpid() and gpu_uuid == GPU
        events.append("own_pid_resident")
        return not faults.get("residency")

    def transition(action, token):
        assert token == "private-test-lease"
        events.append(action)
        if faults.get("transition") == action:
            raise RuntimeError("transition failed")

    class ProcessExit(BaseException):
        pass

    monkeypatch.setattr(kev, "verify_manifest", lambda _value: paths)
    monkeypatch.setattr(kev, "process_is_gpu_resident", resident)
    monkeypatch.setattr(kev, "broker_transition", transition)
    monkeypatch.setattr(kev.os, "_exit", lambda code: (_ for _ in ()).throw(ProcessExit(code)))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    monkeypatch.setenv("OLLAMA_UNIFY_GPU_LEASE", "private-test-lease")
    return events, faults, ProcessExit


def test_kev_native_eager_load_and_leased_probability_readout(native_runtime) -> None:
    events, _faults, _exit = native_runtime
    model = kev.KevModel({})
    assert events == ["cuda_available", "load_weights", "synchronize", "own_pid_resident"]
    events.clear()

    result = model.decide(request())

    assert events == ["encode", "prepare", "native_probs", "synchronize", "own_pid_resident", "ready"]
    assert result == {"model": kev.MODEL, "revision": kev.WEIGHTS_REVISION, "answers": {
        "action": {"type": "choice", "choice": "click", "probabilities": {"click": 0.8, "abstain": 0.2}},
    }}
    assert model.calls == 1
    assert "private-test-lease" not in json.dumps(result)


@pytest.mark.parametrize("scope", ["", GPU + ",GPU-other", "0"])
def test_kev_refuses_missing_or_unscoped_lease_before_cuda(native_runtime, monkeypatch, scope) -> None:
    events, _faults, _exit = native_runtime
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", scope)
    with pytest.raises(RuntimeError, match="one exact broker-leased GPU UUID"):
        kev.KevModel({})
    assert events == []


def test_kev_refuses_cpu_weight_placement(native_runtime) -> None:
    _events, faults, _exit = native_runtime
    faults["weight_device"] = "cpu"
    with pytest.raises(RuntimeError, match="left the leased CUDA device"):
        kev.KevModel({})


def test_kev_bounds_tokens_before_request_cuda_growth(native_runtime) -> None:
    events, faults, _exit = native_runtime
    model = kev.KevModel({})
    events.clear()
    faults["tokens"] = 4097
    with pytest.raises(ValueError, match="strict token bound"):
        model.decide(request())
    assert events == ["encode"]


@pytest.mark.parametrize("fault", ["inference", "residency", "ready"])
def test_kev_failed_cuda_interval_exits_for_broker_owned_cleanup(native_runtime, fault) -> None:
    events, faults, process_exit = native_runtime
    model = kev.KevModel({})
    events.clear()
    if fault == "ready":
        faults["transition"] = "ready"
    else:
        faults[fault] = True

    with pytest.raises(process_exit) as exc:
        model.decide(request())
    assert exc.value.args == (125,)
    assert events.index("prepare") < events.index("native_probs")
    if fault != "ready":
        assert "ready" not in events


def test_kev_invalid_model_never_reaches_native_inference(native_runtime) -> None:
    events, _faults, _exit = native_runtime
    model = kev.KevModel({})
    events.clear()
    wrong = copy.deepcopy(request())
    wrong["model"] = "kev-latest"
    with pytest.raises(ValueError):
        model.decide(wrong)
    assert events == []
