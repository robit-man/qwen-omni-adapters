"""Train a bounded CTC head on cached native audio features without a language pass.

Torch, safetensors and tokenizers are build-only dependencies. Failed candidates
remain local artifacts. Passing this dataset gate does not qualify native head
execution, partial streaming or a model release. Use an exact scoped CUDA lease
of at least 4096 MiB and a fresh readiness/work path.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qwen_omni_adapters.accelerator import process_is_gpu_resident


def minimum_ctc_frames(labels):
    return len(labels) + sum(a == b for a, b in zip(labels, labels[1:], strict=False))


def word_error_rate(reference, hypothesis):
    expected = re.findall(r"[a-z0-9']+", reference.casefold())
    actual = re.findall(r"[a-z0-9']+", hypothesis.casefold())
    if not expected:
        return float(bool(actual))
    previous = list(range(len(actual) + 1))
    for i, word in enumerate(expected, 1):
        current = [i]
        for j, value in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (word != value)))
        previous = current
    return previous[-1] / len(expected)


def load_records(path, *, data=None):
    content = path.read_text() if data is None else data.decode("utf-8")
    rows = [json.loads(s) for s in content.splitlines() if s.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("empty dataset or duplicate record IDs")
    for row in rows:
        if (
            row.get("schema") != "robit.qwen3a-audio-bridge-dataset.v1"
            or row.get("current_visual_input") is not False
        ):
            raise ValueError("audio-only provenance is required")
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("invalid dataset split")
    if {r["split"] for r in rows} != {"train", "validation", "test"}:
        raise ValueError("all three fixed splits are required")
    for split in ("validation", "test"):
        subset = [r for r in rows if r["split"] == split]
        if not any(r.get("transcript", "").strip() for r in subset) or not any(
            not r.get("transcript", "").strip() for r in subset
        ):
            raise ValueError("evaluation splits need speech and empty-reference audio")
    return rows


def validate_feature_manifest(value, path, *, expected_sha256=None):
    if (
        value.get("schema") != "robit.omni.native-audio-feature-cache.v1"
        or value.get("dtype") != "float32"
        or value.get("width") != 1280
    ):
        raise ValueError("native float32 1280-wide features are required")
    if value.get("manifest_sha256") != (
        expected_sha256 or hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("feature cache belongs to a different dataset")
    if (
        value.get("language_weights_loaded") is not False
        or value.get("language_forward_passes") != 0
    ):
        raise ValueError("feature producer must not load or execute a second language model")


def prevent_split_overlap(rows, metadata):
    groups = collections.defaultdict(set)
    for row in rows:
        groups[metadata[row["audio"]]["audio_sha256"]].add(row["split"])
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError("identical audio content appears in different dataset splits")


def require_scope(uuid, token):
    if not uuid.startswith("GPU-") or "," in uuid or not token:
        raise ValueError("use docker gpu run with one exact UUID")
    result = subprocess.run(
        ["docker", "gpu", "status"], capture_output=True, text=True, check=True, timeout=10
    )
    own = next(
        (r for r in json.loads(result.stdout).get("leases", []) if r.get("token") == token), None
    )
    if not own or own.get("gpu_uuids") != [uuid] or own.get("requested_mib", 0) < 4096:
        raise ValueError("training requires its own exact-GPU reservation of at least 4096 MiB")


def wait_for_broker_ready(token):
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "gpu", "status"], capture_output=True, text=True, check=True, timeout=10
        )
        if any(
            r.get("token") == token and r.get("state") == "active"
            for r in json.loads(result.stdout).get("leases", [])
        ):
            return
        time.sleep(0.1)
    raise RuntimeError("training reservation did not become active")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--ready", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--reuse-features", type=Path, required=True)
    p.add_argument("--tokenizer-source", type=Path, required=True)
    p.add_argument("--encoder-binding", type=Path, required=True)
    a = p.parse_args()
    if not 1 <= a.epochs <= 1000:
        p.error("epochs must be 1..1000")
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    from tokenizers import Tokenizer

    tool_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    token = os.environ.get("OLLAMA_UNIFY_GPU_LEASE", "")
    require_scope(uuid, token)
    assert not a.ready.exists()
    a.work.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    manifest = a.manifest
    manifest_bytes = manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    rows = load_records(manifest, data=manifest_bytes)
    binding = json.loads(a.encoder_binding.read_text())
    encoder_hashes = {item["frozen_tensor_sha256"] for item in binding["audio_towers"].values()}
    if len(encoder_hashes) != 1:
        raise ValueError("backbones do not share the same frozen encoder")

    def norm(text):
        return " ".join(re.findall(r"[a-z0-9']+", text.casefold()))

    for row in rows:
        row["text"] = norm(row.get("transcript", ""))
    started = time.monotonic()
    tokenizer_bytes = a.tokenizer_source.read_bytes()
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
    (a.work / "tokenizer.json").write_bytes(tokenizer_bytes)
    for row in rows:
        row["labels"] = tokenizer.encode(row["text"]).ids if row["text"] else []
    feature_manifest_bytes = (a.reuse_features / "feature-manifest.json").read_bytes()
    feature_manifest = json.loads(feature_manifest_bytes)
    validate_feature_manifest(feature_manifest, manifest, expected_sha256=manifest_sha256)
    metadata = feature_manifest["features"]
    if tokenizer.token_to_id("[BLANK]") != 0 or tokenizer.get_vocab_size() not in (512, 2048):
        raise ValueError("unsupported CTC tokenizer")
    cache = {}
    for path in sorted({row["audio"] for row in rows}):
        item = metadata[path]
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != item["audio_sha256"]:
            raise ValueError("source audio changed after feature extraction")
        value = np.fromfile(item["feature_path"], dtype="<f4").reshape(-1, 1280)
        if (
            not np.isfinite(value).all()
            or len(value) != item["frames"]
            or not 1 <= len(value) <= 640
        ):
            raise ValueError("invalid native feature shape or values")
        if hashlib.sha256(value.data).hexdigest() != item["feature_sha256"]:
            raise ValueError("native feature hash mismatch")
        cache[path] = torch.from_numpy(value)
    prevent_split_overlap(rows, metadata)
    infeasible = []
    for row in rows:
        labels = row["labels"]
        if minimum_ctc_frames(labels) > len(cache[row["audio"]]):
            infeasible.append(row["id"])
    assert not infeasible, ("CTC-infeasible alignments", infeasible)

    class TemporalHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(1280, eps=1e-5)
            self.input = torch.nn.Linear(1280, 512)
            self.temporal = torch.nn.Conv1d(512, 512, 3)
            torch.nn.init.zeros_(self.temporal.weight)
            torch.nn.init.zeros_(self.temporal.bias)
            self.output = torch.nn.Linear(512, tokenizer.get_vocab_size())

        def forward(self, x):
            h = torch.nn.functional.gelu(self.input(self.norm(x)))
            causal = self.temporal(torch.nn.functional.pad(h.transpose(1, 2), (2, 0))).transpose(
                1, 2
            )
            return self.output(torch.nn.functional.gelu(h + causal))

    model = TemporalHead().cuda()
    (a.work / "head-config.json").write_text(
        json.dumps(
            {
                "schema": "robit.omni.audio-ctc-head.v1",
                "feature_width": 1280,
                "hidden_width": 512,
                "vocabulary_size": tokenizer.get_vocab_size(),
                "kernel_size": 3,
                "causal_context_frames": 2,
                "norm_epsilon": 1e-5,
                "activation": "gelu_erf",
                "encoder_tensor_sha256": next(iter(encoder_hashes)),
                "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
                "languages": ["en"],
                "production_enabled": False,
            },
            indent=2,
        )
        + "\n"
    )
    assert process_is_gpu_resident(os.getpid(), uuid)
    a.ready.touch()
    wait_for_broker_ready(token)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)
    splits = {
        split: [r for r in rows if r["split"] == split] for split in ("train", "validation", "test")
    }

    def batch_values(batch):
        inputs = torch.nn.utils.rnn.pad_sequence(
            [cache[r["audio"]] for r in batch], batch_first=True
        ).cuda()
        ilens = torch.tensor([len(cache[r["audio"]]) for r in batch], dtype=torch.long)
        labels = torch.tensor(
            [t for r in batch for t in r["labels"]], dtype=torch.long, device="cuda"
        )
        tlens = torch.tensor([len(r["labels"]) for r in batch], dtype=torch.long)
        return inputs, ilens, labels, tlens

    def evaluate(split):
        model.eval()
        values = []
        with torch.inference_mode():
            for start in range(0, len(splits[split]), 32):
                batch = splits[split][start : start + 32]
                inputs, ilens, labels, tlens = batch_values(batch)
                ids = model(inputs).argmax(-1).cpu().tolist()
                for row, sequence, length in zip(batch, ids, ilens.tolist(), strict=True):
                    tokens = []
                    prior = None
                    for t in sequence[:length]:
                        if t != prior and t != 0:
                            tokens.append(t)
                        prior = t
                    predicted = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    values.append(
                        {
                            "id": row["id"],
                            "reference": row["text"],
                            "hypothesis": predicted,
                            "wer": word_error_rate(row["text"], predicted),
                            "empty_reference": not row["text"],
                        }
                    )
        speech = [r for r in values if not r["empty_reference"]]
        noise = [r for r in values if r["empty_reference"]]
        return {
            "speech_mean_wer": sum(r["wer"] for r in speech) / max(1, len(speech)),
            "empty_false_transcript_rate": sum(bool(r["hypothesis"]) for r in noise)
            / max(1, len(noise)),
            "rows": values,
        }

    best = float("inf")
    history = []
    for epoch in range(a.epochs):
        model.train()
        order = list(splits["train"])
        random.shuffle(order)
        losses = []
        epoch_started = time.monotonic()
        for start in range(0, len(order), 32):
            batch = order[start : start + 32]
            inputs, ilens, labels, tlens = batch_values(batch)
            optimizer.zero_grad(set_to_none=True)
            log_probs = model(inputs).log_softmax(-1).transpose(0, 1).contiguous()
            loss = ctc(log_probs, labels, ilens, tlens)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate("validation")
        score = validation["speech_mean_wer"] + 5 * validation["empty_false_transcript_rate"]
        event = {
            "epoch": epoch + 1,
            "mean_train_loss": sum(losses) / len(losses),
            "validation_wer": validation["speech_mean_wer"],
            "validation_empty_false_rate": validation["empty_false_transcript_rate"],
            "elapsed_s": time.monotonic() - epoch_started,
        }
        history.append(event)
        print(json.dumps(event), flush=True)
        if score < best:
            best = score
            save_file(
                {
                    name: value.detach().cpu().contiguous()
                    for name, value in model.state_dict().items()
                },
                str(a.work / "head.safetensors"),
            )
            (a.work / "selected-validation.json").write_text(
                json.dumps(validation, indent=2) + "\n"
            )
        (a.work / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    selected = json.loads((a.work / "selected-validation.json").read_text())
    eligible = (
        selected["speech_mean_wer"] <= 0.15 and selected["empty_false_transcript_rate"] <= 0.05
    )
    report = {
        "schema": "robit.omni.audio-ctc-head-pilot.v1",
        "feature_backend": "native-ggml-float32",
        "feature_manifest_sha256": hashlib.sha256(feature_manifest_bytes).hexdigest(),
        "architecture": "layernorm-affine-gelu-causal-conv3-residual-affine-ctc",
        "parameters": sum(t.numel() for t in model.parameters()),
        "encoder_tensor_sha256": next(iter(encoder_hashes)),
        "vocabulary_size": tokenizer.get_vocab_size(),
        "head_sha256": hashlib.sha256((a.work / "head.safetensors").read_bytes()).hexdigest(),
        "tool_sha256": tool_sha256,
        "manifest_sha256": manifest_sha256,
        "selection": "validation_wer_plus_5_times_empty_false_rate",
        "validation_gate": eligible,
        "validation_maximum_mean_wer": 0.15,
        "validation_maximum_empty_false_rate": 0.05,
        "history": history,
        "native_runtime_integrated": False,
        "partial_streaming_qualified": False,
        "whole_release_qualified": False,
        "elapsed_s": time.monotonic() - started,
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1048576,
    }
    if eligible:
        model.load_state_dict(load_file(str(a.work / "head.safetensors"), device="cuda"))
        report["test"] = evaluate("test")
        report["test_gate"] = (
            report["test"]["speech_mean_wer"] <= 0.15
            and report["test"]["empty_false_transcript_rate"] <= 0.05
        )
    (a.work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("history", "test")}), flush=True)

    if not eligible:
        raise SystemExit(
            "Candidate failed validation; no test evaluation or release activation performed."
        )
    if not report.get("test_gate"):
        raise SystemExit("Candidate failed held-out test qualification.")


if __name__ == "__main__":
    main()
