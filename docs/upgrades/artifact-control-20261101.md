# Artifact retention and resource admission

The upgrade experiments accumulated obsolete feature generations, whole-projector
exports and materialized TTS views. The release directory had 98.30 GiB of
single-link files and the upgrade checkout 14.86 GiB; another audio-head workspace
held earlier feature generations. `/srv` had only about 20 GiB available.
Current host RAM was not exhausted: `MemAvailable` was about 329 GiB despite a
small `MemFree` counter. Roughly 325 GiB was filesystem cache. No upgrade GPU
lease remained active; both full-corpus trials had completed.

`scripts/artifact_guard.py` now checks forecast disk writes plus a reserve,
the combined owned-artifact ceiling, and optional host RAM admission using
`MemAvailable`. `--execute` starts the command only after those checks pass.
On this A100 build host, retain 50 GiB on `/srv`, 10 GiB on the checkout
filesystem, and 32 GiB available host RAM, in addition to each declared forecast.
The combined upgrade artifact ceiling is 120 GiB. CUDA still requires its own
scoped GPU-broker lease; this tool grants none.

Before starting a job, run a check for each filesystem it writes. For example:

```bash
.venv/bin/python scripts/artifact_guard.py check \
  --path /srv/fine_tuning_suite/runs/qwen-edge-v2-release \
  --reserve-gib 50 --expected-write-gib 1 \
  --expected-ram-gib 40 --ram-reserve-gib 32
```

For the owned-artifact ceiling, also supply each inventory root with
`--owned-root` and `--max-owned-gib 120`. Forecast the largest intermediate
generation, not merely the final file. Run the command through `--execute`
after completing the other filesystem checks. Retained temporary outputs must
fit that forecast; finish each experiment by classifying its outputs and
reclaiming disposable files.

## Verified reclamation

The explicit cleanup plan lists 30,975 files and binds their path, inode,
size, modification time, SHA-256, category and retained reconstruction inputs.
Dry run and apply both verify all entries before removing the first file.
The cleaner refuses shared hardlinks, symlinks, protected targets, candidate
checkpoints, missing/changed provenance, and observed open files or mappings.
It unlinks individual listed files and writes a deletion journal.

| Removed artifacts | Allocated GiB |
| --- | ---: |
| Obsolete pre-native13/14 and rejected-head feature arrays | 35.697 |
| Rejected, reconstructable projector exports | 33.700 |
| Stopped qualification component views | 2.761 |
| Total | 72.159 |

Selected projectors, canonical blobs, all learned checkpoints, raw audio,
licenses, manifests, source and numerical/semantic reports remain. An
independent post-cleanup check confirms all targets absent, 91 retained
reconstruction files matching their hashes, and both selected projectors
matching their integrity records. No Ollama blob, unrelated cache or repository
root was pruned.

The real admission check refused the original `/srv` headroom and passed after
cleanup, including the owned-artifact and RAM checks. Root free space also
increased substantially due to an external action before this cleanup; only
the 72.159 GiB listed in this transaction is attributed to this work.

Full plan: `runtime-data/disposable-artifacts-20261101.json`.
Dry-run receipt: `runtime-data/disposable-artifacts-20261101-dry-run.json`.
Apply receipt and journal: `runtime-data/disposable-artifacts-20261101-applied.json`
and its `.jsonl` sibling. Compact committed evidence is in
[artifact-control-20261101.json](artifact-control-20261101.json).
Reclaimed experiment exports can be rebuilt from the retained checkpoint,
frozen inputs and exporter under a new resource budget. Pending final release
weights remain protected until registry upload and remote verification.

This completes the bounded resource fix. It does not clear the models' failed
auditory parity gates or authorize publication of a failed model.
