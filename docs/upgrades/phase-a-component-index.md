# Phase A checkpoint: indexed, verified component reuse

## Change and actual benefit

An unchanged `prepare --reuse --overwrite` now verifies and retains derived GGUF
views instead of reconstructing them. Bootstrap uses this mode. Missing or corrupt
members are repaired individually. Cold materialization shares one parsed tensor
inventory across views; the existing resolver still inspects the source itself.
The original no-reuse overwrite/refusal behavior and standalone unpack remain.

This is the first packaging/loading improvement. It does **not** implement a new
sidecar weight format, direct filtered mmap, cross-profile tensor deduplication,
voice-conditioning reuse, neural heads or a new model release.

## Installed-weight comparison

The benchmark used the existing installed Ornith and Qwen3.8 trained-bridge tags,
not toy tensors or downloaded replacements. Each sidecar yielded the same
1,482,388,192 bytes of TTS model and codec/projector data. The unmodified baseline
was original `origin/main` revision `7554ae0`; source hashes and every timing are
recorded in [the measurement receipt](phase-a-component-index.json).

Both implementations ran in the same Python 3.12.3 / GGUF 0.19.0 environment,
in separate fresh Linux processes. There were three forced-baseline preparations,
one indexed-cache priming preparation and three verified warm preparations per
profile. These are warm-host measurements, not cold boot or device qualification.
An unrelated source-validation run overlapped the benchmark; the results are
observations on this host, not hardware-independent forecasts.

| Installed profile | Baseline median preparation | Verified warm median | Reduction | Bytes reconstructed on a warm run |
| --- | ---: | ---: | ---: | ---: |
| Ornith 1.5 bridge | 28.479 s | 6.034 s | 78.81% | 0 |
| Qwen3.8 E03 bridge | 22.694 s | 5.963 s | 73.73% | 0 |

The clock measures the preparation function in a fresh child process, excluding
interpreter startup. Each warm run still reads and SHA-verifies both complete
output files. Warm component inodes and modification times were unchanged.
The baseline reconstructed 1,482,388,192 bytes per run. This avoided reconstruction
does not mean the final retained cache became smaller; tensor-level deduplication
and removal of duplicate weight artifacts remain future work.

Median child-process peak RSS during preparation fell from 2,118,396 to 584,544 KiB
for Ornith and from 2,117,048 to 587,224 KiB for Qwen3.8. Indexed cold priming peaked
at 2,117,604 and 2,116,600 KiB respectively. These are **CPU preparation process**
peaks, not GPU allocations, whole-service peaks or Orin qualification results.
No CUDA workload was started and no GPU residency claim follows from this test.

## Unchanged cloning weights

Both profiles' materialized files matched the baseline byte for byte:

- TTS Base: `8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129`.
- Codec/projector: `6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2`.

Those hashes also match the pinned TTS entries in `docs/sidecar-manifest.json`.
No speaker/reference inputs, sampling controls, speech chunking, TTS execution
graph, language trunk or media/tool policy changed. Exact weight preservation is
not a new live cloning-fidelity or acoustic-continuity evaluation. The proposed
20% first-audio and 10% whole-runtime memory targets remain unqualified.

## Cache contract and verification

`component-view-index.json` is versioned and bound to the extractor, installed
layer descriptor and source file identity. Outputs require full SHA-256 checks
on every reuse. Invalid/oversized receipts invalidate reuse; changed blobs cannot
reuse stale components; corrupt outputs require explicit overwrite permission;
symlink outputs are rejected; failures/source changes do not publish a new valid
receipt. Existing installed-manifest trust is unchanged.

The focused gates passed 20 component/sidecar cases, including actual Q8 tensor
bytes, metadata/tokenizer reconstruction, warm no-write behavior, selective
repair, invalidation, symlink refusal and failed/source-changing preparation.
An installed-weight CLI `prepare --reuse --overwrite` also reported zero bytes
materialized and both views reused.

The first full fork run recorded 5,014 passes and one inherited failure:
`test_private_helper_kills_its_cpu_process_group_on_deadline_or_cancel[True]`.
No packaging code is called by that test. Its complete 60-test module passed in
isolation on preserved snapshot `cf0183d`; the same module plus the 20 packaging
tests passed on the upgrade (80 tests). This does not prove the cause of the
full-suite failure. The subsequent full gate without concurrent benchmarking
passed: **5,015 tests in 283.40 s**, followed by all JavaScript/VAD/call queue,
playback, browser-cache, npm-installer, shell and contract checks. No library or
test fix was made between those two full runs. Two inherited Pillow deprecation
warnings remain. See [the final validation receipt](phase-a-validation.json).

GitHub Actions on the initial preserved-source push did not start a runner. Its
annotation says account payments failed or the spending limit must be increased:
[run 40974818628](https://github.com/robit-man/qwen-omni-upgrade/actions/runs/40974818628).
This is an external CI availability failure; no CI gate was disabled or weakened.

## Reproduction

For a fresh, explicit benchmark directory and an unchanged baseline checkout:

```bash
.venv/bin/python runtime/benchmark_component_views.py \
  --baseline-repo /home/roko/Documents/qwen-omni-adapters-main \
  --model robit/ornith-1.5-omni-audio-bridge:q4km \
  --model robit/qwen3.8-27b-e03-obliterated-omni-audio-bridge:q4km \
  --work-dir runtime-data/component-index-benchmark-NEW \
  --out runtime-data/upgrade-evidence/component-index-NEW.json --trials 3
```

The benchmark refuses an existing work directory, bounds the number of models
and trials, runs CPU artifact preparation only and never modifies an Ollama
manifest. Derived benchmark views can be removed after confirming no worker
uses that new directory; preserve this report, hashes and the measurement JSON.

After the benchmark and installed-weight CLI check completed, the exact six
run-owned files were listed and measured, then removed: 2,964,780,247 bytes,
including two small generated indexes. The benchmark directory was removed once
empty. Source Ollama blobs were retained. The temporary baseline worktree was
removed after its isolated test completed; its original commit remains in Git.

Next work should be scoped and measured separately: actual duplicate-component
deduplication/filtered-loader engineering, then full speaker-conditioning reuse
and acoustic scheduling. Do not promote this preparation result to a complete
Edge v2 release or claim the proposed neural adapters have been trained.
