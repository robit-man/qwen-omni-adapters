# B4a — bounded idle speech weights

Successful file-reference requests can now retain one native TTS worker for an
operator-selected idle interval, with a 300-second maximum. The default remains
zero. Reusing weights does not reuse generated audio, KV or sampler state. Inline
references remain ephemeral; cancellation, errors, profile changes and explicit
shedding release the relevant worker. An idle monitor never interrupts a request
holding the synthesis lock.

Tegra retention requires at least 6144 MiB of available unified memory. Missing
or failed telemetry declines retention. Health reports distinguish an intentional
headroom release from expiry and incomplete generation. Discrete CUDA retains
the existing exact-UUID broker admission and budget.

## Measured scope

Both layouts prime and unload identically before timing, use the same checkpoints,
seed, 32768-token context, two-frame streams and A/B/A texts. Timing starts at the
actual route handler and ends at the first raw PCM chunk. These are three-request
case studies, not p95/p99, playback or full-concurrency qualification.

| Measurement | Request-scoped | Five-second idle retention |
| --- | ---: | ---: |
| A100 median first PCM | 1.306 s | 0.068 s |
| A100 median completion | 2.584 s | 1.142 s |
| Initial Orin first PCM, median | 3.811 s | 0.274 s |

The A100 retained PID produced byte-identical seeded PCM for every corresponding
request, including repeated A, alternate-reference WAV and inline-reference
streaming. Raw-PCM continuity passed. Cancellation killed only the TTS worker;
the subsequent B was fresh. Expiry released the process. Cold first requests
still load weights: idle retention establishes a follow-up benefit, not a
guaranteed cold-start improvement.

The initial Orin case passed beside the resident primary and harness with at
least 6772 MiB available across the two policies. A subsequent broader run under
greater existing memory load fell below the 6144 MiB retention threshold. It
declined retention, so its warm-PID qualification assertion failed. The complete
failed receipt is retained as `phase-b4-idle-orin-pressure.json`; it is not a
passed warm qualification. The 2048 MiB active-run guard was never crossed.
Production services remained active; no extra restart was used to make this pass.

After B5, a matched 4K/adaptive-context comparison passed the broader real-route
controls on Orin (`phase-b4-idle-orin-adaptive.json`). Its three-request median
first PCM fell from 3.338 to 0.253 seconds (92.43%); median route completion fell
57.96%. The first cold request remained approximately 3.1 seconds. A/B/A PCM was
identical, expiry occurred, alternate-reference WAV matched its baseline, inline
conditioning stayed ephemeral and cancellation killed the active TTS worker.
After the later controls, availability briefly fell below the 6144 MiB retention
threshold, and retention was correctly declined. The active-run minimum was
6083 MiB, safely above the separate 2048 MiB cancellation guard. The setup
constructs the production Config with `adaptive_context=True` in both policies;
its private qualification wrapper hash is recorded. No primary restart occurred.

Consequently this option is not enabled automatically for either backbone.
Ornith needs complete concurrency qualification before a warm default is chosen;
Qwen3.8 remains request-scoped. Full reference-transcript/acoustic-code prompt
conditioning remains a separate unfinished B4 requirement. This change retains
the existing GGUF speaker-conditioning path and does not label it full acoustic
prompt support.

The matched Orin B1/B2 receipt in `phase-e-orin-token-rows.json` separately
confirms all twelve PCM/continuity comparisons and approximately 64.5% lower
TTS-process CPU peak RSS from bounded token rows. That percentage describes the
worker's CPU RSS, not total device memory or full-system peak consumption.

`./scripts/validate.sh` passed all 5041 tests and remaining source/JavaScript/
shell/contract gates. Receipts bind the tested source hashes. Whole-release
parity, training milestones and registry publication remain pending.
