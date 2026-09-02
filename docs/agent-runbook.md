# Agent runbook

This is the shortest reliable path for an agent starting from only this repo.

## 1. Establish state

```bash
git status --short --branch
./portal/start.sh --status || true
df -h .
```

Do not stop a running portal or delete its cache unless the user requested that
state change. Review `AGENTS.md` and the host GPU policy before deployment.

## 2. Bootstrap without inference

```bash
./scripts/bootstrap.sh
.venv/bin/qwen-omni doctor --deployment
```

Bootstrap is idempotent: it reuses the virtual environment, pinned llama.cpp
checkout/build, and installed Ollama tags. `doctor` reports missing commands,
binaries, models, and broker discovery independently.

For a source-only CI check with no model pull or CUDA build:

```bash
./scripts/bootstrap.sh --skip-llama --skip-models
./scripts/validate.sh
```

## 3. Understand the installed artifact

```bash
.venv/bin/qwen-omni resolve \
  robit/qwen3.8-27b-e03-obliterated-omni:q4km
```

Expect exactly one custom sidecar layer, a valid bundle schema, and non-zero
base/comprehension/TTS tensor inventories. Do not point Ollama `FROM` at the
sidecar GGUF.

## 4. Deploy

```bash
./portal/start.sh --daemon
./portal/start.sh --status
```

The supervisor runs broker discovery before CUDA, pins the comprehension and
TTS processes to the acquired UUID, verifies residency, starts only loopback
services, runs smoke gates, and publishes the tunnel last.

## 5. Diagnose in layers

Read `runtime-data/logs/supervisor.log` first, then the failed stage log:

- `comprehension.log`: GGUF/projector load, context, media decode;
- `tts.log`: voice reference, codec windows, broker transitions;
- `adapter.log`: routing and upstream status, without media bodies;
- `portal.log`: authentication, queueing, request lifecycle;
- `pre-tunnel-smoke.log`: final readiness gate;
- `cloudflared.log`: tunnel creation only.

Never add raw prompts, transcripts, thinking, base64, audio, frames, tokens,
IP addresses, or user agents to default logs.

For portal behavior, keep these routing facts straight:

- MP4/WebM without audio is visual-only, not an extraction error.
- Animated GIF is normalized to bounded MP4 before comprehension.
- PDF, DOCX, and UTF-8 text/code are portal-only document inputs. Only bounded
  retrieved excerpts reach the language model; raw document envelopes never
  reach adapter v1.
- Document indexes and diagnostics are separately isolated by the hashed opaque
  browser session and both clear with trash/expire after five idle minutes.
- IndexedDB restores the same browser session after reload. Page leave starts a
  five-minute expiry; trash deletes it immediately. Restored media is
  display-only and must not be submitted into a later request.
- Call turns should answer intent rather than mirror the transcript. Media
  turns keep prior text context, but only the newest attachment is current
  evidence.
- Ordinary turns receive a compact stable behavioral policy. Current date/time,
  CPU, RAM, network-counter, and NVIDIA data is available only through an
  explicit privacy-bounded `get_system_snapshot` tool call; it may never include
  hostnames, addresses, processes, credentials, or session content.
- `stage=tts` means preparing. `audio_start` means the first PCM bytes exist.
- The default TTS stream window is two codec frames. Check `persistent_ready` in
  TTS health and reuse of the resident PID before diagnosing browser buffering.
- `max_frames` is per synthesis block. Long speech has continuous sequence
  numbers and one assembled final WAV; verify `adapter.tts_blocks > 1`.

## 6. Stop and clean

```bash
./portal/start.sh --stop
./scripts/cleanup_runtime.sh
./scripts/cleanup_runtime.sh --apply
```

The first cleanup invocation is a dry run. The apply mode refuses to run while
the recorded supervisor PID is alive and only unlinks known repo-owned derived
files. Ollama tags and blobs remain untouched.
