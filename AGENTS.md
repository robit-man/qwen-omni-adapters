# AGENTS.md — Qwen Omni runtime instructions

This repository is a standalone runtime baseline. An agent receiving only this
checkout must be able to install, validate, deploy, diagnose, and safely stop
the logical Omni model without relying on `fine_tuning_suite`.

## First actions

1. Read `README.md`, `docs/architecture.md`, `docs/runtime.md`, and this file.
2. Inspect `git status` and preserve user changes.
3. Run `./scripts/validate.sh` before and after code changes.
4. For host deployment, run `.venv/bin/qwen-omni doctor --deployment` after
   bootstrap and before starting the supervisor.

Do not restart or stop an existing deployment unless the user asked for it.
`portal/start.sh --status` is read-only. A new checkout uses its own
`runtime-data` directory and ports may conflict with an already running copy.

## Architectural invariants

- `robit/qwen3.8-27b-e03-obliterated-omni:q4km` is one logical Ollama tag.
- Stock Ollama executes the standard Qwen3.8 model/projector layers. It does
  not execute the custom audio/video/TTS sidecar layer.
- The sidecar is one valid namespaced GGUF container, but it is not a valid
  `FROM` target for stock Ollama. The runtime materializes executable views.
- Qwen3.8, Qwen3-Omni, and Qwen3-TTS meet at a semantic-text boundary. Do not
  splice incompatible hidden states or claim a native monolithic graph.
- Adapter v1 is `robit.ollama.omni-adapter.v1`; its portable route requires
  `stream:false`. The portal NDJSON route is an authenticated extension.
- `think` is a native Ollama boolean. Default false, pass it unchanged, keep
  `message.thinking` separate, and never use a system prompt or `/no_think` as
  the control mechanism.
- Keep `cache_prompt:false` on every multimodal comprehension request. Removing
  it reintroduces stale audio/video embeddings across otherwise fresh turns.
- Speech transcript, non-speech audio observation, and visual observation are
  separate tagged evidence. Only the transcript can be attributed to the user.
- Only the newest attached media is current perceptual evidence. Bounded prior
  text dialogue remains available for conversational continuity; cached media
  previews and prior descriptions must never be replayed as new attachments.
- Browser session state is isolated in cookie-scoped IndexedDB, expires five
  minutes after page leave, and is deleted by trash. Restored media is
  display-only.
- Runtime environment context is rebuilt for every turn and must remain
  privacy-bounded: never add hostnames, addresses, routes, sockets, processes,
  credentials, or session content.
- The Qwen3-TTS resident protocol must reset generation memory and samplers and
  create a fresh audio-generation helper between prompts. Matching profiles may
  reuse one PID and loaded weights; decoded request state may not.
- Never synthesize reasoning or an unresolved tool call.

## CUDA deployment policy

This host uses the ollama-unify GPU lease broker. Before creating, starting,
or resizing any Docker/container/service deployment that uses CUDA:

1. Run `docker gpu discover` and inspect broker policy.
2. Use `docker gpu run --owner NAME --vram-mib MIB --gpu GPU_UUID` for a
   foreground workload, or scoped acquire → residency → ready → prepare/resize
   → release for independently supervised services.
3. Set the external process's `CUDA_VISIBLE_DEVICES` to exactly the reserved
   UUIDs. Do not add unreserved GPUs.
4. Read `/usr/local/share/ollama-unify/AGENTS.md` before changing the service
   lifecycle on this host.
5. Do not use a static free-VRAM scan or anonymous CUDA allocation. Keep
   Ollama `num_gpu` automatic (`-1`); it is a layer count, not a GPU selector.

`portal/start.sh` implements the scoped protocol and verifies actual CUDA
residency. Do not weaken that check or introduce a CPU fallback.

The portable `qwen-omni-daemon` refuses direct Linux mode when it detects the
broker. macOS and Windows use the direct supervisor with platform-native GPU
builds and service managers; see `docs/services.md`.

## Safe workflow

```bash
./scripts/bootstrap.sh                 # install/build/pull/resolve
.venv/bin/qwen-omni doctor --deployment
./scripts/validate.sh
./portal/start.sh --daemon
./portal/start.sh --status
./portal/start.sh --stop
```

The supervisor must publish the tunnel only after local health, exact-text,
and GPU TTS smoke gates pass. Component services bind to loopback. Never log or
commit the generated access token or full capability URL.

## Change map and synchronized artifacts

When changing the public request/response contract, update all of:

- `src/qwen_omni_adapters/contract.py`
- `docs/protocol.md`
- `docs/schema/request-v1.schema.json`
- `docs/schema/response-v1.schema.json`
- `runtime/adapter_server.py`
- both clients and contract tests

When changing voice fields, also update `portal/voice-profile.json`, the voice
schema, portal validation/UI, TTS wrapper, and tests. When changing streaming
events, update portal backend, browser parser, smoke test, and protocol docs.

## Required regression gates

- `./scripts/validate.sh` passes.
- `think:false` produces no thinking channel; `think:true` produces a separate
  thinking channel against the real Ollama tag.
- red → blue → red media produces red → blue → red observations.
- quiet/click/steady-noise VAD fixtures submit zero calls; sustained speech and
  sustained alarm submit one each.
- two simultaneous sessions receive only their own marker and expose at most
  aggregate active/queued counts.
- TTS returns valid 24 kHz mono PCM16 WAV; streaming returns ordered PCM chunks,
  repeated matching-profile prompts reuse one resident worker PID, and an
  A → B → A synthesize/transcribe gate returns A → B → A without one-turn lag.
- Browser cache restore/expiry/clear and runtime-environment privacy tests pass.
- CUDA PIDs are resident only on the leased UUID.

## Storage and cleanup

Weights and materialized views are never committed. `runtime-data/components`
is derived from the installed Ollama sidecar and normally removed on clean
shutdown. Run `./scripts/cleanup_runtime.sh` for a dry run, then
`./scripts/cleanup_runtime.sh --apply` only after confirming the supervisor is
stopped.

For build/release work, delete run-local safetensors, full-precision GGUFs,
partials, and redundant quantizations only after Ollama/Hugging Face uploads
and remote verification have succeeded. List exact candidates and measure the
directory first. Never recursively clean a repository root, cache root,
`$HOME`, or `/srv/ollama/models`; never manually delete Ollama blobs. Remove an
obsolete tag with `ollama rm <exact-tag>` after verifying it is no longer used.

Keep source revisions, manifests, digests, Modelfiles, licenses, test reports,
and compact release evidence.
