# arm64 and NVIDIA Jetson (Tegra)

The runtime supports NVIDIA Tegra modules — Jetson AGX Orin, Orin NX/Nano,
Xavier AGX/NX, TX2, TX1/Nano — as a first-class CUDA host. Nothing on this
page changes how a discrete NVIDIA or Apple Metal host behaves; Tegra simply
answers the same questions with the evidence its driver actually publishes.

## Why Tegra needs its own answers

A Tegra module runs an **integrated** GPU on unified memory behind the
`nvgpu`/`nvhost` kernel drivers. `nvidia-smi` exists on JetPack, but:

```text
|   0  Orin (nvgpu)                  N/A  | N/A              N/A |
| N/A   N/A  N/A               N/A /  N/A | Not Supported        |
| Processes:  No running processes found                          |
```

* `--query-compute-apps` returns success with an empty body, permanently.
* `--query-gpu=memory.used,utilization.gpu,power.draw` answers `[N/A]`.
* There is no ollama-unify GPU broker, because there is no discrete device
  to lease and no second GPU to allocate between.

Treating an empty compute-app listing as "not resident yet" is what made a
Jetson start-up fail its residency gate after the full 120-second timeout,
every time.

## What residency means on Tegra

`qwen_omni_adapters.accelerator` keeps one question — *is this pid actually on
the GPU?* — and dispatches on the host:

| Host | Evidence |
|---|---|
| Discrete NVIDIA | `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory`, pinned to the reserved UUID |
| Tegra | the worker's own open handles on the integrated-GPU character devices plus the `nvmap` GPU allocator |
| macOS | the Metal-enabled pinned llama.cpp build enforced at bootstrap |

A CUDA context on Tegra holds `/dev/nvgpu/igpu0/*` (JetPack 6 / L4T R36+) or
`/dev/nvhost-*gpu*` (JetPack 4–5) open for the lifetime of the context, and
`/dev/nvmap` for every GPU-visible allocation. A CPU-fallback build opens
neither. Both are required, because `nvmap` alone is also the general Tegra
multimedia allocator.

This is still a real proof and there is still **no CPU fallback**: a
`llama-server` or `llama-tts` that failed to initialize CUDA never opens those
nodes, and the daemon refuses to reach `ready`.

## Build

`scripts/build_llama_cpp.sh` reads the SoC from the device tree and pins the
exact compute capability, instead of building ggml's discrete-desktop
architecture list:

| SoC | Module | `CMAKE_CUDA_ARCHITECTURES` |
|---|---|---|
| `tegra210` | TX1, Nano | 53 |
| `tegra186` | TX2 | 62 |
| `tegra194` | Xavier AGX/NX | 72 |
| `tegra234` | Orin AGX / NX / Nano | 87 |
| `tegra264` | Thor | 101 |

It also caps build parallelism at one job per 2 GiB of RAM, because on unified
memory `nvcc` competes with everything else on the module and an unbounded
`-j$(nproc)` reliably OOMs a Jetson mid-compile. Override either with
`OMNI_CUDA_ARCHITECTURES` and `LLAMA_CPP_BUILD_JOBS`.

## Deploy

```bash
git clone https://github.com/robit-man/qwen-omni-adapters.git
cd qwen-omni-adapters
./deploy.sh
```

`deploy.sh` detects the Tegra SoC, unified-memory size, installed runtime,
managed service, and current model. Arrow-key menus select install/upgrade,
one of the two trained bridges, and the always-listening harness by default;
only an explicit core-only selection or `--no-harness` omits its indicator. It
pulls the logical Ollama tag, validates the sidecar, runs doctor/regression
gates, installs the direct systemd service, and waits only for local component
health. The desktop indicator service is started immediately after the core
unit and waits for the portal in parallel. Generation smoke is not part of
guided readiness. `cloudflared` is optional; without it the portal stays on
loopback.

The board still proves each worker's Tegra GPU handles while loading, but it
does not generate text or speech before exposing the portal. Set
`OMNI_STARTUP_SMOKE=1` for the blocking ASR → cloned-TTS → ASR and co-residency
diagnostic when deliberately troubleshooting. The default desktop harness
deployment requires its real AppIndicator backend, a reachable
desktop audio server, and a captured microphone frame. Camera nodes are not
opened during this startup:
FFmpeg first touches them only after a structured camera-view request.
The indicator can explicitly open a loopback browser view; every discovered
camera is composed in one left-to-right row rather than a grid. It also polls
`origin/main` and offers a click-to-install fast-forward update. Automatic
updates preserve untracked files and fail closed for tracked edits, a non-main
checkout, or diverged history; use `./deploy.sh` for those cases or for changes
that require reinstalling privileged system-unit definitions.
On Ubuntu appliance desktops, the harness install also adds one narrow polkit
rule for the distro-owned, read-only `package-system-locked` query. This stops
update-notifier from opening an administrator-password dialog during the
autologin/Chromium startup race; it does not authorize package installation,
updates, or configuration.

An upgrade is a controlled handoff. The deployer resolves the live owners of
ports 8892, 8901, 8910, 8920, and 8930 from procfs and refuses an unknown
owner. It stops and records recognized legacy/current Omni units, waits for
the ports to close, and asks Ollama to unload only the selected,
prior-configured, and known legacy Omni tags. It then prints pre/post GPU and
unified-memory snapshots from `qwen_omni_adapters.accelerator`, samples the
Tegra load five times, and requires the exact installed bundle bytes plus a
6 GiB reserve before starting the replacement. Override the conservative
admission values with `OMNI_DEPLOY_MEMORY_RESERVE_MIB` and
`OMNI_DEPLOY_MAX_GPU_UTILIZATION` only when production measurements justify
it. If the replacement fails, the old environment, unit, and managed services
are restored; unmanaged processes are identified but cannot be reconstructed.

`ornith15` is recommended for a 32 GB Jetson. `qwen38` provides the larger E03
trunk but leaves less room for KV cache, graph workspaces, the desktop, and
concurrent tools; see [the trained-bridge budget](#trained-audio-bridge-profile).

The optional Laya shadow decision worker is disabled by default on Tegra. Its
roughly 1.5--2 GiB resident host allocation otherwise consumes the safety band
needed to keep comprehension and cloned TTS resident while browser and shell
tasks run. A larger measured Jetson deployment may opt in with
`OMNI_DECISION_PLANE_ENABLED=1`; the language/tool path remains authoritative
when the shadow worker is absent.

Guided Tegra deployment also creates `.pointing-venv`, installs the NVIDIA
JetPack PyTorch wheel matching the board's L4T release, and downloads the pinned
Moondream 2 point checkpoint. The point worker starts before comprehension so
the live context-window allocator sees its real resident cost. It stays loaded
beside comprehension and cloned TTS; each worker must retain its own Tegra GPU
device handles. Set `OMNI_ENABLE_POINTING=0` only to deliberately restore the
slower two-pass crop fallback. There is no CPU fallback. JetPack 6.0's official
Torch 2.4 lacks the later `enable_gqa` SDPA argument, so the worker applies the
equivalent K/V-head repetition before loading the checkpoint.
Explicit browser verification snapshots and completed visual clicks reuse this
resident worker for a fixed, bounded current-frame reading, avoiding another
model load and preserving exact visible completion text after screenshot bytes
leave the durable task context.

As a service:

```bash
./deploy.sh                           # installs/upgrades and starts the service
.venv/bin/qwen-omni-daemon status
```

## One Ollama slot

Each trained bridge carries its sole language model in the standard Ollama
layer and its combined native-vision/Omni-audio projector in the standard
projector layer. The installer names that same logical tag for both runtime
stages, so one llama.cpp server performs comprehension, language, and tools;
there is no second Ollama language runner.

## Legacy full-Omni memory

Unified memory means the "VRAM" figures below come out of the same pool as the
OS and every other process. Measured from the `ornith15` bundle manifest
(`qwen-omni resolve robit/ornith-1.5-omni:q4km`):

| Component | Runtime | Weights |
|---|---|---|
| Ornith 1.5 9B Q4_K_M + CLIP projector | Ollama (language, vision, tools) | 6.1 GiB |
| Qwen3-Omni-30B-A3B Q4_K_M + projector | `llama-server` comprehension | 18.5 GiB |
| Qwen3-TTS-12Hz-1.7B + codec projector | `llama-tts` | 1.4 GiB |

That is about 26 GiB of weights before any KV cache. A 64 GB AGX Orin has
ample headroom. A **32 GB module is tight**: it fits the three workers, but
not with much else resident, so budget carefully if the host is also running
vision or other CUDA workloads.

Legacy full bundles retain a 65,536-token ceiling. Compact bridge bundles may
have a larger native positional range, but default to a **16,384-token resident
working set**. History beyond that bound is handled by the lossless
virtual-context hierarchy instead of preallocating a nominal 256K KV cache in
the Jetson's shared pool. The direct Jetson daemon runs
`runtime/comprehension_launcher.py`, derives KV bytes per token from the
selected GGUF, and may select a smaller standard tier when current
`MemAvailable` cannot fund 16K plus the shared runtime reserve. The selected
value is published in daemon status as `comprehension_context_tokens`; the
default ceiling is `comprehension_context_ceiling`. The published
`comprehension_parallel_slots` matches llama-server's `--parallel` value; the
default is one so the planner cannot undercount four implicit KV slots. An
explicit `OMNI_COMPREHENSION_CONTEXT_TOKENS` can still override the ceiling for
controlled benchmarks, but larger values are not the supported Jetson default.

`OMNI_COMPREHENSION_CACHE_TYPE_K` and
`OMNI_COMPREHENSION_CACHE_TYPE_V` separately control the llama.cpp L0 KV
formats. The guided 32 GB Tegra deployment uses `q8_0` for the qualified Ornith
audio bridge; other model/platform pairs retain the general `f16` default until
they pass equivalent live gates. The qualified q8 run held a 16,384-token worker
through sealed 256K domain answers, text/audio/image comprehension, resident
cloned and streamed TTS, post-TTS ASR, and structured tools without a restart or
governor downshift. A later foreground plus medium-horizon action soak did
downshift 16K to 8K and then 4K as the full working set became resident. This
qualifies q8 as the cache format, not 16K as a permanent tier; 16K remains the
ceiling and the live governor is authoritative. `q4_0` remains experimental:
its sealed domain answers passed, but the live tool gate selected an incorrect
GUI route and was rejected. The launcher derives its admission slope from exact
block bytes and resets its
calibration when the format contract changes. The pinned build has no 2-bit KV
format, so q4 testing is only a lower-precision cache ablation inspired by the
KIVI direction, not a reproduction of KIVI's 2-bit method.

Guided Ornith/Tegra deployment also sets
`OMNI_BACKGROUND_RESIDENCY_MODE=action`. Before a silent durable-task slice,
the harness asks the loopback TTS wrapper to shed only its independently
reloadable speech graph. The shared Omni language/ASR/vision trunk and the
pointing worker stay resident. A later synthesis request reloads the shipped
voice-clone profile through the normal TTS path. `conversation` keeps TTS warm
and is the default on unqualified model/platform pairs. Action mode does not by
itself authorize automatic context expansion while TTS is absent; expansion
needs a coordinated foreground downshift before speech can safely return.

Note that `-ngl 99` does not increase the footprint here the way it does on a
discrete card: there is one pool, so offloading layers changes which engine
computes them, not how much memory they occupy.

## Trained audio-bridge profile

The lightweight profile removes the separate 18.5 GiB Omni comprehension
model. It retains one quantized target language trunk, the target's native
vision projector, the frozen Omni audio encoder, a trained final audio
projection, and the existing TTS stack. Current artifact-byte projections are:

| Target | Language + combined projector + TTS weights |
|---|---:|
| standard Ornith 1.5 9B | about 8.2 GiB |
| Qwen3.8 27B E03 Obliterated | about 18.3 GiB |

These are file/resident-weight totals, not a general peak-unified-memory
benchmark. KV cache, graph workspaces, CUDA allocations, the OS, and the portal
still consume the shared pool. The guided deployer's admission gate prevents a
start when the post-handoff host cannot retain the installed layers and its
default 6 GiB operating reserve. This admission check is intentionally quick;
it does not run inference or claim that every larger context, concurrent
workload, or long-run peak will fit.

At worker start, the context launcher performs the finer-grained KV admission.
Its first load can select the largest tier justified by complete component
bytes instead of waiting through multiple restarts at 4K/8K/16K. A live sample
then calibrates actual residency; an abnormal exit or insufficient post-load
reserve caps the next supervised start below that failed tier. On Tegra the
selected llama.cpp process and its CUDA character-device handles stay fixed for
the complete service session. JetPack 6 / L4T R36.3 can panic in the `nvgpu`
character-device close path when a CUDA-heavy process exits, so transient tool
pressure must never turn an otherwise ordinary request gap into a model-process
restart. Prompt limits still expand, compact, and shed dynamically inside the
startup-selected KV allocation, and the shared governor refuses new browser,
TTS, memory, or background residency before crossing its hard floor. The next
supervised service start re-evaluates all live capacity and can select a larger
or smaller tier. `OMNI_COMPREHENSION_RUNTIME_RESIZE=1` is an explicit diagnostic
override for a JetPack release whose repeated process-teardown stress gate has
passed; it is not a production default.

Discrete-memory Linux hosts retain live one-tier downshift/expansion. Their
resizes occur only while every inference slot is idle (except an emergency hard
floor) and after the pressure/cooldown interval, with request/task context
compacted against the currently published active tier before inference.

`qwen-omni doctor` reports the accelerator, and omits the broker tooling
(`docker`, `jq`, `ss`) that does not apply:

```bash
.venv/bin/qwen-omni doctor \
  --model robit/ornith-1.5-omni-audio-bridge:q4km \
  --language-model robit/ornith-1.5-omni-audio-bridge:q4km
```

```json
{
  "accelerator": {
    "machine": "aarch64",
    "tegra": true,
    "tegra_soc": "tegra234",
    "l4t_release": "R36.3.0",
    "cuda_architectures": "87",
    "gpu_memory_model": "unified",
    "residency_backend": "tegra-device-handles"
  },
  "deployment_mode": "direct"
}
```

## Running without the comprehension worker

Qwen3-Omni comprehension is by far the largest component. Measured on a 32 GB
AGX Orin (29.98 GiB usable):

| Component | Resident |
|---|---|
| baseline (OS + desktop) | 6.1 GiB |
| Qwen3-Omni comprehension @8K context | 16.8 GiB |
| Qwen3-TTS worker, voice cloning | 6.7 GiB |
| Ollama language on the logical tag | 5.6 GiB |

All of it at once is about 35 GiB, and the kernel OOM-killer settles that
argument -- on this host it took the desktop down with it. `OMNI_ENABLE_COMPREHENSION=0`
runs the adapter for language and speech alone: the logical tag still serves
text, vision and tools through Ollama, and Qwen3-TTS still provides the voice,
for roughly 12 GiB total. Audio, video and image comprehension then report
unavailable rather than failing obscurely, and the health gate treats the
absent worker as intended rather than unhealthy.

`OMNI_TTS_PERSISTENT=0` additionally lets the TTS worker exit between
utterances, so its memory is only held while actually speaking.

Where the host manages the comprehension worker itself -- starting and
stopping it around demand -- set `OMNI_COMPREHENSION_URL` explicitly alongside
`OMNI_ENABLE_COMPREHENSION=0`: the adapter will use the worker whenever it is
up, and the supervisor will not try to own its lifetime.

Set `OMNI_STARTUP_SMOKE=0` when that host also brokers TTS memory. The daemon's
normal startup gate performs real language and speech generations; disabling
that gate keeps service startup lightweight so only the host's admitted
requests can load model weights. Readiness endpoints remain available and the
host should probe each route after it admits the corresponding component.

## Language on the comprehension model

`OMNI_LANGUAGE_API=openai` points the language stage at any OpenAI-compatible
endpoint instead of Ollama, including the comprehension server's own
`/v1/chat/completions`. Qwen3-Omni is an Instruct model, so a host that cannot
afford both sets of weights can run language on the one it has already loaded
rather than adding a second. Ollama-only request fields (`keep_alive`,
`think`, `options`, `format`) are dropped, and the sampling controls the
OpenAI schema does define are carried across.

Set a systemd `MemoryMax=` on the unit regardless. Unified memory means an
unbounded component does not merely get itself killed -- it can take the
machine with it.

## Host facts

`get_system_snapshot` reports the integrated GPU from sysfs — per-mille load,
devfreq frequency, the GPU thermal zone, and unified memory marked
`"memory_model": "unified"` — because `nvidia-smi --query-gpu` would otherwise
report no GPU at all on this host. The same privacy bounds apply: no
hostnames, addresses, routes, sockets, processes, or session content.
