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
./deploy.sh ornith15
```

`deploy.sh` probes for the broker and routes accordingly: with `docker gpu`
present it keeps the existing broker-managed `portal/start.sh --daemon`
lifecycle unchanged; without it — every Tegra module, and any unmanaged
workstation — it runs the portable direct supervisor in the foreground, which
prints the authenticated URL when the stack is ready. `cloudflared` is optional
there; without it the portal stays on loopback.

`ornith15` is the profile that fits a Jetson -- see [Memory](#memory) for the
actual budget. The `qwen38` profile's 27B base needs roughly 12 GiB more
unified memory on top of the same comprehension and TTS components.

As a service:

```bash
./scripts/bootstrap.sh
./services/linux/install.sh          # auto-selects direct mode on Tegra
.venv/bin/qwen-omni-daemon status
```

## One Ollama slot

The logical tag carries the language model in its standard layers: `ollama
show --modelfile` gives `robit/ornith-1.5-omni:q4km` and `robit/ornith-1.5:9b`
byte-identical base and projector blobs. Ollama keys a loaded runner by tag
*name*, though, so naming the core tag for the language stage loads a second
resident copy of the same weights.

On Tegra the language stage therefore defaults to the logical tag itself --
one name, one runner, one copy. `OMNI_LANGUAGE_MODEL` still overrides it, and
discrete hosts keep the explicit core tag they have always used.

## Memory

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

The comprehension window therefore defaults to 16384 tokens on Tegra rather
than 65536 -- the KV cache is the part of that budget still worth spending
carefully. `OMNI_COMPREHENSION_CONTEXT_TOKENS` overrides it, and a 64 GB
module can comfortably raise it.

Note that `-ngl 99` does not increase the footprint here the way it does on a
discrete card: there is one pool, so offloading layers changes which engine
computes them, not how much memory they occupy.

`qwen-omni doctor` reports the accelerator, and omits the broker tooling
(`docker`, `jq`, `ss`) that does not apply:

```bash
.venv/bin/qwen-omni doctor --model robit/ornith-1.5-omni:q4km \
  --language-model robit/ornith-1.5-omni:q4km
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
