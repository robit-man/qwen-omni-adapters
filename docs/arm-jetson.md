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

`ornith15` (9B) is the profile that fits a Jetson. The 27B `qwen38` profile
needs more unified memory than an Orin has once the language backend, TTS
worker, and OS are also resident.

As a service:

```bash
./scripts/bootstrap.sh
./services/linux/install.sh          # auto-selects direct mode on Tegra
.venv/bin/qwen-omni-daemon status
```

## Memory

The comprehension window defaults to 32768 tokens on Tegra rather than 65536,
because the integrated GPU shares the module's system RAM with the Ollama
language backend, the TTS worker, and the OS.
`OMNI_COMPREHENSION_CONTEXT_TOKENS` still overrides it.

`qwen-omni doctor` reports the accelerator, and omits the broker tooling
(`docker`, `jq`, `ss`) that does not apply:

```bash
.venv/bin/qwen-omni doctor --model robit/ornith-1.5-omni:q4km \
  --language-model robit/ornith-1.5:9b
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

## Host facts

`get_system_snapshot` reports the integrated GPU from sysfs — per-mille load,
devfreq frequency, the GPU thermal zone, and unified memory marked
`"memory_model": "unified"` — because `nvidia-smi --query-gpu` would otherwise
report no GPU at all on this host. The same privacy bounds apply: no
hostnames, addresses, routes, sockets, processes, or session content.
