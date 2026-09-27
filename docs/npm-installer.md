# `omnindicator` npm installer

`omnindicator` is a small, dependency-free Node.js host doctor and installer
for this repository. It does not contain model weights and it does not replace
the Python/CUDA runtime. Its job is deliberately narrow:

1. identify the operating system, architecture, Tegra versus discrete NVIDIA
   memory model, broker mode, system memory, model-class capacity, disk space,
   desktop-session signal, and required host tools;
2. stop before cloning or pulling model data when the coarse compute floor,
   disk floor, platform, or prerequisites are not satisfied;
3. prepare a release-pinned checkout of this repository;
4. set the selected logical model as both the Omni and language tag, avoiding a
   redundant language runner; and
5. hand control to the repository-native deployer, which remains responsible
   for exact artifact validation, live memory admission, GPU residency, service
   readiness, rollback, and the model download itself.

The fastest guided path is one command:

```bash
npx omnindicator@latest
```

Or install the CLI once:

```bash
npm install --global omnindicator
omnindicator
```

There is intentionally no npm `postinstall` hook. Installing a JavaScript
dependency must not silently clone source, request administrator access, pull
tens of gigabytes of weights, or start a microphone-enabled service. `npx
omnindicator` installs and immediately invokes the visible guided command; a
global install exposes the same command for an explicit invocation.

## Commands

```bash
omnindicator doctor
omnindicator doctor --json
omnindicator install
omnindicator install --profile ornith15 --yes
omnindicator status
```

Use `--dry-run` with `install` to print the checkout and native deploy command
without cloning, pulling weights, or changing services. `--repo-dir PATH` uses
an existing checkout without changing its Git state. On Linux, `--core-only`
is the only way to opt out of the desktop indicator.

## Platform scope

| Platform | Compute path | Managed runtime | Always-listening tray/top-bar indicator |
|---|---|---|---|
| NVIDIA Jetson/Tegra Linux | CUDA on integrated GPU/unified memory | systemd direct mode | Yes; GTK/AppIndicator harness is the default |
| Discrete NVIDIA Linux | broker-scoped CUDA when ollama-unify is present, direct otherwise | systemd | Yes; GTK/AppIndicator harness is the default |
| macOS | pinned llama.cpp Metal build | per-user launchd agent | No native tray implementation yet; portal only |
| Windows x64 + NVIDIA | pinned llama.cpp CUDA build | scheduled task or pywin32 service | No native tray implementation yet; portal only |

The package refuses to describe macOS or Windows as indicator deployments.
Interactive users must acknowledge the core-only boundary; unattended use must
pass `--core-only --yes`.

## Capacity screening

The npm doctor uses conservative class-level floors, not an optimistic free
VRAM snapshot. The compact Ornith bundle is about 8.15 GiB and is the preferred
32 GB Jetson profile. The compact Qwen3.8 E03 bundle is about 18.33 GiB and
needs substantially more headroom for KV cache, TTS, pointing, CUDA workspaces,
the desktop, and tools. A 32 GB-class Tegra can be screened as conditional for
Qwen3.8; that is not a promise that a useful live context tier will fit.

After the coarse screen, the native deployment performs the authoritative
checks using installed layer bytes, current unified/GPU memory, the operating
reserve, GPU utilization policy, per-worker residency evidence, and readiness.
It may still refuse or roll back a host that passed the npm screen. Conversely,
the npm package offers no `--force` switch that can erase a compute failure.

## Checkout and update behavior

The default checkout lives in the platform application-data directory and is
pinned to the release tag corresponding to the npm version. It stays on a
tracking `main` branch so the Linux indicator's explicit **Update available**
action can fast-forward it later. `omnindicator` will not overwrite tracked
local changes, switch an unexpected branch, or update an unexpected Git
remote. Untracked runtime data is preserved.

No registry credential, portal access token, model credential, voice clip, or
host identifier is embedded in the npm package.
