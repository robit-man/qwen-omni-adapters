# Ollama-like service and dashboard

The repository includes a foreground supervisor, `qwen-omni-daemon`, designed
to be owned by the native service manager. On every start it:

1. checks Ollama for the logical Omni and language tags and pulls either when
   absent;
2. verifies that the language tag shares the logical tag's standard blobs;
3. resolves and validates the custom sidecar layer;
4. materializes missing comprehension and TTS views;
5. preloads and warms the optional resident Laya decision worker, then starts
   and health-checks comprehension, TTS, adapter, and portal workers; Laya
   failure leaves the deliberative path available;
6. proves direct GPU residency on NVIDIA systems -- per-process compute-app
   accounting on discrete cards, integrated-GPU device handles on Tegra (see
   [arm64 and NVIDIA Jetson](arm-jetson.md)) -- or relies on the
   Metal-enabled pinned build on macOS;
7. runs text, tagged-ASR, direct ASR-to-TTS, WAV, and streaming speech smoke gates;
8. starts Cloudflared last and records the authenticated dashboard URL.

Configuration is read from environment variables and an optional repository
`.env` copied from `.env.example`. Process state and the capability URL live in
the permission-restricted `runtime-data/state` directory.

## Linux systemd

On Jetson, the preferred entry point is the guided service installer:

```bash
./deploy.sh
```

It detects fresh versus installed state, offers install/upgrade and both
trained bridge models through arrow-key menus, persists the selected logical
tag, invokes the Linux installer in direct mode, and waits only for local
component health. With the optional desktop harness it starts the user service
immediately after the core unit, installs
the GTK/AppIndicator and PulseAudio bindings, requires an actual status-notifier
host, and waits until the harness has consumed a microphone frame. Before
cutover it inventories the five runtime
ports, stops only recognized Omni services/processes, unloads the selected and
prior-configured Ollama runners, and proves the ports are free. On Tegra it
then samples GPU utilization from sysfs and admits the exact installed layer
bytes only when unified memory retains a 6 GiB runtime reserve. A failed
startup restores the prior `.env`, systemd unit, and managed services. The
lower-level commands below remain available for staged and custom deployments.

These post-install gates execute on the target host. In particular, a Jetson
must prove its own `nvgpu`/`nvmap` residency and run the real speech route using
the arm64/CUDA binaries; passing the portable suite on an x86 development host
is necessary but not treated as Jetson runtime evidence.

On an ollama-unify host, the systemd unit deliberately uses the broker-aware
`portal/start.sh --foreground` lifecycle:

```bash
./scripts/bootstrap.sh
./services/linux/install.sh
sudo systemctl status qwen-omni-adapters
./portal/start.sh --status
```

The launcher runs `docker gpu discover`, acquires an exact UUID, verifies CUDA
residency, and releases the lease only after workers stop. Direct mode is
refused when the broker is present.

For unmanaged NVIDIA Linux and every NVIDIA Tegra module:

```bash
./services/linux/install.sh            # auto-selects broker or direct
./services/linux/install.sh --direct   # pin direct mode
./services/linux/install.sh --broker   # pin broker mode
.venv/bin/qwen-omni-daemon status
```

The installer now resolves the mode from the host: broker where `docker gpu
discover` succeeds, direct otherwise. Direct mode verifies the comprehension
PID with `nvidia-smi` compute-app accounting on a discrete card, or with the
integrated-GPU device handles on Tegra, where that accounting does not exist.
It has no CPU fallback, and it is still refused when the broker is present.

Remove the unit without removing models or runtime evidence:

```bash
./services/linux/uninstall.sh
```

## macOS launchd

Build llama.cpp with Metal and install a per-user LaunchAgent so it runs in the
same user domain as Ollama and its model store:

```bash
./scripts/bootstrap.sh
./services/macos/install.sh
.venv/bin/qwen-omni-daemon status
```

The pinned build script enables `GGML_METAL`. The daemon starts llama.cpp with
all layers offloaded (`-ngl 99` / `--gpu-layers -1`); there is no intentional
CPU inference path. Remove it with `./services/macos/uninstall.sh`.

## Windows

Run PowerShell from the repository:

```powershell
.\scripts\bootstrap.ps1
.\services\windows\install.ps1 -Mode Task
.\.venv\Scripts\qwen-omni-daemon.exe status
```

Task mode is recommended because desktop Ollama and its model store normally
belong to the signed-in user. It creates a managed logon task with restart
policy and the same foreground supervisor.

A true Windows Service is also available through pywin32:

```powershell
.\services\windows\install.ps1 -Mode Service
```

The installer requests the Windows account that owns/can access Ollama. A
LocalSystem service commonly cannot see a user's Ollama model directory, which
is why the identity must be explicit. Both modes use the CUDA-enabled pinned
llama.cpp build and require `nvidia-smi` proof for comprehension residency and
the persistent TTS worker.

Uninstall with the matching mode:

```powershell
.\services\windows\uninstall.ps1 -Mode Task
```

## Portable controls

```bash
qwen-omni-daemon serve             # macOS/Windows foreground
qwen-omni-daemon serve --no-tunnel
qwen-omni-daemon status
qwen-omni-daemon stop              # writes a graceful cross-platform stop request
```

The daemon does not expose a separate administration UI. The existing phone
portal is its dashboard and test console, so local and Cloudflared users see the
same health, queue activity, chat, call, camera, microphone, voice, and
reasoning controls.
