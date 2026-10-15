# Live desktop validation: 2026-10-14

**Full desktop acceptance has not passed.** These runs used the installed
applications and generated speech from the deployed Omni TTS route, resampled
to 16 kHz and submitted to actual ASR, language planning, and desktop tools.
No typed transcript or host-written task plan replaced the spoken instruction.
They do not constitute physical microphone acceptance.

Qwen, Ornith, and Kev ran on separate broker-leased A100s. Moondream 3.1 used
each Omni deployment's scoped GPU through the native Photon worker. The local
ignored artifacts retain exact model, GPU, process, frame, action, and file
receipts; no weights, credentials, desktop recordings, or private browser data
are included in this source report.

## Models and reference

| Component | Exact identity |
| --- | --- |
| Qwen | `robit/qwen3.8-27b-e03-obliterated-omni-audio-bridge:q4km` |
| Ornith | `robit/ornith-1.5-omni-audio-bridge:q4km` |
| Pointing | `moondream/moondream3.1-9B-A2B@1da27e6c2d2f9118de14560df456881dc62d91f3` |
| Kev | `jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52` |
| Omnius reference | `0ae974bd6c08ed9a129075fd184e63462f6b0021`, including October 12–14 work |

The demonstrated Omnius host path compiles a plan once and executes it through
fresh finite controllers. This port exposes that workflow executor and the
ordinary observe/execute API. The tested speech loop initially steered toward
individual calls; current guidance prefers bounded workflows and semantic
targets. Original Omnius fixture success is not speech-to-desktop acceptance
for these deployments.

## Recorded outcomes

Artifact names below are relative to this checkout's ignored `runtime-data/`.
Earlier failed reports remain unchanged. A fresh independent terminal check
does not rewrite an unresolved action receipt.

| Model and task | Actual result | Artifact |
| --- | --- | --- |
| Qwen, Gedit selection | Full document selection verified; buffer and file unchanged. | `real-desktop-21cc7c48d470` |
| Ornith, visual word double-click | Moondream-grounded pointer input selected `alpha`; fresh owned-document selection and unchanged file verified. Original action effect remained unresolved. | `real-desktop-dfbf475d5758` |
| Qwen, visual word double-click | Same physical goal verified independently. Slow run exhausted its tool budget and returned empty final content; conversational completion remains incomplete. | `real-desktop-090a8f64fc06` |
| Qwen, replace and Save | Real pointer Save changed the file, but transcription produced wrong text and extra newline. Failed. | `real-desktop-53d6b3bc07ee` |
| Ornith, context menu | Actual right-click exposed menu items and Escape closed the menu. Historical verification did not pass; no fresh full-case pass claimed. | `real-desktop-cd366dea71c0` |
| Ornith, Calculator | After the coordinate repair, real pointer Clear input worked. The model copied a wrong Plus ID and entered `172`; arithmetic failed. Owned-window recording retained. | `real-calculator-bd482f00c68b` |
| Ornith, Files | Workbench click/double-click were blocked by native hit testing. Files unchanged; subsequent selection, rename, and drag stages were not reached. | `real-files-6a9e8c6c524f` |
| Ornith, default-profile Firefox | After enabling host accessibility, actual address-bar focus and mouse clicks worked. Accessibility text setting left the field empty; documentation and memory stages were not reached. Recording retained. | `real-firefox-b5b58e920249` |
| Kev, recorded Calculator state | Real CUDA inference returned Clear at native probability 0.3573 across 29 choices. The unchanged 0.95 threshold caused abstention. No desktop action; classification diagnostic only. | `kev/recorded-calculator-live-diagnostic.json` |

The earlier sequential Ornith batch is retained at
`ornith-real-desktop-suite-20261015-0047/manifest.json`. Its failures exposed
the old 45-second client timeout and selection-oracle mismatch. It is not a
passing full-scope manifest.

## Fixes and limits of their validation

- Broker startup/profile/readiness and CUDA lifecycle changes were committed
  and pushed as `e4d2d0ac3815393d67fc02e3658c2ef4f2c94191`.
- GTK4 coordinates now require an exact X11/frame-margin/window-relative
  binding. Real whole-Calculator observation succeeded; real mouse input
  subsequently reached Clear. Occluded Files targets still reject.
- Local llama.cpp generation now reserves output space using actual rendered
  token counts. Incomplete tool arguments reject before execution or replay.
- Kev's lossless projection fits its unchanged strict encoder bounds:
  1,135 state tokens and 2,553 total tokens. Real inference succeeded and
  confidence remained below execution acceptance for the recorded broad case.
- GNOME accessibility was enabled. Firefox then registered a real AT-SPI
  root and both bridge and portal observations succeeded.
- Guarded `type_text` and workflow guidance are implemented. **Their newest
  live speech acceptance is pending**, not inferred from mechanical tests.
- The final source gate passed **1,698 tests** and both deployment doctors.
  This is regression evidence, not a substitute for live model acceptance.

The source checkpoint before this report is
`runtime-data/source-checkpoint-keyboard-workflow.json`, SHA-256
`bdcb8b9e98cfdee0b2b65974482285cd9e86696a62dcf80620c9001538f91714`.
Complex validation was configured for the existing 50-round tool bound.

## Hardware block and remaining acceptance

During the coordinated Qwen reload, the broker reported its A100 missing and
the remaining NVIDIA devices in `Reboot` recovery state. CUDA admission was
blocked. The startup retry loop and our validation services were stopped;
their leases were released. No GPU reset, driver reload, reboot, or further
CUDA load was attempted. Fault evidence is retained in
`runtime-data/gpu-health-fault-keyboard-reload.json`.

After operator GPU recovery, repeat fresh cases on both models with the final
source: keyboard typing, complete editing/save/context tasks, arithmetic,
visual Calculator input, active Kev-selected GUI input, Files selection and
rename/drag, and real documentation navigation followed by sourced memory
write and a second spoken recall. A complete passing suite manifest is still
required. Clipboard, modifier-assisted selection, broader applications, and
physical microphone acceptance also remain unverified.
