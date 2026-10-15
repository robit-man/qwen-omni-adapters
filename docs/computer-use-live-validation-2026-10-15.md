# GPU 1 live desktop campaign: 2026-10-15

**In progress; full acceptance has not passed.** This campaign executes the
[100-case plan](computer-use-100-case-plan.md) against Qwen and Ornith for 200
model/case results. Failed attempts remain preserved when a later repeat is
run. No previous day's success is imported into this campaign.

Only host GPU 1, `GPU-170a99ee-850f-2182-1050-4e8d3c87b6b0`, is reserved for
the selected Omni trunk, TTS, Moondream and Kev. Qwen and Ornith are sequential.
The global operator inference hold and fault recorder remain enabled.
The live controller owns teardown order: free Kev first, then the portal's
workers, then release the scoped lease. No GPU reset/reboot is authorized.

The initial production checkpoint is `d691b73`. Source regression prerequisites
passed 1,698 tests before live execution; those tests are not live results.
The next frozen source checkpoint passed all validation gates, including 1,853
tests in 89.71 seconds. Its 244 source files were unchanged throughout that
gate (source manifest SHA-256
`a650d734957c2e2c3bf4deb6fc8cb580f938aa2ceb07365c901a4e164219939f`).
The fresh campaign manifest and per-attempt notes are retained under the
ignored `runtime-data/gpu1-acceptance/campaign-20261015/` and
`runtime-data/gpu1-acceptance/runs/` directories. Raw speech, native action
receipts, frames and exact owned file readbacks remain local.

| Case | Model | Result | Independent evidence | Duration |
| --- | --- | --- | --- | --- |
| CU-001 | Qwen | Pass | Actual generated TTS/ASR; native key chord; fresh full 18-character selection; buffer and disk unchanged; owned window closed. | 61.86 s |
| CU-001, scoped repeat | Qwen | Pass | Fresh generated speech and native selection; all six criteria passed; authoritative final echoed exactly `native_ui` and `tool_search`; source hashes stable. Original attempt retained. | 58.27 s |
| CU-002 | Qwen | Pass | Current-frame pinned Moondream grounding; physical double-click; exact `alpha` selection at offsets 0..5; pointer movement; buffer and disk unchanged; owned window closed. | 123.62 s |
| CU-003 | Qwen | Fail | Correct ASR and exact requested buffer edit, but Save was accessibility `invoke`, not the required physical click. File had two final newlines, 20 bytes instead of 19. Model's success claim did not satisfy the oracle. | 85.68 s |
| CU-004 | Qwen | Pass | Actual right-click, positively observed native context menu, Escape dismissal, fresh unchanged buffer/file, pointer movement and owned-window cleanup. | 82.34 s |
| CU-005 | Qwen | Fail | Physical Save click worked. ASR added a letter to the spelled word; model saved `copper finch aready` instead of the source instruction. Context-menu effects remained unresolved and a shell fallback violated the required native method. | 251.15 s |
| CU-006 | Qwen | Fail | Actual streamed ASR was correct, but repeated accessibility activations and unresolved/stale effect checks exhausted the 300-second action deadline. No exact `42` terminal result or authoritative final; Calculator cleanup and request idleness verified. | 309.32 s |
| CU-007 | Qwen | Fail | Correct actual ASR and five guarded physical clicks; model bypassed the spoken visual reference with native identities. No pinned Moondream receipt or fresh owned-Calculator display proof. Cleanup and request idleness verified. | 244.76 s |
| CU-086 | Qwen | Pass | Actual 16 kHz Pulse monitor capture: 208,960 frames with quiet, click and steady noise; production VAD trace; zero submitted turns, stream events or GUI dispatch; call worker and owned Pulse module released. This is not physical-mic acceptance. | 13.16 s |
| CU-087 | Qwen | Blocked before capture | Owned-window discovery was repaired and passed fresh setup; physical headset metadata then hit the ALSA-only test guard. No recording, blip, ASR or model task occurred. These attempts remain retained. | Setup only |

CUDA process evidence and fresh broker discovery after each completed attempt
proved the owned Qwen/Kev processes resident only on GPU 1, without a reported
new GPU fault. Each attempt has a separate `after-test-note.md` and JSON note.
The manifest retains all 200 slots; rows not recorded above remain pending,
not passing.

The CU-003 failures are being triaged separately: model action-method fidelity
and Gedit's implicit final-newline semantics. Neither the saved bytes nor the
physical click criterion is normalized away. Independent cases continue only
after request idleness, cleanup and GPU health are confirmed.

CU-005 also exposed a method-authority gap: the server actually executed one
read-only shell inspection of the owned test file despite the test's GUI-only
system instruction and `portal_shell_bridge:false`. That flag controls bridge
routing, not all discovered shell tools. The result remained failed, and the
inspection was never counted as GUI task progress. The portal now enforces an
immutable, explicit per-request tool allowlist across discovery and dispatch,
including detached and streamed continuations. A bounded actual-weight Qwen
probe returned no shell tools, only the permitted desktop tool, and no input
dispatch. After one controlled reload, the fresh spoken CU-001 repeat passed
with the exact applied scope in its authoritative final. This verifies the
scope fix; it does not turn CU-005 into a pass. Omitted scope retains the
ordinary default tool contract.

Physical-microphone acceptance remains pending. The user agreed to speak after
a cue, which is issued only after actual microphone frames and the owned target
window are ready. Opt-in capture/VAD/submitted-audio/stream evidence and an
isolated voice-session scope are implemented and source-validated. The runner
plays one 80 ms, 8 kHz blip, then waits for captured quiet before publishing its
speech-ready marker. Ordinary deployment recording stays disabled.

CU-087 exposed a setup failure before recording: Gedit created one mapped
document window and one unmapped helper with the same class and PID. The
owned-app launcher now selects new visible windows while excluding every
preexisting class-matching XID and retaining exact process identity checks.
A fresh live CU-087 attempt passed window setup and cleaned up its owned
window, then stopped at the separate ALSA-only microphone metadata guard.
Read-only discovery confirmed that the default Bluetooth headset is connected;
its stale PipeWire connection field is not authoritative. No microphone audio
or attention cue was emitted in these blocked attempts.
The headset's current 8 kHz speech-mode output cannot reproduce an 8 kHz tone;
the user explicitly selected a lower cue through that headset. The fresh
physical test will use a recorded 1 kHz blip, retaining 8 kHz as the runner's
ordinary default and checking the selected output's bandwidth before playback.
The Bluetooth admission, cue selection/propagation and CU-088/CU-090 method
oracle corrections passed the next full source gate: 1,904 tests in 90.95
seconds, with all 221 checked files unchanged (manifest SHA-256
`3782fcd840acba12c8d3d674211ae7d5e702227de89fbca372f932c8ba9d70e8`).
Bluetooth admission requires a fresh connection query for the exact typed
hardware identity; virtual sources and monitors cannot satisfy it. Arithmetic
capture cases reject keyboard or setter input followed by an incidental click.

The helper and expanded perception runners passed the next frozen source gate:
1,872 tests in 99.84 seconds, with 221 checked source files unchanged (manifest
SHA-256 `b8b073c34165ca658dd47dfa60cf540a0bc56c2894f8b278a98611166d04b90d`).
This remains a source gate, not additional live acceptance. LibreOffice Calc
24.2.7 was installed for future spreadsheet cases: 43 approved new packages,
no existing package version changes, and a clean package audit. Calc has not
been launched or tested in this campaign yet.
