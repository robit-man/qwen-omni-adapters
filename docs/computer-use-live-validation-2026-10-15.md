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
| CU-008 | Qwen | Fail | Actual ASR was correct. Model did not exercise the required ambiguous role-only Kev selection, returned no accepted Kev receipts or authoritative final, and exhausted the 300-second action deadline. Native-only scope held; owned Calculator cleanup and request idleness verified. | 309.96 s |
| CU-086 | Qwen | Pass | Actual 16 kHz Pulse monitor capture: 208,960 frames with quiet, click and steady noise; production VAD trace; zero submitted turns, stream events or GUI dispatch; call worker and owned Pulse module released. This is not physical-mic acceptance. | 13.16 s |
| CU-087 | Qwen | Fail, document identity oracle | Real Bluetooth capture, one VAD utterance, actual ASR, verified physical `type_text`, and exact visible `copper finch ready` in the owned editor. Strict initial document identity changed from generation 1 to 2 after focus transitions; that one criterion failed. Cleanup and idleness verified; all attempts retained. | 117.69 s |

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

Physical-microphone full acceptance remains pending, although the actual
capture → ASR → physical keyboard path ran. The user agreed to speak after
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
or attention cue was emitted in those pre-capture blocked attempts.
The headset's current 8 kHz speech-mode output cannot reproduce an 8 kHz tone;
the user explicitly selected a lower cue through that headset. The actual
physical test used a recorded 1 kHz blip, retaining 8 kHz as the runner's
ordinary default and checking the selected output's bandwidth before playback.
The Bluetooth admission, cue selection/propagation and CU-088/CU-090 method
oracle corrections passed the next full source gate: 1,904 tests in 90.95
seconds, with all 221 checked files unchanged (manifest SHA-256
`3782fcd840acba12c8d3d674211ae7d5e702227de89fbca372f932c8ba9d70e8`).
Bluetooth admission requires a fresh connection query for the exact typed
hardware identity; virtual sources and monitors cannot satisfy it. Arithmetic
capture cases reject keyboard or setter input followed by an incidental click.

The first physical attempt played its single cue at 19:20:21 UTC and published
speech-ready at 19:20:22 UTC. The submitted microphone utterance was 4.6 seconds.
ASR damaged the opening app name but preserved the requested words. The model
performed only focus and physical keyboard typing, both with verified effects;
the fresh owned-window buffer contained exactly the requested 18 characters.
All other criteria passed. The strict document-generation failure remains
failed while unsuccessful-observation identity bookkeeping is investigated;
the result is not relabeled as a full acceptance pass.
The source cause has been reproduced: an unsuccessful native observation
commits partial identity history before validating a unique active window.
That falsely records the document's absence and renews its generation on the
next successful read. A staged-history repair is under validation; complete
successful observations must still renew identities after genuine absence.
The repair and guarded Calc launcher passed full validation: 1,928 tests in
90.87 seconds, with all 222 checked source files unchanged (manifest SHA-256
`fcde2cf79d9c5342b8e9cf86d4699dd2b72bee9be89fa0d681fc37c5764815a0`).
The launcher disables only the spreadsheet child's OpenCL path and verifies
that an existing LibreOffice process cannot bypass that environment by taking
over the new window. It preserves normal desktop profiles and global settings.

The bounded physical repeat imported the repaired observer but stopped before
any speech submission: two transient samples after Bluetooth playback prevented the
readiness predicate from accepting 7.688 seconds of subsequent quiet. The
retained trace has no VAD start/active/utterance and no model calls. The original
predicate checks every sample since playback returned, so a transient never
ages out. This is a separate readiness defect; the microphone threshold is
unchanged and live document-identity acceptance remains pending.
The transient's attribution to delayed cue audio is a timing-based inference,
not a spectral measurement. The readiness repair checks a continuous trailing
quiet second with actual captured sample counts, frame continuity and freshness;
it retains the same threshold, eight-second bound and no-admitted-speech guard.
Full source validation passed 1,931 tests in 89.45 seconds, with all 222 checked
files unchanged (manifest SHA-256
`98076ac9518ddb5f303036ce2fedbe5c1ca317f1bc7e966a4250a31cff640b0e`).
Fresh physical acceptance of that repair remains pending.

The helper and expanded perception runners passed the next frozen source gate:
1,872 tests in 99.84 seconds, with 221 checked source files unchanged (manifest
SHA-256 `b8b073c34165ca658dd47dfa60cf540a0bc56c2894f8b278a98611166d04b90d`).
This remains a source gate, not additional live acceptance. LibreOffice Calc
24.2.7 was installed for future spreadsheet cases: 43 approved new packages,
no existing package version changes, and a clean package audit. Calc has not
been launched or tested in this campaign yet.
