# GPU 1 live desktop campaign: 2026-10-15

**In progress; full acceptance has not passed.** This campaign executes the
[100-case plan](computer-use-100-case-plan.md) against Qwen and Ornith for 200
model/case results. Failed attempts remain preserved when a later repeat is
run. No previous day's success is imported into this campaign.
The [recent Omnius layer audit](computer-use-omnius-layer-audit-2026-10-15.md)
identifies the exact reference commits and distinguishes model-driven browser
evidence from fixed-planner native executor demonstrations.

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
| CU-009 | Qwen | Fail, later foreground confound | Correct actual ASR, but no dispatched task action. Initial owned-target click was rejected as occluded; later observations moved to another window and freshness checks failed. No authoritative final within 300 seconds. Owned Files window closed; outstanding inference subsequently became idle. | 319.59 s |
| CU-010 | Qwen | Fail | Correct actual TTS/ASR and six verified native inputs across ten rounds. Navigation, Find, scrolling, sourced memory write and second-turn recall did not complete within 300 seconds. Owned Firefox window closed politely; inference subsequently became idle. | 322.99 s |
| CU-011 | Qwen | Fail, foreground confound | Actual TTS/ASR completed, but the model observed another foreground window and dispatched no input. Independent capture proves focus and pointer changes without model input; the actor is unknown. File unchanged, owned Files window closed, inference idle. | 110.44 s |
| CU-011, native-thinking repair repeat | Qwen | Fail, native hit test | Correct actual TTS/ASR and structured native rounds. The exact owned `alpha.txt` target was rejected as occluded; every model observation remained in the owned Files window and the pointer stayed at one position. No task input or authoritative final before the 300-second deadline. File unchanged; polite cleanup and idleness verified. | 306.93 s |
| CU-013, first attempt | Qwen | Fail, test observer | The test frontend's observer stopped on an assertion after synthesis reached the queue. Its active-only lease check conflicted with legal pending transitions; the exact failing phase was not captured. No instruction WAV, ASR or task input completed. Owned editor closed; original GPU deployment stayed intact. | 1.96 s |
| CU-013, observer repair repeat | Qwen | Fail, incomplete selection | Actual TTS/ASR and seven dispatched inputs. Initial double-clicks selected `ready`; a later selection reached `alpha`, then Shift-Right extended only to offsets 0..6 rather than 0..12. No authoritative final before the action deadline. Buffer/file unchanged; editor and test frontend closed; deployment idle and unchanged. | 306.69 s |
| CU-017 | Qwen | Fail, wrong edit method | Actual TTS/ASR, four native inputs and an authoritative final. The model selected `copper`, physically pressed Delete, then Ctrl-Z restored the original buffer. Guarded XTEST receipts and the two fresh document states were verified, but Delete did not satisfy the required Ctrl-X cut. Disk unchanged; editor and test frontend closed; deployment idle. | 95.41 s |
| CU-001 | Ornith | Fail, interrupted stream | Actual Omni instruction synthesis and correct ASR. Repeated observe/verify calls dispatched no input; no selection or authoritative final. The stream ended with an incomplete chunked read. Saved file unchanged and owned editor closed; recovery proved a clean controller shutdown. | 231.31 s |
| CU-002 | Ornith | Fail, foreground confound, cancelled | Correct actual TTS/ASR. Pinned Moondream 3.1 produced a grounding receipt, but observations moved to unrelated windows. The proposed double-click was rejected for changed UI identity/revision before input. Operator cancelled only the test client; no actual input, final or selected-word proof. Owned editor closed and requests became idle. | 202.05 s |
| CU-004 | Ornith | Pass | Actual Omni TTS/ASR; guarded XTEST right-click opened an observed menu; physical Escape and independent positive popup-hidden evidence proved dismissal. Fresh document/file unchanged, pointer movement and authoritative final verified. The unresolved historical Escape receipt was retained. Owned editor closed and requests became idle. | 45.42 s |
| CU-030 | Qwen | Fail, setup, two attempts | Owned Calc launch passed its OpenCL guard, but initial native observation failed before speech or task input. Installing the GTK3 accessibility plugin did not make the immediate second attempt pass. | 1.47 / 0.93 s |
| CU-086 | Qwen | Pass | Actual 16 kHz Pulse monitor capture: 208,960 frames with quiet, click and steady noise; production VAD trace; zero submitted turns, stream events or GUI dispatch; call worker and owned Pulse module released. This is not physical-mic acceptance. | 13.16 s |
| CU-087 | Qwen | Fail, document identity oracle | Real Bluetooth capture, one VAD utterance, actual ASR, verified physical `type_text`, and exact visible `copper finch ready` in the owned editor. Strict initial document identity changed from generation 1 to 2 after focus transitions; that one criterion failed. Cleanup and idleness verified; all attempts retained. | 117.69 s |
| CU-087, latest physical repeat | Qwen | Fail, malformed model tool arguments | Real Bluetooth capture and ASR preserved the requested words. The next model continuation returned `{` as structured arguments; strict validation rejected the entire batch before execution. No typing or authoritative final. Cleanup and idleness verified. | 110.21 s |

CUDA process evidence and fresh broker discovery after each completed attempt
proved the owned Omni/Kev processes resident only on GPU 1, without a reported
new GPU fault. Each attempt has a separate `after-test-note.md` and JSON note.
The manifest retains all 200 slots; rows not recorded above remain pending,
not passing.

At this checkpoint, 19 distinct local model/case slots have been attempted:
five passed and fourteen failed; 181 remain unrun. These counts include setup
and interrupted attempts and exclude repeats. Ornith's context-menu case passed;
full desktop acceptance remains incomplete.

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
plays one 80 ms blip (the user selected 1 kHz in this headset), then waits for captured quiet before publishing its
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
next successful read. The staged-history repair is source-validated; complete
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
The next physical repeat passed cue readiness and produced actual ASR, then
stopped at a separate malformed model continuation. The native generation
repair still lacks a successful post-repair typing trajectory; no earlier
failed attempt is promoted to a pass. Retained server timing shows the last
continuation evaluated exactly its 4,096-token output cap. Upstream stop reason
and usage were not retained, so budget exhaustion is a hypothesis rather than
a proven cause of the malformed arguments. Raising the limit or repairing JSON
by guessing has not been applied.

A later read-only query of the actual loaded Qwen chat template showed that
omitting `enable_thinking` enables reasoning at its default `xhigh` effort.
The language adapter omitted that native flag for `think:false` in the Qwen
profile while suppressing reasoning from the returned stream. Two bounded
actual-weight parameter probes reproduced the mismatch: explicit false returned
the requested short answer in two completion tokens with no reasoning; omission
returned the same answer in 23 completion tokens with 89 reasoning characters.
Neither probe dispatched GUI input or counts as campaign acceptance. This
proves a native-control defect, but not the exact cause of an earlier malformed
or empty continuation. The language-only translation repair passed full source
validation: 1,992 tests in 94.09 seconds, with the adapter, regression tests and
runtime documentation unchanged during the gate (log SHA-256
`0871806ff8531a41b85d339c130cd2838781f2b2a6e8ff628b57515a3d17e0f9`).
The repair is pushed in `0f2267c`. After one fresh startup, actual authenticated
portal requests passed both native controls: false returned the requested short
answer with zero thinking in 1.032 seconds; true returned the answer and a
separate 83-character thinking channel in 1.462 seconds. Neither request used
audio, TTS or tools, and neither counts as a GUI case. The subsequent genuine
spoken CU-011 repeat above still failed at the independent native hit guard.
The native-control criterion is verified; physical file clicking remains
unverified. The currently healthy stack is retained for a separate bounded
hit-test diagnosis, without another model reload.

A separate read-only owned-Files diagnostic reproduced the rejected hit in
1.37 seconds, with no model calls or task input. The native provider returned
a panel outside the requested file's ancestry, then no child at the declared
point. A second 1.84-second discriminator recorded that panel as showing,
visible, non-defunct and childless; ownership, foreground, coordinate mapping
and target bounds stayed stable. Its full snapshot revision changed, however,
so the second diagnostic failed its postcheck and remains unverified. Both
owned windows closed politely. These probes do not establish task acceptance
or justify removing hit guards; the underlying provider failure remains open.
The matching installed GTK 4.14.5 source always sets AT-SPI showing/visible
flags, and its hit traversal does not apply the same presence filtering as
exported child enumeration. Positive flags therefore cannot exclude an
inactive branch. This is a provider limitation, not proof that the particular
returned panel was hidden. No coordinate substitution or guard bypass was
applied.

Before the first CU-013 run, its independent oracle was tightened to require
an observed alpha selection followed by actual Shift-arrow extension to alpha
copper. It rejects wrong documents, generations, scopes, stage order and
substituted selection methods. Independent observer IDs and portal IDs are
bound within their own bridges, with agreement on the owned window and buffer.
The oracle and portal structured-batch recovery passed the joint full gate:
2,098 tests in 95.92 seconds, with all six reviewed source fingerprints
unchanged (log SHA-256
`fee1e1be07c0cafc1cfd738f27e7a4d42dcf321cf4ea9671d469b4bc2a0b4c59`).
These checks do not add a passing live case.

The updated portal was tested on a separate loopback frontend against the
already loaded GPU 1 graphs because the broker supervisor cannot reload only
its Python children. The first observer required an active lease even during
legal preparation; it stopped on an assertion after synthesis began. A pending
phase rejection is supported by the code and timing, but the exact phase and
assertion line were not captured. That failed attempt stays recorded. The corrected observer
allows pending only for the same lease during monitoring, retaining active-only
startup/cleanup and all identity, scope and hardware guards. The second frontend
survived its complete live window and was then stopped. Only active phases were
sampled, so no captured pending transition is claimed. Every original worker,
supervisor and lease identity stayed unchanged. The second run exposed missing
key-chord backend attestation as well as the independent wrong/incomplete
selection. It produced no malformed-batch correction event, so live recovery
remains unexercised. No previous failure is promoted to a pass.

The keyboard receipt now reports its guarded XTEST route, retaining uncertain
post-attempt failures and every pre-input guard. CU-017's oracle now requires
cutting only copper at offsets 6..12, the exact `alpha  ready` intermediate,
and physical Ctrl-Z restoration in the same document. Seven previously accepted
false-positive trajectories are rejected. The joint full source gate passed
2,123 tests in 90.35 seconds with all seven reviewed file fingerprints unchanged
(log SHA-256
`015c70f082258aca7d91377ea0d9c533bf40560fb318639547ddc7cb844c5398`).
The subsequent genuine Qwen CU-017 run verified the new keyboard attestation
on physical Delete and Ctrl-Z, without another dispatch or fallback. Fresh
observations proved the exact `alpha  ready` intermediate and restoration to
`alpha copper ready`. The task still failed because the model substituted
Delete for the requested cut. Keyboard attestation is verified; cut/undo task
completion remains pending. This run produced no malformed-batch correction
event, so live structured recovery remains unexercised.

After Qwen cleanup and idleness, the controller performed one planned switch
to `robit/ornith-1.5-omni-audio-bridge:q4km`: Kev and the Qwen workers exited,
their CUDA allocations disappeared and the old lease was released before the
new scoped lease was acquired. Fresh readiness evidence proved Ornith and Kev
resident only on GPU 1, with the pinned Moondream 3.1 worker ready and shed.
The first local Ornith CU-001 run then failed as recorded above. The daemon
interruption was followed by a read-only recovery check; the prior Ornith
workers were absent. Their earlier readiness is historical evidence, not a
claim that weights remain loaded after the interruption. Recovery subsequently
proved a clean Kev-first controller shutdown, no remaining owned CUDA process,
an empty lease list and no reported new hardware fault. The inference hold and
fault recorder remained enabled. A fresh doctor and one supervised Ornith
startup passed, with new trunk/Kev identities resident only on GPU 1 and the
pinned Moondream 3.1 worker ready and shed. The bounded CU-002 attempt above
used that stack; cleanup and idleness were verified while retaining the healthy
deployment.

The helper and expanded perception runners passed the next frozen source gate:
1,872 tests in 99.84 seconds, with 221 checked source files unchanged (manifest
SHA-256 `b8b073c34165ca658dd47dfa60cf540a0bc56c2894f8b278a98611166d04b90d`).
This remains a source gate, not additional live acceptance. LibreOffice Calc
24.2.7 was installed for future spreadsheet cases: 43 approved new packages,
no existing package version changes, and a clean package audit. A subsequent
installation added only the GTK3 accessibility plugin, without changing
existing package versions. The two CU-030 setup failures above remain failed.
A separate read-only diagnostic later observed 376 native Calc nodes in 0.418
seconds with the GTK3 plugin loaded; the snapshot was truncated and mostly
menus, so spreadsheet cell access is not yet verified.

Cleanup previously used `xdotool windowclose`, which destroys the X window.
An observed Calc abort during cleanup exposed this harness defect. Cleanup now
sends the application's advertised `WM_DELETE_WINDOW` message to the exact
owned window after checking PID and process start identity twice. Closure must
still be observed separately; refusal or timeout remains failed cleanup.
The bounded live diagnostic verified polite Calc closure, process exit and no
abort or unexpected-destruction error in 1.232 seconds total. It submitted no
model requests or task inputs. Full source validation passed 1,950 tests in
90.20 seconds with 249 source fingerprints unchanged (validation log SHA-256
`9b0ac11b3cc6ee280e182a88621bb70545fba949fec1899d8b1a57389b4f3f8c`).

Remote Egg-host Ornith runs are recorded separately; they do not fill local
GPU 1 slots or establish pinned Moondream 3.1/Kev acceptance.

The source inventory at `d6c0b2e` routes all 100 cases to runner modules, but
does not establish that all are executable. Twelve cases still have concrete
capability or evidence gaps: CU-012 text-range dragging; CU-014/015 held-modifier
clicks; CU-022 physical window move/resize; CU-023 minimized-window restoration;
CU-029 media slider/playback evidence; CU-059 exact pointer passage selection;
CU-073/077 deterministic pending-point barriers; CU-074 repeated-Apply setup;
CU-080 interrupted-dispatch barrier; and CU-089 acoustic interruption evidence.
The other 88 have candidate execution paths, not 88 live passes. These source
gaps and all unrun model/case slots remain incomplete.

The next bounded architecture change addresses repeated UI working context,
which occupied 86–90% of recorded remote tool content. A request-local origin
ledger now retires older server-produced maps and candidate offers only after
a fresh canonical `observe`/`verify` result. Execute/workflow bodies, including
cached results, cannot trigger retirement. The newest pair, raw traces and
immutable action/policy/effect/verification receipts remain intact; explicitly
referenced controls retain bounded historical readbacks. Unknown prefix changes
disable retirement for that request. This is a local projection policy, not a
verbatim claim about demonstrated Omnius behavior or a fix for wrong model
choices.

The frozen source gate passed 2,184 tests in 94.65 seconds, with all six reviewed
file fingerprints unchanged and every validation gate passing (log SHA-256
`974f2aa0250f41b2d811a692700a428a116ea4fbc7cfb0bbdff5d851aa47339d`).
Source checks include both actual portal routes and detached streaming, forged
client-origin rejection, cached-result freshness, exact receipt preservation,
new-request isolation, accepted ASR/native think preservation and prefix-repack
fallback.

One CPU-only portal frontend then loaded `68cc25c` against the existing Ornith
graphs and exact GPU 1 lease, without restarting a GPU worker. The genuine
CU-004 run above passed. Paired measurements at a fresh observation reduced
the same working history from 84,729 to 34,671 UTF-8 JSON bytes, with six full
maps becoming one. Across the run, nine older snapshot bodies were retired.
All nine measured newest assistant/result pairs and immutable receipt digests
stayed equal; the raw stream remained independently retained. These are byte
measurements, not token counts or a causal latency comparison. The task used
native semantic targeting; it did not exercise a Moondream point or Kev choice.

The editor closed, both portal queues and trunk slots became idle, and TTS
weights were shed. The test frontend was then stopped while preserving the
original healthy Ornith workers, lease, inference hold and fault recorder.
Historical unresolved action statuses were not rewritten. Summary SHA-256:
`a9f6d8443235d1dce1e46c6b606a694d8dc717fdd959a13488aa6c626dc4e216`.
Projection-metrics SHA-256:
`17a386fd2ca8be613f0074269066f604627f905db9dea3e2e64e2d9b73acb8f7`.
