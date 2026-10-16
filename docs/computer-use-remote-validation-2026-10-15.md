# Remote Ornith desktop validation: 2026-10-15

**Full acceptance has not passed.** These Egg-host results are separate from
the local GPU 1 campaign. Both hosts use Tegra device-handle residency checks.
Their configured point head is Moondream 2, and Kev is disabled; these runs
cannot establish pinned Moondream 3.1/Kev acceptance.

## Egg31

The active checkout was still `5bdf6957`, without the native computer-use
layer. After staged ARM validation passed 1,931 tests, the checkout was updated
to `7e1f814`. Only Python adapter/portal services were reloaded. The supervisor,
Ornith comprehension worker, pointing wrapper and cold TTS wrapper retained
their process identities. Untracked files were preserved. No automatic update
timer was found or enabled.

The first owned Gedit selection attempt failed at native observation in 1.33
seconds, before TTS, ASR or input dispatch. The original file remained unchanged
and the owned window closed. A read-only provider diagnostic proved that the
installed AT-SPI 2.44 lacks `clear_cache_single()`.

Commit `c16c815` adds capability-based cache invalidation: prefer the newer
single-node API, otherwise use the older documented recursive API. Missing APIs
and provider exceptions still reject the observation; exceptions are never
retried through another API. Read budgets, freshness and input guards remain.
Full source validation passed 1,963 tests in 92.31 seconds (log SHA-256
`11e4dd978d5853617622dbe40d4bbf97a27f2bf533dca0dd374d2c2edededea0`).
After a source-only update, a fresh observer successfully read the owned editor.
The single genuine TTS-to-ASR repeat with `think:true` nevertheless failed after 195.69 seconds,
including its 180-second action deadline. ASR preserved the spoken intent;
Ornith produced five abstained outcomes, no dispatched input and no
authoritative final. The original 19-byte file stayed unchanged and the owned
editor closed. The observer child started after the update and imported the
exact validated source; supervisor and model worker identities stayed unchanged
without a service reload. Health, idleness and Tegra residency passed afterward.
This verifies the cache compatibility repair, not task completion. The separate
edit/Save chain was not attempted after the selection gate failed.

The retained trace explains the five abstentions: Ornith dropped the `atspi:`
prefix from observed document identities, yielding zero eligible exact targets.
Read-only enumeration against that same captured state yields one deterministic
candidate with the complete observed ID. This was `no_matching_control`, not
an ambiguity requiring Kev. Incorrect selection endpoints and confusion between
node, step and offered action IDs were also retained. No guessed ID repair or
weaker dispatch guard was applied.

A separate default-mode `think:false` selection scenario passed in 127.88
seconds on unchanged `c16c815`, using actual generated speech and ASR. An exact
offered candidate dispatched native AT-SPI Text selection at offsets 0..18;
fresh independent verification proved that selection, the unchanged 18-character
buffer and exact original 19-byte file. All six selection criteria passed;
the owned editor closed and worker identities, health, idleness and Tegra
residency remained intact. This establishes native accessibility selection,
without a physical keyboard or pointer claim. It does not replace the failed
`think:true` result or establish an edit/Save chain pass.

The later `0f2267c` update passed deployment doctor with explicit deployed
Ornith model/language tags; an initial invocation had checked the CLI's default
Qwen profile and is retained as an operator invocation failure. A single
code-only adapter/portal reload completed in 4.98 seconds, preserving the
supervisor, comprehension, pointing and TTS wrapper identities. The prior
harness state was restored; health, idleness and residency passed. The separate
physical edit/Save attempt is recorded below.

The separate `edit-save-line` runner scenario is source-validated: 2,030 tests
passed in 96.28 seconds, with runner and new oracle tests unchanged during
validation (log SHA-256
`888c5e1b987d64aa606c451b5a2fde8c428d17a091566bd069082358f101b30a`).
Its spoken instruction asks for no Return. Acceptance requires actual keyboard
typing, a physical click on the exact Save button, a fresh unchanged document
identity containing the exact 18 characters, and exact 19-byte saved content.
Semantic Save, Save As, wrong newlines and intervening contradictions fail.
These source checks are not a live edit/Save result.

The first genuine `edit-save-line` attempt on `e27834f`, with `think:false`,
failed after 87.57 seconds. Actual generated speech and ASR preserved the
instruction. One native observation completed; the next model response contained
malformed workflow JSON, rejected as a whole before any typing, Save click or
pointer input. The owned editor closed, the original 19-byte file stayed
unchanged, and service health, eventual idleness, worker identities and Tegra
residency passed. No fresh terminal UI snapshot followed the protocol error;
none is claimed. The proposed selection endpoint was also incorrect, but that
proposal never executed. Output length and upstream stop reason were not
retained, so truncation is not established. The failed attempt and after-test
note remain preserved (summary SHA-256
`5c72c0e36eb74acc6761d4a7a63ebe73a2957c9cf7a0cd2f7b4e8e43138b24b8`).

The portal now allows one model-generated correction of a malformed structured
batch before any call in that batch executes. It preserves earlier receipts,
the accepted spoken command and execution scope; it never guesses JSON or
replays the rejected proposal. Missing speech, visual media or prior output
audio forbid recovery. A second malformed batch still fails. Focused checks
and the joint full gate passed: 2,098 tests in 95.92 seconds, with all six
reviewed source fingerprints unchanged (log SHA-256
`fee1e1be07c0cafc1cfd738f27e7a4d42dcf321cf4ea9671d469b4bc2a0b4c59`).
Live post-change recovery and edit/Save acceptance remain unverified.

After `b9bf58a`, one code-only Python reload preserved the supervisor and all
model/wrapper identities. A verifier initially expected module launch arguments;
the actual fixed script paths, working directories, start times and hashes
proved adoption without another reload. The genuine `think:false` Save repeat
failed at its original 300-second action bound: 320.15 seconds total, including
17.739 seconds of TTS. ASR was correct, but there was one unique unchanged focus
input, no typing, Save or pointer input, and no authoritative final. The owned
editor closed; file bytes, workers, health and eventual idleness were preserved
(summary SHA-256
`c79f130622e5e55ce2356e9f2cca2ae948f1365bdcd43393d37373b89370ce43`).

Exact trace analysis found two unoffered execute IDs and four correctly copied
current offer IDs. All four focus offers shared one idempotency key: the first
executed, while three returned explicitly replayed receipts. No old-offer,
stale-scope/revision or argument-shape failure was observed. Thirteen snapshots
of the same owned revision occupied 162,162 of 180,203 recorded result bytes;
this establishes duplication without proving it caused the focus loop. There
were zero correction resets or typed recovery events: live malformed-batch
recovery was not exercised by this repeat. Full edit/Save acceptance remains
failed.

## Egg41

| Scenario | Actual evidence | Result |
| --- | --- | --- |
| Calculator, Ornith `think:true` | Actual Omni TTS in 17.04 seconds, correct ASR, two verified physical clicks using offered action IDs, pointer movement. | Failed at the 300-second action deadline; 318.75 seconds total. No authoritative final or fresh exact-answer proof. Owned window closed. |
| Files chain, Ornith `think:true` selected | Owned window and initial filesystem observed; actual instruction synthesis requested. | Failed at the 180-second synthesis timeout; 183.08 seconds total. No instruction WAV, ASR or task input. Owned window closed. |

Both attempts used `robit/ornith-1.5-omni-audio-bridge:q4km` and generated
speech, without substituting a typed task goal. The configured point head was
`vikhyatk/moondream2@9a7d4024050840e001defacec2b00727e89149e6`.
The clean source stayed at `7e1f814` and the same Ornith worker remained
Tegra-resident throughout both attempts.

Calculator rejected seven attempts to execute a model-authored seven-character
step ID rather than a 32-character offered action ID. Those IDs were invented,
not reused stale offers. Later valid offered IDs executed successfully. The
last cancelled continuation contained 55,570 prompt tokens and took 82.54
seconds to prefill; no context truncation was reported.

The 15 completed rounds retained 142,945 bytes of model-facing tool content.
Repeated UI snapshots account for 86.25% of that content: ten snapshot
occurrences contain only three revisions. This proves historical snapshot
accumulation; it does not establish that accumulation caused the incorrect IDs.
All ten snapshots belonged to the exact owned Calculator window, so this
measurement was not confounded by another foreground application.

Files synthesis continued after the client timed out. The same TTS child was
still running after seven minutes, with only 97 frames generated at 7 minutes
7 seconds and 358,667 major page faults in a captured sample. Available memory
was about 5.07 GiB of 29.95 GiB, with 1.56 GiB swap used. These measurements
establish severe slowdown without proving a single memory or driver cause.
The blocking server route has separate startup/generation deadlines of up to
900 seconds; the client timeout did not promptly cancel its work.

The child exited before guarded recovery could send a signal. No signal or
service restart was performed. Subsequent checks proved no TTS child, zero
portal activity/queue, and the unchanged healthy Ornith worker. Exact server
completion duration was not recovered and is not claimed.

Compact notes and measurements are retained privately under
`runtime-data/gpu1-acceptance/remote-egg41-20261015/`; full speech and UI journals
remain on the remote host. Setup-only output-directory and transient observer
failures are preserved separately and do not count as executed speech tests.

After `0f2267c` was pushed, Egg41's external updater changed the clean checkout
and restarted its services. New adapter/portal processes started after that
checkout update, executed from the expected repository, and matched the pushed
source hashes. Health and Tegra residency passed. Another task was active at
the latest check; no new inference, GUI action or lifecycle action was sent by
this test operator. This proves deployment adoption, without claiming a new
native-thinking or desktop task acceptance result on Egg41.

A later read-only check confirmed clean `e27834f` adoption, healthy services,
Tegra-resident Ornith and zero active, inflight or queued requests. No TTS child
was running. No inference or reload was sent by this operator; this remains
deployment evidence rather than an additional desktop test result.

## Egg31 cut and undo at `d6c0b2e`

The checkout was fast-forwarded to `d6c0b2e` after the preceding test became
idle. The existing supervisor, comprehension, pointing, TTS, adapter and portal
workers were preserved. Fresh native bridge process identities and source
hashes proved that the guarded keyboard receipt change was loaded without a
GPU-worker restart.

The genuine Ornith `think:false` CU-017 run failed at its original 300-second
action bound, taking 317.01 seconds including 11.84 seconds of actual Omni
instruction synthesis. ASR preserved the cut, verification and undo request.
Five unique native inputs selected `copper`, incorrectly typed `ready` over
it, selected the whole changed document and finally deleted it. Fresh native
observations recorded `alpha ready ready` and then an empty buffer. No Ctrl-X,
Ctrl-Z, correct cut intermediate, restoration or authoritative final occurred.
The saved 19-byte file remained unchanged.

Guarded Ctrl-A and Delete receipts carried the new
`x11_guarded_xtest` / `native_key_input_invoked` attestation. That receipt
criterion passed independently; the task failed. The owned editor closed,
natural activity drained to zero, worker identities and Tegra residency stayed
intact, and no operator task action, cancellation, retry or reload was used.
The historical failed attempts remain unchanged. No malformed-batch correction
event occurred, so structured recovery remains unexercised. This host still
uses Moondream 2 with Kev disabled; this is not pinned Moondream 3.1/Kev or local
GPU 1 acceptance.

Private evidence is retained at
`runtime-data/gpu1-acceptance/remote-egg31/CU017-ornith-false-20261015T220610Z-af1867/`.
Summary SHA-256:
`7f2c99362b7edf2b36c8ff0c3488093589165a566287e91b9bf63aa64e9b89c2`.
After-test note SHA-256:
`6abb68d6d48447392d1741eff27cb2d20eaf39dfe8e3c2a9c7198298a1840b50`.

## Egg31 punctuation typing at `d6c0b2e`

The independent CU-019 `think:false` run failed at its unchanged 300-second
action deadline, taking 317.43 seconds total. Actual Omni TTS took 9.64 seconds
and produced a valid six-second WAV. The source instruction was
`Type exactly: Copper, finch! Then add a newline and type: Ready?`;
ASR returned `type exactly copper finch then add a new line and type ready.`
The written fixture did not explicitly speak punctuation or capitalization
directions. This is an input-fixture ambiguity, not proof that ASR dropped
spoken words such as “comma” or “question mark.” The original expected buffer
was retained without normalization.

Independently of that ambiguity, the model made one unique focus input and no
typing. Six fabricated execute IDs were rejected as unknown candidates; an
earlier malformed workflow was rejected for missing valid effect predicates.
The fresh owned document remained empty, no authoritative final arrived, and
the keyboard criteria failed. The owned editor closed, worker identities and
Tegra residency remained intact, and the active harness was preserved. The
outstanding request drained naturally to zero activity, inflight and queued
requests at 22:40:01 UTC before further work.
No operator correction, task replay, source change or service reload occurred.

## Egg31 Flatpak foreground observation at `6f39e49`

The exact staged source passed 2,842 ARM checks in 163.33 seconds and the
explicit Ornith deployment doctor. One supported Python-only reload adopted
the new adapter and portal. Supervisor, resident Ornith, pointing, TTS and
tunnel process starts were preserved, as were configuration hashes and all
18,030 untracked paths with their complete NUL-separated digest. The previously
active harness was restored. Health, native Tegra residency and zero queue
activity passed after adoption.

One unchanged production `NativeUiAdapter.observe` then passed against the
existing, independently rechecked active Firefox XID `60817411`, host PID
`1510469` and start `10898437`. It used a fresh bridge and exact window scope,
with pointing disabled and no prepared action. It returned 71 nodes without
truncation, including a real `DocumentWeb`/DocURL capability. The whole observer
driver took 4.239 seconds, including pre/post checks and bridge startup/cleanup;
isolated native-read latency was not measured.
The verified namespace-provider digest bound the native revision; raw UI text,
URLs and private proof metadata were excluded from the retained report.

There were zero input, focus, browser-close, pointing or model calls. The bridge
closed and the original Firefox window, all mapped client IDs, source, worker
starts, configuration and untracked inventory remained unchanged. This passes
foreground identity/observation only. Inactive Flatpak switching and owned
Flatpak launch/cleanup remain unavailable, and no remote long task passed.
The deployed remote stack still uses Moondream 2 with Kev unavailable; it is
distinct from the local Moondream 3.1/Kev condition.

Evidence is under
`runtime-data/gpu1-acceptance/remote-egg31-namespace-adoption-20261016T053907Z/`.
After-test note SHA256:
`d93f63b597e079fc11c72e2e489485dd3b2d3ad13ee9170838dd67a3532d4726`.
The 15-file evidence manifest SHA256 is
`243ed9fb785fc0fd14e9a564de52e595121528c601b20b2518d0441303e394c4`.
