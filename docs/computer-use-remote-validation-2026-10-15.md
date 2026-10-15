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
harness state was restored; health, idleness and residency passed. A separate
physical edit/Save scenario remains pending.

The separate `edit-save-line` runner scenario is source-validated: 2,030 tests
passed in 96.28 seconds, with runner and new oracle tests unchanged during
validation (log SHA-256
`888c5e1b987d64aa606c451b5a2fde8c428d17a091566bd069082358f101b30a`).
Its spoken instruction asks for no Return. Acceptance requires actual keyboard
typing, a physical click on the exact Save button, a fresh unchanged document
identity containing the exact 18 characters, and exact 19-byte saved content.
Semantic Save, Save As, wrong newlines and intervening contradictions fail.
These source checks are not a live edit/Save result.

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
