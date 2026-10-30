# Actual Ornith GPU1 LHN-047 baseline

The unchanged original task failed. This was an experimental cross-model
baseline using frozen077 source, not adoption of failed077/078 repairs and not
a controlled comparison with the earlier073 Qwen whole-task run.

The loaded profile was `robit/ornith-1.5-omni-audio-bridge:q4km` on GPU1 UUID
`GPU-170a99ee-850f-2182-1050-4e8d3c87b6b0`. One supported cold handoff drained
the earlier Qwen workers, allocations, heartbeat and lease before starting the
Ornith profile. Actual Ornith model/projector arguments, its new eight-worker
epoch and new scoped lease were independently checked. The CPU frontend used
the frozen582-file solver inventory, separately from the actual MAIN adapter.

## Task result

The original LHN-047 signature remained
`840e87f62a4aef5d747ddbf83b017e6647423896797ff5bfa9cc3a46534462aa`.
No goal, criterion, oracle, phase limit, failure limit, timeout or confidence
threshold changed. Input was text, as currently requested; no microphone,
ASR or TTS was used for the task.

The task ran for149.034 seconds and returned an authoritative blocked final.
There were14 matched actual adapter request/response pairs: seven first-stage
and seven second-stage requests, all HTTP200 with `think:false`. Six native
inputs switched windows. Phases5 and6 repeated previously verified transitions;
phase7 requested replanning without dispatch. The three-consecutive-failure
bound then stopped the task. This was not a model transport outage or an
observed external focus-ownership violation.

None of the five required outputs exists. No input document acquisition,
editorial changes, source-page navigation, keyboard typing, physical click,
reopening, archive or memory completion is supported. The independent ten-criterion grade passes only C08, input/unrelated-state
preservation: 1/10 criteria, 0/5 outputs. Its exact raw archive is sealed and read back. Configured
Kev and Moondream availability receives no task-use credit: there were no
corresponding delegation receipts. Real cursor movement does not substitute for
a verified physical click.

## Supported diagnosis and limits

All seven first-stage proposals reached their corresponding second-stage
requests unchanged. The planner retained the objective, unmet requirements,
previous operation and host results. Phases6/7 also received repeated-transition
feedback. Missing handoff context therefore does not explain this run.

There are observable semantic inconsistencies. Phase3 names an Input window
but binds Firefox; later proposals misdescribe current focus and claim an
`Input.txt` entry that the observation does not contain. These are observed
plan/target and evidence mismatches, not proof of the model's internal reason.

A distinct compression ambiguity is source-proven: the raw Input control has
the current window's `scope_id`, but its compact representation exposes only
`active_window:false`. Genuine inactive window roots have the same flag, and
the serialized child ancestors omit the window root. There is no explicit
child-to-current-window join. This is a bounded candidate hypothesis for a
separate paired test, not a causal conclusion or an implemented repair. Raw
AT-SPI evidence supplies no typed folder/file/MIME field; do not invent one.

## Application cleanup and unchanged runtime

Fresh mandatory default-ledger admission preceded setup. The run registered
three exact test windows before activation, then politely closed all three.
Fresh postflight independently verified their absence and preservation of all16
pre-existing managed window identities. The default ledger has zero pending
and five resolved runs. Shared application processes remained alive; no
process-name cleanup, user-window closure or unsaved-buffer discard occurred.
This does not prove historical ownership completeness or cleanliness on Eggs.

All eight actual Ornith core workers and both CPU frontend workers retained
their exact PID/start/UID identities. Trunk and Kev remained resident only on
the leased GPU1; the recorder stayed active and both portals were idle. Fresh
byte/path checks preserved all582 solver files and installed derived helpers.
The existing5240-test source gate was reused after exact parity, not rerun.

An earlier setup attempt failed before task submission because the private
candidate runtime-data ancestor was0775. The unchanged ownership directory
guard correctly refused it. Only that private directory was changed to0700;
the failed evidence remains failed and preserved. It is not task success.

## Durable evidence

The independent manifest/readback covers135 retained files (18,103,715 bytes)
and617 external immutable source/predecessor references. Root verified every
manifested file and external reference against exact bytes and SHA256.
Manifest `d378bd1f418e1a66edb1bc02289435524bde6bdb8a1c17e4a950db78e5c40a0d`;
readback `985d106cea4e67f0e12ce50b4484f4554cb55cafe72de5c137602e9a7b74946b`.

- Independent ten-criterion grade: research root
  `gpu1-Ornith-whole078-independent-terminal-review-20261017/independent-grade.json`,
  SHA256 `a17aa9bfbb393322d66640a25b7b4562856374ec6d0fa8c0c1ab0bb3a22e31be`.
- Independent source/method/lifetime audit SHA256
  `ff3879735db2ed9ec68c8ffde63b107eb5f06db217eb7f4221060ebf48072010`.

Research root: `/home/roko/Documents/qwen-omni-research/2026-10-17`.

- Actual run: private077 candidate
  `runtime-data/gpu1-acceptance/long-horizon/ornith-LHN-047-077baseline078-20261017T230909627400Z`;
  summary SHA256 `3b85dde39e01d5a97e5118a17be1890f0c288a86baef1e871ec1fd9ae31c4efb`.
- Original after-test note SHA256
  `9280cdfe2e3c8b53578e8a72efc74edb949adecb4721e1c4d947fa27c9c3c655`.
- `gpu1-Ornith-whole078-root-postflight-r2-20261017/postflight.json`:
  `b06b29fc7a6d66d95426583b317befe364f78448657de345f36694d34d5be28a`.
- `Ornith078-first-six-calls-readonly-diagnosis-20261017/diagnosis.json`
  includes distinguished initial-six and remaining-eight call analyses:
  `084414645e61f8d501004da43600869d7e7cc614b9607f5f5eade2d4a1874385`.

The whole requested outcome remains unverified. Preserve this baseline before
another repair. Keep cleanup075 publication separate from any unaccepted
planner change; both Egg whole tasks and the original breadth matrix remain owed.
