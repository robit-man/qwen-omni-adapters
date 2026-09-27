# Jetson memory-pressure tool recovery

Status: active repair, 2026-09-26.

This file is the durable implementation contract for the live Egg failure. Do
not close the incident from a unit test or a manually staged download. Close it
only after the model completes the original spoken request on the Jetson.

## Preserved failure evidence

- The speaker asked for the Android Hell music file on the Desktop.
- The current file is `/home/egg/Desktop/04 Android Hell.mp3` (3,678,417 bytes).
- The language prompt was roughly 3K--4K tokens in a 16K resident transformer
  context. Context exhaustion did not cause the failure.
- The indicator's `Active context: 16,384 / 65,536` label described reserved
  physical context capacity and its configured ceiling, not used prompt tokens.
- Jetson unified memory had about 0.5--0.9 GiB available. NVMAP reported about
  17.1 GiB allocated: comprehension 8.32 GiB, pointing 3.70 GiB, and resident
  TTS 3.75 GiB. GNOME Shell PSS was about 3.44 GiB.
- Shell requires the 2 GiB hard floor plus a 0.5 GiB launch reserve. All three
  model-authored shell calls were rejected before execution with
  `resource_pressure`.
- Exact-argument duplicate detection treated rewritten shell commands as new
  attempts. Three nonproductive rounds then triggered
  `MAX_STALLED_TOOL_ROUNDS` and returned a 502.
- After the first concrete shell call, follow-up routing discarded
  `workspace_file`; the bounded filesystem route could not recover.
- `workspace_file` was itself classified as `bounded`, so it would have been
  rejected below the 2 GiB hard floor even if rediscovered.
- Context compaction cannot unload resident pointing/TTS weights or release a
  llama.cpp KV allocation. It therefore could not resolve this host-memory
  admission failure.

## Repair checklist

- [x] Start the Moondream control plane cold; load its graph only inside an
      explicit point/observe request and shed it in `finally`.
- [x] Start the TTS control plane cold; reuse its framed worker only within one
      WAV/PCM response and close it in `finally`, including cancellation and
      failure paths.
- [x] Remove daemon startup gates that require idle pointing/TTS GPU residency;
      diagnostic smoke now requires comprehension resident and TTS shed.
- [ ] On the Egg, verify NVMAP before, during, and after one pointing call and
      one cloned-TTS response, with comprehension resident throughout.
- [ ] Rename indicator context telemetry so reserved capacity is never shown as
      used or full context.
- [ ] Make `workspace_file` genuinely fixed-footprint (bounded traversal and
      bounded text reads) and available below the model-growth floor.
- [ ] Return a typed capability-change receipt for memory-pressure rejection;
      it must say the attempted action did not execute.
- [ ] Retire a pressure-rejected executor for the remainder of the foreground
      turn, including through subsequent `tool_search` calls.
- [ ] Preserve a discovered/staged file-delivery route across a failed sibling
      branch.
- [ ] Replace the browser-specific foreground recovery instruction with a
      generic typed-capability recovery instruction.
- [ ] Add regression coverage reproducing shell pressure -> filesystem list ->
      exact file delivery without three shell retries.
- [x] Remove the generic consecutive-nonproductive-round cap from JSON and
      streaming agent loops; repeated calls execute normally while typed
      per-tool limits, request timeout, and disconnect cancellation remain.
- [ ] Run the complete validation gate after implementation.
- [ ] Deploy to the Egg without unloading comprehension; pointing and TTS must
      be cold before the foreground tool request.
- [ ] Repeat the original request and verify one exact Desktop discovery plus
      a session-authenticated Download result for `04 Android Hell.mp3`.

## Non-solutions

- Do not lower the global OOM safety floor merely to make the test pass.
- Do not hard-code `Android Hell`, the Egg username, or the Desktop filename.
- Do not use lexical suffix checks or canned command rewriting.
- Do not claim context compaction fixed residency pressure.
- Do not declare success from direct API calls that bypass model tool choice.
