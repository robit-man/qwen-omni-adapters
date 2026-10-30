# 076: Test semantic previews at the control-discovery boundary

Status: proposed next bounded experiment; no implementation, adoption or new model
calls. Cleanup075 is frozen at its separate passing live-lifecycle milestone.

## Failing acceptance criterion and evidence

On the five frozen scenes, the loaded Qwen first-stage planner must choose useful,
source-grounded immediate operations without losing constraints. The difficult
navigation scenes must advance beyond activating their already-active window.
Existing useful prerequisites must remain useful.

The074 real comparison failed: baseline3/5 useful versus candidate2/5, both
navigation cases unchanged, one held-out regression and zero catalogue selections.
All ten actual requests/responses and their source, token and GPU-epoch joins were
independently verified. Final review SHA256:
`bc464491abf8807dc2a019e548c0ae15e7f1b0b272c56984878ac820e8578e0c`.

Actual candidate wires expose14 role/count groups covering285 eligible records,
but no accessible-name examples in those groups. Full projection omits the
Workbench control. The model explicitly recognizes the active window yet selects
it again. This is observed before Kev or input dispatch. It does not establish
that missing names are the sole cause: coarse pointer/activation semantics and
historical state remain alternative explanations.

## One bounded hypothesis and patch

Test whether exact accessible-name excerpts make the existing discovery choices
useful. Start from the unchanged582-source074r3/075r2 cohort
`e714963144bb550b95e63c51e5359ebb67e7c7090bfda9e9c4ce9d543fe13c7a`, whose
full gate passed5,211 tests. Permit changes only to
`portal/ui_control_catalogue.py` and its existing dedicated test module.

Enrich existing root role-group entries with optional `name_examples` and mark
them as partial accessible-name excerpts. Use the same eligible source records,
original traversal order, nonblank names and preserved duplicates. Each excerpt
is an exact prefix of at most120 UTF8 bytes, without a split codepoint. Each
group's canonical name array is at most256 UTF8 bytes. Preserve the original
complete offered group list first; add examples only while the entire view stays
within4096 UTF8 bytes and16 entries. If metadata cannot fit, retain the original
view. No goal matching, label sorting, deduplication, app-specific policy or role
whitelist is permitted.

Keep references, counts, ordering, source identities, finite schemas, system and
caller instructions, full projected controls, histories, expansion API, fresh
binding checks and input authority unchanged. Names are untrusted evidence, never
action aliases. Keep .95,40 phases, three failures and the current model profile.

The source-only prototype measured3856/2490/3126/4087/4087 bytes across the five
scenes. This is feasibility evidence, not a model improvement. Its frozen proposal
is `/home/roko/Documents/qwen-omni-research/2026-10-17/control-discovery074-actual-negative-diagnosis-20261017/proposal.json`
(SHA256 `9337898d54adc83e162fb92606ba5e0c75c11e10b5fad88e4f8f7fe704567d12`).

## Gates and one evidence window

1. Preserve582 source, previous failures, exact fixtures and oracles. Run focused
   Unicode/size/order/duplicate/blank/ineligible/foreign/stale-source controls and
   prove no existing reference, schema or execution authority changes. Obtain an
   independent source/method review and the required final full source gate.
2. Freeze a paired comparison against the five retained actual074 candidate
   outcomes. Mark those as historical observations, never new POSTs. Use the
   identical five scenes, settings and stop rules; allow at most ten new candidate
   POSTs from the already frozen source sequences, no retries or resampling. The
   independent method must preregister exact calls, wire parity outside preview
   metadata, token admission and per-call/aggregate deadlines before execution.
3. Use the current healthy GPU1 lease/worker epoch only after fresh admission.
   Keep this comparison read-only and first-stage-only; host-simulated catalogue
   continuation is not a second model, physical effect or whole-task result.
4. Require useful navigation decisions and preservation of all three positive
   controls. An inspection counts only as discovery; exact full binding must
   follow through the unchanged API. Independently grade all outcomes and retain
   the first result, including regressions.

If useful names are present but the model repeats active-window activation or
regresses, preserve the failure and stop this patch cycle. Do not tune labels,
thresholds, histories or schema order in the same condition. Only a passing
comparison admits the unchanged whole-task check, with fresh075 admission,
actual imported-source proof and original task criteria. Local Ornith, both Eggs
and the original breadth matrix remain separate required work.
