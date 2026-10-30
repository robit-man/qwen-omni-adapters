# CU.TRACE.RETENTION.063 — verified isolated storage repair

Status: storage/source acceptance PASS; not integrated or deployed. Failed062
semantic accuracy and all desktop/model-task criteria remain unchanged.

The implementation retains exact canonical traces up to 512 KiB in private,
session-generation/task-epoch-owned sealed records. Bounded references replace
oversized inline audit events; private listing/expansion verifies the complete
body, reference and ownership. The existing 65,536-byte inline guard remains.

Lifecycle controls cover Trash, expiry, close during reads, late callbacks,
atomic lease renewal, quotas, exact Unicode pagination and fresh-process
recovery. Native activity renews an already-bound session through the existing
bounded worker without filesystem IO in native lookup. Archive failure cannot
preempt a valid native result. A bound epoch survives helper recreation and
cannot silently mint a replacement generation after expiry.

Concrete read-after-close, inaccurate clear-count and lease-shortwrite failures
were preserved before their narrow repairs. Independent real-file/thread
controls confirm those repairs. The original five full-store saved-trace cases
now recover every exact body. A 118,376-byte resource-union fixture reconstructs
across 22 pages without clipping. These are explicit CPU fixtures, not actual
model requests.

Validation:

- 377 focused CPU controls pass.
- Independent storage/lifecycle review passes.
- Separate mixed-mode CPU acceptance passes 4/4: visual acknowledgment then
  fresh native exact-value write; lost acknowledgment prevents write; focus
  alone cannot verify a failed setter; expired binding cannot rebind silently.
- Full `./scripts/validate.sh`: 5,897 tests pass in 314.95 seconds; all script
  gates pass. The 568-file source is unchanged before/after validation.

Candidate: `/tmp/qwen-trace063-20261017`, inventory SHA256
`cd474ff8d4fccf6e43f645abb033e5b720942846eb8cf1645ea234a97d6c10df`.
Exactly seven approved changed/new paths relative to the frozen062 baseline;
model instructions, semantic quality, policy and input guards are unchanged.

Durable report:
`/home/roko/Documents/qwen-omni-research/2026-10-17/trace-retention063-implementation-20261017/implementation-and-acceptance.json`,
SHA256 `6b4ed1b64c480e623eefbe9b5706a31f85d254219cd169480e9b5e4015d6595e`.
Full gate receipt SHA256
`5a1cf265734a04c0aa742c38a6601bd11e3c5a99c3f563f70017edd0a3fee809`.

Release still needs explicit integration with the visual source dependencies
and its combined regression/actual acceptance. Publishing only these hooks
would leave missing modules; publishing failed visual semantics would not
satisfy desktop acceptance. No main commit/push, production adoption, actual
model or GUI action occurred under063. Stop storage edits at this milestone.
