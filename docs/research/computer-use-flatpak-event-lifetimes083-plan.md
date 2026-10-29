#083: Preserve X11 event lifetimes during subscription

Failing acceptance: a genuine CreateNotify followed by DestroyNotify before a
single event drain must not poison ownership discovery by subscribing to the
already destroyed window. The real isolated-Xvfb experiment reproduced
BadWindow3/request2 at events:post_sync. Its paired live-window control passed.
Evidence: R/flatpak-create-destroy-causal-20261017/report.json, SHA
0488b96228d94a969347837649045a29a07a68d34fe7b45f5d9ea67c73a03eef.
The exact Egg31 historical event sequence remains inferred, not proven.

Bounded hypothesis: collect one complete bounded event batch before new-XID
subscriptions; skip a subscription only for an actual ordered Create lifetime
proved destroyed by that same batch. Retain the original ordered events and
grant no ownership or selected-window authority for the vanished lifetime.
Preserve XID reuse, selected existing windows, synthetic-event rejection,
bounds and fail-closed protocol behavior for unknown disappearance or other
errors. Do not clear a poisoned connection or relax PID/session/scope guards.

Start an isolated source candidate at exact published8bbb605, covering457
source files. Run the required full baseline before editing. Prefer changes
only to runtime/owned_flatpak_firefox.py and focused tests. Do not combine
private082/080 source or alter source/services on either Egg during construction.

Validate focused negative controls, independent review and the full after-edit
gate. Reuse the exact real-X11 causal protocol on a private authenticated Xvfb
display, checking the formerly failing and live-window cases with actual event
batches and cleanup. Add a real reuse case if needed for the proposed algorithm.
Do not repeat already green broad tests without a source change or concern.

After acceptance, root may publish this bounded fix separately to main under
the existing user authorization, observe automatic service uptake once, and
perform a separately admitted original Egg task. Preserve Egg31's pending old
unregistered launch until its ownership or absence is positively proved. No
window deletion, manual ledger editing, GPU reset, CPU fallback or unscoped
model allocation belongs to083. This is a registration prerequisite; it does
not establish computer-use or whole-task success.
