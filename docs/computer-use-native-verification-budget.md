# Native desktop verification observation budget

Store-created desktop controllers use a server-owned 10-second verification
window. Browser controllers and direct generic controllers retain 2.5 seconds.
The core deadline, cancellation checks, 20-observation ceiling and 30-second
constructor maximum are unchanged; clients and models receive no new setting.

The bound addresses a retained Egg41 original LHN-041 receipt: one actual native
post-input observation returned after 4.406875 seconds (another after 4.400156),
exceeding the prior 2.5-second window before the controller evaluated a true
visible-window predicate. The 10-second bound allows approximately two such
reads with bounded headroom. It does not guarantee every host or task will
finish within that window. Observations returning after the deadline remain
unverified; existing exact-deadline equality semantics are preserved.

CPU controls replay the queried root, predicate and semantic revisions with a
controlled clock; irrelevant nodes are omitted. No host GUI or model runs.
Source verification of an already-visible window is not proof of opening a
file, source acquisition, delivery or whole-task completion. The unchanged
semantic-revision control receives zero new-effect and task-progress credit.
Idempotency still prevents uncertain input from being dispatched again, and
cancellation remains authoritative before and after an observation.

This timing criterion is separate from local LHN-091, whose clicked editor root
remained unfocused and did not expose a focus capability. Changing the timing
bound does not supply that capability or make its focused predicate true.
Actual native adoption and long-task acceptance remain pending independent
source review, a full gate and a separately authorized live window.
