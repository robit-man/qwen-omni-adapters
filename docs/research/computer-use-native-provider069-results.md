# CU.NATIVE.PROVIDER.069 — direct provider comparison complete

Status: actual provider comparison PASS; native dialog observation still FAILED.
The repaired probe made seven successful direct GetState calls. No production
native repair, model inference, service change or whole-task pass occurred.

Before the sole physical Save As shortcut, the plain editor returned raw state
words `[1124073474, 0]`, including ACTIVE. All four GI/cache stages agreed, and
the mandatory plain-provider prerequisite passed.

After Save As opened, the associated editor frame and dialog each returned
`[1124073472, 0]`: the same state with ACTIVE absent. All three modal samples
and every fresh GI/cache variant agreed. A text control was FOCUSED under the
dialog through five recorded parent links. The unchanged native bridge rejected
all three samples with native_active_window_unavailable.

All seven calls used the actual declared Accessible.GetState()->au interface,
provider credentials matching PID3036686/UID1000, and raw decoded state words.
The live introspection reply had the canonical D-Bus DOCTYPE and one GDBus
comment. The probe now accepts that real format, rejects malformed/expanding
declarations, and reports typed failure stages. Earlier probe failures and the
comment-parser CPU RED remain preserved.

This identifies a concrete compatibility failure: our adapter assumes every
usable foreground window supplies ACTIVE, while this GTK dialog's fresh
provider state does not. GTK4.14.5's source adds ACTIVE only for APPLICATION
roles; GtkDialog selects DIALOG. The exact installed Ubuntu patch series leaves
the relevant code unchanged, and the proved provider process maps the installed
GTK4.14.5 library. Repeated cache refresh cannot manufacture that missing bit.

The capture took 3.746 seconds. Both new windows closed once; 14 preexisting
managed windows, 12 protected processes and frozen source remained unchanged.
Each descendant scan hit 128 nodes and was truncated. The positive focused
ancestry is verified; global focus uniqueness and production XID-to-accessible
authority are not established by this diagnostic.

Next070 adds a separately validated activation proof from complete, current
host/provider/root association plus fresh focused ancestry. Preserve the
standard ACTIVE path. Revalidate the same proof before publishing controls and
at every activation/input boundary. Title, PID, geometry or a remembered focus
slot alone cannot grant authority. Input and independent effect readback remain
separate acceptance steps.

Durable actual review:
`/home/roko/Documents/qwen-omni-research/2026-10-17/native-dialog069-actual-review-20261017/actual-review.json`,
SHA256 `b02b9a01b3f6b818be6387f225b00ab45d1f9fbd7e252c4fe49b3fbdebb9be07`.
Independent audit SHA256
`026c6c219e51a943745de5127852b6a79264b692622212c17af280cd41e06b8e`.
Raw report:
`runtime-data/gpu1-acceptance/operator-audit/execution-sop-20261017/native-dialog069/runs/20261017T180427592083Z/report.json`.
