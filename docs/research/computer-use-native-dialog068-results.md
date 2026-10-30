# CU.NATIVE.DIALOG.068 — owned Save As state comparison

Status: live metadata capture completed; native discovery still FAILED and the
direct-provider comparison remains INCOMPLETE. No production patch or model
inference occurred. A metadata-capture pass is not task success.

The single physical Ctrl+Shift+S opened one new, positively owned Save As
transient. The existing process/start/UID and literal parentage were verified.
Capture and cleanup took3.679 seconds. Both created XIDs closed through one
polite request each; preexisting managed windows,12 protected process identities
and frozen source stayed unchanged.

The plain editor's frame was ACTIVE before the shortcut. In all three modal
samples, the uniquely associated editor and dialog roots were not ACTIVE, both
before and after clear_cache_single, bounded GLib processing and the attempted
provider query. The unchanged native reader rejected each sample with
native_active_window_unavailable. A text descendant was FOCUSED in each scan.
Every descendant scan hit its128-node limit, so this is not proof that no
eligible node exists anywhere in the application's accessibility tree.

All seven direct-provider attempts returned DiagnosticError. No direct GetState
words or accepted signature were captured. Provider ACTIVE is therefore
UNKNOWN. The diagnostic's rejection of XML containing `<!` may reject the
standard D-Bus DOCTYPE, but actual XML was not retained and this precise error
cause is not yet proven. Preserve this failure without recasting failed calls
as valid false states.

Pinned upstream GTK4.14.5 provides a concrete role/state hypothesis:
collect_states adds ACTIVE for internal APPLICATION roles; GtkWindow defaults
to that role while GtkDialog selects DIALOG. GtkFileChooserDialog inherits
GtkDialog. The upstream signal path can emit active changes outside that role
condition. Installed Ubuntu patches were not compared, so the upstream source
alone is not proof of the loaded binary's behavior. See
[GTK state serialization](https://github.com/GNOME/gtk/blob/4.14.5/gtk/a11y/gtkatspicontext.c)
and [dialog role](https://github.com/GNOME/gtk/blob/4.14.5/gtk/deprecated/gtkdialog.c).

The initial invocation failed before any GUI setup because isolated system
Python could not import httpx. Normal repository Python without `-I` passed
the import-only check and ran the unchanged diagnostic; the system-GI child
retained isolation. The failed invocation and original marker remain preserved.

Next069 repairs only provider-probe compatibility and diagnostics. Require a
successful, consistent plain-editor provider read before another Save As input.
Keep068's consumed physical budget and failed comparison immutable; no native
ACTIVE/ownership/freshness guard changes are justified by this capture alone.

Durable actual review:
`/home/roko/Documents/qwen-omni-research/2026-10-17/native-dialog068-actual-review-20261017/actual-review.json`,
SHA256 `78841487501d3f699376d939ce3dea6ea17e927281f05e46255df5fb14c14ff6`.
Raw report: `runtime-data/gpu1-acceptance/operator-audit/execution-sop-20261017/native-dialog068/runs/20261017T175201483115Z/report.json`.
