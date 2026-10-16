# Owned Flatpak Firefox setup

Add `--browser flatpak-firefox` to an existing
`runtime.verify_long_horizon_native` invocation when the declared browser is the
system-installed Flatpak Firefox. This is an explicit test-host setup condition;
ordinary `firefox` and `chromium` launch paths and model tools are unchanged.
It opens only one ordinary-profile `about:blank` window. It does not navigate
sources, author deliverables, change application preferences, enable debugging,
or create a private browser profile.

The host needs a recovered Linux X11 desktop session, `/usr/bin/flatpak`, the
system `org.mozilla.firefox` stable deployment for `aarch64` or `x86_64`, and
`/usr/bin/python3` with AT-SPI GI bindings. Xlib, XRes 1.2, the managed-client
property and the existing native namespace proof must be available. A complete
prelaunch X window tree includes hidden, unmanaged and unknown-owner resources
in the exclusion set. The single launch must produce one new managed XID with
actual create/map events, a unique native frame and committed Document
`DocURL=about:blank`. Registration additionally binds the installed deployment
and executable, XRes host PID/start, authenticated Flatpak namespace/provider
proof and source hashes. Titles, classes and relay PIDs cannot substitute.

Use an exclusive operator setup interval. The event witness is bounded evidence
of the single launch and new window; it is not atomic causal proof against an
indistinguishable concurrent launch by another actor. Unknown or concurrent
activity withholds ownership. No launch is replayed automatically.

A dedicated fixed system-Python helper retains the private ownership ledger.
The runner rechecks it before freezing/submitting the exact native task window
scope. Journals contain JSON evidence; live helper handles remain private.
Requests and replies bind their operation and sequence. Timeout, EOF, oversized
or malformed replies permanently close that transport and stop only its CPU
helper. An uncertain browser window may remain; neither a late reply nor a new
helper may silently restore ownership or count cleanup as successful.

After the runner proves task drain, cleanup refreshes the exact ownership proof
and requests one polite `WM_DELETE_WINDOW` for the owned XID. It never kills the
shared Firefox process or closes other same-PID windows. Closure requires actual
absence from a complete X tree and preservation of preexisting resources; a
missing class-search result is insufficient. Unknown ownership or closure stays
an explicit failed/uncertain cleanup result.

This capability is setup and cleanup only. All subsequent task work remains
model-driven native GUI interaction under the normal controller guards. CPU
controls are source validation; live owned-Flatpak launch/cleanup and long-task
acceptance remain pending until separately executed and independently graded.
