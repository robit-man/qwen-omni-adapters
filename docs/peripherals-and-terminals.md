# Device awareness and dedicated terminals

The local harness uses the same authenticated, opted-in tool catalog as the
portal. The `peripherals` family exposes `peripheral_inventory`, `serial_io`,
`bluetooth_io` and `network_io`; the `terminal` family exposes
`terminal_session`. Shell remains available for other protocols and host work.
The reasoner selects families semantically. There are no product-name prompts,
USB-ID application classifiers, expected-device lists or canned probe payloads.

## Discover hardware from evidence

Start with `peripheral_inventory` using `scope=all`. This is passive: it reports
the executing service user's access, hardware TTYs and stable aliases, observed
port holders, USB interfaces and driver bindings, Bluetooth state, RF blocks,
network interfaces, and a summary of every available sysfs class and bus.

For complete details choose `scope=system`, a returned `category` such as
`class/net`, and follow `next_cursor`. Every category comes from the host,
including GPIO, I2C, SPI, CAN, PCI, storage, sound, video, input, power and virtual
interfaces where the kernel exposes them. Categories are not a product list.
Bus and class entries may alias the same physical device. Page counts and
completion markers disclose partial results. `scope=network` exposes addresses;
the separate privacy-bounded `get_system_snapshot` contract is unchanged.

Distinguish these claims:

| Observation | What it establishes |
| --- | --- |
| USB name or advertisement | A device supplied that label; firmware identity remains unverified |
| Interface with no driver | The kernel sees the interface, but its transport is not bound |
| Read/write permission | This service user has OS access; communication is not yet verified |
| Port holder | A visible process owns the endpoint; do not compete with it |
| Received bytes | Actual data from this transport; application meaning still needs interpretation |
| Protocol response | A caller may validate the actual response against an independently chosen protocol |

No names, identifiers or payloads from a connected device become instructions.
Inventory does not open unknown ports, sweep baudrates, stop owners, pair,
join a network, or send application commands. A short scan that sees nothing
does not prove a physical device is absent.

## Generic transport operations

`serial_io` opens an observed physical TTY at an **explicit** baudrate and
returns an opaque handle. Reuse it for `read`, `write`, `exchange`, `status`
and `close`. The handle belongs to the calling session, remains open across
requests, expires after five idle minutes, and closes on Trash. The store caps
open handles, refuses observed holders and system consoles, and uses advisory
locking plus kernel TTY exclusivity. Ownership observation is limited to process
descriptors visible to the service user; it is not a universal device broker.
Stable aliases are preferable to renumbered TTY names.

Reads cap at 16 KiB, writes at 4 KiB and I/O waits at ten seconds. UTF-8, hex
and base64 inputs are supported; base64 readback preserves arbitrary binary
frames. Opening configures baud, parity and stop bits, without explicit DTR/RTS
toggles. Close-time hangup is disabled. Driver behavior on opening a port still
depends on the hardware. An OS-accepted write is not remote effect verification.

`bluetooth_io` supports bounded BLE scanning, generic GATT service/characteristic
enumeration, reads, writes with ATT response, and notifications. Use addresses
and UUIDs from fresh observations. Existing connected devices are reported busy
rather than taken over. Owned connections close after each operation; scans
release discovery. Power changes are explicit and include state readback.
There is no automatic pairing. Advertisements and GATT descriptions remain
untrusted metadata, and ATT acknowledgements do not establish application effects.
The implementation uses [Bleak's scanner](https://bleak.readthedocs.io/en/stable/api/scanner.html)
and [client](https://bleak.readthedocs.io/en/stable/api/client.html) APIs.

`network_io` supplies interface/radio status, Wi-Fi scans, local mDNS service
discovery, explicit Wi-Fi power changes, and bounded TCP/UDP exchanges. Exchange
targets use explicit IP literals and ports. It never extracts Wi-Fi credentials
or joins a network automatically. A read is a bounded transport observation,
not a guarantee of a complete application response. The public-only web fetcher
retains its separate restrictions.

On Linux, OS metadata and utilities include sysfs, `ip`, `rfkill`, NetworkManager
(`nmcli`), BlueZ (`bluetoothctl`) and optionally `avahi-browse`. Missing utilities,
permissions, blocked radios, missing drivers and occupied ports remain explicit
failures. BLE support is installed through the runtime's Bleak dependency.
Hardware TTY access and complete sysfs enumeration require a POSIX/Linux host.
Grant the service user the appropriate device group, usually `dialout`; avoid
world-writable device rules. Install drivers matching the running kernel rather
than forcing incompatible modules. Driver repair is host setup, not firmware
recognition or a device-specific adapter.

## Persistent terminal and SSH sessions

`terminal_session` provides `create`, `list`, `read`, `send`, `key`, `resize`,
`interrupt` and `close`. Create an interactive shell or use `mode=ssh` with a
host, `user@host`, or an existing SSH-config alias. A private tmux server keeps
the PTY alive independently of Python workers; fresh harness instances can list
and inspect it after a reload. Each portal session has its own private socket
namespace. The tool never attaches to an unrelated tmux server. Local humans
can use the returned `attach_command` to view the same dedicated terminal.

Reads return fresh bounded screen/history snapshots, a digest, process IDs,
PTY state, observed exit status or signal, and the requested SSH destination.
PTY closure alone does not establish process exit. When necessary the manager
asks its own tmux server to refresh child status before returning wait results.
No prompt, command name or input-delivery acknowledgement verifies task success.
SSH startup is distinct from successful authentication or verified remote identity.

Normal input is literal, with optional Enter submission. Credentials should use
the existing SSH configuration/agent or `secret_file`: a private, service-owned,
single-line regular file. Its contents go through a temporary tmux input buffer,
never argv or a result, and the buffer is deleted. File input requires an
echo-disabled canonical prompt; readline/TUI raw mode is insufficient proof of a
credential prompt. The child environment omits runtime credential variables.
Host-key checking stays enabled. Terminal history is capped, with eight sessions
per owner and 32 private servers; close unused sessions. Trash terminates the
calling session's dedicated terminals. These are operational boundaries rather
than a sandbox around the already-authorized host shell.

The Linux guided deployer installs `tmux`; other POSIX installations can install
it with their system package manager. A missing terminal backend is reported as
unavailable. No neural worker restart is needed to adopt these Python tools.

## Qualification

Tests use unrelated synthetic descriptors, real test-owned PTYs and TCP/UDP
sockets, mocked generic GATT servers, and real private tmux terminals. They
check binary fidelity, explicit baudrate, ownership, cross-session rejection,
idle cleanup, paging, secret non-echo, actual process wait status, and terminal
survival through a fresh manager. Production code and tool descriptions contain
no expected-device identities. Live qualification must let the reasoner choose
its own generic discovery steps and preserve unknown or blocked states; a
user-supplied device name or a plausible descriptor is not a passing protocol test.
