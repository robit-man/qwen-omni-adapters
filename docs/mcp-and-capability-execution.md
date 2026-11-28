# MCP connections and capability execution

Observatory's **Tools → MCP connections** accepts a remote endpoint and an
optional key. Choose **This browser** for cookie-scoped access, or **Device and
daemon** to make a connection discoverable to the local agent. Registration
initializes the server and discovers its tools; it does not execute them.
The same authenticated portal API supports automation:

```http
POST /api/mcp/servers
Content-Type: application/json

{"endpoint":"https://example.org/mcp","key":"optional private key","scope":"session","name":"Archive"}
```

Preserve the returned session cookie. `GET /api/mcp/servers` lists accessible
connections. `POST /api/mcp/servers/<id>` refreshes discovery;
`DELETE /api/mcp/servers/<id>` removes the registration and its credential.
Keys belong in this human/API registration flow, never in a model prompt.

## Protocol and credentials

The bridge uses the official Python MCP SDK for version negotiation, request
correlation and result validation. It tries Streamable HTTP and actual legacy
GET-SSE; keyed registration tries Bearer and `x-api-key` authentication. An
endpoint without a key sends no authentication header. Endpoints must have an
HTTP(S) origin and cannot contain user information, a query or a fragment.
Transport redirects are disabled and credential-bearing requests are pinned to
the registered origin. Proxy environment settings are not inherited.

Registration metadata contains references, not secrets. Credentials are stored
separately in owner-only files under `runtime-data/state/mcp` (directory 0700,
files 0600). Known key values are redacted from model-visible results and public
metadata. Browser Trash removes that browser's registrations and sessions;
device registrations persist until explicitly removed. Each caller nevertheless
gets a separate protocol session and must discover its own current tool schema.

Discovery follows pagination, bounded to 32 pages and 1,024 entries. Calls validate
arguments against the server's current schema and require rediscovery when a
definition changes. Exact remote names, annotations and output schemas survive
discovery. Remote `isError` remains a failed execution. A failed or timed-out
action is not automatically replayed: its effects must be inspected before
another action. Sessions, queues, response bytes and operation times are bounded.

The model-facing `mcp_connections`, `mcp_catalog`, `mcp_call` and `mcp_resource`
tools expose connections, tool schemas, resources, templates and prompts.
Standard resource bodies are read through the originating MCP connection.
Standard image content and image resource blobs enter the actual image input
channel; receipt text alone does not count as image comprehension. Remote audio
or arbitrary binary bytes do not automatically become a perceptual observation.
An arbitrary HTTP URL in a remote response does not inherit the registration's
credentials. Nonstandard authenticated REST result handles need a separate,
explicit host policy; this bridge does not silently authorize that behavior.

The implementation follows the [MCP transport contract](https://modelcontextprotocol.io/specification/latest/basic/transports)
and [server tools contract](https://modelcontextprotocol.io/specification/latest/server/tools).
The intake and private-reference patterns were also reviewed against
`robit-man/open-agents` main at `1c1a1504`; its working checkout was left untouched.

## Camera QR intake on the Egg

Set `OMNI_CALL_SCAN_QR=1` in the Egg checkout's `.env` and reload its native
call harness. The scan follows the tray's camera control and the native mute
and service-pause state. It samples individual cameras at up to 640 px every
1.5 s when quiet and up to 1280 px every 300 ms for five seconds after motion
or a new QR result. These are application sampling targets. Camera-native
timing depends on the capture driver. Existing requested stills and clips
use their normal capture path.

The scanner keeps frames in memory or a short-lived RAM-backed workspace.
It decodes QR codes locally on the CPU and reports each held code once across
the array. The native voice uses the configured cloned TTS after the
conversation yields. Ordinary links are recognized by hostname and recent
short text is available through `camera_qr` for ten minutes. The tool labels
the payload as untrusted historical camera data. Private code formats only
produce a type and a short spoken notice.

Remote MCP QR codes can carry a plain URL ending in `/mcp` or `/sse`, an
explicit `mcp+https://` URL, or one structured endpoint. Example payloads
with optional credentials are shown below.

```json
{"type":"mcp","endpoint":"https://example.org/mcp","key":"optional API key"}
```

```json
{"mcpServers":{"archive":{"url":"https://example.org/mcp","headers":{"Authorization":"Bearer optional-key"}}}}
```

An explicitly marked scan initializes the remote server and lists its tools.
An authenticated native owner can then use those tools by voice through the
ordinary MCP discovery and call path. The connection persists across daemon
token changes with native-owner scope. The separate owner-only credential file
is written after successful discovery. Repeated scans of the same connection
reuse the registration. The owner can remove a registration through
`mcp_connections` before replacing its key.

Automatic registration accepts public HTTPS endpoints. Configure an exact
comma-separated set of trusted LAN origins through
`OMNI_CALL_QR_TRUSTED_MCP_ORIGINS` in the portal's environment when needed.
For example, `http://192.168.1.80:8080` authorizes that origin. Public host
resolution is checked on each transport request and the connection uses a
checked address with the original TLS hostname. The scanned payload cannot
add an origin exception. Discovery executes no remote tools. Omnius's
`packages/execution/src/mcp/intake.ts` and `secret-store.ts` informed the
configuration shape and credential separation. The Egg uses its existing
Python MCP registry and transport negotiation.

## Camera tuning and roll correction

The scanner discovers supported V4L2 controls separately on each camera.
It enables supported automatic exposure, focus, gain, and white balance.
It trims brightness and contrast within bounds after repeated poor readings
and restores edited controls when tuning stops. Set
`OMNI_CALL_CAMERA_AUTO_TUNE=0` to keep the current hardware controls. The QR
decoder still retries missed frames with local contrast enhancement.

Run `bash scripts/install-camera-orientation.sh` on the Egg to install the
checksum-pinned RightWayUp Pico INT8 model and its CPU ONNX Runtime. The
native harness automatically uses that installed model while camera scanning
is active. A custom local Pico model path can be set with
`OMNI_CALL_CAMERA_ROLL_MODEL`. Each camera needs three confident, consistent
readings before native stills, motion clips, or the local camera view are
rotated. Ambiguous scenes retain the last confirmed roll for up to two
minutes. The estimator measures in-plane roll. Pitch, relative rig pose, and
IMU fusion require their own calibration. [RightWayUp model contract](https://github.com/ortusaitech/rightwayup) · [V4L2 user controls](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/control.html)

## Tool selection and current visual evidence

The capability catalog describes typed families and their discovery and
execution operations. Tool errors can declare a capability failure, available
alternatives and whether the task itself is blocked. The model is instructed to
inspect those contracts, use research, shell or a terminal when appropriate,
verify actual effects and avoid a capability refusal without attempting a
supported route. This does not manufacture missing hardware or authentication.

`capture_view` captures the current desktop or the runtime-owned browser and
stages an immutable, session-owned downloadable PNG. It returns dimensions,
SHA-256 and capture authority, plus a bounded visual overview for the model.
It is a captured computer surface, not a physical camera observation. On GNOME
Wayland a short-lived owned Mutter/PipeWire stream captures one real compositor
frame and is stopped in cleanup. X11 uses its own capture path. No global portal
permission or desktop setting is changed. Linux bootstrap checks the required
GStreamer/PipeWire bindings and plugins.

The explicit `after_capture=answer` plan enters a closed answer phase only after
a successful staged capture. `continue` keeps tools available for subsequent
inspection or actions. No prompt-keyword test decides that transition.

Physical serial tools keep their hardware-TTY and ownership guards. A
pseudo-terminal receives a typed unsupported-transport result with generic
terminal/shell alternatives; it is never relabeled as physical UART evidence.
Wi-Fi and Bluetooth operations report observed interfaces, blocked radios,
missing drivers and permissions without guessing a product or protocol.

## Execution display and verification

Tools projects the controller's committed frontier and successor steps, tool
choice rationale, acceptance criteria, evidence references and environment
revision. Selecting a step shows its supporting receipts. The display does not
invent a dependency graph, progress percentage or private reasoning transcript.
Acquiring information, changing an environment and independently verifying the
change remain distinct states. Failed tool receipts remain failures in the
activity timeline.

Its dispatch display links calls only through retained request IDs and tool
rounds. Select a dispatch to inspect its source time, status and retention.
Counters describe the bounded retained sample, rather than lifetime totals.
States remains dedicated to voice phases, runtime state and the work ledger.

Each dispatch has an **Arguments & output** disclosure. Its content is fetched
on demand from `GET /api/observatory/tool-call` using the request, round and
call IDs. Only the calling cookie can read it. Arguments and outputs live in a
bounded volatile store, expire with tool inactivity and clear with Trash;
late results from a cleared request cannot recreate them. Diagnostics retain
only call IDs, names, rounds and status. Credentials and media bytes are omitted
from inspection, and large displayed values are explicitly truncated. Nothing
in the inspector dispatches a tool or treats a return as proof of completion.

States also has horizontal **heard / done / said** source-time rails. Each
readout connects its exact recorded time to the rail and exposes source identity,
actor, authority and hash. Typed input, validated transcripts, client-reported
speech, tool returns and reply text remain distinguishable. A supplied audio
transcript is labeled generated speech; generated audio has no physical-playback
claim. The source text endpoint is included only when Observatory requests its
States activity view, is bounded to 36 records from 15 minutes, and remains
cookie-scoped. The native owner session can inspect device voice sources; other
browsers see their own events. No text, tool output or reasoning is added to the
content-free status journal.

Unit/protocol coverage is in `test_mcp_bridge.py`, `test_desktop_capture.py`,
`test_peripheral_tools.py`, `test_omni_portal.py` and `test_observatory.py`.
Local live fixtures separately exercise actual model source consumption,
downloaded capture hashes, peripheral inspection, isolated browser actions and
a multi-step file task. These are local acceptance checks, not a claim of
arbitrary-task reliability or a reproduced research benchmark.

## Learned behavior evidence

Memory's **Learned behaviors & tools** view separates actual tool returns from
audited approaches. Failed executions remain warnings; information reads and
mutations awaiting verification remain candidates. A verified-effect record
requires a separate successful controller verification that matches the recorded
contract. It does not assert complete-task success, generalized skill mastery or
current permission. Acceptance semantics and environment revision remain visible
in the source audit. The display never assigns invented confidence scores.

`behavior-memory.sqlite3` retains immutable, hash-checked, occurrence-addressed
source records, scoped by a hash of the authorized session. The native controller
uses its parent voice scope while its transient tool sessions stay task-local.
Actual call IDs join approaches to their audits. Later observations of the same
contract receive directed lineage links; those links do not assert causality.
Only tool names and typed contract/audit metadata enter this store: raw argument
values, output previews, credentials and binary media are omitted. Browser Trash
clears only that browser's behavior records; device-owner records remain private
and survive browser cleanup. Legacy tasks with unknown ownership are not imported
as the current owner's experience.

The source commits before optional indexing into the corresponding episodic
graph. Those historical tool-source events can then be encoded and retrieved
through `episodic_search`, with its existing attribution and time filters.
Replaying an occurrence cannot add another independent success, and conflicting
evidence under the same identity fails. A missed optional index write leaves the
behavior source intact for retry.

The reference review covered `open-agents` at `1c1a1504`: its
`proceduralMemoryStore.ts`, `successful-procedure-memory.ts`, `lesson-bank.ts`
and their runtime producers. They are separate mechanisms, rather than one
uniformly verified success/failure graph. This implementation adopts immutable
lineage, private scopes and unverified present applicability; it does not copy
fixed-confidence promotion, lexical contradiction heuristics or automatic
permission from historical success.
