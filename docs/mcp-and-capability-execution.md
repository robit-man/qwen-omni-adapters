# MCP connections and capability execution

Observatory's **States → MCP connections** accepts a remote endpoint and an
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

States projects the controller's committed frontier and successor steps, tool
choice rationale, acceptance criteria, evidence references and environment
revision. Selecting a step shows its supporting receipts. The display does not
invent a dependency graph, progress percentage or private reasoning transcript.
Acquiring information, changing an environment and independently verifying the
change remain distinct states. Failed tool receipts remain failures in the
activity timeline.

Unit/protocol coverage is in `test_mcp_bridge.py`, `test_desktop_capture.py`,
`test_peripheral_tools.py`, `test_omni_portal.py` and `test_observatory.py`.
Local live fixtures separately exercise actual model source consumption,
downloaded capture hashes, peripheral inspection, isolated browser actions and
a multi-step file task. These are local acceptance checks, not a claim of
arbitrary-task reliability or a reproduced research benchmark.
