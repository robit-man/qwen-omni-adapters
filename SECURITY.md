# Security model

The reference portal is an authenticated temporary test endpoint, not a
multi-tenant identity service.

- All component services bind to `127.0.0.1`; only the portal is tunneled.
- A high-entropy token is delivered in the URL fragment, retained in browser
  session storage, and sent only as a same-origin Bearer credential.
- The portal pins the model and exposes only an explicit bounded tool
  allowlist: read-only time/capabilities/web/document operations plus temporary
  session-memory read/write. Media, pages, documents, and memory cannot modify
  system or tool policy.
- Web discovery uses an ephemeral local Chromium/Chrome profile and a fixed
  public search-page template, not a search API or caller credential. Provider
  challenges fail closed. Page fetch accepts only public HTTP(S), revalidates
  DNS and every redirect, blocks local/private/metadata destinations, rejects
  binary/oversized bodies, runs fetched-page JavaScript nowhere, and sends no
  caller credentials. Network-level egress policy remains recommended.
- Request/media state is local to one request. Conversation history is browser
  local, never a server-global array.
- Per-session diagnostics contain timings and modality flags only. They exclude
  content, media, transcript, thinking, IP address, and user agent.
- Temporary memory, attached-document indexes, fetched-page indexes/caches, and
  bounded tool receipts are hashed by opaque session, clear with Trash, and
  expire after five idle minutes.
- Uploaded voice references are bounded, validated, written into a temporary
  generation directory, and deleted after synthesis.
- Browser-supplied server paths are rejected.

Cloudflare Quick Tunnel URLs and their fragments are credentials. Do not put
them in issues, commits, screenshots intended for publication, or persistent
logs. Stop the deployment when testing is finished.

For a vulnerability report, contact the repository owner privately rather
than publishing secrets or a working endpoint in a public issue.
