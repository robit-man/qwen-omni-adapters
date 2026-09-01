# Security model

The reference portal is an authenticated temporary test endpoint, not a
multi-tenant identity service.

- All component services bind to `127.0.0.1`; only the portal is tunneled.
- A high-entropy token is delivered in the URL fragment, retained in browser
  session storage, and sent only as a same-origin Bearer credential.
- The portal pins the model and exposes only an explicit read-only tool
  allowlist. Media-derived text cannot modify system or tool policy.
- Request/media state is local to one request. Conversation history is browser
  local, never a server-global array.
- Per-session diagnostics contain timings and modality flags only. They exclude
  content, media, transcript, thinking, IP address, and user agent.
- Uploaded voice references are bounded, validated, written into a temporary
  generation directory, and deleted after synthesis.
- Browser-supplied server paths are rejected.

Cloudflare Quick Tunnel URLs and their fragments are credentials. Do not put
them in issues, commits, screenshots intended for publication, or persistent
logs. Stop the deployment when testing is finished.

For a vulnerability report, contact the repository owner privately rather
than publishing secrets or a working endpoint in a public issue.
