# Observatory instruments

The app uses the display vocabulary extracted from
[Nick Gallo's 2026 clock](https://www.nickgallo.dev/projects/clock/) on 2026-11-16.
The reference's clock, seconds/day/year meters, random radio decodes, frequency
simulation and tilt are replaced by actual runtime instruments. No remote page,
analytics, third-party script or external font is loaded by the app.

## Module mapping

| Extracted module | Observatory use |
| --- | --- |
| Black glass, #FFA800 phosphor, bloom and pixel grid | Shared instrument surfaces |
| Chip controls and inverse selected state | Four keyboard-accessible review tabs |
| DSEG7 and 8% ghost segments | Actual loaded graphs, weight sizes, event and memory counts |
| DSEG14 readouts | Worker readiness, voice-loop phases and current activity |
| Floating-label frames and right-angle wires | Logical model connected to observed execution workers |
| Hatched meters | Measured KV budgets, RAM, queue admission and recall strength |
| Graduated plotting grid | Time-positioned sensory events and source-time/strength memory points |

Fonts are self-hosted under `portal/static/fonts/`: DSEG7 Classic Bold and DSEG14
Classic Regular from DSEG 0.46.0, with `DSEG-LICENSE.txt`, and VT323 Regular from
Google Fonts, with `VT323-OFL.txt`. Source attribution remains in source comments
and this document; the app has no external attribution link in its footer.

`lcd-surface.js` composites one full-window, pointer-transparent WebGL mask over
the app and embedded chat. It matches the reference's 3px grid with 38% horizontal
and 22% vertical black attenuation. It reads no DOM pixels, text or media and has
no continuous render loop: it redraws only on resize/context restoration, with a
12-megapixel drawing-buffer bound. CSS supplies the original amber radial
backlight, phosphor bloom and six-second stepped flicker. A CSS mask provides the
same treatment if WebGL is unavailable. Reduced motion removes flicker; forced
colors remove the optical layer. The reference itself uses CSS compositing,
not Three.js or a canvas renderer.

## App and navigation contract

The shell owns `100dvh`, with a fixed header, navigation and footer. The document
never scrolls. The selected view fills the remaining workspace. Its instrument
and expandable detail ledger have bounded independent scrolling. The four peer
tabs implement the tablist/tabpanel model, one tab stop, arrows, Home and End.
Selected view and event filters remain URL query state. Cards and grids reflow;
state lettering sizes against its card through container query units, including
when the chat pane narrows the instrument. No page or instrument is scaled down
as a picture.

Open chat appears in the header and footer. Desktop uses a split workspace;
narrow screens use a full chat pane with an explicit Close chat control. The
existing chat client is loaded lazily at `/?surface=observatory#access=…`, sharing
the same cookie, storage scope and bearer fragment as an ordinary portal visit.
`observatory-chat.css` provides the amber variant. This one root-page variant
permits `frame-ancestors 'self'` and `X-Frame-Options: SAMEORIGIN`; ordinary chat
and other routes retain frame denial. Closing chat unloads its frame so capture,
playback and live streams cannot continue invisibly. Its IndexedDB and detached
ordinary turns retain the existing reload/Trash behavior. Copy chat link and
Open in new tab use the current authenticated origin; capability URLs are not
logged or placed in query strings. Capture remains opt-in.

## Live data and authority

`/api/observatory` retains `robit.omni.observatory.v1` and adds `agent`, `weights`,
and typed timeline `signal`/`event` fields. It remains authenticated and no-store.
The browser polls every two seconds while visible, with single-flight fetches,
a 25-second deadline, explicit pause/manual refresh, and stale retained snapshots
on failure. Environment and model inventories cache for five seconds.

`RuntimeModelReader` performs bounded GETs to the configured worker `/props`,
TTS health and Ollama `/api/ps`. It never starts, warms, resizes or unloads a
model. Model paths are reduced to basenames before returning to the browser.
A reported local artifact size requires an actual GGUF header. TTS readiness and
its request-scoped release policy stay distinct from service health; GPU
residency is shown only when the worker reports it explicitly. Reported bytes
are not represented as a complete GPU-allocation inventory.

The event raster offers one-, five- and fifteen-minute windows. Markers use real
source timestamps and select a source/retention receipt. Record filters and
pagination live in an expandable ledger. Typed instrumentation covers received
input modality flags, comprehension/language/TTS stages, perception completion,
recognized-speech presence, sound/visual-evidence presence, and audio start/end.
No transcript, acoustic description, visual description, model delta or PCM is
recorded. Sound-only observations never become heard-speech events. Audio output
is labeled output/synthesis; it is not proof that physical speakers were heard.
Voice-loop status transitions are a bounded in-memory observation tail, retaining
the status source's timestamp. Stale state is never highlighted as current.

The state instrument combines fresh daemon reports with active portal request
stages across browser chats and calls. The embedded chat also sends content-free,
same-origin phase notifications so short turns appear immediately between polls.
Microphone capture requires recent delivered PCM frames; text-chat readiness and
an unlocked playback context never claim that the microphone is listening.
Native capture heartbeats preserve the phase-transition timestamp separately.
Concurrent input, inference and playback can illuminate multiple activity cards.
The instrument includes actual inference/admission counts and host RAM. Detailed
state and the work/archive ledger remain available in the expandable section.
Memory distinguishes durable passive memories, this browser's session memory,
working context and aggregate host corpus storage. Passive plot points use source
times and measured recall strengths, with a selectable source excerpt. Missing
or empty stores have explicit text. Event filters do not hide older memories.

## Agent identity

The authenticated snapshot supplies the heading and title; unauthenticated HTML
does not disclose the system account. `runtime_agent_name()` defaults to the
runtime service account, preserving an existing `OMNI_AGENT_NAME` deployment
configuration when no saved override exists. Double-click, Enter/Space/F2, or
Rename opens an inline editor. Cancel/Escape discards the draft; confirmed Save
restores focus and changes subsequent language self-state and live addressee
routing. It does not change the host login, model tag or memory/task ownership.

Authenticated `GET /api/agent-identity` returns `name`, `default_name` and `source`.
`PUT /api/agent-identity` accepts only `{"name":"Mira"}`. Names contain 1–64
letters, numbers, spaces or `_ . @ + -`; invalid or failed saves retain the old
identity. Atomic private-file replacement persists the override under
`runtime-data/state/agent-identity.json` in `OMNI_REPO_ROOT` or the runtime working
directory. Set `OMNI_AGENT_IDENTITY_FILE` consistently when processes use different
working directories. Each lookup rereads the file, requiring no service restart
after this code is installed. Bad records fall back with an explicit warning.
