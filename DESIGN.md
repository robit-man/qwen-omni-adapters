---
version: alpha
name: "Omni Portal"
description: "A phone-first technical console for private multimodal conversation with a locally hosted Omni runtime."
colors:
  primary: "#FACC15"
  background: "#050505"
  panel: "#0A0A0A"
  raised: "#111111"
  text: "#F5F5F2"
  accent: "#FACC15"
  danger: "#FB7185"
  success: "#86EFAC"
  observatory-amber: "#FFA800"
  observatory-background: "#000000"
  observatory-muted: "#C88F39"
  observatory-rule: "#5C3E0E"
typography:
  sans:
    fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
  utility:
    fontFamily: "IBM Plex Mono, JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace"
  observatory-label:
    fontFamily: "VT323, ui-monospace, monospace"
  observatory-digits:
    fontFamily: "DSEG7, monospace"
  observatory-segments:
    fontFamily: "DSEG14, monospace"
rounded:
  DEFAULT: "0.75rem"
  control: "0.5625rem"
  panel: "1rem"
  dialog: "1.25rem"
spacing:
  control-gap: "0.3125rem"
  panel-padding: "0.875rem"
  shell-max: "51.25rem"
components:
  icon-button:
    size: "2.375rem"
    textColor: "{colors.text}"
  composer:
    backgroundColor: "{colors.raised}"
    textColor: "{colors.text}"
    rounded: "1.125rem"
  dialog:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text}"
    rounded: "{rounded.dialog}"
  primary-action:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.background}"
  recording-state:
    textColor: "{colors.danger}"
  success-state:
    textColor: "{colors.success}"
  observatory-display:
    backgroundColor: "{colors.observatory-background}"
    textColor: "{colors.observatory-amber}"
  observatory-telemetry:
    textColor: "{colors.observatory-muted}"
  observatory-rule:
    backgroundColor: "{colors.observatory-rule}"
---

# Omni Portal Design System

## Overview

### Creative North Star

The portal should feel like a compact instrument panel for a local machine: dark optical surfaces, fine technical rules, terse telemetry, and one high-visibility control color. Its established reference is the NOCLIP documentation language described in `portal/README.md`, translated into a task-focused chat console rather than a branded landing page.

### Product context and register

- **Audience and primary job:** A person operating an Omni runtime from the host, a phone, or another authenticated browser needs to talk, attach media, inspect live sources, and receive text or speech without exposing internal services.
- **Target market(s) and evidence:** Global technical users; the repository defines an English interface and no market-specific behavior.
- **Locale(s) and language policy:** English UI. User and model content may be multilingual; controls remain short, literal, and sentence case.
- **Usage scene:** Phone-first, frequently in a live microphone/camera context, with desktop support and a dense single-screen layout.
- **Register:** Product. Task clarity, state truth, privacy boundaries, and fast recovery lead.
- **Memorable signature:** Near-black grid-like media surfaces with compact mono telemetry and a single yellow action signal.
- **Restraint:** Media previews, chat content, capture state, and errors stay visually dominant; decoration must never compete with them.
- **Anti-references:** Generic gradient SaaS dashboards, oversized marketing typography, glossy glass cards, hidden hover-only actions, and ornamental AI motifs.
- **Token ownership/runtime mapping:** The existing runtime CSS in `portal/static/portal.css` remains canonical. This file mirrors accepted values and explains intent; shared portal components consume the CSS custom properties declared in `:root`.

## Colors

`background`, `panel`, and `raised` create hierarchy through small tonal steps reinforced by translucent white rules. `text` is the high-contrast reading color. `primary` is the design-document alias for the runtime `accent` token; both intentionally resolve to the same yellow used for safe selected modes, primary send actions, and focus. `danger` is reserved for recording, errors, and interruption; `success` identifies live/healthy states. Color never carries state without text, an icon, shape, or accessible name.

## Typography

The sans stack owns conversation and controls. The utility mono stack owns compact status, metadata, measurements, and technical labels, usually with modest tracking and uppercase only where it functions as telemetry. Body content remains sentence case. System fallbacks prevent late font movement.

## Layout

The single chat column is capped at `shell-max` and uses one bounded conversation scroller between a fixed top control rail and a natural-height composer. Safe-area insets protect phone chrome. Popovers are anchored without affecting flow; dialogs stay within the visual viewport and keep controls reachable. Media reserves its intrinsic aspect ratio and uses `object-fit: contain` where cropping would remove evidence.

The observatory is a full-window application. Its approved visual reference is [Nick Gallo's 2026 LCD clock](https://www.nickgallo.dev/projects/clock/), with the extracted frame, segment, phosphor, wire, pixel-grid and hatched-meter vocabulary repurposed for runtime data. Four top chips select Active weights, Inference events, States, or Memory. Each view owns a different responsive instrument, rather than a shared clock or a scaled picture. The document does not scroll: the header, tabs and footer remain in the app frame; visualizations and expandable detail ledgers own bounded inner scrolling. On phones the tab chips form two columns. Chat opens beside the active instrument on desktop and occupies the workspace on narrow screens.

## Elevation & Depth

Hierarchy comes primarily from tonal layers and one-pixel borders. The sticky top bar may use blur. Composer, popover, and modal surfaces may use deep, diffuse shadows because they float above active content; static messages and cards remain flat. Dialog backdrops are dark enough to isolate camera content without hiding orientation.

## Shapes

Circular icon controls are the compact rail grammar. Utility controls use the `control` radius, contained panels use `panel`, and modal surfaces use `dialog`. Fine strokes and simple line icons match the instrument-panel reference. Pills are reserved for small live/recording badges, not ordinary buttons.

## Components

### Foundational visual states

Every control defines default, hover, focus-visible, pressed/selected, disabled, busy, and error states as applicable. Yellow focus rings remain visible. Busy states preserve geometry. Live media uses a stable reserved frame; failures replace status copy without moving controls. Loading uses concise status text in its reserved region rather than decorative skeletons.

### Buttons and actions

Icon buttons use the shared 38px control and an accessible name; important camera capture controls expand into labeled targets. Yellow indicates a safe primary/selected action. Rose indicates active recording or an error. Disabled controls retain their footprint and lose pointer affordance.

### Navigation and data display

The top rail owns global session actions. The conversation owns vertical scrolling. Reload and returning visits restore the same conversation, partial response, composer draft, attachments, and scroll-follow state; only the explicit Trash action represents deletion. Camera source discovery is an anchored source deck: two explicit origins first, then camera choices with truthful live/placeholder states. No camera activates merely because the page loaded.

### Forms and overlays

The composer uses an auto-growing, non-resizable textarea and explicit attachment removal. Interactive choices belong in popovers, not tooltips. Camera and settings dialogs use native modal semantics, an explicit close control, bounded content, Escape behavior, and focus restoration. Scrollbars inherit the global dark theme and remain visible.

### Iconography

Use the existing inline 24px outline SVG language with round caps and joins. Icons supplement visible labels in unfamiliar or consequential capture controls. Icon-only rail actions always have an accessible name and a tooltip title.

### Motion

Motion communicates state: a short reveal for new messages and a restrained pulse only while actively recording. Routine popover and camera polling updates do not animate. Reduced-motion mode collapses animations and transitions to effectively immediate updates.

### Content and data visualization

Copy is direct and operational: “Hold to record,” “Take still,” “Camera unavailable.” Status text states what is happening and how to recover. Do not personify errors or claim a capture completed before an attachment exists.

The observatory's signature is the **sodium-vapor LCD instrument**: pure black, amber #FFA800, DSEG7 measurements, DSEG14 state lettering, VT323 labels, 8% inaccessible ghost segments, a 3px scanline/pixel grid, fine glowing frames, right-angle SVG wires, and hatched meters. These modules visualize live model inventory, a multi-lane sensory/inference event raster, combined chat and daemon activity, and source-backed memory structure. There is no clock, synthetic frequency display, tilt, whole-panel scaling, or ornamental waveform. Body detail text remains unsegmented and outside the grid for legibility. State lettering uses container-relative sizing so narrowing a view beside chat cannot overflow a state card.

`portal/static/observatory.css` is canonical for this named surface variant: `colors.observatory-*` map to `--lcd`, `--lcd-bg`, `--lcd-muted`, and `--lcd-rule`; the three `typography.observatory-*` roles map to `--label`, `.d7`, and `.d14`. All three fonts are self-hosted under `portal/static/fonts/` with their licenses. The source extraction, modules, and intentional adaptations are recorded in `docs/observatory-lcd.md`.

The optical treatment covers the entire app, including framed chat. `lcd-surface.js` renders a static, pointer-transparent WebGL pixel/scanline mask using the reference's 3px pitch and 38%/22% attenuation, with a matching CSS fallback. It never samples the DOM or media. Backlight, bloom and the reference's bounded six-second flicker remain CSS compositing. Reduced motion disables flicker; forced colors remove the render layer.

The extracted radio and time modules carry observed runtime facts. Weight inventory reads llama.cpp properties, TTS readiness and Ollama's running-model list without warming models; local GGUF byte counts require an actual GGUF header. Model loading and GPU residency are separate observations. Sensory lanes use explicit typed request flags, perception evidence flags, stream stages and audio events; transcripts, media bytes and reasoning never enter the journal. Voice-loop transitions are bounded observations with their own source timestamps. Memory stores, source times, recall strengths, retention scopes and session boundaries remain distinct. Missing and stale sources never become invented activity.

The observatory heading uses the authenticated runtime identity, defaulting to the service account name, rather than a product-name constant. Double-clicking the name opens an inline editor; Enter/Space/F2 and the visible Rename action provide keyboard and touch access. Save changes the shared agent identity for following turns; Cancel/Escape discard the draft and restore focus. Pending saves preserve geometry and block duplicate submission; failed saves retain the draft with an inline error. Telemetry remains observational; identity rename is an explicit configuration action. The embedded chat reuses the existing portal and cookie/session boundaries, carries access only in the URL fragment, and uses a scoped amber chat stylesheet. Only the observatory chat variant permits same-origin framing; other routes retain frame denial. Closing chat unloads its frame so microphone, playback and live streams cannot continue invisibly. Copy chat link and Open in new tab use the same current-origin capability as the existing portal access plumbing.

### Memory structure and source inspection

Memory uses two explicit scope chips: **Device memory** and **This browser**. Device memory exposes aggregate owner-archive structure; source excerpts and derived facts are included only for the authorized owner session. This browser exposes only its own retained evidence. With the episodic service enabled, each scope shows committed source events, episodes, entities, active/history relation counts and semantic-vector counts. The older store projection remains the fallback: device source archive, temporal views and derived semantic cache; or browser conversation, saved notes and working context, with their separate lifetimes and Trash boundary.

The graph is a bounded recent neighborhood of reported entities and relations. Right-angle wires show typed relations or source support; layout distance carries no similarity score. Selecting a keyboard-accessible node highlights its connected wires and underlines its label, then updates the adjacent collapsible inspector. The inspector shows source IDs, recorded times, authority and actor labels, relation type, epistemic status, active/superseded validity, and quoted supporting excerpts when available. Relations remain model-derived claims supported by inspectable sources. Source authority and speaker attribution remain distinct; the graph does not establish either by appearance. Excerpts are bounded text, with no raw media or hidden reasoning.

Source counts remain separate from graph encoding and semantic indexing. The status names background encoding, pending work that yields to foreground activity, deferred encoding errors, or a current index; the inspector separates sources awaiting graph encoding from those awaiting vectors. Until relations exist, modality nodes can show retained source structure. Empty scopes and private aggregate-only views use explicit text. Cache retention strength is separate from factual confidence, and neither graph geometry nor counts imply recall quality or human-level memory performance.

The map and inspector retain the amber frames, VT323 labels and DSEG7 counts. Selected state adds an underline and text feedback rather than relying on color alone. The workbench places map and inspector side by side, stacking them when its container is at most 900px wide, including beside chat. The graph reflows from three to two columns below 600px of available width, with its source root spanning both columns. The visualization and inspector scroll internally; on narrow workbenches the inspector is capped at 280px. Long labels and quoted text wrap without expanding the app frame. The separate retention guide keeps its existing bounded detail scroller.

### States execution plan

States keeps its voice-loop and admission instruments and adds a task selector with the controller's current committed frontier. The header reports task status, phase, next transition, plan revision and environment version. Connected step cards show explicit states: planned, evidence acquired, awaiting verification, verified, or a reported action/effect/verification failure. Current and selected steps use amber borders; verified status also gains an underline. A selected step exposes its tool family, operation, retained selection explanation, expected effect, target, acceptance test, verification family and executor receipt IDs, alongside requirement statuses and their evidence. This makes tool choice and substep completion inspectable without displaying private reasoning. Missing plans, explanations or receipts have explicit text; discovery and command success alone never become invented progress or completion.

The step list and evidence inspector each scroll within a 450px bound inside the existing States visualization. Their two-column workbench stacks at viewport widths of 900px or less. Labels and evidence wrap within the app frame; the four peer tabs and whole-window scrolling contract remain unchanged.

### MCP connections

The collapsible **MCP connections** section stays within States. Its form has a labeled endpoint, optional password-style API key, **Available to** scope selector and Connect action. Connection discovery, success and failure appear in a live status region; unavailable service disables the form. The server list reports name, transport, whether a key is configured, and scope, with explicit Refresh and Remove actions. The key field clears after either submission outcome. Credentials remain private server-side and are omitted from the public connection list; Remove deletes the registration and its credential.

**Device and daemon** connections persist for the shared runtime. **This browser** connections belong to the cookie-scoped session and clear with its Trash action; they do not become another browser's connections. The form retains square amber controls and VT323 labels, reflows to two columns at 900px and one at 480px, and scrolls with the bounded States instrument rather than expanding the document.

## Do's and Don'ts

- **Do:** Keep privacy-sensitive camera activation behind an explicit source and device choice.
- **Do:** Reuse the existing attachment path so stills and clips behave exactly like uploaded media.
- **Do:** Keep bounded visual observations available for later questions while labeling them as historical evidence.
- **Do:** Distinguish live samples, short-lived session journals, durable context, relevance-decayed memory, and task archives wherever they meet.
- **Don't:** Auto-open cameras, replay cached raw media as current evidence, or expose host device paths.
- **Don't:** Present retained history as a complete recording of Egg’s lifespan or infer certainty from a graph edge.
- **Don't:** Crop live evidence, hide capture controls behind hover, or introduce new decorative colors for individual features.
