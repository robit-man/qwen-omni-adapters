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
  observatory-amber: "#FFAA20"
  observatory-background: "#020101"
  observatory-muted: "#C58037"
  observatory-rule: "#79340C"
typography:
  sans:
    fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
  utility:
    fontFamily: "IBM Plex Mono, JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace"
  scope-controls:
    fontFamily: "Courier New, Courier, monospace"
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

The scope uses the supplied **RF Scope / Amber Signal Monitor**
(`index(9).html`) canvas and stylesheet. One portrait instrument is centered in
the viewport. Label-width filters wrap inside its frame, pointed channel
labels form the right rail, and a nonlinear time ruler runs below the plot.
When the screen fits it, a rail label unfolds into a packet history the size
of the plot. The plot uses the reference bitmap lettering. The settings and
packet history use the reference Courier New stack.

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

The Observatory's signature is an amber scope with a stable frame and open space between traces. Event time sets horizontal position. Recorded channel sets vertical position. Measured duration sets the length of the hatched part of a trace. New stems grow from real events, branches yield around nearby stems, and callouts expand, compact, and fade. A selected callout holds its source and measured data. Pausing freezes the displayed time and reduced motion updates the image at snapshot boundaries.

`portal/static/reference-rf-observatory.css` and
`portal/static/reference-rf-observatory.js` carry the reference palette,
bitmap lettering, and motion. The native settings and source dialog expose
the recorded events to keyboards and assistive technology. See
[Scope data mapping](docs/observatory-lcd.md).

### Memory structure and source inspection

Memory uses two explicit scope chips: **Device memory** and **This browser**. Device memory exposes aggregate owner-archive structure; source excerpts and derived facts are included only for the authorized owner session. This browser exposes only its own retained evidence. With the episodic service enabled, each scope shows committed source events, episodes, entities, active/history relation counts and semantic-vector counts. The older store projection remains the fallback: device source archive, temporal views and derived semantic cache; or browser conversation, saved notes and working context, with their separate lifetimes and Trash boundary.

The graph is a bounded recent neighborhood of reported entities and relations. Right-angle wires show typed relations or source support; layout distance carries no similarity score. Selecting a keyboard-accessible node highlights its connected wires and underlines its label, then updates the adjacent collapsible inspector. The inspector shows source IDs, recorded times, authority and actor labels, relation type, epistemic status, active/superseded validity, and quoted supporting excerpts when available. Relations remain model-derived claims supported by inspectable sources. Source authority and speaker attribution remain distinct; the graph does not establish either by appearance. Excerpts are bounded text, with no raw media or hidden reasoning.

Source counts remain separate from graph encoding and semantic indexing. The status names background encoding, pending work that yields to foreground activity, deferred encoding errors, or a current index; the inspector separates sources awaiting graph encoding from those awaiting vectors. Until relations exist, modality nodes can show retained source structure. Empty scopes and private aggregate-only views use explicit text. Cache retention strength is separate from factual confidence, and neither graph geometry nor counts imply recall quality or human-level memory performance.

The map and inspector retain the amber frames, VT323 labels and DSEG7 counts. Selected state adds an underline and text feedback rather than relying on color alone. The workbench places map and inspector side by side, stacking them when its container is at most 900px wide, including beside chat. The graph reflows from three to two columns below 600px of available width, with its source root spanning both columns. The visualization and inspector scroll internally; on narrow workbenches the inspector is capped at 280px. Long labels and quoted text wrap without expanding the app frame. The separate retention guide keeps its existing bounded detail scroller.

### Tools execution plan

The Tools disclosure includes a task selector with the controller's current committed frontier. The Runtime disclosure holds voice phases and admission readings. The header reports task status, phase, next transition, plan revision and environment version. Connected step cards show explicit states: planned, evidence acquired, awaiting verification, verified, or a reported action/effect/verification failure. Current and selected steps use amber borders; verified status also gains an underline. A selected step exposes its tool family, operation, retained selection explanation, expected effect, target, acceptance test, verification family and executor receipt IDs, alongside requirement statuses and their evidence. This makes tool choice and substep completion inspectable without displaying private reasoning. Missing plans, explanations or receipts have explicit text; discovery and command success alone never become invented progress or completion.

The inspector scrolls the plan steps and evidence together. They stack in narrow layouts, with labels and evidence wrapping within the drawer.

### MCP connections

The collapsible **MCP connections** section stays within Tools. Its form has a labeled endpoint, optional password-style API key, **Available to** scope selector and Connect action. Connection discovery, success and failure appear in a live status region; unavailable service disables the form. The server list reports name, transport, whether a key is configured, and scope, with explicit Refresh and Remove actions. The key field clears after either submission outcome. Credentials remain private server-side and are omitted from the public connection list; Remove deletes the registration and its credential.

**Device and daemon** connections persist for the shared runtime. **This browser** connections belong to the cookie-scoped session and clear with its Trash action; they do not become another browser's connections. The form retains square amber controls and VT323 labels, reflows to two columns at 900px and one at 480px, and scrolls with the bounded Tools instrument rather than expanding the document.

## Do's and Don'ts

- **Do:** Keep privacy-sensitive camera activation behind an explicit source and device choice.
- **Do:** Reuse the existing attachment path so stills and clips behave exactly like uploaded media.
- **Do:** Keep bounded visual observations available for later questions while labeling them as historical evidence.
- **Do:** Distinguish live samples, short-lived session journals, durable context, relevance-decayed memory, and task archives wherever they meet.
- **Don't:** Auto-open cameras, replay cached raw media as current evidence, or expose host device paths.
- **Don't:** Present retained history as a complete recording of Egg’s lifespan or infer certainty from a graph edge.
- **Don't:** Crop live evidence, hide capture controls behind hover, or introduce new decorative colors for individual features.

## Tools review

The Tools disclosure shows the executor signal, retained-return counts, request-linked dispatches, and MCP connections. Committed task plans appear below the dispatches. Recorded request and round IDs establish the chains.
All surfaces retain inner scrolling, responsive chat splits and labeled keyboard
scroll regions. Tool failure is visibly distinct from a successful return;
neither establishes task completion without current verification.

Memory also has Experience and Learned behaviors & tools modes. The latter
shows separately recorded invocations, verified effects, failed approaches and
unverified candidates, with exact audit references and source hashes in the
inspector. Directed links mean later evidence for the same committed contract;
they do not imply that a failure caused a recovery. Present applicability and
permission stay unverified. Owner contents are private; unknown legacy ownership
is not reassigned. Embedded chat controls use the same sodium-yellow palette on
black, including selected and recording states; the idle mic helper is absent.

Dispatch modules expand into private, on-demand argument and output readouts.
The central scope maps Input, Heard, Inference, Tools, Memory, Media, Said, Tasks and Voice loop to distinct rails. Type filters change the visible channels and markers retain their source timestamps. Inference marks describe recorded processing; Said records replies, synthesis, and audio output with their recorded evidence labels. Selecting a callout opens the available source excerpt and measurements.
