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
typography:
  sans:
    fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
  utility:
    fontFamily: "IBM Plex Mono, JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace"
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

## Do's and Don'ts

- **Do:** Keep privacy-sensitive camera activation behind an explicit source and device choice.
- **Do:** Reuse the existing attachment path so stills and clips behave exactly like uploaded media.
- **Do:** Keep bounded visual observations available for later questions while labeling them as historical evidence.
- **Don't:** Auto-open cameras, replay cached raw media as current evidence, or expose host device paths.
- **Don't:** Crop live evidence, hide capture controls behind hover, or introduce new decorative colors for individual features.
