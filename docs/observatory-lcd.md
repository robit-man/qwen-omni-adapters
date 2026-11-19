# Observatory scope

The supplied **RF Scope / Amber Signal Monitor** (`index(9).html`) provides
the 910 × 1630 canvas, pointed channel rails, canopy branches,
and trace and callout animation. The renderer and stylesheet are served locally
as `reference-rf-observatory.js` and `reference-rf-observatory.css`.
The scope and controls share one amber text color and use Helena Zhang's
[Departure Mono](https://departuremono.com/), bundled with its SIL Open Font License.

## Display mapping

| Channel | Recorded source |
| --- | --- |
| Input | Submitted text and audio arrival |
| Heard | Verified adapter transcripts, sound observations, and recognition receipts |
| Inference | Actual perception, language and synthesis stages, and request telemetry |
| Tools | Dispatches, returns, and authorized tool receipts |
| Memory | Retained records visible to the current scope |
| Media | Image, video, and document activity |
| Said | Reply text, generated speech, and audio output receipts |
| Tasks | Background task state and archive updates |
| Voice loop | Observed daemon state transitions |

Horizontal position uses the source timestamp and the selected window (five
minutes by default, with one-, ten-, and fifteen-minute options under the
leftmost time label). The scope holds up to 15 minutes of redacted trace
metadata. Widening the ruler can reveal those older traces. Trash removes the
session's traces. A measured duration
can produce a hatched interval.
Instantaneous events use a stem. Pointed rails show observed counts for the
visible window. Callouts use the event type, available timing or tool round, and
recorded error status. Recent labels compact and fade as their traces age. Use the
arrow keys to tune channels and Enter to inspect a trace. Settings also has a
keyboard-accessible event selector. Source shows authorized details. Generated audio and verified physical playback have distinct meanings.

The bottom blocks anchor two dot traces for host CPU utilization and the
busiest reported GPU. CPU uses the change in Linux aggregate counters
between readings. GPU uses the runtime's accelerator readings, including
Tegra's integrated GPU. Each column covers five seconds. New samples enter
from the right. Blank columns mark missing samples, and the numeric readout
shows `--` when a reading is missing or stale.

The weights readout shows loaded/total runtime components. Open it to see
the reported state of each model, media projector, speech model and codec.
When language and perception share a worker, its model is counted once.

## Controls

The scope fills the page. Label-width filters wrap inside the instrument and
toggle independently. **ALL** enables all channels from a partial selection
and clears the plot when every channel is already enabled.
When a square or wider viewport has room for a readable history, the pointed
right label travels outward along its connector, then unfolds into a box the
size of the plot. The triangular collapse control stays at the connector
endpoint while the name moves to the top. Each packet
shows the recorded time, source and available measurement. The **PACKET** box
widens to the preview edge and then expands down to the retained detail and
metadata. Selecting a trace opens and
reveals its packet. The left arrow, Escape, and an outside click fold the
history back. On narrower screens the labels filter the trace and a selected
trace opens its source dialog. Callouts ease between positions and fade as
they enter or leave the timeline. Reduced motion settles them immediately.

Tool packets show the recorded dispatch, outcome, call ID, and available
timing. Retained arguments and output are available to their browser session
for the scope window, subject to the existing size limits, credential and
media redaction, and Trash cleanup.

The original toolbar exposes pause, refresh, PNG capture, fullscreen, and
settings. Pause
holds the sample time, and reduced motion updates the plot when samples arrive.
Settings provides the time window, event selection, and capture export.

## Live data and authority

The authenticated `/api/observatory?activity=1` snapshot retains the
`robit.omni.observatory.v1` schema and uses a two-second poll while the page is
visible. It performs read-only health and inventory probes. The diagnostic
journal carries typed flags, stages, timestamps and bounded measurements.
Private excerpts arrive through the authorized source-event projection, with
cookie-scoped browser records and owner-scoped durable memory. Trash clears
the browser's plotted private sources. A retained event identifies its source
and retention. Missing or stale data has explicit status.

## Agent identity

Authenticated `GET /api/agent-identity` returns `name`, `default_name` and `source`.
`PUT /api/agent-identity` accepts only `{"name":"Mira"}`. Names contain 1–64
letters, numbers, spaces or `_ . @ + -`; invalid or failed saves retain the old
identity. Atomic private-file replacement persists the override under
`runtime-data/state/agent-identity.json` in `OMNI_REPO_ROOT` or the runtime working
directory. Set `OMNI_AGENT_IDENTITY_FILE` consistently when processes use different
working directories. Each lookup rereads the file, requiring no service restart
after this code is installed. Bad records fall back with an explicit warning.
