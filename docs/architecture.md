# Architecture and capability ownership

## System boundary

The project exposes one logical model name through an Ollama-shaped adapter,
while each architecture runs in the runtime that understands it:

```text
client / phone
    |
    v
authenticated portal (optional) ---> isolated queue + diagnostics
    |
    v
unified adapter
    |-- text ------------------------> Ollama / Qwen3.8
    |-- audio, image, video ---------> Qwen3-Omni comprehension
    |                                   |
    |                            tagged evidence
    |                                   v
    +--------------------------------> Ollama / Qwen3.8
                                        |
                              final speech requested?
                                        v
                                   Qwen3-TTS
```

The comprehension result is untrusted evidence, not a new system message.
Transcribed speech, non-speech acoustics, and visual observations use distinct
tags. This prevents an encoder's suggested answer from becoming the user's
utterance and prevents text seen or heard inside media from changing tool or
system policy.

## Artifact layout

The Ollama tag retains normal runnable layers and adds one custom media layer:

```text
application/vnd.robit.ollama.omni.bundle.v1+gguf
```

The custom GGUF uses namespaces for byte-preserving component views:

| Namespace | View |
|---|---|
| unprefixed | base model |
| `b.p.*` | base projector |
| `a.c.m.*` | comprehension model |
| `a.c.p.*` | comprehension projector |
| `s.t.m.*` | TTS model |
| `s.t.p.*` | TTS codec/projector |

`qwen-omni prepare` reconstructs only the runtime views. They are disposable;
the attached sidecar remains the source of truth.

## State and concurrency

The adapter is stateless between HTTP requests. Each request owns its parsed
media bytes, stage outputs, tools, voice settings, cancellation, and response
stream. The portal keeps each browser's text history and rendered state in a
same-origin IndexedDB record keyed by a one-way cookie-derived scope. Reloads
restore that record, page leave starts a five-minute expiry, and trash deletes
it immediately. Cached media is display-only and cannot silently become new
model input. The optional document index uses the same isolation boundary,
bounded in memory, cleared by trash, and expired after disconnect. Raw
documents and retrieved passages never cross into another browser session.
The demonstration tool harness uses that same boundary for temporary recall and
its short web-fetch cache. Tool schemas are server-pinned, results are appended
as ordinary tool-role messages, and no executor state belongs to the model or
adapter worker.

The reference deployment serializes GPU inference with one active lane and
admits four active/queued requests. This is bounded concurrency, not shared
context. The count shown in the UI is aggregate only.

llama.cpp prompt caching is disabled for comprehension because the pinned
multimodal slot cache can retain decoded media embeddings. This is a correctness
requirement even when ordinary token-prefix caching would be safe.

Every request receives a newly captured trusted runtime-environment system
message. Its schema exposes date/time, OS/architecture, CPU/load, RAM, bounded
interface counters, and NVIDIA utilization while explicitly omitting hostnames,
addresses, routes, sockets, processes, credentials, and user/session content.
It is operational context, never an authorization source.

## Streaming scope

The portable adapter v1 response is turn-based. The phone portal adds an NDJSON
transport that relays language deltas and live Qwen3-TTS PCM windows while
retaining one authoritative final Ollama-shaped response. It also relays
bounded tool-round start/completion events; only a final response with no
pending calls may enter TTS. Camera-call mode
sends a current frame with each confirmed speech turn; it does not feed an
unbounded camera stream into one ever-growing context. Long speech replies are
split at sentence boundaries before the Qwen3-TTS per-generation frame limit;
their PCM windows share one monotonically increasing sequence and are assembled
into the final replay WAV. Silent video is valid, and animated GIF input is
normalized into a bounded temporal video before comprehension.

Call turns include bounded prior dialogue plus a system instruction to answer
the user's intent without echoing, transcribing, or paraphrasing by default.
Media chat likewise keeps prior textual conversation, but only the newest
attachment is labelled as current perceptual evidence. Qwen3-TTS uses a
persistent framed subprocess protocol so matching-profile requests reuse the
resident model and the browser can schedule the first two-frame PCM window as
soon as it arrives.
