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
stream. The portal keeps conversation history only in that browser page.

The reference deployment serializes GPU inference with one active lane and
admits four active/queued requests. This is bounded concurrency, not shared
context. The count shown in the UI is aggregate only.

llama.cpp prompt caching is disabled for comprehension because the pinned
multimodal slot cache can retain decoded media embeddings. This is a correctness
requirement even when ordinary token-prefix caching would be safe.

## Streaming scope

The portable adapter v1 response is turn-based. The phone portal adds an NDJSON
transport that relays language deltas and live Qwen3-TTS PCM windows while
retaining one authoritative final Ollama-shaped response. Camera-call mode
sends a current frame with each confirmed speech turn; it does not feed an
unbounded camera stream into one ever-growing context.
