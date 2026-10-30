# Phase A2 — shared, indexed component packaging

Both installed trained-bridge releases contained the same TTS checkpoint and
codec bytes in separate 1.48 GB sidecars. Backbone names in the embedded manifest
made their complete blob digests differ. The v2 packer places checkpoint identity
in a deterministic component index and language/projector bindings in each tag's
layer annotation. It never repacks or substitutes the reasoning weights.

The real installed candidates are:

- `robit/ornith-1.5-omni:edge-v2-candidate-20261029`
- `robit/qwen3.8-27b-e03-omni:edge-v2-candidate-20261029`

Both resolve one physical blob with SHA-256
`5f39fca31532abfb5c162b3c258e26bc7ab959a94708ab32000185301d35ebf7`.
The CPU-fenced Ollama metadata endpoint reports both updated tag digests. No
inference runner was loaded and no existing service was restarted.

| Payload in the pair | Original | Candidate | Reduction |
| --- | ---: | ---: | ---: |
| Sidecar bytes | 2,964,792,672 | 1,482,396,544 | 50.00% |
| Unique language/projector/sidecar bytes | 28,433,847,712 | 26,951,451,584 | 5.21% |

The [receipt](phase-a2-sharing.json) proves exact language/projector identities,
both original TTS checkpoint hashes after extraction, and zero reconstruction
for the second tag using the same verified runtime cache. These measurements
describe the payload required by a new two-release installation. Legacy tags and
blobs were retained; current disk space was not reclaimed.

Reproduction uses `qwen-omni pack-tts`, independent `ollama cp` targets, and
`runtime/qualify_indexed_sidecar.py --pair SOURCE TARGET --pair SOURCE TARGET
--bundle BUNDLE --work-dir NEW_DIRECTORY --out RECEIPT`. The qualifier checks both
targets' standard identities before attaching either payload and never mutates
the source tags.

Tests cover deterministic renamed inputs, byte-identical F32/Q8 extraction,
cross-tag reuse, changed language/projector/sidecar bindings, missing bindings,
invalid view namespaces, preservation of the old output after a write failure,
and Ollama v2/legacy lookup. No compatibility or voice-quality result is inferred
from packaging labels. Full cloned speech still uses the existing checkpoint
and backend; official full acoustic-prompt support remains B4. Direct filtered
native views remain the next packaging milestone.
