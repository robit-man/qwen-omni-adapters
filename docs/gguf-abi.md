# GGUF Sidecar and Ollama Layer ABI

This document defines artifact schema `robit.ollama-monolithic-omni.v3` and
its delivery as a custom layer in one Ollama model manifest. The public HTTP
schema is independently versioned as `robit.ollama.omni-adapter.v1`.

## Indexed trained-bridge sidecar v2

`robit.ollama-audio-bridge.v2` is an additional, backward-compatible artifact
schema for reduced trained-bridge profiles. It contains the two existing TTS
graphs. The language trunk and matching trained vision/audio projector remain
standard Ollama layers. Existing v1 and legacy full-Omni bundles still resolve.

The v2 manifest has a `robit.gguf-component-index.v2` inventory with each
component's architecture, tensor and metadata prefixes, tensor count, original
GGUF size and SHA-256. Its sources are content digests; input paths, filenames
and backbone names do not change the payload. Thus identical TTS checkpoints
share one content-addressed Ollama blob across independently versioned backbones.
This does not combine their trained language/audio projections.

Each tag carries a small content-addressed JSON layer with media type
`application/vnd.robit.ollama.omni.binding.v2+json`. Its
`robit.omni.layer-binding.v2` object binds the sidecar digest to the exact
language/projector layer digests and sizes. Stock Ollama push drops descriptor
annotations but preserves this layer. Attachment also writes the existing
`io.robit.omni.binding.v2` annotation for compatibility. Existing annotation-only
v2 tags remain readable; new attachments include the portable layer.

Resolution checks the binding blob's digest, size and bounded JSON payload,
rejects duplicate layers, and rejects any conflicting binding copy. Neither a
valid annotation nor a valid layer masks a damaged or mismatched counterpart.
Attachment records selected identities; it does not prove that an arbitrary
projector was trained for that trunk. Begin with the corresponding qualified
baseline and preserve its standard layers. The TTS GGUF bytes remain shared;
the per-backbone binding adds less than one KiB, with no neural weights.

```bash
.venv/bin/qwen-omni pack-tts \
  --tts components/tts-model.gguf \
  --tts-projector components/tts-projector.gguf \
  --out release/shared-tts-sidecar-v2.gguf

ollama cp robit/ornith-1.5-omni-audio-bridge:q4km robit/ornith-1.5-omni:edge-v2-candidate
.venv/bin/qwen-omni attach robit/ornith-1.5-omni:edge-v2-candidate release/shared-tts-sidecar-v2.gguf
.venv/bin/qwen-omni prepare robit/ornith-1.5-omni:edge-v2-candidate --out runtime-data/components --reuse
```

After changing parameters with `ollama create`, run `qwen-omni attach` again and
verify with `qwen-omni resolve`. Creation drops custom-layer annotations and may
retain a binding for an earlier base/projector pair. For the edge profiles,
package `PARAMETER num_ctx 16384` rather than inheriting the large backbone
default; this sets the default resident window without reducing the model's
native context metadata. Recheck after stock push/pull before publication.

The resolver follows Ollama 0.40's v2-first lookup, including its canonical
public host and encoded custom-registry ports, with legacy fallback. Existing
flat GGUF manifests remain the supported tag shape. This follows the
[pinned Ollama path implementation](https://github.com/ollama/ollama/blob/v0.40.11/manifest/paths.go).

[Qualification evidence](upgrades/phase-a2-sharing.json) records both installed
candidate identities, unchanged backbone layers, byte-identical extracted TTS
checkpoints and reuse across tags. The pair avoids 1,482,396,128 payload bytes
on a new installation. Existing legacy blobs remain; this is not a claim of
current disk reclamation or lower GPU allocation.

### Native indexed views

The reference-encoder extension may additionally carry `tts_reference_encoder`
under `s.t.e.*` with metadata under
`robit.audio_bundle.component.tts_reference_encoder.kv.*`. It requires
`robit.qwen3tts.reference-encoder.v1`, `clip` architecture and the
`qwen3tts_refenc` audio projector type. The inspector rejects unknown optional
components and incompatible reference-encoder ABI metadata. Older two-component
v2 sidecars remain valid. Readers predating this extension reject its expanded
inventory; use the matching runtime and native patch set.

Build the component offline from the pinned official tokenizer checkpoint using
`runtime/export_tts_reference_encoder.py`. That build-only tool accepts local
safetensors and adds no runtime download or Torch dependency. `pack-tts` accepts
`--tts-reference-encoder FILE` to include it in the same authenticated model tag.
Preparation includes its materialized view or a third direct source alias.
`llama-tts-reference-encode` selects that indexed role explicitly, retaining
original tensor offsets. Ordinary model and codec loaders select their existing
roles. The encoder emits 16 acoustic codes per 12.5 Hz frame from bounded
24 kHz mono references; this component alone does not add transcript-conditioned
cloning to the speech request protocol. See
[`phase-b4b1-reference-encoder.md`](upgrades/phase-b4b1-reference-encoder.md).

The optional English `audio_ctc_head` component uses tensor prefix `a.h.c.*`
and metadata prefix `robit.audio_bundle.component.audio_ctc_head.kv.*`. Its ABI is
`robit.omni.audio-ctc-head.v1`, architecture `robit_audio_ctc`, with eight F32
tensors and an exact frozen-encoder identity. The exported component also carries
its training-only lexicon, tokenizer and qualification identities. Include it
with `pack-tts --audio-ctc-head FILE`. It remains listed in
`runtime.disabled_components`; ordinary runtime preparation selects the existing
TTS components only. Explicit `prepare --view audio_ctc_head --direct-views`
creates one source alias. Materialization remains available and reproduces the
original head GGUF exactly.

The matching `runtime/audio_ctc_probe.cpp` selects the head's metadata and tensor
namespace while preserving the container offsets. It executes the small head
without reconstructing a separate file or loading the sibling speech weights.
The model/tag binding still preserves each backbone's own language weights and
trained projector. The shared frozen audio-feature interface was qualified for
both corresponding backbones; other feature interfaces require new evaluation.
This component does not enable an ASR route or establish streaming/multilingual
capability. See [head qualification](upgrades/phase-c1-native-head.md) and
[single-tag attachment](upgrades/phase-c1b-indexed-head.md).

The runtime's `llama.cpp-indexed-tts-views.patch` adds
`robit.indexed-tts-views.v1`: role-aware native loading of the v2 TTS envelope.
The model loader selects `s.t.m.*`; the codec loader selects `s.t.p.*`. Both
restore their original metadata namespace, tokenizer, tensor names, shapes and
quantization while retaining the original container offsets. This is a patched
native loader, not stock Ollama execution of a heterogeneous GGUF.

On POSIX, `prepare --direct-views` creates disposable source aliases and a
versioned receipt instead of reconstructed payloads. Bootstrap, the portable
supervisor and broker launcher prefer this mode for v2. Existing v1 bundles and
Windows retain materialization. Ordinary materialization still rejects aliases;
use a fresh output directory for an explicitly extracted view. Cleanup unlinks
the aliases and leaves the installed blob intact. Direct mode requires the
updated native build from this checkout.

The native view is read-only and must not be serialized. Its metadata/tensor
inventory is filtered in memory; its file offsets and physical alignment belong
to the original sidecar. Full-file mmap prefetch is disabled only for indexed
model views, preventing a transient duplicate read of sibling codec tensors.
See the [native view qualification](upgrades/phase-a3-native-views.md).

## Invariants

- The sidecar is one valid GGUF v3 file with a contiguous tensor index.
- It contains six reproducible views: base model, base projector,
  comprehension model/projector, and TTS model/projector.
- Each view materializes to the same bytes as its independently tested source
  GGUF when the source was produced by this packer.
- The stock Ollama model and projector remain ordinary standard layers.
- The sidecar is a custom layer; it is never a Modelfile `FROM` target.
- Quantization occurs per component before packing. Never quantize the bundle.

The logical model therefore has several OCI layers, while the custom media
payload is one physical GGUF. “One tag” and “one sidecar GGUF” are accurate;
“one directly runnable GGUF containing every architecture” is not.

## Tensor views

| View | Stored prefix | Example visible name after extraction |
|---|---|---|
| Base model | none | `blk.0.attn_q.weight` |
| Base projector | `b.p.` | `v.blk.0.attn_q.weight` |
| Comprehension model | `a.c.m.` | `blk.0.attn_q.weight` |
| Comprehension projector | `a.c.p.` | `a.blk.0.attn_q.weight` |
| TTS model | `s.t.m.` | `blk.0.attn_q.weight` |
| TTS projector/codec | `s.t.p.` | model-specific acoustic tensor |

Prefixes are stripped only while creating a filtered/materialized component
view. The bundle is read-only at runtime. The packer rejects a base file that
already uses a reserved prefix and rejects names longer than the GGML limit
after namespacing.

## Metadata

Base metadata remains unprefixed. Each embedded component's metadata is copied
under:

```text
robit.audio_bundle.component.<view>.kv.<original-key>
```

Bundle-level keys are:

| Key | Purpose |
|---|---|
| `robit.audio_bundle.schema` | Exact artifact schema identifier |
| `robit.audio_bundle.manifest` | Compact JSON contract, sources, and component map |

The component materializer removes the metadata namespace and restores the
original key. It must reproduce architecture, tokenizer, tensor names, shapes,
types, order, and packed tensor bytes exactly.

## Manifest shape

The embedded manifest includes:

```json
{
  "schema": "robit.ollama-monolithic-omni.v3",
  "physical_bundle_artifacts": 1,
  "container": "GGUF v3",
  "container_format": "robit-namespaced-multigraph-gguf-v1",
  "ollama_delivery": "attach as application/vnd.robit.ollama.omni.bundle.v1+gguf layer",
  "runtime": {
    "stock_ollama_direct_bundle_import": false,
    "stock_ollama_logical_tag": true,
    "custom_media_handler_required": true
  }
}
```

`component_files` records source label, filename, byte size, SHA-256,
architecture, tensor count, and storage prefix for every view. Release records
add immutable upstream revisions, licenses, toolchain commit, and live tests.

## Packing

```bash
.venv/bin/qwen-omni pack \
  --base-gguf ./components/base.gguf \
  --base-projector-gguf ./components/base-projector.gguf \
  --comprehension-gguf ./components/comprehension-model.gguf \
  --comprehension-projector-gguf ./components/comprehension-projector.gguf \
  --tts-gguf ./components/tts-model.gguf \
  --tts-projector-gguf ./components/tts-projector.gguf \
  --base-source org/base@revision \
  --base-projector-source org/base-projector@digest \
  --comprehension-source org/omni@revision:q4_k_m \
  --comprehension-projector-source org/omni@revision:mmproj-q8_0 \
  --tts-source org/tts@revision:q4_k_m \
  --tts-projector-source org/tts@revision:mmproj-q8_0 \
  --out ./release/omni-sidecar.gguf

.venv/bin/qwen-omni inspect ./release/omni-sidecar.gguf
```

The writer uses a sibling `.partial` file and atomically renames only after the
GGUF is complete. `omni-inspect` validates the schema, manifest declarations,
and non-empty tensor views. It does not substitute for inference tests.

## Attaching to Ollama

First create or copy a normal Ollama tag whose standard layers already pass
text, vision, tools, and thinking. Then attach the validated custom layer:

```bash
.venv/bin/qwen-omni attach \
  robit/example-omni:q4km \
  ./release/omni-sidecar.gguf

.venv/bin/qwen-omni resolve robit/example-omni:q4km
```

`omni-attach` computes the content digest, hard-links or copies the blob into
the local Ollama store, replaces any older Robit Omni sidecar entry, and
atomically updates the manifest. It preserves every standard layer.

The resulting descriptor is:

```json
{
  "mediaType": "application/vnd.robit.ollama.omni.bundle.v1+gguf",
  "digest": "sha256:<bundle-digest>",
  "size": 38843038144,
  "annotations": {
    "io.robit.omni.schema": "robit.ollama-monolithic-omni.v3"
  }
}
```

Stop concurrent model create/copy operations while attaching. This is a local
manifest operation because upstream Ollama's Modelfile grammar has no custom
layer directive.

## Runtime materialization

Runtimes without a filtered mmap implementation create disposable views:

```bash
.venv/bin/qwen-omni prepare robit/example-omni:q4km \
  --out ./runtime-cache
```

Equivalent single-view extraction is available through `omni-unpack`. The
runtime cache is not a release artifact and must be removed after workers stop.
The tag retains the source sidecar and can recreate it at any time.

### Indexed preparation in the upgrade fork

`prepare --reuse --overwrite` reuses verified component files and repairs only
missing, changed or unverified views. Bootstrap uses this path. Preparation
without `--reuse` retains the previous overwrite/refusal behavior. All misses
share one parsed sidecar tensor inventory; the existing resolver still performs
its own schema inspection. The standalone `unpack` operation remains explicit.

The generated `component-view-index.json` uses schema
`robit.omni.component-view-index.v1`. It binds the extractor version, installed
Ollama layer descriptor, source path and source inode/size/timestamps to each
output filename, SHA-256 and materialization report. Every reuse reads and hashes
the complete output; same-size corruption cannot be accepted merely because a
file exists. A changed source, unsupported index or malformed bounded receipt
invalidates reuse. Source changes during preparation prevent publication of a
new receipt. Output symlinks are rejected. This receipt inherits the resolver's
trust in the installed Ollama descriptor; it is not an authenticity signature.

The preparation result adds `component_index` with reused/materialized view
lists and byte totals, and `reused` on each view. `--overwrite` permits repair;
with `--reuse`, valid views are retained. Reused reports retain the original
extraction timing, so use external wall timing to measure preparation latency.
A warm hit leaves the component files and index unchanged. Receipts are
disposable alongside their views; delete them only after their workers stop.

This is validated materialization reuse, not direct filtered mmap, tensor-content
deduplication or reduced GPU residency. Both existing sidecar schemas and their
matching model/codec tensors remain unchanged. The benchmark
`runtime/benchmark_component_views.py` compares an original checkout with this
path using installed weights, fresh CPU processes and an explicitly new cache.

## Why direct stock loading is rejected

Stock llama.cpp chooses one graph from `general.architecture` and then checks
the GGUF tensor inventory. The v3 sidecar has heterogeneous inventories by
design, so a direct Ollama import/load cannot satisfy that graph's expected
tensor count. The following alternatives were tested and rejected:

- an all-namespaced tensor model layer: imports but fails required-tensor
  accounting at load;
- bytes appended after the base GGUF: removed when Ollama normalizes/imports
  the model;
- a pre-tensor opaque gap: rejected because tensor offsets must be contiguous;
- large byte arrays in GGUF metadata: eagerly loaded into host memory and not a
  practical 20+ GB payload mechanism.

Changing this outcome requires upstream loader/runtime work, not a metadata
flag. The custom manifest layer is therefore part of the ABI.

## Verification checklist

1. Record the sidecar SHA-256 and byte size.
2. Confirm exact expected tensor counts for all six views.
3. Materialize every view and compare it with its pinned source digest.
4. Confirm `ollama show` still advertises completion, vision, tools, and
   thinking from the standard layers.
5. Run live audio, image, video, and TTS component probes.
6. Push the tag and pull it back; `omni-resolve` must find the same layer digest.
7. Remove only disposable materialized views after all workers stop.
