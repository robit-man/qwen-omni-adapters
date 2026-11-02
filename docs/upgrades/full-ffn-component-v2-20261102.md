# Full FFN audio component binding

The trained research adapters cover all 32 Ornith or 64 Qwen3.8 FFN blocks.
The original component validator accepts only the last eight blocks, so it
cannot prepare these adapters for native GGUF qualification.

`bind-audio-adapter --scope full-ffn` now explicitly writes
`robit.omni.conditional-audio-lora.v2` with
`robit.audio_adapter.scope=full-ffn`. The default remains `last8-ffn`, using the
unchanged v1 ABI. Tensor count never implicitly enables the larger scope.

| Component ABI | Required layout | File limit |
| --- | --- | --- |
| v1 | Last eight blocks, exactly 48 tensors | 64 MiB |
| v2 | Every base-model block, exactly six tensors per block | 256 MiB |

Both layouts require qwen35, rank eight, alpha eight, finite F32 tensors,
complete gate/up/down pairs, exact base-derived shapes and matching base and
trained-projector digests. The v2 bound accommodates Qwen's approximately
132 MiB tensor payload. It does not enlarge the v1 limit. Impossible tensor
counts are rejected before constructing a full-block inventory.

The existing audio-component media type and required v3 layer binding are
retained. Older readers reject the new required component ABI. Current readers
verify it through attach, annotation-free descriptor resolution and direct
component preparation. Binding preserves every tensor value and neither copies
the base/projector nor changes audio-only execution scope.

The new contract tests fail before the change and pass afterward. They cover
32/64-block binding, unchanged tensor bytes, explicit selection, incomplete early
blocks, inconsistent ABI/scope declarations, descriptor round trips, both size
limits and the CLI. Large-size admission uses sparse fixture padding; it is not
a real Qwen model or an inference-quality test.

The actual Ornith B5500 export preserves all 192 tensors exactly. Its pinned
patched Ollama runner loads all 192 tensors and completes 58 raw validation
cases. Five speech outputs match the existing native controls byte-for-byte,
including full reference-audio/text cloning, both defaults and A → B → A.
The first harness stopped after the 58 cases because its optional silence parser
returned `None` where the API correctly returned an empty string. Saved XML
proves equivalence; a second window runs only the missing controls. Both windows
shut down and release their own leases normally. See
[the native evidence](native-ornith-full-ffn-192-20261102.json).

Native Q4 speech has 27 errors per 594 words versus source BF16's 28 and legacy
nine. Environmental and temporal parity still fails. Native 384-tensor Qwen,
final Orin and full capability/resource qualification remain gates. This is
pinned patched Ollama support, not unmodified stock compatibility. No new model
selection or public weight publication is enabled by this change.
