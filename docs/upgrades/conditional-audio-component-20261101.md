# Conditional audio adapter packaging

The indexed tag can now carry a required audio-only FFN LoRA without replacing
the reasoning checkpoint or duplicating the shared TTS sidecar. Before this
change, trained LoRA exports could only be supplied by qualification scripts;
the deployable runtime could silently omit that required computation.

`qwen-omni bind-audio-adapter SOURCE --base-model BASE --projector PROJECTOR --out OUT`
copies the small adapter export with all tensor values unchanged and records the
exact model/projector digests plus `robit.omni.conditional-audio-lora.v1` ABI.
It accepts the matching Qwen3.5-family last-eight-layer rank8/alpha8 FFN pairs,
requires all 48 tensors to be finite F32 with matching dimensions, and requires
the trained-prefill.v2 projector contract. Binding does not establish quality.

`qwen-omni attach TAG SIDECAR --audio-adapter OUT` installs that required component
as `application/vnd.robit.ollama.omni.audio-adapter.v1+gguf`. The same hashed JSON
binding layer now uses schema `robit.omni.layer-binding.v3` for these tags. Its
existing media type ensures older resolvers reject the required new ABI. Tags
without an adapter keep their v2 binding and existing execution behavior.

The portable supervisor and Linux portal load the adapter disabled. Only audio
perception requests enable it; native vision, final language, thinking and tools
explicitly send a nonempty scale0 setting. Mixed media splits visual and acoustic
perception with scale0 and scale1, respectively, through the same primary worker.
The adapter requires a single local trained-bridge endpoint for both stages.
No global Ollama ADAPTER layer is used.

Both Ornith and Qwen A100 mechanism gates passed six real TCP/native checks each, including
off→on→off text/vision/tool controls, separated thinking, mixed media, and
concurrent audio-on/text-off requests through one PID. All 12 backend requests
carry explicit scales and multimodal requests retain `cache_prompt:false`.
Each worker exited after its bounded window. Each used16K total context with two8K slots; this is a mechanism/concurrency check, not the final profile qualification.

Stock Ollama0.40.11 pushed and pulled the 12,587,008-byte bound adapter and
772-byte binding through a loopback registry. Both layers actually uploaded and
downloaded; existing immutable neural layers were prepositioned with hardlinks.
The pulled manifest resolves and prepares direct TTS views with zero materialized
bytes. The shared TTS sidecar digest is unchanged. The actual prior resolver at
65e86bc refuses the new required binding. Temporary registry aliases and CPU
servers were stopped/removed; no public registry or original deployment changed.

Source validation passes 6,380 tests and the full validation script. The first
native harness attempt used the wrong tool-argument representation; a corrected
harness passed without changing production source or model requests. The first
registry fixture omitted Ollama's required Location header; the corrected fixture
passed without a production patch. Preserve these failed attempts as evidence.

The mechanism uses already-rejected weights and a TRAIN fit fixture. It clears
packaging and execution isolation only. It does not clear environmental-audio
parity, assigned-target qualification or either public release gate. See
[the compact live evidence](conditional-audio-component-20261101.json).
