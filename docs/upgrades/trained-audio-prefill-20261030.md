# Exact trained audio framing — 2026-10-30

The native rendered request differed from the training sequence in three ways:
extra audio boundary tokens, retained caller text on describe requests, and a
missing leading newline on the extraction directive. Lower training loss did
not address these inference-only mismatches.

Trained audio requests now use the same directive and native no-thinking
prefill for chat, describe and transcribe. The audio pass of split video
perception uses that directive too. Caller text stays in the language stage.
The language stage's native `think` and tool behavior remain independent.

Newly exported trained projectors declare:

- `robit.audio_bridge.prefill_abi=robit.qwen3a.trained-prefill.v2`
- `clip.audio.add_boundary_tokens=false`
- `robit.audio_bridge.prompt_sha256=9f73862652e0226ec3f9690f0a783d1c21dc1113285b4dc18d0edd51f2766758`

The explicit flag controls native Qwen3A boundaries. Existing projectors retain
their default boundary policy. The native library exports a CPU-only ABI query;
portal startup, the portable supervisor and the Tegra context launcher reject a
new projector when its prompt signature or matching native library is unsupported.
The metadata writer requires a fresh destination and verifies every ordinary
and quantized tensor byte before publishing the output without overwriting files.

Example preparation and verification:

```bash
.venv/bin/python -m qwen_omni_adapters.audio_prefill prepare \
  --source /path/to/trained-projector.gguf --output /path/to/new-projector.gguf
.venv/bin/python -m qwen_omni_adapters.audio_prefill verify \
  --projector /path/to/new-projector.gguf --binary vendor/llama.cpp/build/bin/llama-server
```

The A100 native integration compares rendered prefix/suffix strings directly
against the pinned fine-tuning source for all three tasks. They match exactly.
Native tokenization emits no extra text tokens with the new flag; the otherwise
identical legacy projector still emits IDs 248070/248071. Every native worker is
proven resident on the leased GPU before its encode. A real archived old native
library is refused before model startup. Six newly generated full-reference
TTS outputs pass raw PCM continuity and cache-on/off equality.

All 15 patches compose on the pinned backend, and the changed files match the
compiled sources. The full source/contract/browser/installer gate passes
**6,280 tests** in 286.35 seconds. The first native harness omitted the required
`stream:false`; its corrected fixture passes without another production patch.

The step-400 Ornith projection's 860 tensor payloads remain byte-identical
through metadata export. On the actual native path, synthetic beeps and the
real clock alarm are recovered, and clean speech remains exact. Dog barking
is still called a bird; church bells are still called music. Silence/noise
descriptions are also not a full semantic pass. **Publication remains gated.**
These are component checks on A100; final installed Ornith/Orin and Qwen/A100
qualification must use the selected, repaired checkpoints.

See [native controls and integrity](trained-audio-prefill-20261030.json) and
[the retained sound outputs](trained-audio-prefill-probes-20261030.json).
