# Verified model profiles

The guided Jetson launcher provides two audited trained-audio-bridge profiles:

| Launcher profile | Logical model and sole language trunk | Artifact weights | Base identity |
|---|---|---:|---|
| `qwen38` | `robit/qwen3.8-27b-e03-obliterated-omni-audio-bridge:q4km` | 18.33 GiB | Qwen3.8 E03 obliterated |
| `ornith15` | `robit/ornith-1.5-omni-audio-bridge:q4km` | 8.15 GiB | standard Ornith 1.5 9B |

```bash
./deploy.sh                         # arrow-key guided install/upgrade
./deploy.sh ornith15                # non-interactive profile selection
./deploy.sh qwen38
```

The launcher sets `OMNI_MODEL` and `OMNI_LANGUAGE_MODEL` to the same logical
bridge tag. Its standard model layer is the sole language trunk, so no adjacent
Ollama language runner is loaded. The resolver proves the standard language
and combined native-vision/Omni-audio projector layers before CUDA startup.

Both profiles use the same adapter wire contract and Qwen3-TTS sidecar. What
changes is the language, native-image, tool, and optional-thinking trunk:

- ordinary `/api/chat` text/tool/image behavior comes from the selected trunk;
- the frozen Omni audio encoder and trained final projection feed that trunk;
- tagged speech and acoustic evidence remain provenance-separated;
- spoken output remains text-conditioned Qwen3-TTS;
- the full Omni Thinker and a second Ollama language copy are absent.

That makes these releases drop-in replacements at the logical adapter/daemon
boundary. They are not standalone stock-Ollama audio/TTS models: stock Ollama
does not execute the custom sidecar, so clients that need Omni audio or speech
must continue to use this repository's daemon API.

Standard Ornith advertises a 262,144-token model context; the published bridge
tag bounds default generation to 16,384 tokens. Runtime memory admission still
selects a safe active window for the Jetson.

Release artifacts and cards are mirrored at
[Ornith on Hugging Face](https://huggingface.co/cudabenchmarktest/Ornith-1.5-9B-Omni-Audio-Bridge-GGUF)
and [Qwen3.8 on Hugging Face](https://huggingface.co/cudabenchmarktest/Qwen3.8-27B-E03-Obliterated-Omni-Audio-Bridge-GGUF).

Legacy full-Omni logical tags remain supported through explicit advanced
`OMNI_MODEL`/`OMNI_LANGUAGE_MODEL` configuration, but they are intentionally
absent from the guided Jetson installer menu. After the desktop service is
installed, the indicator's **Models** submenu exposes those two legacy tags
alongside the compact releases, including download progress and exact-tag
activate/load/unload/delete actions. The rejected Ornith obliterated release is
not in either catalog.
