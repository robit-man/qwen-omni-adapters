# Verified model profiles

The adapter runtime is graph-agnostic at the semantic boundary, but a logical
tag must be paired with the exact standard language/projector tag from which
its sidecar was packed. The launcher provides these audited profiles:

| Launcher profile | Logical model | Language backend | Base identity |
|---|---|---|---|
| `qwen38` | `robit/qwen3.8-27b-e03-obliterated-omni:q4km` | `robit/qwen3.8-27b-obliterated-e03:27b` | Qwen3.8 E03 obliterated |
| `ornith15` | `robit/ornith-1.5-omni:q4km` | `robit/ornith-1.5:9b` | stock Ornith 1.5 9B |
| `ornith15-obliterated` | `robit/ornith-1.5-obliterated-omni:q4km` | `robit/ornith-1.5-obliterated:9b` | OBLITERATUS Ornith 1.5 9B |

```bash
./deploy.sh ornith15
./deploy.sh ornith15-obliterated
```

The launcher exports `OMNI_MODEL` and `OMNI_LANGUAGE_MODEL`; the supervisor
then proves that both resolve to the same standard model/projector blobs before
loading CUDA media workers. A mismatched pair fails closed.

All three profiles use the same wire contract and the same pinned
Qwen3-Omni/Qwen3-TTS media graphs. What changes is the Ollama-owned language,
native-image, tool, and optional-thinking base. Consequently:

- ordinary `/api/chat` text/tool/image behavior comes from the selected base;
- audio/video comprehension remains a separate Qwen3-Omni graph;
- tagged semantic evidence is passed to the selected base;
- spoken output remains text-conditioned Qwen3-TTS;
- no profile claims native hidden-state fusion.

The two Ornith profiles have 262,144-token contexts and use
`num_predict=-1`, Ollama's unlimited model-level generation setting. Clients
may still choose a bounded per-request limit.

The build records and exact six-view SHA-256 inventories live in the
[`fine_tuning_suite` Ornith release documentation](https://github.com/robit-man/fine_tuning_suite/blob/main/docs/omni-adapter/ornith15-release.md).
