# Runtime cleanup and weight retention

The repository distinguishes published source artifacts from disposable local
runtime data.

## Normal runtime cleanup

`portal/start.sh --stop` removes the portal-owned component views by default,
after all workers exit and the GPU lease is released. Set `OMNI_KEEP_CACHE=1`
only when a near-term restart justifies retaining the derived GGUFs.

Inspect and remove leftover derived files with:

```bash
./scripts/cleanup_runtime.sh
./scripts/cleanup_runtime.sh --apply
```

The script is deliberately narrow. It refuses a live supervisor, prints exact
candidates, and never traverses an Ollama model directory.

## Build/release cleanup gate

If an agent uses this repo alongside a model build, cleanup begins only after:

1. local model creation and all expected capability tests pass;
2. every requested Ollama push completes and the remote tag is fetched;
3. every requested Hugging Face upload completes and its remote inventory and
   model card are verified;
4. source revision, quantization, component digests, licenses, configuration,
   and test results are recorded.

Then remove only files owned by that completed run: downloaded safetensor
shards, merged checkpoints, full-precision GGUF intermediates, partials,
redundant quantizations, and materialized views. Measure the exact run directory
before and after. Shared Hugging Face caches and donor weights require a
reference check before deletion.

Never manually remove `/srv/ollama/models/blobs`, another Ollama blob tree, or
manifest files. Use `ollama rm <exact-obsolete-tag>` after proving the tag is no
longer referenced.
