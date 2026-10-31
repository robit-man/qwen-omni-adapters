# Preserve indexed binding through stock Ollama delivery

Stock Ollama 0.40.11 drops layer annotations during push. The previous indexed
v2 package therefore lost `io.robit.omni.binding.v2` on a real push/pull round
trip, and the receiving runtime rejected it before loading. Keeping the binding
only in local staging did not make the published artifact executable.

Attachment now includes a small content-addressed JSON binding layer,
`application/vnd.robit.ollama.omni.binding.v2+json`, alongside the unchanged
GGUF sidecar. It binds the exact standard language/projector digests and sizes
to the sidecar digest. The selected Ornith binding is 379 bytes. Both backbones
can keep sharing their identical TTS GGUF; no neural weights or native graph
changed. Existing annotation-only v2 and legacy tags remain readable.

Resolution verifies binding size, bounded JSON and SHA-256, rejects duplicate
layers, and checks every supplied binding copy against the selected weights.
A correct annotation cannot hide a damaged JSON layer; a correct layer cannot
hide a contradictory annotation. Tests also reject changed backbones/projectors,
missing blobs, invalid JSON, path-like digests, and excessive payload sizes.
Repeated attachment retains one binding and one shared sidecar.

The real stock delivery test passed:

1. Stock push uploaded the binding through POST/PATCH/PUT to a loopback registry.
2. Stock pull downloaded it into a fresh derived model store that initially held
   hardlinks to the preexisting neural/configuration blobs, without that binding.
3. The pushed manifest remained byte-identical after pull, with descriptor
   annotations absent and the binding layer present.
4. Resolution and direct component preparation succeeded with zero materialized
   component bytes.
5. The pulled model loaded on the broker-reserved A100 GPU0 and passed native
   `think:false`, separated `think:true`, and typed tool-call checks. It used the
   model's **16,384-token default** without a request context override. Vulkan
   was disabled in this isolated test so only its reserved CUDA device was used.

This deliberately tests transfer of the new metadata blob. The large neural
blobs were already present; public weight upload and remote verification remain
separate release requirements. No public registry changed. The shared Ollama
service and the restored Orin deployment were preserved.

The source gate passed before the patch (6,282 tests) and after it (6,295),
including all validation-script harnesses. The focused sidecar suite passed
39 tests. [The receipt](registry-binding-20261031.json) records the actual failed
and repaired round trips and the exact binding payload.

Run `qwen-omni attach` after `ollama create`, then verify a stock push/pull and
`qwen-omni resolve` before publishing a new tag. The edge Modelfile must set
`PARAMETER num_ctx 16384`; changing this resident default retains native context
metadata. See [the ABI contract](../gguf-abi.md).
