# Expanded legacy full-Omni comparison

The same 13 diagnostic WAVs used for the conditional-trunk adapter checkpoints
were evaluated with the unchanged legacy full-Omni model and encoder. Every
input ID and SHA-256 matches the candidate reviews. Native16 runs on a scoped
A100 at 16K context, F16 KV, with actual worker GPU residency proved.
The legacy model is a benchmark reference only; its worker and lease were
released after the comparison.

The broader 160-clip speech and 240-clip environmental legacy references
already exist in [the earlier qualification](rich-projection-qualification-20261031.json).
They were not rerun. This check matches the newer adapter diagnostic's exact
13 inputs and 16K context; it adds no unique benchmark recordings.

| Input | Legacy observation | Rejected candidate examples |
| --- | --- | --- |
| Church bells | Church bells ringing | Piano, bowed string instrument, struck chord |
| Printer | Printer printing | Drill, fireworks, zipper |
| Group voices | Children speaking and playing | Bark, bird, crow |
| Gunfire | Rapid automatic gunfire | Boom or generic explosion |
| Passing siren | Vehicle siren approaches, passes and fades | Engine, car or train without siren |

The legacy model also returns the exact sentence fixture and identifies the
dog bark, clap, static noise, electronic alarm and sustained tone. It
hallucinates an electronic beep on the silence fixture; that failure remains
in the [receipt](legacy-expanded-audio-20261031.json). Its human-voice transcript
is unverified output, never a reference transcript or new training truth.
This small repeated diagnostic set does not establish broad accuracy or
justify relaxing any candidate gate.

The first private comparator attempt incorrectly reused the trained-bridge
request profile, including `enable_thinking=false`. Existing runtime code
explicitly excludes that override for the legacy multimodal model. Empty
responses from that attempt are retained as invalid benchmark evidence. The
separate successful rerun uses the existing legacy request defaults and no
adapter. Production code and service configuration were unchanged.

The valid comparison confirms that the rejected checkpoints lose capabilities
present in their legacy reference. Concurrent training prevents an isolated
latency or memory comparison. Publication remains gated on repaired semantic
parity and the complete target-host qualification.
