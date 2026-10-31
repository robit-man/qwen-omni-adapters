# Completed acoustic repair trials, 2026-10-31

The Qwen richer-target run completed 1,600 steps in 11,189.6 seconds. Validation
loss selected checkpoint 750 (0.19299 versus initial 0.39992). Final checkpoint
1600 was also tested on the assigned A100 at the target 16K context: speech and
bark are correct, but church bells are still described as a synthesizer tone.
Children are described as animals and a clap as a gunshot. These results do not
clear semantic parity; no model was published.

Only the matching final audio projection changes. Its final-checkpoint SHA-256
is `ff36810539f12fbad1e21f1fc3508380e97ace510586cc0c1fdb21fecfecac2c`;
858 other audio/vision tensors, the reasoning checkpoint, and TTS weights remain
unchanged. The final checkpoint is not promoted over the selected checkpoint.

## Speech validation on the assigned A100

Same 160 inputs, including 154 speech clips with 3,336 reference words and six
negative controls, 4K context and F16 KV. Original source/native binaries and
projector are compared against the repaired native path and checkpoint 750.

| Speech source | Original Qwen bridge WER | Repaired Qwen 750 | Full Omni |
| --- | ---: | ---: | ---: |
| All speech | 18.765% | 7.224% | 20.504% |
| Synthetic speech with procedural ambience | 3.355% | 1.842% | 8.026% |
| Natural LibriSpeech with local augmentation | 6.734% | 6.229% | 1.515% |
| TTS word-list recovery | 43.781% | 14.403% | 45.254% |

All three runs produce zero transcripts on the six negative controls. The
natural-speech improvement is only three fewer errors across 594 reference
words; no statistical significance is claimed. Natural speech still trails
full Omni. Validation and aggregate WER do not establish complete release parity.

## Trials rejected without runtime changes

- Training both existing audio projection layers for 400 steps matches the
  initial graph exactly, but never beats initialization on validation loss.
  Native output regresses the garden sentence, bark and bell recognition.
  The four-tensor export is valid; that does not make its predictions useful.
- A cached-feature batch of four preserves the intended per-example loss
  mathematically and runs the frozen-text step 2.77 times faster, but fails the
  predeclared numerical gate: gradient cosine 0.99145, relative RMSE 0.13065,
  maximum individual loss difference 0.00702. No optimizer step or trainer
  change was applied. The speedup excludes encoder and data I/O.
- A single nonlinear classifier trial over frozen audio mean/std features
  selects validation mAP 0.45839 and annotated top-1 34.58%, versus the earlier
  ridge's 0.41385 and 32.5%. It misses its preset improvement thresholds, the
  recorded alarm/bark/bell checks, and silence rejection. It is not integrated
  or packaged. Its scores are uncalibrated, its labels are weak, and pooled
  features lose temporal order.
- The selected Ornith850 bridge also makes several of the same mistakes with
  the full-precision frozen training trunk. Deployment quantization alone is
  therefore insufficient to repair those errors.

The next repair needs stronger acoustic supervision and actual native quality
improvement. No rejected classifier, projection trial or optimized training path
is enabled. [The receipt](completed-repair-trials-20261031.json) retains actual
outputs, parameters, rejected thresholds and source metrics.

## Portable binding on Orin

The actual stock-pushed manifest from the passing A100 transport test was
installed on Orin with all layer hashes checked. Source `59cedcd` resolves its
JSON binding with descriptor annotations absent, prepares component aliases with
zero materialized bytes, passes the trained-prefill ABI2 guard and deployment
doctor, and sees the packaged 16K default. This was a CPU metadata/preparation
check; it started no new CUDA worker. The original core remained healthy.
[The Orin receipt](registry-binding-orin-20261031.json) records that scope.
