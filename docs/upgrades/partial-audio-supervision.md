# Acoustic labels with unavailable speech transcripts

Weak sound-event annotations do not establish whether speech is absent or what
words were spoken. Training an empty transcript for every such recording teaches
an unsupported ASR target; discarding every uncertain recording removes useful
human-activity and environmental examples.

The optional build integration in
[`partial_supervision.py`](../../src/qwen_omni_adapters/partial_supervision.py)
retains acoustic supervision while masking unknown transcript targets. Mark a
record `"transcript_supervision": "unavailable"`, leave its transcript placeholder
empty, and supply its acoustic observation. The collator masks the complete
speech target, including a BPE token that also contains the following tag's
opening bracket. Acoustic and stop-token labels remain scored. Ordinary replay
and known empty-transcript controls keep their original supervision.

In the optional audio-bridge build environment, install the wrapper before
constructing the trainer's collator:

```python
from training_suite.training import audio_bridge_training as training
from qwen_omni_adapters.partial_supervision import install_partial_speech_collator

install_partial_speech_collator(training)
report = training.train_audio_bridge(config)
```

The caller supplies the trainer and its offset-capable tokenizer. This module
imports neither Torch nor the fine-tuning suite and adds no inference dependency.
Reject records containing unverified words, a missing placeholder, mismatched
token alignment, or no remaining acoustic target before any optimizer step.

Both supported real tokenizers pass the CPU integration audit: encoder/replay
tensors remain identical, nine speech tokens are masked including the merged
boundary token, and acoustic/stop labels remain exact. The repository regression
uses captured token IDs and character offsets. The versioned implementation's
executable AST matches the audited helper used by both ongoing repair jobs.

This establishes correct partial-label handling. It does not establish improved
model quality or clear the publication gate. See the
[training checkpoint](class-guided-training-20261031.md).
