# Exact digital silence

Native inference invented a click on A100 and a device-interface sound on Orin
from an all-zero WAV. The adapter now proves silence directly for complete,
16 kHz mono PCM16 input whose samples are all zero. This avoids those phantom
observations and unnecessary inference. It uses no VAD or amplitude threshold.

The actual HTTP check completes silent live turns with no comprehension,
language or TTS requests. A one-LSB nonzero control still attempts comprehension.
The check also verifies the complete RIFF container and a single data stream.
Multiple data/format chunks, alternative WAV streams and incomplete/trailing
bytes cannot establish silence. Ordinary metadata chunks remain supported.
Mixed media, incomplete WAVs, explicit text questions and reference-voice input
retain their normal paths. Full validation passes 6,358 tests and all remaining
source gates. No existing deployment was restarted for this change.

[Machine-readable evidence](digital-silence-20261101.json) retains the bounded
HTTP outcomes and source hashes. This fixes only exact silence; native event
errors and all other release gates remain open. It does not qualify a model.
