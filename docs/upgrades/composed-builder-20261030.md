# Rebuilding the composed native patch set

Acceptance: rerunning the native installer against its fully patched checkout
must succeed without undoing existing features. The frame-alignment patch failed
its reverse-apply check after the later CNN-tail patch changed the chunk-size
line in its context. Fresh sequential application worked; rebuilding the same
tree failed before compilation.

Both builders now recognize the composed audio alignment through four feature
markers across its three source files, following the existing TTS composition
checks. Incomplete sources still fall through to patch validation. A regression
fixture applies audio alignment, adds the later context change, proves reverse
application fails and then verifies a second installer pass succeeds unchanged.

The real Linux builder completes twice on the full sixteen-patch checkout.
The native server and mtmd shared-library hashes remain identical to the earlier
video/PCM qualification: the installer change does not change the execution
graph. All 6,282 source tests pass in 285.02 seconds. Windows receives the same
composition check; native Windows execution was not tested here.

The contemporaneous projection experiments remain unqualified. On 154 replay
validation speech clips (3,336 words), Ornith's label-trained checkpoint has
7.73% WER and richer-target step 400 has 7.28%. The natural-speech subset worsens
from 4.38% to 5.56%; full Omni scores 1.52% on that subset. Aggregate WER therefore
does not establish natural-speech parity. The word-list and synthetic-speech
subsets have different behavior and are retained separately in the local receipts.

Richer-target step 800 restores the fixed clean sentence and dog bark but calls
church bells a gong/steel drum. A diagnostic mean/std ridge classifier over the
frozen encoder reaches validation macro AP 0.414 across 152 represented classes,
yet also misses the known real sounds and gives unsound labels for silence. It
is not enabled or packaged as a fix. Temporal pooling loses order, so this does
not prove the encoder lacks the necessary information. Full scheduled training
and subsequent independent qualification remain pending; nothing is published.
