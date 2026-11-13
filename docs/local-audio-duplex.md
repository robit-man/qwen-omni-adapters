# Local workstation audio handoff — 2026-11-12

This records a verified repair on the Razer Blade 15 Advanced Mid 2021
(RZ09-0409) workstation. It is not a default configuration for other hosts.
Re-enumerate the actual devices and drivers before applying any host change.
Keep this machine's setup out of generic device discovery and tool routing.

## Failure and required invariant

The built-in microphone array was absent from ALSA while an external analog
jack appeared as a selectable microphone. The firmware NHLT table described a
digital microphone endpoint, but three active modprobe files forced
`snd-intel-dspcfg dsp_driver=1` (legacy HDA):

- `/etc/modprobe.d/99-force-hda.conf`
- `/etc/modprobe.d/99-hda-legacy.conf`
- `/etc/modprobe.d/99-audio-hda.conf`

The existing speaker amplifier needs a vendor-coefficient initialization
sequence **followed by an audio-controller reset**. Switching to SOF while
running the amplifier sequence without that reset produced distorted capture
and silent speakers. The old reset helper explicitly rejected the SOF driver.
Restoring files and mixer settings alone did not establish audible recovery.

Both directions must work concurrently. Validate known spoken playback captured
by the actual microphone; do not substitute a sink monitor, an available-device
flag, a successful playback command, or a single spectral peak for that check.
Keep the reference transcript separate from observation tags when comparing
ASR; malformed encoder tags can otherwise create a false negative.

## Accepted persistent configuration

- `/etc/modprobe.d/90-omni-audio-sof.conf` selects
  `options snd-intel-dspcfg dsp_driver=3`. The three legacy-forcing `.conf`
  files above were removed after backup. Old `.disabled` copies are not active.
- `/usr/local/sbin/alc298-speaker-init.sh` and its existing coefficient sequence
  remain the amplifier initialization source.
- `/usr/local/sbin/alc298-audio-reset` retains its exact PCI identity guard and
  now resolves the controller's **actual bound driver**. It accepts only
  `snd_hda_intel` or the observed `sof-audio-pci-intel-tgl`; it unbinds/rebinds
  that same driver. Do not replace this with an unconditional legacy-HDA reset.
- The enabled `/etc/systemd/system/alc298-speaker-init.service` keeps the full
  initialization → reset order before the desktop/user audio services start.
- `/etc/systemd/system-sleep/alc298-post-resume` also keeps that full order.
  An initialization failure does not proceed into a controller reset.
- `/etc/udev/rules.d/99-razer-alc298-hda-power.rules` applies the existing
  keep-awake policy on `add`, `change`, and `bind`. The controller's
  `power/control` is `on`; a rebind must not silently leave the amplifier under
  runtime autosuspend.
- Initramfs images are rebuilt with the accepted SOF selection. Recheck them
  after changing boot-time module configuration.

Do not run the coefficient sequence alone against active desktop streams.
Quiesce audio clients for a manual handoff and preserve the complete sequence.
Card numbers changed during testing: use observed ALSA card identities rather
than assuming `card 0` is the Intel codec.

## Desktop defaults

The SOF card uses the `HiFi` profile. WirePlumber persists these selections in
`~/.local/state/wireplumber/default-profile` and `default-nodes`:

```text
card:   alsa_card.pci-0000_00_1f.3-platform-skl_hda_dsp_generic
input:  alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp_6__source
output: alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp__sink
port:   [Out] Speaker
```

The input is **Digital Microphone**, not **Headphones Stereo Microphone**.
Existing recording streams also need to move to the selected input; changing
the default alone does not necessarily move an already-open application.
The user requested full speaker volume during verification. Do not silently
reset that preference. Omni was temporarily muted to prevent its own test
playback from becoming a conversational turn, then restored to its original
unmuted state.

## Verification and recovery evidence

Direct ALSA capture of all four physical DMIC channels recovered the speech
played through the physical speakers with no clipping. A second test used
ordinary default-device `pacat` playback and mono 16 kHz `parecord` capture:
14.56 seconds captured, zero clipped samples, and local ASR recovered the known
spoken content (normalized word-sequence match 0.747). The user also confirmed
simultaneous microphone and speaker operation. Neither reboot nor suspend was
performed as part of that live verification; boot/resume configuration was
persisted and checked separately.

Root-owned backups, original ALSA state, and compact verification receipts:

```text
/var/backups/omni-audio-duplex-20261112T144106/
```

Its `accepted` marker disarms the temporary rollback controller at
`/usr/local/sbin/omni-audio-duplex-control`. The temporary rollback timers must
remain stopped after acceptance. Do not re-arm an old legacy rollback merely
because the microphone is momentarily absent during a handoff: it restores
speaker-only legacy mode and removes the digital microphone again.

Any future rollback must be explicit about that loss of input, preserve the
current accepted state first, and verify actual speaker playback afterward.
Do not reload GPU/model services to repair these host audio routes.

## Follow-up: microphone heard, conversational reply absent

After the hardware repair, the public browser call was also checked through
the full comprehension → language → TTS path. A controlled spoken arithmetic
question returned “Four.” with ordered audio events and both
`speech_synthesized=true` and `audio_streamed=true`. The same question played
through the physical speakers and recorded from the default digital microphone
also completed that route. The capture had no clipped samples. A browser call
then independently completed all three stages, and the user confirmed that
the agent responded. No model weights, audio drivers, or running services were
changed during this follow-up.

Do not infer a missing language model from an HTTP 200 or an absent reply.
Inspect the final adapter metadata: `route`, `input_transcript`,
`speech_addressee`, `tts_skipped_reason`, and the streamed-audio flags. Sound-only
input and speech confidently addressed elsewhere can intentionally stop before
language/TTS. Also inspect request timing and the queue: one observed browser
turn spent approximately 116 seconds processing while another request waited.
The cause of the earlier intermittent silence was not established by the
successful probes; do not describe it as a repaired weight or routing defect.
Preserve the working duplex configuration when investigating a recurrence.

## Later recurrence: capture gain and invalid speech evidence

Subsequent live evidence established additional failures that the earlier
successful playback test did not resolve. The digital microphone was at 25%,
which PulseAudio reports as **−36.12 dB**, while loud speaker playback still
registered. After restoring 100% / 0 dB, its volume later changed to 60% /
−13.33 dB. The portal requested browser automatic gain control. All four portal
microphone paths now explicitly disable AGC while retaining echo cancellation
and noise suppression. An existing browser stream needs a page reload and a new
call to acquire the updated constraints. Keep the speaker settings independent.

One live turn contained the purported transcript “Empty transcript: no
intelligible speech was present in the supplied audio.” The old semantic gate
accepted this as an utterance, and the conversational model repeated “Four.”
from previous history. The gate now has a typed `INVALID_SPEECH` result that
removes the false speech evidence before either browser history or reply
generation. The resident language model passed all 26 contrastive evaluation
cases, including genuine microphone questions, quoted diagnostics, ambiguous
speech, names used as subjects, and speech addressed elsewhere. This does not
prove every future ASR result will be correct.

Use an isolated test session and exclude the open browser call from controlled
speaker tests; otherwise the test utterance becomes real conversation history.
Restore each temporarily muted recording stream afterward. A newly opened
`parecord` can inherit the existing recorder's muted stream state: explicitly
check the new stream's mute before treating all-zero samples as hardware
failure. The corrected normal-level check captured 8.24 seconds at −24.26 dBFS,
peak 0.612, with zero clipping and recognizable reference speech. Speaker master
and Firefox playback remained at 100%.

The working AGX Orin was inspected read-only. At comparison time it ran the
same `c37b5f9` source and `ornith15` audio-bridge model tag; its comprehension
context was 65,536 tokens versus 16,384 here. This is configuration evidence,
not proof that the earlier false routing decision was caused by context size.

After the local adapter/portal reload, two spoken requests sent through the
public portal, each with the old arithmetic exchange in its history, returned
“Paris.” and “Seven.” with comprehension → language → TTS and nonempty ordered
PCM. Unamplified TTS measured −16.38 dBFS RMS, peak 0.798, with zero clipping.
The reload preserved the resident comprehension/TTS worker PIDs and tunnel URL;
no audio driver or service on the Orin was restarted.

## Recognized speech was still being deliberately silenced

A later browser turn correctly transcribed “Hello there. Testing.” Its semantic
route was `AMBIGUOUS`, with valid speech, but the language model returned exactly
`<observe_only/>`; the adapter correctly converted that control into silence.
Replaying the recognized words reproduced the failure. Text requests did not
use that room-response control and answered normally. The earlier successful
arithmetic probes did not establish that greetings or connection checks worked.

`omni.live_engaged` already existed and was parsed but never supplied to the
response policy. It now selects positive active-conversation context instead of
the catalog-owned room-silence control. Browser calls explicitly set it true;
older open tabs are upgraded at the adapter when they report engagement. The
local harness keeps its actual engagement-window state, and speech explicitly
addressed elsewhere still stops before reply generation. Final live metadata
includes `adapter.live_engaged` for diagnosis; this never establishes speaker
identity. The public endpoint now returned spoken replies for the exact greeting
and “Can you hear me?” with `live_engaged=true`, including with old arithmetic
dialogue in the request history.

A separate capture stall was reproduced from microphone frame levels: the
old VAD escaped calibration on its first loud frame, retained a noise floor
of 0.003, and held 12.2 seconds of audio without submitting a turn. All observed
frame RMS values were above 0.0409. Calibration now establishes an observed
baseline before its early-onset check. The same levels no longer falsely open
a recording. Loud steady-room fixtures submit zero requests; speech separated
by pauses in that room submits one and then two requests. Existing quiet,
click, speech, alarm and early-speech fixtures still pass. Both the shared VAD
and its browser fallback carry this calibration rule. A page reload is needed
to update an already-running capture graph; restarting model weights cannot
change JavaScript already loaded in a tab.
