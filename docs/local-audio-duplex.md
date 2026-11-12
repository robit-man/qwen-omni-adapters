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
