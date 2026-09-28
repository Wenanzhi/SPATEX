# Data and evaluation protocol

## Data

Audio is sampled at 8 kHz. Each two-second training mixture contains one target
and two independently scaled interferers. The target is the reverberant source
image at every valid microphone. A three-second enrollment segment is used by
the auxiliary training branch; inference requires only the target direction.
LibriSpeech `train-clean-100`, `dev-clean` and `test-clean` provide the respective
speaker-disjoint training, validation and test sources.

| Saved subset | Microphones | Geometry |
|---|---|---|
| `1_4ch_fixed` | 4 | Linear, 4 cm spacing |
| `3_var_match` | 2–8 | Linear/planar, 2.5–8 cm spacing |
| `4_var_unmatch` | 2–8 | Sparse/random, 1 cm position perturbations |

Training uses linear/planar arrays; validation includes sparse/random layouts.
Sparse arrays sample positions from a 5-by-5 planar grid with grid spacing
2.5–8 cm. Random arrays use aperture parameter range 5–25 cm. Full saved-set
parameters are in [data_protocol](../data_protocol).

```text
3_var_match/
  meta.json
  mixture/000000.wav
  target/000000.wav
  enrollment/000000.wav
  meta/000000.json
```

Each sample's metadata includes microphone coordinates, valid microphone count
and target azimuth. WAVs contain valid channels only. The loader pads audio,
coordinates and masks together. Preserve their shared channel order.
The original builder writes PCM16 with clipping; inference writes FLOAT WAV.
Do not independently normalize output channels when measuring spatial fidelity.
Published scores used saved WAVs; newly generated files have not been verified
to match those original WAVs byte for byte.

## Released training recipes

The released epoch-85 model and the six configurations retain the actual
experiment settings. The main recipe uses Adam with initial learning rate
0.0005, global batch size 8 and gradient clipping at 0.5. Its spatial-loss weights
are 0.15 for IPD and 0.03 for coherence. The configurations also retain the
auxiliary activity loss with weight 0.1 and the corresponding speaker/activity
modules. These modules are needed for training and strict checkpoint loading;
enrollment speech is not required for DOA-only inference.

The original trainer's `epochs=100` loop uses epoch labels 0 through 100
inclusive. Checkpoint selection uses validation SI-SNRi averaged across valid
output channels. The released weights omit optimizer, scheduler, AMP and
metric-history state, so use a full training checkpoint for exact resumption.

Basic interface, masking and checkpoint checks can be run after installation
with `python -m pytest -q`.

## Signal and spatial metrics

`--paper_metrics` enables reference-channel SI-SNRi, SNRi, NB-PESQ and STOI.
Use `reference_scale_invariant_signal_noise_ratio_i` and
`reference_signal_noise_ratio_i` in detailed CSV output. Unprefixed SI-SNRi/SNRi
average valid output channels, as used for training checkpoint selection.

Pairwise spatial metrics first average valid unordered microphone pairs within
each scene, then average scenes:

- ILD: whole-utterance energy-ratio error in dB.
- IPD: absolute circular phase error in radians, rectangular 256/128 STFT.
- ITD: whole-utterance GCC-PHAT at 8 kHz, a ±1 ms search and integer lags
  (125 microseconds per step).

Spatial training losses use Hann windows and target-active bins. The evaluation
IPD metric differs from the IPD training loss.

## Direction preservation and cue errors

`src.training.eval_doa_preservation` uses SRP-PHAT with 1024-sample frames,
256-sample hop, 200–3500 Hz, a 1-degree azimuth grid, label-derived active-frame
selection at -40 dB and sound speed 343 m/s. Rank-one horizontal arrays use
front/back reflection handling. `Pres@5` is the fraction of scenes with
output-to-target DOA drift at most 5 degrees.

For cue-error tests, use signed offsets `-15 -10 -5 0 5 10 15` on the same fixed
four-microphone scenes. Average positive and negative offsets of equal magnitude
when plotting absolute cue error. Keep label caches separate for different
subsets or localizer settings. `src.training.eval_doa_mismatch` also supports
signal/spatial evaluation under perturbed cues.

Use explicit checkpoint paths for the released weights. The `best` selector
requires validation history from a full training checkpoint. See the
[README](../README.md) for complete commands.
