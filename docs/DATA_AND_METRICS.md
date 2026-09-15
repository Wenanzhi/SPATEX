# Data and metric protocol

The input target is the reverberant source image at every valid microphone.
Training samples contain one target and two independently scaled interferers,
at 8 kHz, with two-second mixtures and a three-second enrollment segment used
only by the archived training auxiliary branch. The inference separator is
conditioned on target direction, not speaker identity.

| Saved condition | Configuration |
|---|---|
| `1_4ch_fixed` | Four microphones, linear array, 4 cm spacing |
| `3_var_match` | 2–8 microphones, linear/planar, 2.5–8 cm spacing |
| `4_var_unmatch` | 2–8 microphones, sparse/random, 1 cm position perturbations |

The precise generator parameters are in `data_protocol/*/meta.json`. In the
implementation, sparse layouts sample microphone positions from a 5-by-5 planar
grid with spacing drawn from `planar_spacing_range=[0.025, 0.08]` metres,
whereas random layouts use `random_aperture_range=[0.05, 0.25]` metres. The
paper's summary “2.5–25 cm apertures” should be reconciled with these different
layout-specific definitions. These values have not been silently changed.

Saved subset layout:

```text
3_var_match/
  meta.json
  mixture/000000.wav
  target/000000.wav
  enrollment/000000.wav
  meta/000000.json
  ...
```

Each sample metadata file includes microphone coordinates, valid microphone
count and target azimuth. Mixture and target WAVs contain valid channels only;
the saved-set loader pads them together with coordinates and mask. All channel
permutations must be applied consistently to waveforms, coordinates and mask.

## Signal and spatial scores

`--paper_metrics` computes reference-channel SI-SNRi/SNRi, NB-PESQ and STOI.
Use `reference_scale_invariant_signal_noise_ratio_i` and
`reference_signal_noise_ratio_i`, not the unprefixed all-channel metrics.

Pairwise metrics average valid unordered microphone pairs within each scene,
then average scenes. ILD is whole-utterance energy-ratio error in dB. IPD is
absolute circular STFT phase error in radians with a rectangular 256/128 STFT.
ITD uses whole-utterance GCC-PHAT at 8 kHz, a ±1 ms search and integer lags
(125 microseconds per step). It is separate from the interpolated DOA localizer.
Training spatial losses use Hann windows and target-active bins; evaluation
IPD is not the training IPD-loss value.

## Direction preservation / DOA errors

The frozen SRP-PHAT localizer uses 1024-sample frames, 256-sample hop,
200–3500 Hz, 1-degree azimuth grid, -40 dB label-derived active-frame selection,
and speed of sound 343 m/s. Rank-one horizontal geometry receives front/back
reflection handling. `Pres@5` reports output-versus-label direction drift at
most 5 degrees, not accuracy against the input target clue.

```bash
python -m src.training.eval_doa_preservation pact_full \
  --project_root . --testset data/Testset/seg_2s/3_var_match \
  --output_root outputs/doa --label_cache outputs/labels_match.npz \
  --build_label_cache --pretrain_path best \
  --estimator srp_phat --device cpu --num_samples 500
```

For the paper's fixed-array clue-error study, use `1_4ch_fixed` and
`--offsets -15 -10 -5 0 5 10 15`. Average the positive and negative offsets of
equal magnitude across the same scenes for Figure 2. Keep caches separate for
different saved subsets and localizer configurations. These commands permit
reevaluation; they do not assert equality with every archived floating-point
export, whose original settings should be checked in its source manifest.
