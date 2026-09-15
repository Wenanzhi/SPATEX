# Reproduction status

## Objective discrepancy

The supplied manuscript, section 2.3, defines PCM + 0.15 IPD + 0.03 Coh.
The archived main, independent, PCM, IPD, coherence and fixed-four configs have
`act_loss_weight=0.1` and `use_activity_head=true`. Both the current network and
the main model's training snapshot add the corresponding BCE loss when a target
is supplied. `cue_mode=doa_only` disables activity gating in separation, but
does not disable this auxiliary training loss. Earlier author discussions
acknowledge this archive/manuscript discrepancy; a matching no-Act main-model
run has not been identified in the evidence inspected for this candidate.

The release therefore preserves two explicit config sets. No checkpoint is
relabeled as having been trained without Act. The manuscript-aligned set changes
only the Act weight after data-path relocation and requires fresh training.
Keeping the auxiliary modules preserves strict state-dict compatibility.

The separate `abl_m2_coatt_logit_mean_pure_pcm/97.pt` experiment does disable Act:
its reference SI-SNRi is 11.4860 / 10.9440 dB on matched/unmatched geometry.
These are not the paper PCM row (11.0310 / 10.7341 dB). This separate experiment
is not substituted into the paper table.

## Version selection

Final PACT uses `coattention_aggregation=mean`. The older Method 2 defaults to
`sqrt_count`; the older PAPER_PLAN discussion and its coherence-only divergence
do not describe the final mean-logit coherence-only result (99.pt).

The current core network differs from the main model's saved network snapshot
by an optional `disable_geometry_conditioning` switch, false by default.
The main architecture and parameter names are retained. Training and data code
also accumulated reproducibility and numerical-check changes after some earlier
runs. Reusing an archived hyperparameter file with today's trainer is not a
claim of bitwise recreation of that historical training trajectory.

The legacy `epochs=100` trainer loops over epoch indices 0 through 100 inclusive.
Thus the configuration name/budget should not be interpreted as proof that every
historical run made exactly 100 training passes. The manuscript's “at most 100
epochs” wording needs author reconciliation with this convention. Reported
checkpoints retain their original integer epoch labels.

## Data and numerical protocol

- LibriSpeech training/validation/test splits are speaker-disjoint.
- Training geometry is linear/planar; validation includes sparse/random arrays.
  Unmatched test geometry is outside the training distribution, not unseen
  throughout model development.
- The saved two-second WAV sets were generated before later training random
  protocol changes. Regenerated files have not been compared byte for byte.
- Saved test audio is PCM16 with hard clipping in the original builder. The
  public inference utility writes FLOAT WAV and does not perform channelwise
  normalization. These are different roles and do not alter archived metrics.
- PACT checkpoint selection uses validation SI-SNRi averaged across valid output
  channels; paper signal results use the first microphone. Select the same
  checkpoint before comparing the first-microphone test scores.
- Archived metric exports from different inference batch sizes/precision modes
  have small differences even with identical weights. The results source ledger
  selects the exports corresponding to the supplied PDF; it does not merge
  nearby evaluations or infer missing precision metadata.
- Existing results are single-run checkpoints. Scene confidence intervals, when
  present in source analyses, do not measure training-seed variability.

## What this candidate has and has not established

The author review includes test logs, strict checkpoint loading and numerical
comparison against the source model. No full training run or full 500-scene
metric reevaluation is initiated by preparing this package. Dependency versions
were observed in the experiment environment; a clean-machine install and
cross-version support still need validation before claiming them.

Pretrained weights and original test audio are not bundled. Their release
locations must be added only when actual approved artifacts exist.
