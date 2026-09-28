# SPATEX

**Spatial Preservation for Array-Flexible Target Speech Extraction**

[English](README.md) | [简体中文](README_zh.md)

Training, inference and evaluation code for **SPATEX: Array-Flexible
Multichannel-to-Multichannel Target Speech Extraction with Spatial Cue
Preservation**. Given a multichannel mixture, microphone coordinates and a target
azimuth, SPATEX reconstructs the reverberant target waveform at each microphone.
The main model is trained with 2–8 microphones at 8 kHz.

![SPATEX architecture](docs/figures/spatex_architecture.png)

## Installation

The experiment environment uses Python 3.8.20, PyTorch/torchaudio 2.4.1 and CUDA
12.1. Install the matching PyTorch/torchaudio build for your machine, then run:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
# Online room simulation / test-set generation:
python -m pip install -r requirements-data.txt
# NB-PESQ and STOI:
python -m pip install -r requirements-eval.txt
```

Run the commands below from the repository root.

## Pretrained model

Only the main variable-array checkpoint is distributed:

| Model | Configuration | Checkpoint | Training epoch label | Parameters |
|---|---|---|---:|---:|
| SPATEX variable-array | [spatex_variable.json](configs/spatex_variable.json) | [Download weights](https://github.com/Wenanzhi/SPATEX/raw/refs/heads/main/checkpoints/spatex_variable.pt) | 85 | 4,567,660 |

The weight file is included in a normal Git clone; Git LFS is not required.
Verify it from the repository root with:

```bash
sha256sum -c checkpoints/SHA256SUMS
```

[Model metadata](checkpoints/manifest.json) records the size and SHA-256.
The release preserves every tensor from the original epoch-85 checkpoint and
omits optimizer, scheduler, AMP and metric-history state. Use it for inference
and evaluation; exact training resumption requires a full training checkpoint.
The Python package retains its earlier name `pact` for compatibility.

## Inference

Input audio must be 8 kHz. WAV channels and microphone-coordinate rows must have
the same order. Coordinates are in metres; azimuth is measured from +y toward
+x, with +z vertical. Enrollment speech is not required for inference.

```bash
python -m pact.infer \
  --config configs/spatex_variable.json \
  --checkpoint checkpoints/spatex_variable.pt \
  --mixture /path/to/mixture.wav \
  --geometry examples/geometry_fixed4.json \
  --azimuth 30 --output outputs/target.wav --device cpu
```

The example geometry is a four-microphone linear array with 4 cm spacing; replace
it with your recording geometry. Output uses floating-point WAV to preserve
channel amplitudes. The Python API exposes `pact.load_model` and `pact.extract`.

## Training

Prepare LibriSpeech with `train-clean-100`, `dev-clean` and `test-clean` under one
directory. Materialize a configuration with your dataset path:

```bash
python scripts/prepare_experiment.py \
  --config configs/spatex_variable.json \
  --output experiments/spatex_variable \
  --librispeech-root /path/to/LibriSpeech

python -m src.training.train experiments/spatex_variable --use_cuda --gpu_ids 0
```

For a global batch size of eight across eight GPUs:

```bash
torchrun --standalone --nproc_per_node=8 -m src.training.train \
  experiments/spatex_variable --use_cuda --find_unused_parameters
```

The main recipe uses Adam, initial learning rate 0.0005, gradient clipping at
0.5, and PCM + 0.15 IPD + 0.03 coherence + 0.1 auxiliary activity loss. All
released configurations preserve their actual experiment settings. The auxiliary
speaker/activity modules are retained for training and checkpoint compatibility.
The original `epochs=100` loop uses epoch labels 0 through 100 inclusive.
Checkpoint selection uses validation SI-SNRi averaged across valid channels.

| Configuration | Temporal attention | Spatial losses | Training arrays |
|---|---|---|---|
| `spatex_variable` | Shared mean logits | IPD + Coh | Variable, 2–8 microphones |
| `spatex_fixed4` | Shared mean logits | IPD + Coh | Fixed four-mic linear |
| `independent_full` | Independent | IPD + Coh | Variable, 2–8 microphones |
| `spatex_pcm` | Shared mean logits | None | Variable, 2–8 microphones |
| `spatex_ipd` | Shared mean logits | IPD | Variable, 2–8 microphones |
| `spatex_coherence` | Shared mean logits | Coh | Variable, 2–8 microphones |

The five additional recipes are supplied for training; their weights are not
included. Each recipe also retains the auxiliary activity loss.

## Build and evaluate test sets

```bash
python -m src.training.build_testsets \
  --librispeech_root /path/to/LibriSpeech --out_root data/Testset \
  --segments 2 --num_samples 500 \
  --subsets 1_4ch_fixed 3_var_match 4_var_unmatch
```

Evaluation needs the experiment configuration but not the original LibriSpeech
files once saved test WAVs are available:

```bash
mkdir -p experiments/spatex_variable
cp configs/spatex_variable.json experiments/spatex_variable/config.json
cp checkpoints/spatex_variable.pt experiments/spatex_variable/85.pt

python -m src.training.run_testsets spatex_variable \
  --testset_root data/Testset --segment seg_2s \
  --subsets 1_4ch_fixed 3_var_match 4_var_unmatch \
  --checkpoint 85.pt --paper_metrics --detailed \
  --out_root outputs/metrics --detailed_out_root outputs/per_scene \
  --use_cuda --gpu_ids 0
```

Use the explicit `85.pt` selector with the released checkpoint. The `best`
selector is for full checkpoints containing validation history. Omit the CUDA
options for CPU evaluation. For direction preservation and cue-error evaluation:

```bash
python -m src.training.eval_doa_preservation spatex_variable \
  --project_root . --testset data/Testset/seg_2s/3_var_match \
  --output_root outputs/doa --label_cache outputs/labels_match.npz \
  --build_label_cache --pretrain_path 85.pt \
  --estimator srp_phat --device cpu --num_samples 500
```

Use `1_4ch_fixed`, a separate label cache and
`--offsets -15 -10 -5 0 5 10 15` for the cue-error experiment.
See [data and metric definitions](docs/DATA_AND_METRICS.md) for geometry,
normalization and aggregation conventions. LibriSpeech recordings and saved test
WAVs are not bundled. Regenerated test sets have not been verified to reproduce
the original saved evaluation WAVs byte for byte.

## Basic checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The checks cover checkpoint loading, forward/loss/metric execution, microphone
masking and permutation equivariance. Full historical training trajectories and
paper-wide metric reevaluation are separate from these checks.

## License and citation

See [LICENSE](LICENSE), [third-party notices](THIRD_PARTY_NOTICES.md) and
[CITATION.cff](CITATION.cff). The code retains the GPL v2 license supplied with
the upstream M2M-TSE implementation.
