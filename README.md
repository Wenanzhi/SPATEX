# SPATEX

**Array-Flexible Multichannel-to-Multichannel Target Speech Extraction Using Direction Clue**

Submitted to **ICASSP 2027**.

[Code](https://github.com/Wenanzhi/SPATEX) | [Demo](https://wenanzhi.github.io/SPATEX/demo/) | [Pretrained model](https://github.com/Wenanzhi/SPATEX/raw/refs/heads/main/checkpoints/spatex_variable.pt)

## Install

Python 3.8; run the following commands from the repository root after cloning.
The main epoch-85 checkpoint is included in `checkpoints/spatex_variable.pt`.

```bash
git clone https://github.com/Wenanzhi/SPATEX.git
cd SPATEX
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## Train

Place LibriSpeech `train-clean-100`, `dev-clean` and `test-clean` under
`/path/to/LibriSpeech`, then prepare a new experiment directory:

```bash
python scripts/prepare_experiment.py \
  --config configs/spatex_variable.json \
  --output experiments/spatex_train \
  --librispeech-root /path/to/LibriSpeech
python -m src.training.train experiments/spatex_train --use_cuda --gpu_ids 0
```

## Evaluate the released checkpoint

Generate the fixed four-microphone and matched/unmatched 2–8 microphone test sets:

```bash
python -m src.training.build_testsets \
  --librispeech_root /path/to/LibriSpeech --out_root data/Testset \
  --segments 2 --num_samples 500 \
  --subsets 1_4ch_fixed 3_var_match 4_var_unmatch

mkdir -p experiments/spatex_variable
cp configs/spatex_variable.json experiments/spatex_variable/config.json
cp checkpoints/spatex_variable.pt experiments/spatex_variable/85.pt

python -m src.training.run_testsets spatex_variable \
  --testset_root data/Testset --segment seg_2s \
  --subsets 1_4ch_fixed 3_var_match 4_var_unmatch \
  --checkpoint 85.pt --paper_metrics --out_root outputs/metrics \
  --use_cuda --gpu_ids 0
```

## Inference on your audio

Use an 8-kHz multichannel WAV and microphone coordinates in metres, ordered to
match the WAV channels. Azimuth is measured from +y toward +x. Replace the example
geometry with your recording's array coordinates.

```bash
python -m pact.infer \
  --config configs/spatex_variable.json \
  --checkpoint checkpoints/spatex_variable.pt \
  --mixture /path/to/mixture.wav --geometry examples/geometry_fixed4.json \
  --azimuth 30 --output outputs/target.wav --device cpu
```

See [data, metrics and additional evaluation](docs/DATA_AND_METRICS.md) for the
protocol and reproduction limits. [License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)
