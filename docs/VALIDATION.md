# Candidate validation — 2026-09-15

Environment: Linux, Python 3.8.20, PyTorch 2.4.1, torchaudio 2.4.1; CPU
validation. Test execution used pytest 8.3.5 in an isolated temporary dependency
directory, without changing the experiment environment.

| Check | Result |
|---|---|
| Regression suite | 33 passed, 4 warnings |
| Original paper checkpoints | Six strict state-dict loads passed |
| Restricted checkpoint loading | Six `weights_only=True` loads passed |
| Main model versus training network snapshot | Maximum output difference 0.0 on saved scene `3_var_match/000000` |
| Real-scene output | Finite; shape `[1, 8, 16000]` |
| Public WAV command | Passed with original main checkpoint and synthetic four-channel input; FLOAT WAV output |
| Train/eval/data/inference CLI help | Seven entrypoints passed |
| Experiment preparation | Path relocation and manuscript Act=0 setting passed |
| Wheel packaging | Built successfully without installing dependencies |
| Paper table lookup | 171/172 numeric cells round to the PDF; one DeFTAN-II ILD discrepancy documented in `results/README.md` |

The regression suite covers masked/pairwise losses, finite gradients, silence,
permutation equivariance, padding invariance, variable channel count, stable
sample metadata, data-worker reproducibility, checkpoint scaler state, exact
cyclic clue conventions, strict checkpoint errors and enrollment-free inference.

The runtime's existing STFT deprecation warnings remain; the source currently
uses the PyTorch 2.4-era `return_complex=False` interface. Later PyTorch support
requires separate compatibility work and has not been claimed.

No full training, fresh full-testset evaluation, clean-machine installation,
GPU/DDP reproduction or external-baseline retraining was performed in this
packaging task. The six checkpoint files were hashed before and after the
read-only verification and were unchanged.
