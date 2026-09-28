"""Export real epoch-85 outputs, audio and visualizations for the static demo."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from pact import extract, load_model
from src.helpers.utils import scale_invariant_signal_noise_ratio as si_snr
from src.training.network_TACDeFTAN2 import metrics


SCENES = [
    ('fixed', 'Fixed linear array', '1_4ch_fixed', 'linear', 4),
    ('matched', 'Matched planar array', '3_var_match', 'planar', 6),
    ('unmatched', 'Unmatched random array', '4_var_unmatch', 'random', 8),
]


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def envelope(x, bins=360):
    return [[round(float(part.min()), 5), round(float(part.max()), 5)]
            for part in np.array_split(x, bins)]


def spectrogram(x):
    tensor = torch.from_numpy(np.ascontiguousarray(x))
    return torch.stft(tensor, 256, 128, window=torch.hann_window(256),
                      return_complex=True).abs().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--testset-root', type=Path, required=True,
                        help='Saved six-second test root, e.g. Testset/seg_6s')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/spatex_variable.json')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/spatex_variable.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/demo')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    model = load_model(args.config, args.checkpoint, args.device)
    rows = []
    for key, title, subset, geometry, count in SCENES:
        # Deterministic selection based on geometry, never on output scores.
        choices = sorted((args.testset_root / subset / 'meta').glob('*.json'))
        selection = next((p for p in choices if
            (lambda d: d['geometry_type'] == geometry and d['n_valid_mics'] == count)(
                json.loads(p.read_text()))), None)
        if selection is None:
            raise FileNotFoundError('No matching scene for ' + key)
        meta = json.loads(selection.read_text())
        source = args.testset_root / subset
        mixture, sr = sf.read(source / 'mixture' / (selection.stem + '.wav'),
                              dtype='float32', always_2d=True)
        target, target_sr = sf.read(source / 'target' / (selection.stem + '.wav'),
                                    dtype='float32', always_2d=True)
        assert sr == target_sr == 8000 and mixture.shape == target.shape
        assert mixture.shape[1] == count
        mix = torch.from_numpy(mixture.T.copy()).unsqueeze(0)
        reference = torch.from_numpy(target.T.copy()).unsqueeze(0)
        xyz = torch.tensor(meta['mic_xyz'], dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            prediction = extract(model, mix, xyz, meta['azim_deg']).cpu()
        assert prediction.shape == mix.shape and torch.isfinite(prediction).all()
        scores = metrics(mix, prediction, reference)
        estimate = prediction[0].numpy().T
        arrays = {'mixture': mixture, 'estimate': estimate, 'reference': target}
        peak = max(float(np.abs(x).max()) for x in arrays.values())
        gain = min(1.0, 0.98 / max(peak, 1e-8))
        audio_dir = args.output / 'audio' / key
        image_dir = args.output / 'spectrograms' / key
        audio_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        sf.write(audio_dir / 'spatex_multichannel.wav', estimate, sr, subtype='FLOAT')
        specs = {kind: [spectrogram(x[:, c]) for c in range(count)]
                 for kind, x in arrays.items()}
        spec_peak = max(float(x.max()) for values in specs.values() for x in values)
        channels = []
        for c in range(count):
            improvement = float((si_snr(prediction[:, c:c+1], reference[:, c:c+1]) -
                                 si_snr(mix[:, c:c+1], reference[:, c:c+1])).item())
            channel = {'index': c + 1, 'si_snri': round(improvement, 3), 'tracks': {}}
            for kind, array in arrays.items():
                wav = audio_dir / ('%s_mic%d.wav' % (kind, c+1))
                sf.write(wav, array[:, c] * gain, sr, subtype='PCM_16')
                png = image_dir / ('%s_mic%d.png' % (kind, c+1))
                db = 20 * np.log10(np.maximum(specs[kind][c], 1e-8) / spec_peak)
                # Shared -70..0 dB color range across all signals and microphones.
                plt.imsave(png, db, origin='lower', cmap='magma', vmin=-70, vmax=0)
                channel['tracks'][kind] = {
                    'audio': wav.relative_to(args.output).as_posix(),
                    'spectrogram': png.relative_to(args.output).as_posix(),
                    'envelope': envelope(array[:, c] * gain),
                }
            channels.append(channel)
        row = {'id': key, 'title': title, 'subset': subset,
               'source_scene': selection.stem, 'microphones': count,
               'geometry': geometry, 'sample_rate': sr,
               'duration': len(mixture) / sr, 'azimuth': meta['azim_deg'],
               'mic_xyz': meta['mic_xyz'], 'rt60': meta['rt60'],
               'input_sir_db': meta['sir_db'], 'playback_gain': gain,
               'waveform_peak': peak * gain,
               'delta_ipd': scores['delta_IPD_mean'][0],
               'delta_ild': scores['delta_ILD_mean'][0],
               'delta_itd_us': scores['delta_ITD_gccphat_mean'][0] * 1e6,
               'multichannel_download': (audio_dir / 'spatex_multichannel.wav').relative_to(args.output).as_posix(),
               'channels': channels,
               'source_hashes': {kind: checksum(source / kind / (selection.stem + '.wav'))
                                 for kind in ('mixture', 'target')}}
        rows.append(row)
        print(key, selection.stem, count, 'mics', len(mixture)/sr, 's',
              'reference SI-SNRi:', channels[0]['si_snri'], flush=True)
    payload = {'model': 'SPATEX variable-array', 'epoch': 85,
               'checkpoint_sha256': checksum(args.checkpoint),
               'configuration_sha256': checksum(args.config),
               'selection': 'First saved scene matching each geometry and microphone count; no score filtering.',
               'playback': 'One shared attenuation gain per scene across signals and microphones; PCM16 playback, original float multichannel estimate downloadable.',
               'spectrogram': {'n_fft': 256, 'hop': 128, 'window': 'Hann',
                               'db_range': [-70, 0], 'reference': 'Maximum magnitude across every signal and microphone in the scene.'},
               'scenes': rows}
    # A script assignment also works when index.html is opened directly from disk.
    (args.output / 'data.js').write_text('window.SPATEX_DEMO = ' + json.dumps(
        payload, separators=(',', ':'), allow_nan=False) + ';\n')
    manifest = {k: v for k, v in payload.items() if k != 'scenes'}
    manifest['scenes'] = [{k: v for k, v in s.items() if k != 'channels'} for s in rows]
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
