"""Extract a target from one multichannel WAV file."""
import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from .inference import extract, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mixture', required=True)
    parser.add_argument('--geometry', required=True, help='JSON with mic_xyz, optionally mic_mask')
    parser.add_argument('--azimuth', required=True, type=float)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error('Output already exists; choose a new output path.')
    waveform, sample_rate = sf.read(args.mixture, dtype='float32', always_2d=True)
    if sample_rate != 8000:
        parser.error('Input must be sampled at 8000 Hz; resample all channels together first.')
    geometry = json.loads(Path(args.geometry).read_text())
    xyz = torch.tensor(geometry['mic_xyz'], dtype=torch.float32).unsqueeze(0)
    mask = geometry.get('mic_mask')
    if mask is not None:
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    model = load_model(args.config, args.checkpoint, args.device)
    output = extract(model, torch.from_numpy(waveform.T).unsqueeze(0), xyz, args.azimuth, mask)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    # Float WAV avoids clipping or separately normalizing microphone gains.
    sf.write(args.output, output[0].cpu().numpy().T, sample_rate, subtype='FLOAT')
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
