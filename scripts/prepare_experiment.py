"""Materialize a portable config into a fresh experiment directory."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--librispeech-root', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(args.librispeech_root).expanduser().resolve()
    for section in ('train_data', 'val_data', 'test_data'):
        config[section]['input_dir'] = str(root)
        for subset in config[section]['subsets']:
            if not (root / subset).is_dir():
                parser.error('Missing LibriSpeech subset: ' + str(root / subset))
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print('Prepared:', destination)
    print('Activity loss weight:', config['model_params']['act_loss_weight'])


if __name__ == '__main__':
    main()
