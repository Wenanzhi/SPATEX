"""Evaluate frozen array-agnostic TSE checkpoints under signed DOA errors."""

import argparse
import glob
import importlib
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.helpers import utils
from src.training.dataset import (SavedTSETestset,
                                  array_agnostic_collate_fn)
from src.training.eval import (_atomic_write_per_sample_csv, test_epoch)


class DOAOffsetDataset(torch.utils.data.Dataset):
    """Re-encode the saved ground-truth azimuth with a fixed signed offset."""

    def __init__(self, base_dataset, offset_deg):
        self.base_dataset = base_dataset
        self.offset_deg = float(offset_deg)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        item = self.base_dataset[index]
        metadata = dict(item['metadata'])
        true_azimuth = float(metadata['azim_deg'])
        clue_azimuth = (true_azimuth + self.offset_deg) % 360.0
        result = dict(item)
        result['metadata'] = metadata
        result['azim_vec'] = self.base_dataset._get_azim_vector(  # pylint: disable=protected-access
            clue_azimuth)
        metadata['clue_azim_deg'] = clue_azimuth
        return result


def _best_checkpoint(exp_dir, base_metric):
    checkpoints = sorted(
        glob.glob(os.path.join(exp_dir, '*.pt')),
        key=lambda path: int(Path(path).stem),
    )
    if not checkpoints:
        raise FileNotFoundError('No integer-named checkpoints in %s' % exp_dir)
    payload = torch.load(checkpoints[-1], map_location='cpu')
    values = np.asarray(payload['val_metrics'][base_metric], dtype=np.float64)
    if not np.isfinite(values).any():
        raise RuntimeError('No finite validation %s values' % base_metric)
    epoch = int(np.nanargmax(values))
    checkpoint = os.path.join(exp_dir, '%d.pt' % epoch)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    return checkpoint, epoch, float(values[epoch])


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(
        prefix='.%s.' % path.name, suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write('\n')
        os.replace(temp_path, path)
    except BaseException:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def evaluate_experiment(experiment, args):
    exp_dir = Path(args.project_root) / 'experiments' / experiment
    params = utils.Params(str(exp_dir / 'config.json'))
    network = importlib.import_module(params.model)
    model = network.Net(**params.model_params).to(args.device)
    checkpoint, best_epoch, best_value = _best_checkpoint(
        str(exp_dir), params.base_metric)
    utils.load_checkpoint(checkpoint, model, data_parallel=False)

    original_test = params.test_data
    base_dataset = SavedTSETestset(
        saved_path=args.testset,
        num_samples=args.num_samples,
        max_n_mics=original_test.get(
            'max_n_mics', original_test.get('n_mics', 8)),
        azim_type=original_test.get('azim_type', 'cycpos'),
        d_model=original_test.get('d_model', 40),
        alpha=original_test.get('alpha', 80),
        append_doa_unit_vector=original_test.get(
            'append_doa_unit_vector', False),
    )

    all_rows = []
    aggregates = []
    for offset in args.offsets:
        dataset = DOAOffsetDataset(base_dataset, offset)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            pin_memory=args.device.type == 'cuda',
            collate_fn=array_agnostic_collate_fn,
        )
        rows = []
        metrics = test_epoch(
            model=model,
            device=args.device,
            test_loader=loader,
            n_items=None,
            metrics_fn=network.metrics,
            is_main_process=True,
            per_sample_rows=rows,
            per_sample_context={
                'experiment': experiment,
                'subset': Path(args.testset).name,
                'doa_offset_deg': float(offset),
                'checkpoint': checkpoint,
            },
            paper_metrics=True,
            sample_rate=args.sample_rate,
        )
        if len(rows) != len(base_dataset):
            raise RuntimeError(
                'Expected %d rows, got %d for offset %s' %
                (len(base_dataset), len(rows), offset))
        all_rows.extend(rows)
        aggregates.append({
            'experiment': experiment,
            'doa_offset_deg': float(offset),
            **{name: float(value) for name, value in metrics.items()},
        })

    output_dir = Path(args.output_root) / experiment
    _atomic_write_per_sample_csv(all_rows, output_dir / 'per_sample.csv')
    _atomic_write_per_sample_csv(aggregates, output_dir / 'summary.csv')
    _atomic_write_json(output_dir / 'manifest.json', {
        'experiment': experiment,
        'checkpoint': checkpoint,
        'best_epoch': best_epoch,
        'best_validation_metric': params.base_metric,
        'best_validation_value': best_value,
        'testset': str(Path(args.testset).resolve()),
        'num_samples': len(base_dataset),
        'offsets_deg': [float(value) for value in args.offsets],
        'sample_rate': args.sample_rate,
        'signal_metric_scope': 'reference_channel_0',
    })


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiments', nargs='+')
    parser.add_argument('--project_root', required=True)
    parser.add_argument('--testset', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--offsets', nargs='+', type=float,
                        default=[-20, -10, -5, 0, 5, 10, 20])
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--sample_rate', type=int, default=8000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def main():
    args = _parse_args()
    args.device = torch.device(args.device)
    if args.device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_device(args.device)
    logging.basicConfig(level=logging.INFO)
    for experiment in args.experiments:
        evaluate_experiment(experiment, args)


if __name__ == '__main__':
    main()
