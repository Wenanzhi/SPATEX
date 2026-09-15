"""
Run an experiment from this project on every saved subset under
``data/Testset/seg_2s/`` and write per-subset metrics to
``<project>/test/<subset>.txt``.

Layout: the ``test/`` folder contains exactly one file per test subset
(``1_4ch_fixed.txt`` … ``6_8ch_unfixed.txt``). Each file is an aligned
table with one row per model, so all models tested on a subset sit side
by side. Re-running a model updates its row in place and leaves the other
rows untouched, so models accumulate across runs.

The experiment name is given relative to ``experiments/`` and is the only
thing that needs to change to test a different model.

Usage
-----
    python -m src.training.run_testsets tac_deftan2_mimo_tse/baseline_tac_pcm \
        --use_cuda --gpu_ids 0

You can list multiple experiments to test in one go.
"""

import argparse
import importlib
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

from src.helpers import utils
from src.training import eval as eval_mod


PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
DEFAULT_TESTSET_ROOT = 'data/Testset'
DEFAULT_SEGMENT = 'seg_2s'  # TAC-M2M models train at segment_len=2.0s.
SUBSETS = ['1_4ch_fixed', '2_4ch_unfixed', '3_var_match', '4_var_unmatch',
           '5_8ch_fixed', '6_8ch_unfixed']

MIN_COL_W = 12  # minimum width of a metric column in the rendered table.


def _read_subset_file(path):
    """Parse an existing ``<subset>.txt`` table back into ``{model: {metric: value}}``.

    Comment lines start with ``#``; the first non-comment line is the header
    naming the metric columns; every following line is one model's row.
    Missing cells are written as ``-`` and skipped here.
    """
    if not path.exists():
        return {}
    rows = {}
    header = None
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.split()
        if header is None:
            header = parts[1:]  # drop the leading 'model' column label
            continue
        model, values = parts[0], parts[1:]
        metrics = {}
        for name, value in zip(header, values):
            if value == '-':
                continue
            try:
                metrics[name] = float(value)
            except ValueError:
                pass
        rows[model] = metrics
    return rows


def _write_subset_file(path, subset, saved_path, num_samples, rows):
    """Render ``{model: {metric: value}}`` as an aligned, one-row-per-model table."""
    metric_keys = sorted({k for m in rows.values() for k in m})
    model_w = max([len('model')] + [len(m) for m in rows])
    col_w = {k: max(len(k), MIN_COL_W) for k in metric_keys}

    header_cells = '  '.join(f'{k:>{col_w[k]}}' for k in metric_keys)
    lines = [
        f'# Subset : {subset}',
        f'# Testset: {saved_path}',
        f'# Samples: {num_samples}',
        f'# Updated: {datetime.now().isoformat(timespec="seconds")}',
        '#',
        f'{"model":<{model_w}}  {header_cells}',
    ]
    for model in sorted(rows):
        cells = []
        for k in metric_keys:
            if k in rows[model]:
                cells.append(f'{rows[model][k]:>{col_w[k]}.6f}')
            else:
                cells.append(f'{"-":>{col_w[k]}}')
        lines.append(f'{model:<{model_w}}  ' + '  '.join(cells))
    path.write_text('\n'.join(lines) + '\n')


def _update_subset_file(path, subset, experiment, metrics, saved_path,
                        num_samples):
    """Merge one model's result into the subset table, preserving other rows."""
    rows = _read_subset_file(path)
    rows[experiment] = {k: float(v) for k, v in metrics.items()}
    _write_subset_file(path, subset, saved_path, num_samples, rows)
    logging.info('Updated %s [%s]', path, experiment)


def _saved_test_data(original, saved_path, num_samples):
    """Build a saved-testset config without changing its direction encoding."""
    n_mics_budget = original.get('max_n_mics', original.get('n_mics', 8))
    config = {
        'saved_path': saved_path,
        'num_samples': num_samples,
        'max_n_mics': n_mics_budget,
    }
    for key in ('azim_type', 'd_model', 'alpha', 'append_doa_unit_vector'):
        if key in original:
            config[key] = original[key]
    return config


def _detailed_per_sample_path(detailed_out_root, segment, subset,
                              experiment):
    return (Path(detailed_out_root) / segment / subset / experiment
            / 'per_sample.csv')


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(
        prefix=f'.{path.name}.', suffix='.tmp', dir=str(path.parent))
    os.close(descriptor)
    try:
        Path(temp_path).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + '\n')
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def _update_detailed_manifest(path, entry):
    """Upsert one completed experiment/subset evaluation record."""
    path = Path(path)
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        payload = {'format_version': 1, 'entries': []}
    key = (entry['experiment'], entry['segment'], entry['subset'])
    entries = [
        existing for existing in payload.get('entries', [])
        if (existing.get('experiment'), existing.get('segment'),
            existing.get('subset')) != key
    ]
    entries.append(entry)
    payload['entries'] = sorted(
        entries,
        key=lambda item: (item['segment'], item['subset'],
                          item['experiment']))
    payload['updated_at'] = datetime.now().isoformat(timespec='seconds')
    _atomic_write_json(path, payload)


def run_one(experiment, subset, args_template):
    exp_dir = os.path.join(PROJECT_ROOT, 'experiments', experiment)
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f'Experiment dir not found: {exp_dir}')

    saved_path = os.path.join(args_template.testset_root,
                              args_template.segment, subset)
    if not os.path.isdir(saved_path):
        raise FileNotFoundError(f'Saved testset not found: {saved_path}')

    args = argparse.Namespace(**vars(args_template))
    args.exp_dir = exp_dir
    params = utils.Params(os.path.join(exp_dir, 'config.json'))
    for k, v in params.__dict__.items():
        setattr(args, k, v)

    # Runtime resource controls must take precedence over training config.
    for key in ('use_cuda', 'gpu_ids', 'n_workers', 'eval_batch_size',
                'paper_metrics', 'use_amp'):
        setattr(args, key, getattr(args_template, key))
    # Analytic baselines do not have a meaningful checkpoint.  Let their
    # config opt out explicitly instead of manufacturing a dummy ``0.pt``.
    if getattr(args, 'parameter_free', False):
        args.pretrain_path = ''
    elif args_template.checkpoint == 'best':
        args.pretrain_path = 'best'
    else:
        checkpoint = os.path.expanduser(args_template.checkpoint)
        args.pretrain_path = (
            checkpoint if os.path.isabs(checkpoint)
            else os.path.join(exp_dir, checkpoint))
    args.profiling = False
    args.n_items = None

    # Override test_data with saved-path config while preserving the channel
    # budget and direction encoding expected by the model.  Method 3 appends a
    # 3-D Cartesian DOA suffix, so dropping these fields would silently create
    # a 40-D/43-D mismatch at evaluation time.
    args.test_data = _saved_test_data(
        args.test_data, saved_path, args_template.num_samples)
    per_sample_path = None
    if getattr(args_template, 'detailed', False):
        per_sample_path = _detailed_per_sample_path(
            getattr(args_template, 'detailed_out_root',
                    os.path.join(PROJECT_ROOT, 'test_detailed')),
            args_template.segment, subset,
            experiment)
    args.per_sample_path = str(per_sample_path) if per_sample_path else None
    args.per_sample_context = {
        'experiment': experiment,
        'segment': args_template.segment,
        'subset': subset,
    }

    network = importlib.import_module(args.model)
    metrics = eval_mod.evaluate(network, args)
    if not metrics:
        # evaluate() swallows exceptions and returns None — surface that as a
        # failure rather than silently writing an empty results file.
        raise RuntimeError(
            f'evaluate() returned no metrics for {experiment} / {subset} '
            '(check the eval log; common cause: CUDA OOM)')

    return ({k: float(v) for k, v in metrics.items()}, saved_path,
            args.pretrain_path, per_sample_path)


def _build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiments', nargs='+', type=str,
                        help='Experiment names relative to experiments/ '
                        '(e.g. tac_deftan2_mimo_tse/baseline_tac_pcm).')
    parser.add_argument('--testset_root', type=str, default=DEFAULT_TESTSET_ROOT)
    parser.add_argument('--segment', type=str, default=DEFAULT_SEGMENT,
                        choices=['seg_2s', 'seg_6s'])
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--subsets', nargs='+', default=SUBSETS)
    parser.add_argument('--out_root', type=str,
                        default=os.path.join(PROJECT_ROOT, 'test'))
    parser.add_argument('--detailed', '--export_per_sample',
                        dest='detailed', action='store_true',
                        help='Atomically export per-sample CSV files and '
                             'update test_detailed/manifest.json.')
    parser.add_argument('--detailed_out_root', type=str,
                        default=os.path.join(PROJECT_ROOT, 'test_detailed'))
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument(
        '--checkpoint', type=str, default='best',
        help='Checkpoint filename/path to evaluate, or "best" (default).')
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument(
        '--use_amp', dest='use_amp', action='store_true',
        help='Use automatic mixed precision during inference (default).')
    precision.add_argument(
        '--no_amp', dest='use_amp', action='store_false',
        help='Run inference in full precision for numerical checks.')
    parser.set_defaults(use_amp=True)
    parser.add_argument(
        '--paper_metrics', action='store_true',
        help='Add reference-channel SNR/SI-SNR, PESQ-NB, and STOI for '
             'paper tables. This is slower and is disabled by default.')
    return parser


def main():
    args = _build_arg_parser().parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for experiment in args.experiments:
        # Keep the per-run log next to the experiment so test/ holds only the
        # four subset tables.
        exp_dir = Path(PROJECT_ROOT) / 'experiments' / experiment
        if exp_dir.is_dir():
            utils.set_logger(str(exp_dir / 'run_testsets.log'))
        logging.info('=== %s ===', experiment)
        for subset in args.subsets:
            try:
                metrics, saved_path, checkpoint, per_sample_path = run_one(
                    experiment, subset, args)
                _update_subset_file(
                    out_root / f'{subset}.txt', subset, experiment,
                    metrics, saved_path, args.num_samples)
                if args.detailed:
                    detailed_root = Path(args.detailed_out_root)
                    _update_detailed_manifest(
                        detailed_root / 'manifest.json', {
                            'experiment': experiment,
                            'checkpoint': os.path.abspath(checkpoint),
                            'segment': args.segment,
                            'subset': subset,
                            'saved_path': os.path.abspath(saved_path),
                            'num_samples': args.num_samples,
                            'per_sample_csv': os.path.relpath(
                                per_sample_path, detailed_root),
                            'aggregate_metrics': metrics,
                            'completed_at': datetime.now().isoformat(
                                timespec='seconds'),
                        })
            except Exception:
                logging.exception('Failed on %s / %s', experiment, subset)


if __name__ == '__main__':
    main()
