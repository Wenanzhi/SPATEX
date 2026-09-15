"""
Test script to evaluate the model.
"""

import argparse
import importlib
import multiprocessing
import os
import glob
import logging
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = object
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm  # pylint: disable=unused-import
from torch.cuda.amp import autocast

from src.helpers import utils
from src.training.dataset import (LibriSpeechTSEDataset, SavedTSETestset,
                                  tensorboard_add_metrics,
                                  array_agnostic_collate_fn, unpack_batch,
                                  model_auxiliary_kwargs)
from src.training.external_baseline_common import reference_channel_paper_metrics


def _saved_dataset_kwargs(test_data):
    """Keep saved-set and direction-encoding fields required by the model."""
    allowed = {
        'saved_path', 'num_samples', 'max_n_mics',
        'azim_type', 'd_model', 'alpha', 'append_doa_unit_vector',
    }
    return {key: value for key, value in test_data.items() if key in allowed}


_PER_SAMPLE_METADATA_KEYS = (
    'sample_id', 'scene_id', 'sample_key', 'subset', 'n_valid_mics',
    'geometry_type', 'seen_unseen_geometry', 'sensor_position_error_std',
    'rt60', 'sir_db', 'min_angle_gap', 'azim_deg',
)


def _metric_values_for_samples(value, batch_size):
    """Return one scalar per sample, or ``None`` for batch aggregates."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    values = np.asarray(value).reshape(-1)
    if values.size != batch_size:
        return None
    return values.tolist()


def _append_per_sample_rows(rows, metrics_batch, metadata, mic_mask,
                            batch_size, sample_offset, context=None):
    """Append aligned sample metrics without fabricating per-sample loss."""
    context = context or {}
    sample_metrics = {}
    for name, value in metrics_batch.items():
        if name == 'loss':
            continue
        values = _metric_values_for_samples(value, batch_size)
        if values is not None:
            sample_metrics[name] = values

    for index in range(batch_size):
        sample_metadata = (
            metadata[index] if metadata is not None and index < len(metadata)
            else {})
        row = dict(context)
        for name, values in sample_metrics.items():
            row[name] = values[index]
        for key in _PER_SAMPLE_METADATA_KEYS:
            if key in sample_metadata:
                row[key] = sample_metadata[key]

        fallback_id = sample_offset + index
        row.setdefault('sample_id', fallback_id)
        row.setdefault('scene_id', row['sample_id'])
        row.setdefault('subset', context.get('subset', ''))
        row.setdefault(
            'sample_key',
            '%s/%s' % (row['subset'], row['sample_id'])
            if row['subset'] else str(row['sample_id']))
        if mic_mask is not None:
            row['n_valid_mics'] = int(mic_mask[index].sum().item())
        else:
            row.setdefault('n_valid_mics', None)
        rows.append(row)


def _atomic_write_per_sample_csv(rows, path):
    """Write a complete CSV then atomically replace the destination."""
    output_path = os.path.abspath(os.fspath(path))
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(
        prefix='.%s.' % os.path.basename(output_path),
        suffix='.tmp', dir=output_dir)
    os.close(descriptor)
    try:
        pd.DataFrame(rows).to_csv(temp_path, index=False)
        os.replace(temp_path, output_path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise



def test_epoch(model: nn.Module, device: torch.device,
               test_loader: torch.utils.data.dataloader.DataLoader,
               n_items: int, metrics_fn,
               profiling: bool = False, epoch: int = 0,
               writer: SummaryWriter = None, data_params = None,
               distributed: bool = False, is_main_process: bool = True,
               per_sample_rows=None, per_sample_context=None,
               paper_metrics: bool = False,
               sample_rate: int = 8000,
               use_amp: bool = True,
               amp_dtype: torch.dtype = torch.float16,
               fail_on_nonfinite: bool = False) -> float:
    """
    Evaluate the network.
    """
    model.eval()
    metric_sums = {}
    metric_counts = {}
    sample_offset = 0
    
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Test', ncols=130, disable=not is_main_process) as t:
            for batch_idx, batch in enumerate(test_loader):

                mixed, gt, enrollment, azim_vec, mic_mask, mic_xyz, metadata = unpack_batch(batch, device)
                auxiliary_kwargs = model_auxiliary_kwargs(batch, device)

                # Run through the model
                with autocast(
                        enabled=use_amp and device.type == 'cuda',
                        dtype=amp_dtype):
                    output, loss = model(mixed, gt, enrollment, azim_vec,
                                         mic_mask=mic_mask, mic_xyz=mic_xyz,
                                         **auxiliary_kwargs)

                if fail_on_nonfinite:
                    finite_flag = (
                        torch.isfinite(loss.detach()).all()
                        & torch.isfinite(output.detach()).all()
                    ).to(dtype=torch.int32)
                    if distributed:
                        dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
                    if finite_flag.item() == 0:
                        raise FloatingPointError(
                            "Non-finite validation output/loss at epoch=%d "
                            "batch=%d" % (epoch, batch_idx))

                # Compute metrics
                try:
                    metrics_batch = metrics_fn(mixed, output, gt,
                                               mic_mask=mic_mask,
                                               mic_xyz=mic_xyz,
                                               metadata=metadata)
                except TypeError:
                    metrics_batch = metrics_fn(mixed, output, gt)
                if paper_metrics:
                    metrics_batch.update(reference_channel_paper_metrics(
                        mixed, output, gt, sample_rate=sample_rate))
                if per_sample_rows is not None:
                    _append_per_sample_rows(
                        per_sample_rows, metrics_batch, metadata, mic_mask,
                        mixed.shape[0], sample_offset, per_sample_context)
                    sample_offset += mixed.shape[0]
                metrics_batch['loss'] = [loss.mean().item()]
                for k in metrics_batch.keys():
                    values = np.asarray(metrics_batch[k], dtype=np.float64).reshape(-1)
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(values.sum())
                    metric_counts[k] = metric_counts.get(k, 0.0) + float(values.size)

                if writer is not None and is_main_process:
                    tensorboard_add_metrics(
                        writer, tag='Test', metrics=metrics_batch,
                        step=epoch)
                    
                # Show current loss and metrics in the progress meter
                avg_metrics_batch = {}
                for k in metrics_batch.keys():
                    values = np.asarray(metrics_batch[k], dtype=np.float64).reshape(-1)
                    avg_metrics_batch[k[:4]] = '%.05f' % (float(values.mean()) if values.size > 0 else 0.0)
                t.set_postfix(**avg_metrics_batch)
                t.update()

                if n_items is not None and batch_idx == (n_items - 1):
                    break

        if distributed:
            for k in sorted(metric_sums.keys()):
                reduced = torch.tensor(
                    [metric_sums[k], metric_counts[k]],
                    dtype=torch.float64,
                    device=device
                )
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                metric_sums[k] = reduced[0].item()
                metric_counts[k] = reduced[1].item()

        avg_metrics = {}
        for k in metric_sums.keys():
            if metric_counts[k] > 0:
                avg_metrics[k] = metric_sums[k] / metric_counts[k]
            else:
                avg_metrics[k] = 0.0

        avg_metrics_str = "Test:"
        for m in avg_metrics.keys():
            avg_metrics_str += ' %s=%.04f' % (m, avg_metrics[m])
        logging.info(avg_metrics_str)

        return avg_metrics



def evaluate(network, args: argparse.Namespace):
    """
    Evaluate the model on a given dataset.
    """

    # Load dataset: SavedTSETestset if test_data points at a pre-generated set,
    # otherwise the on-the-fly LibriSpeechTSEDataset.
    if args.test_data.get('saved_path'):
        saved_kwargs = _saved_dataset_kwargs(args.test_data)
        data_test = SavedTSETestset(**saved_kwargs)
        logging.info("Loaded saved testset from %s containing %d samples" %
                     (saved_kwargs['saved_path'], len(data_test)))
        use_array_metadata = True
    else:
        data_test = LibriSpeechTSEDataset(**args.test_data)
        logging.info("Loaded test dataset at %s containing %d samples" %
                     (args.test_data['input_dir'], len(data_test)))
        use_array_metadata = args.test_data.get('return_array_metadata', False)

    # Set up the device and workers.
    use_cuda = args.use_cuda and torch.cuda.is_available()
    if use_cuda:
        gpu_ids = args.gpu_ids if args.gpu_ids is not None\
                        else range(torch.cuda.device_count())
        device_ids = [_ for _ in gpu_ids]
        data_parallel = len(device_ids) > 1
        device = torch.device('cuda:%d' % device_ids[0])
        torch.cuda.set_device(device_ids[0])
        logging.info("Using CUDA devices: %s" % str(device_ids))
    else:
        data_parallel = False
        device = torch.device('cpu')
        logging.info("Using device: CPU")

    # Set multiprocessing params
    num_workers = min(multiprocessing.cpu_count(), args.n_workers)
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': True
    } if use_cuda else {}

    collate_fn = array_agnostic_collate_fn if use_array_metadata else None

    # Set up data loader
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.eval_batch_size,
                                              collate_fn=collate_fn,
                                              **kwargs)

    # Set up model
    model = network.Net(**args.model_params)
    if use_cuda and data_parallel:
        model = nn.DataParallel(model, device_ids=device_ids)
        logging.info("Using data parallel model")
    model.to(device)

    # Load weights
    if args.pretrain_path == "best":
        ckpts = glob.glob(os.path.join(args.exp_dir, '*.pt'))
        ckpts.sort(
            key=lambda _: int(os.path.splitext(os.path.basename(_))[0]))
        val_metrics = torch.load(ckpts[-1], map_location='cpu')['val_metrics'][args.base_metric]
        best_epoch = max(range(len(val_metrics)), key=val_metrics.__getitem__)
        args.pretrain_path = os.path.join(args.exp_dir, '%d.pt' % best_epoch)
        logging.info(
            "Found 'best' validation %s=%.02f at %s" %
            (args.base_metric, val_metrics[best_epoch], args.pretrain_path))
    if args.pretrain_path != "":
        utils.load_checkpoint(
            args.pretrain_path, model, data_parallel=data_parallel)
        logging.info("Loaded pretrain weights from %s" % args.pretrain_path)

    # Evaluate
    try:
        per_sample_path = getattr(args, 'per_sample_path', None)
        per_sample_rows = [] if per_sample_path else None
        per_sample_context = dict(
            getattr(args, 'per_sample_context', None) or {})
        per_sample_context.setdefault('checkpoint', args.pretrain_path)
        metrics = test_epoch(
            model, device, test_loader, args.n_items, # network.loss,
            network.metrics, args.profiling,
            per_sample_rows=per_sample_rows,
            per_sample_context=per_sample_context,
            paper_metrics=bool(getattr(args, 'paper_metrics', False)),
            sample_rate=int(args.test_data.get('sr', 8000)),
            use_amp=bool(getattr(args, 'use_amp', True)),
            amp_dtype=(
                torch.bfloat16
                if str(getattr(args, 'amp_dtype', 'float16')).lower()
                in ('bfloat16', 'bf16')
                else torch.float16),
            fail_on_nonfinite=bool(getattr(
                args, 'fail_on_nonfinite', False)))
        if per_sample_path:
            _atomic_write_per_sample_csv(per_sample_rows, per_sample_path)
            logging.info('Wrote %d per-sample rows to %s',
                         len(per_sample_rows), per_sample_path)
        return metrics
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as _:  # pylint: disable=broad-except
        import traceback  # pylint: disable=import-outside-toplevel
        traceback.print_exc()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data Params
    parser.add_argument('experiments', nargs='+', type=str,
                        default=None,
                        help="List of experiments to evaluate. "
                        "Provide only one experiment when providing "
                        "pretrained path. If pretrained path is not "
                        "provided, epoch with best validation metric "
                        "is used for evaluation.")
    parser.add_argument('--results', type=str, default="",
                        help="Path to the CSV file to store results.")
    parser.add_argument('--test_input_dir', type=str, default="",
                        help="Override `test_data.input_dir` in config.json "
                        "without modifying the experiment config.")

    # System params
    parser.add_argument('--n_items', type=int, default=None,
                        help="Number of items to test.")
    parser.add_argument('--pretrain_path', type=str, default="best",
                        help="Path to pretrained weights")
    parser.add_argument('--profiling', dest='profiling', action='store_true',
                        help="Enable or disable profiling.")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true',
                        help="Whether to use cuda")
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None,
                        help="List of GPU ids used for training. "
                        "Eg., --gpu_ids 2 4. All GPUs are used by default.")
    args = parser.parse_args()

    results = []

    for exp_dir in args.experiments:
        eval_args = argparse.Namespace(**vars(args))
        eval_args.exp_dir = exp_dir

        utils.set_logger(os.path.join(exp_dir, 'eval.log'))
        logging.info("Evaluating %s ..." % exp_dir)

        # Load model and training params
        params = utils.Params(os.path.join(exp_dir, 'config.json'))
        for k, v in params.__dict__.items():
            vars(eval_args)[k] = v
        if args.test_input_dir != "":
            eval_args.test_data['input_dir'] = args.test_input_dir
            logging.info("Overriding test_data.input_dir to %s", args.test_input_dir)

        network = importlib.import_module(eval_args.model)
        logging.info("Imported the model from '%s'." % eval_args.model)

        curr_res = evaluate(network, eval_args)
        curr_res['experiment'] = os.path.basename(exp_dir)
        results.append(curr_res)

        del eval_args

    if args.results != "":
        print("Writing results to %s" % args.results)
        pd.DataFrame(results).to_csv(args.results, index=False)
        
