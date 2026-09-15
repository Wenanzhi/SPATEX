"""
The main training script for training on synthetic data
"""

import argparse
import importlib
import multiprocessing
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm  # pylint: disable=unused-import

from src.helpers import utils
from src.training.eval import test_epoch
from src.training.dataset import (LibriSpeechTSEDataset as Dataset,
                                  array_agnostic_collate_fn, unpack_batch,
                                  model_auxiliary_kwargs)



def train_epoch(model: nn.Module, device: torch.device, optimizer: optim.Optimizer,
                train_loader: torch.utils.data.dataloader.DataLoader,
                metrics_fn,
                n_items: int, epoch: int = 0,
                scaler: GradScaler = None,
                distributed: bool = False,
                is_main_process: bool = True,
                grad_clip_norm: float = 0.5,
                use_amp: bool = True,
                amp_dtype: torch.dtype = torch.float16,
                fail_on_nonfinite: bool = False,
                skip_nonfinite_gradients: bool = False,
                max_nonfinite_gradient_skips_per_epoch: int = 0) -> float:

    """
    Train a single epoch.
    """
    # Set the model to training.
    model.train()

    # Training loop
    metric_sums = {}
    metric_counts = {}
    nonfinite_gradient_skips = 0

    with tqdm(total=len(train_loader), desc='Train', ncols=130, disable=not is_main_process) as t:
        for batch_idx, batch in enumerate(train_loader):

            mixed, gt, enrollment, azim_vec, mic_mask, mic_xyz, metadata = unpack_batch(batch, device)
            auxiliary_kwargs = model_auxiliary_kwargs(batch, device)

            # Reset grad
            optimizer.zero_grad()
            model.zero_grad()

            with autocast(
                    enabled=use_amp and device.type == 'cuda',
                    dtype=amp_dtype):
                # Run through the model
                output, loss = model(mixed, gt, enrollment, azim_vec,
                                     mic_mask=mic_mask, mic_xyz=mic_xyz,
                                     **auxiliary_kwargs)

            loss_value = loss.mean()
            if fail_on_nonfinite:
                local_finite = torch.isfinite(loss_value.detach()).all()
                finite_flag = local_finite.to(dtype=torch.int32)
                if not bool(local_finite.item()):
                    def _tensor_diagnostic(name, value):
                        detached = value.detach()
                        finite = torch.isfinite(detached)
                        finite_values = detached[finite]
                        abs_max = (
                            float(finite_values.abs().max().item())
                            if finite_values.numel() else float('nan'))
                        return (
                            '%s(shape=%s,dtype=%s,finite=%d/%d,absmax=%g)'
                            % (name, tuple(detached.shape), detached.dtype,
                               int(finite.sum().item()), detached.numel(),
                               abs_max))

                    module = model.module if hasattr(model, 'module') else model
                    bad_parameters = [
                        name for name, parameter in module.named_parameters()
                        if not torch.isfinite(parameter.detach()).all()
                    ]
                    process_rank = (
                        dist.get_rank() if distributed and dist.is_initialized()
                        else 0)
                    logging.error(
                        'Local non-finite diagnostic epoch=%d batch=%d rank=%d: '
                        '%s %s %s loss=%s bad_parameters=%s',
                        epoch, batch_idx, process_rank,
                        _tensor_diagnostic('mixture', mixed),
                        _tensor_diagnostic('target', gt),
                        _tensor_diagnostic('output', output),
                        str(loss_value.detach()), bad_parameters[:20])
                if distributed:
                    dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
                if finite_flag.item() == 0:
                    raise FloatingPointError(
                        "Non-finite training loss at epoch=%d batch=%d" %
                        (epoch, batch_idx))
            # Backpropagation
            scaler.scale(loss_value).backward()

            # Unscale before clipping so clip_grad_norm_ sees real gradient magnitudes.
            scaler.unscale_(optimizer)
            grad_norm = None
            if grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_norm)

            # GradScaler protects FP16 training, but it is intentionally
            # disabled for BF16.  A finite BF16 forward can still produce an
            # Inf/NaN gradient.  Without this check clip_grad_norm_ propagates
            # the bad value and Adam corrupts every parameter; the failure is
            # then only observed in the following batch's forward pass.
            if fail_on_nonfinite or skip_nonfinite_gradients:
                if grad_norm is None:
                    gradients = [
                        parameter.grad.detach()
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ]
                    if gradients:
                        grad_norm = torch.linalg.vector_norm(torch.stack([
                            torch.linalg.vector_norm(gradient.float(), 2)
                            for gradient in gradients
                        ]), 2)
                    else:
                        grad_norm = torch.zeros((), device=device)
                local_gradient_finite = torch.isfinite(
                    grad_norm.detach()).all()
                gradient_finite_flag = local_gradient_finite.to(
                    dtype=torch.int32)
                if distributed:
                    dist.all_reduce(
                        gradient_finite_flag, op=dist.ReduceOp.MIN)

                if gradient_finite_flag.item() == 0:
                    module = model.module if hasattr(model, 'module') else model
                    bad_gradients = []
                    if not bool(local_gradient_finite.item()):
                        bad_gradients = [
                            name for name, parameter in module.named_parameters()
                            if parameter.grad is not None
                            and not torch.isfinite(
                                parameter.grad.detach()).all()
                        ]
                    process_rank = (
                        dist.get_rank()
                        if distributed and dist.is_initialized() else 0)
                    sample_metadata = []
                    if isinstance(metadata, (list, tuple)):
                        for item in metadata:
                            if isinstance(item, dict):
                                sample_metadata.append({
                                    key: item.get(key) for key in (
                                        'sample_seed', 'dataset_epoch',
                                        'azim_deg', 'rt60', 'sir_db')
                                    if key in item
                                })
                    logging.error(
                        'Non-finite gradient epoch=%d batch=%d rank=%d '
                        'local_finite=%s grad_norm=%s bad_gradients=%s '
                        'sample_metadata=%s',
                        epoch, batch_idx, process_rank,
                        bool(local_gradient_finite.item()),
                        str(grad_norm.detach()), bad_gradients[:20],
                        sample_metadata)

                    # All ranks must make the same decision after the global
                    # MIN reduction, otherwise DDP optimizer states diverge.
                    optimizer.zero_grad()
                    model.zero_grad()
                    scale_change = backoff_grad_scaler_on_global_nonfinite(
                        scaler, device, distributed=distributed)
                    if scale_change is not None and is_main_process:
                        logging.warning(
                            'Backed off synchronized AMP scale from %.0f to '
                            '%.0f', scale_change[0], scale_change[1])

                    can_skip = (
                        skip_nonfinite_gradients
                        and nonfinite_gradient_skips
                        < max_nonfinite_gradient_skips_per_epoch)
                    if can_skip:
                        nonfinite_gradient_skips += 1
                        if is_main_process:
                            logging.warning(
                                'Skipped globally non-finite gradient at '
                                'epoch=%d batch=%d (%d/%d allowed)',
                                epoch, batch_idx,
                                nonfinite_gradient_skips,
                                max_nonfinite_gradient_skips_per_epoch)
                        t.set_postfix(
                            loss='%.05f' % loss_value.item(),
                            grad_skip=nonfinite_gradient_skips)
                        t.update()
                        if n_items is not None and batch_idx == n_items:
                            break
                        continue

                    raise FloatingPointError(
                        'Non-finite training gradient at epoch=%d batch=%d '
                        '(skipped=%d, allowed=%d)' % (
                            epoch, batch_idx, nonfinite_gradient_skips,
                            max_nonfinite_gradient_skips_per_epoch))

            # scaler.step internally skips optimizer.step() if any grad is non-finite.
            scaler.step(optimizer)
            scaler.update()

            metric_sums['loss'] = (
                metric_sums.get('loss', 0.0) + float(loss_value.item()))
            metric_counts['loss'] = metric_counts.get('loss', 0.0) + 1.0

            # Show current loss in the progress meter
            t.set_postfix(loss='%.05f' % loss_value.item())
            t.update()

            if n_items is not None and batch_idx == n_items:
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

    avg_metrics_str = "Train:"
    for m in avg_metrics.keys():
        avg_metrics_str += ' %s=%.04f' % (m, avg_metrics[m])
    logging.info(avg_metrics_str)
    if nonfinite_gradient_skips:
        logging.warning(
            'Epoch %d completed with %d globally skipped non-finite '
            'gradient batch(es)', epoch, nonfinite_gradient_skips)

    return avg_metrics


def train_epoch_accumulated(
        model, device, optimizer, train_loader, metrics_fn, n_items,
        epoch=0, scaler=None, distributed=False, is_main_process=True,
        grad_clip_norm=0.5, use_amp=True, amp_dtype=torch.float16,
        fail_on_nonfinite=False, skip_nonfinite_gradients=False,
        max_nonfinite_gradient_skips_per_epoch=0, accumulation_steps=1):
    """Train using micro-batches; the default training path stays untouched."""
    del metrics_fn
    if accumulation_steps <= 1:
        raise ValueError('accumulation_steps must be greater than one')
    model.train()
    total = len(train_loader)
    if n_items is not None:
        total = min(total, n_items + 1)
    metric_sum = 0.0
    metric_count = 0.0
    skipped = 0
    optimizer.zero_grad()
    model.zero_grad()
    with tqdm(total=total, desc='Train', ncols=130,
              disable=not is_main_process) as t:
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= total:
                break
            group_start = batch_idx // accumulation_steps * accumulation_steps
            group_size = min(accumulation_steps, total - group_start)
            mixed, gt, enrollment, azim_vec, mic_mask, mic_xyz, metadata = \
                unpack_batch(batch, device)
            auxiliary_kwargs = model_auxiliary_kwargs(batch, device)
            with autocast(enabled=use_amp and device.type == 'cuda',
                          dtype=amp_dtype):
                output, loss = model(
                    mixed, gt, enrollment, azim_vec, mic_mask=mic_mask,
                    mic_xyz=mic_xyz, **auxiliary_kwargs)
            loss_value = loss.mean()
            finite = torch.isfinite(loss_value.detach()).all().to(torch.int32)
            if distributed:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if finite.item() == 0:
                raise FloatingPointError(
                    'Non-finite training loss at epoch=%d batch=%d' %
                    (epoch, batch_idx))
            scaler.scale(loss_value / float(group_size)).backward()
            metric_sum += float(loss_value.item())
            metric_count += 1.0
            if batch_idx - group_start + 1 == group_size:
                scaler.unscale_(optimizer)
                grad_norm = None
                if grad_clip_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip_norm)
                if fail_on_nonfinite or skip_nonfinite_gradients:
                    if grad_norm is None:
                        norms = [torch.linalg.vector_norm(p.grad.float(), 2)
                                 for p in model.parameters()
                                 if p.grad is not None]
                        grad_norm = (torch.linalg.vector_norm(
                            torch.stack(norms), 2) if norms else
                            torch.zeros((), device=device))
                    grad_finite = torch.isfinite(grad_norm).all().to(torch.int32)
                    if distributed:
                        dist.all_reduce(grad_finite, op=dist.ReduceOp.MIN)
                    if grad_finite.item() == 0:
                        optimizer.zero_grad()
                        model.zero_grad()
                        backoff_grad_scaler_on_global_nonfinite(
                            scaler, device, distributed=distributed)
                        if (skip_nonfinite_gradients and skipped <
                                max_nonfinite_gradient_skips_per_epoch):
                            skipped += 1
                            t.update()
                            continue
                        raise FloatingPointError(
                            'Non-finite accumulated gradient at epoch=%d '
                            'batch=%d' % (epoch, batch_idx))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                model.zero_grad()
            t.set_postfix(loss='%.05f' % loss_value.item(),
                          accum='%d/%d' %
                          (batch_idx - group_start + 1, group_size))
            t.update()
    if distributed:
        reduced = torch.tensor([metric_sum, metric_count],
                               dtype=torch.float64, device=device)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        metric_sum, metric_count = reduced.tolist()
    result = {'loss': metric_sum / metric_count if metric_count else 0.0}
    logging.info('Train: loss=%.04f', result['loss'])
    return result


def setup_distributed(args: argparse.Namespace):
    use_cuda = args.use_cuda and torch.cuda.is_available()
    distributed = use_cuda and int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = 0
    local_rank = 0
    world_size = 1
    device_ids = []
    device = torch.device('cpu')

    if use_cuda:
        if distributed:
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))

            if args.gpu_ids is not None:
                if len(args.gpu_ids) < local_world_size:
                    raise ValueError(
                        "When using torchrun, len(gpu_ids) must be >= LOCAL_WORLD_SIZE. "
                        "Got len(gpu_ids)=%d, LOCAL_WORLD_SIZE=%d." %
                        (len(args.gpu_ids), local_world_size)
                    )
                device_id = args.gpu_ids[local_rank]
                device_ids = list(args.gpu_ids[:local_world_size])
            else:
                device_id = local_rank
                device_ids = list(range(local_world_size))

            torch.cuda.set_device(device_id)
            dist.init_process_group(backend=args.dist_backend, init_method="env://")
            device = torch.device("cuda:%d" % device_id)
        else:
            gpu_ids = args.gpu_ids if args.gpu_ids is not None \
                            else range(torch.cuda.device_count())
            device_ids = [_ for _ in gpu_ids]
            if len(device_ids) > 0:
                device = torch.device("cuda:%d" % device_ids[0])
                torch.cuda.set_device(device_ids[0])
            else:
                use_cuda = False
                device = torch.device('cpu')

    data_parallel = distributed or (use_cuda and len(device_ids) > 1)
    is_main_process = (rank == 0)
    return use_cuda, distributed, rank, local_rank, world_size, device, device_ids, data_parallel, is_main_process


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _process_seed(seed: int, rank: int = 0) -> int:
    return (
        int(seed) * 6364136223846793005
        + int(rank) * 1442695040888963407
    ) % ((1 << 63) - 1)


def _epoch_seed(seed: int, epoch: int) -> int:
    return (
        int(seed) * 6364136223846793005
        + int(epoch) * 1442695040888963407
    ) % ((1 << 63) - 1)


def seed_data_worker(_worker_id: int):
    worker_seed = torch.initial_seed() % (1 << 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def configure_deterministic_environment(deterministic: bool):
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def resolve_amp_dtype(name):
    normalized = str(name).lower()
    if normalized in ('float16', 'fp16', 'half'):
        return torch.float16
    if normalized in ('bfloat16', 'bf16'):
        return torch.bfloat16
    raise ValueError("Unsupported amp_dtype '%s'" % name)


def backoff_grad_scaler_on_global_nonfinite(
        scaler: GradScaler, device: torch.device,
        distributed: bool = False):
    """Apply the same loss-scale backoff on every DDP rank.

    ``GradScaler`` records overflows locally. When one DDP rank overflows but
    another does not, calling ``scaler.update()`` independently makes their
    scales diverge even if the optimizer step is skipped globally. Use the
    minimum current scale as the common starting point and back it off once on
    every rank.
    """
    if scaler is None or not scaler.is_enabled():
        return None

    current_scale = float(scaler.get_scale())
    if distributed:
        scale_tensor = torch.tensor(
            current_scale, dtype=torch.float32, device=device)
        dist.all_reduce(scale_tensor, op=dist.ReduceOp.MIN)
        current_scale = float(scale_tensor.item())

    new_scale = max(
        current_scale * float(scaler.get_backoff_factor()), 1.0)
    # Supplying new_scale also clears GradScaler's per-optimizer overflow
    # bookkeeping, preparing it for the next iteration.
    scaler.update(new_scale=new_scale)
    return current_scale, new_scale


def set_random_seed(seed: int, use_cuda: bool, rank: int = 0,
                    deterministic: bool = False):
    configure_deterministic_environment(deterministic)
    process_seed = _process_seed(seed, rank)
    random.seed(process_seed)
    np.random.seed(process_seed % (1 << 32))
    torch.manual_seed(process_seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.enabled = True
    if deterministic:
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(
                deterministic, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(deterministic)
    if use_cuda:
        torch.cuda.manual_seed(process_seed)
        torch.cuda.manual_seed_all(process_seed)


def train(network, args: argparse.Namespace):
    """
    Train the network.
    """
    args.random_seed = int(getattr(args, 'random_seed', 230))
    args.validation_seed = int(getattr(args, 'validation_seed', 42))
    args.deterministic = bool(getattr(args, 'deterministic', False))
    configure_deterministic_environment(args.deterministic)
    distributed = args.distributed
    rank = args.rank
    world_size = args.world_size
    use_cuda = args.use_cuda_enabled
    device = args.device
    data_parallel = args.data_parallel
    is_main_process = args.is_main_process

    train_data_params = dict(args.train_data)
    train_data_params['random_seed'] = int(args.random_seed)
    val_data_params = dict(args.val_data)
    val_data_params['random_seed'] = int(args.validation_seed)
    args.train_data = train_data_params
    args.val_data = val_data_params

    # Load dataset
    data_train = Dataset(**train_data_params)
    logging.info("Loaded train dataset at %s containing %d samples" %
                 (args.train_data['input_dir'], len(data_train)))
    data_val = Dataset(**val_data_params)
    logging.info("Loaded val dataset at %s containing %d samples" %
                 (args.val_data['input_dir'], len(data_val)))

    if use_cuda:
        if distributed:
            logging.info("Using DDP: rank=%d local_rank=%d world_size=%d device=%s" %
                         (rank, args.local_rank, world_size, str(device)))
        else:
            logging.info("Using CUDA devices: %s" % str(args.device_ids))
    else:
        logging.info("Using device: CPU")

    # Set multiprocessing params
    num_workers = min(multiprocessing.cpu_count(), args.n_workers)
    if distributed and num_workers > 0:
        num_workers = max(1, num_workers // world_size)
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': True
    } if use_cuda else {}

    train_sampler = None
    val_sampler = None
    train_batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    accumulation_steps = int(getattr(args, 'gradient_accumulation_steps', 1))
    if accumulation_steps < 1:
        raise ValueError('gradient_accumulation_steps must be positive')

    if distributed:
        if args.batch_size % world_size != 0:
            raise ValueError(
                "For DDP, args.batch_size must be divisible by world_size. "
                "Got batch_size=%d, world_size=%d." % (args.batch_size, world_size)
            )
        if args.eval_batch_size % world_size != 0:
            raise ValueError(
                "For DDP, args.eval_batch_size must be divisible by world_size. "
                "Got eval_batch_size=%d, world_size=%d." % (args.eval_batch_size, world_size)
            )

        train_batch_size = args.batch_size // world_size
        eval_batch_size = args.eval_batch_size // world_size
        train_sampler = DistributedSampler(
            data_train,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(args.random_seed),
        )
        val_sampler = DistributedSampler(
            data_val,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=int(args.validation_seed),
        )

        logging.info(
            "Per-process batch size (train/eval): %d/%d, total workers per process: %d",
            train_batch_size, eval_batch_size, num_workers
        )

    if train_batch_size % accumulation_steps != 0:
        raise ValueError(
            'Per-process train batch size must be divisible by '
            'gradient_accumulation_steps. Got %d and %d.' %
            (train_batch_size, accumulation_steps))
    train_batch_size //= accumulation_steps
    logging.info(
        'Gradient accumulation: steps=%d micro_batch=%d effective_global_batch=%d',
        accumulation_steps, train_batch_size,
        train_batch_size * accumulation_steps * world_size)

    collate_fn = array_agnostic_collate_fn if (
        args.train_data.get('return_array_metadata', False)
        or args.val_data.get('return_array_metadata', False)) else None

    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(
        _process_seed(args.random_seed, rank))
    val_loader_generator = torch.Generator()
    val_loader_generator.manual_seed(
        _process_seed(args.validation_seed, rank))

    # Set up data loaders
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=train_batch_size,
                                               shuffle=(train_sampler is None),
                                               sampler=train_sampler,
                                               collate_fn=collate_fn,
                                               worker_init_fn=seed_data_worker,
                                               generator=train_loader_generator,
                                               **kwargs)
    val_loader = torch.utils.data.DataLoader(data_val,
                                             batch_size=eval_batch_size,
                                             sampler=val_sampler,
                                             collate_fn=collate_fn,
                                             worker_init_fn=seed_data_worker,
                                             generator=val_loader_generator,
                                             **kwargs)

    # Set up model
    model = network.Net(**args.model_params)
    model.to(device)

    if distributed:
        cue_mode = args.model_params.get('cue_mode', 'both')
        has_optional_cue_branches = getattr(
            network, 'HAS_OPTIONAL_CUE_BRANCHES', True)
        need_unused = args.find_unused_parameters or (
            cue_mode != 'both' and has_optional_cue_branches)
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=need_unused
        )
        logging.info("Using DistributedDataParallel model")
    elif use_cuda and data_parallel:
        model = nn.DataParallel(model, device_ids=args.device_ids)
        logging.info("Using data parallel model")

    # Set up the optimizer
    logging.info("Initializing optimizer with %s" % str(args.optim))
    optimizer = network.optimizer(model, **args.optim, data_parallel=data_parallel)
    logging.info('Learning rates initialized to:' + utils.format_lr_info(optimizer))

    lr_sched_type = str(getattr(
        args, 'lr_sched_type', 'reduce_on_plateau')).lower()
    lr_sched_interval = int(getattr(args, 'lr_sched_interval', 1))
    if lr_sched_interval <= 0:
        raise ValueError("lr_sched_interval must be positive")
    if lr_sched_type == 'reduce_on_plateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **args.lr_sched)
    elif lr_sched_type == 'exponential':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, **args.lr_sched)
    else:
        raise ValueError(
            "Unsupported lr_sched_type '%s'" % lr_sched_type)
    logging.info(
        "Initialized %s LR scheduler with params: fix_lr_epochs=%d "
        "interval=%d %s",
        lr_sched_type, args.fix_lr_epochs, lr_sched_interval,
        str(args.lr_sched))

    use_amp = bool(getattr(args, 'use_amp', True))
    amp_dtype = resolve_amp_dtype(getattr(args, 'amp_dtype', 'float16'))
    fail_on_nonfinite = bool(getattr(args, 'fail_on_nonfinite', False))
    skip_nonfinite_gradients = bool(getattr(
        args, 'skip_nonfinite_gradients', False))
    max_nonfinite_gradient_skips_per_epoch = int(getattr(
        args, 'max_nonfinite_gradient_skips_per_epoch', 0))
    if (skip_nonfinite_gradients
            and max_nonfinite_gradient_skips_per_epoch < 1):
        raise ValueError(
            'max_nonfinite_gradient_skips_per_epoch must be at least one '
            'when skip_nonfinite_gradients is enabled')

    amp_init_scale = float(getattr(args, 'amp_init_scale', 65536.0))
    amp_growth_factor = float(getattr(args, 'amp_growth_factor', 2.0))
    amp_backoff_factor = float(getattr(args, 'amp_backoff_factor', 0.5))
    amp_growth_interval = int(getattr(args, 'amp_growth_interval', 2000))
    if amp_init_scale <= 0:
        raise ValueError('amp_init_scale must be positive')
    if amp_growth_factor <= 1.0:
        raise ValueError('amp_growth_factor must be greater than one')
    if not 0.0 < amp_backoff_factor < 1.0:
        raise ValueError('amp_backoff_factor must be between zero and one')
    if amp_growth_interval < 1:
        raise ValueError('amp_growth_interval must be positive')

    # The scaler must span epochs. Recreating it in the epoch loop resets the
    # dynamic scale to 65536 and repeatedly reintroduces FP16 overflows.
    scaler = GradScaler(
        init_scale=amp_init_scale,
        growth_factor=amp_growth_factor,
        backoff_factor=amp_backoff_factor,
        growth_interval=amp_growth_interval,
        enabled=use_cuda and use_amp and amp_dtype == torch.float16)
    logging.info(
        'AMP configuration: enabled=%s dtype=%s init_scale=%g '
        'growth_factor=%g backoff_factor=%g growth_interval=%d',
        scaler.is_enabled(), str(amp_dtype), amp_init_scale,
        amp_growth_factor, amp_backoff_factor, amp_growth_interval)

    base_metric = args.base_metric
    train_metrics = {}
    val_metrics = {}

    # Load the model if `args.start_epoch` is greater than 0. This will load the
    # model from epoch = `args.start_epoch - 1`
    assert args.start_epoch >=0, "start_epoch must be greater than 0."
    if args.start_epoch > 0:
        checkpoint_path = os.path.join(args.exp_dir,
                                       '%d.pt' % (args.start_epoch - 1))
        _, train_metrics, val_metrics = utils.load_checkpoint(
            checkpoint_path, model, optim=optimizer, lr_sched=lr_scheduler,
            data_parallel=data_parallel, scaler=scaler)
        logging.info("Loaded checkpoint from %s" % checkpoint_path)
        logging.info("Learning rates restored to:" + utils.format_lr_info(optimizer))
        if scaler.is_enabled():
            logging.info(
                'AMP GradScaler scale after checkpoint load: %.0f',
                scaler.get_scale())

    # Training loop
    try:
        torch.autograd.set_detect_anomaly(args.detect_anomaly)
        for epoch in range(args.start_epoch, args.epochs + 1):
            if bool(getattr(args, 'reset_optimizer_each_epoch', False)):
                # The public SteerNet recipe launches a fresh one-epoch
                # process for every modelXXX.bin and serializes model weights
                # only. Clearing Adam's moments here reproduces that behavior
                # while retaining this trainer's resumable checkpoints.
                optimizer.state.clear()
                logging.info(
                    "Reset optimizer state for the one-epoch recipe")
            data_train.set_epoch(epoch)
            current_epoch_seed = _epoch_seed(args.random_seed, epoch)
            set_random_seed(
                current_epoch_seed,
                use_cuda=use_cuda,
                rank=rank,
                deterministic=args.deterministic,
            )
            train_loader_generator.manual_seed(
                _process_seed(current_epoch_seed, rank))
            val_loader_generator.manual_seed(
                _process_seed(args.validation_seed, rank))
            if distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            logging.info("Epoch %d:" % epoch)
            checkpoint_file = os.path.join(args.exp_dir, '%d.pt' % epoch)
            assert not os.path.exists(checkpoint_file), \
                "Checkpoint file %s already exists" % checkpoint_file

            if scaler.is_enabled():
                logging.info(
                    'AMP GradScaler scale at epoch %d start: %.0f',
                    epoch, scaler.get_scale())
            train_epoch_fn = (train_epoch_accumulated
                              if accumulation_steps > 1 else train_epoch)
            train_epoch_kwargs = ({'accumulation_steps': accumulation_steps}
                                  if accumulation_steps > 1 else {})
            curr_train_metrics = train_epoch_fn(model, device, optimizer,
                                             train_loader, network.metrics,
                                             args.n_train_items, epoch=epoch,
                                             scaler=scaler,
                                             distributed=distributed,
                                             is_main_process=is_main_process,
                                             grad_clip_norm=(
                                                 None if getattr(
                                                     args, 'grad_clip_norm',
                                                     0.5) is None
                                                 else float(getattr(
                                                     args, 'grad_clip_norm',
                                                     0.5))),
                                             use_amp=use_amp,
                                             amp_dtype=amp_dtype,
                                             fail_on_nonfinite=fail_on_nonfinite,
                                             skip_nonfinite_gradients=(
                                                 skip_nonfinite_gradients),
                                             max_nonfinite_gradient_skips_per_epoch=(
                                                 max_nonfinite_gradient_skips_per_epoch),
                                             **train_epoch_kwargs)
            curr_test_metrics = test_epoch(model, device, val_loader,
                                           args.n_test_items,
                                           network.metrics, epoch=epoch,
                                           writer=args.writer,
                                           data_params=args.val_data,
                                           distributed=distributed,
                                           is_main_process=is_main_process,
                                           use_amp=use_amp,
                                           amp_dtype=amp_dtype,
                                           fail_on_nonfinite=fail_on_nonfinite)
            # LR scheduler
            if (epoch >= args.fix_lr_epochs
                    and (epoch - args.fix_lr_epochs + 1)
                    % lr_sched_interval == 0):
                if lr_sched_type == 'reduce_on_plateau':
                    lr_scheduler.step(curr_test_metrics[base_metric])
                else:
                    lr_scheduler.step()
                logging.info(
                    "LR after scheduling step: %s" %
                    [_['lr'] for _ in optimizer.param_groups])

            # Write metrics to tensorboard
            if args.writer is not None:
                args.writer.add_scalars('Train', curr_train_metrics, epoch)
                args.writer.add_scalars('Val', curr_test_metrics, epoch)
                args.writer.flush()

            for k in curr_train_metrics.keys():
                if not k in train_metrics:
                    train_metrics[k] = [curr_train_metrics[k]]
                else:
                    train_metrics[k].append(curr_train_metrics[k])

            for k in curr_test_metrics.keys():
                if not k in val_metrics:
                    val_metrics[k] = [curr_test_metrics[k]]
                else:
                    val_metrics[k].append(curr_test_metrics[k])

            if max(val_metrics[base_metric]) == val_metrics[base_metric][-1]:
                logging.info("Found best validation %s!" % base_metric)

            if is_main_process:
                utils.save_checkpoint(
                    checkpoint_file, epoch, model, optimizer, lr_scheduler,
                    train_metrics, val_metrics, data_parallel, scaler=scaler)
                logging.info("Saved checkpoint at %s" % checkpoint_file)

                utils.save_graph(train_metrics, val_metrics, args.exp_dir, args.epochs)

            if distributed:
                dist.barrier()

        return train_metrics, val_metrics

    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as _:  # pylint: disable=broad-except
        import traceback  # pylint: disable=import-outside-toplevel
        traceback.print_exc()
        raise



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data Params
    parser.add_argument('exp_dir', type=str,
                        default='./experiments/fsd_mask_label_mult',
                        help="Path to save checkpoints and logs.")

    parser.add_argument('--n_train_items', type=int, default=None,
                        help="Number of items to train on in each epoch")
    parser.add_argument('--n_test_items', type=int, default=None,
                        help="Number of items to test.")
    parser.add_argument('--start_epoch', type=int, default=0,
                        help="Start epoch")
    parser.add_argument('--pretrain_path', type=str,
                        help="Path to pretrained weights")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true',
                        help="Whether to use cuda")
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None,
                        help="List of GPU ids used for training. "
                        "Eg., --gpu_ids 2 4. All GPUs are used by default.")
    parser.add_argument('--detect_anomaly', dest='detect_anomaly',
                        action='store_true',
                        help="Whether to use cuda")
    parser.add_argument('--wandb', dest='wandb', action='store_true',
                        help="Whether to sync tensorboard to wandb")
    parser.add_argument('--dist_backend', type=str, default='nccl',
                        help="Distributed backend for DDP.")
    parser.add_argument('--find_unused_parameters', dest='find_unused_parameters',
                        action='store_true',
                        help="Enable find_unused_parameters in DDP.")

    args = parser.parse_args()

    # Set up checkpoints
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    # Load model and training params
    params = utils.Params(os.path.join(args.exp_dir, 'config.json'))
    for k, v in params.__dict__.items():
        vars(args)[k] = v

    random_protocol_configured = all(
        key in params.__dict__
        for key in ('random_seed', 'validation_seed', 'deterministic')
    )
    args.random_seed = int(getattr(args, 'random_seed', 230))
    args.validation_seed = int(getattr(args, 'validation_seed', 42))
    args.deterministic = bool(getattr(args, 'deterministic', False))
    configure_deterministic_environment(args.deterministic)

    try:
        use_cuda, distributed, rank, local_rank, world_size, device, device_ids, data_parallel, is_main_process = setup_distributed(args)
        args.use_cuda_enabled = use_cuda
        args.distributed = distributed
        args.rank = rank
        args.local_rank = local_rank
        args.world_size = world_size
        args.device = device
        args.device_ids = device_ids
        args.data_parallel = data_parallel
        args.is_main_process = is_main_process

        if is_main_process:
            utils.set_logger(os.path.join(args.exp_dir, 'train.log'))
        else:
            logging.getLogger().handlers.clear()
            logging.basicConfig(level=logging.WARNING, format='%(message)s')

        if args.start_epoch > 0 and not random_protocol_configured:
            logging.warning(
                "Resuming a legacy config without the explicit random "
                "protocol; sample generation will differ from the original "
                "run. Use a fresh seed directory for paper experiments."
            )

        # Set the random protocol before constructing the model and datasets.
        set_random_seed(
            args.random_seed,
            use_cuda=use_cuda,
            rank=rank,
            deterministic=args.deterministic,
        )
        logging.info(
            "Random protocol: train_seed=%d validation_seed=%d "
            "deterministic=%s rank=%d",
            args.random_seed,
            args.validation_seed,
            args.deterministic,
            rank,
        )

        # Initialize tensorboard writer (rank 0 only)
        tensorboard_dir = os.path.join(args.exp_dir, 'tensorboard')
        args.writer = SummaryWriter(tensorboard_dir, purge_step=args.start_epoch) if is_main_process else None

        if args.wandb and is_main_process:
            import wandb
            wandb.init(
                project='Semaudio', sync_tensorboard=True,
                dir=tensorboard_dir, name=os.path.basename(args.exp_dir))
        else:
            wandb = None

        network = importlib.import_module(args.model)
        logging.info("Imported the model from '%s'." % args.model)

        train(network, args)

        if args.writer is not None:
            args.writer.close()
        if args.wandb and is_main_process and wandb is not None:
            wandb.finish()
    finally:
        cleanup_distributed()
        
