"""Evaluate geometry-aware DOA preservation of multichannel TSE outputs.

The protocol estimates the target azimuth once from the reverberant
multichannel label and once from the model output.  The primary preservation
metric is the direct, geometry-aware drift between those two estimates.  The
absolute output-to-ground-truth error and the signed excess error

    output DOA error - label DOA error

are retained as localization diagnostics, but the signed difference is not a
spatial-preservation distance and must not be used as the paper's primary
metric.

The default localizer is a frame-aggregated broadband SRP-PHAT search.  It
keeps the complete GCC-PHAT function for every microphone pair and combines
all pairs at every candidate azimuth, rather than committing to one lag peak
per pair before fitting a direction.  A fixed label-derived activity mask is
used for both label and output so systems cannot choose easier frames.

The implementation handles the 2--8 microphone benchmark geometries.  A
rank-one horizontal geometry has a front/back ambiguity, so its error is
minimized over the physically indistinguishable reflected direction.
Rank-two geometries use ordinary circular azimuth error.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import importlib
import importlib.util
import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from src.helpers import utils
from src.training.dataset import (
    SavedTSETestset,
    array_agnostic_collate_fn,
    model_auxiliary_kwargs,
    unpack_batch,
)
from src.training.eval_doa_mismatch import DOAOffsetDataset


def circular_error_deg(estimate, reference):
    """Smallest absolute angular distance on a 360-degree circle."""
    return abs((float(estimate) - float(reference) + 180.0) % 360.0 - 180.0)


def _valid_geometry(mic_xyz, mic_mask=None):
    xyz = np.asarray(
        mic_xyz.detach().cpu() if torch.is_tensor(mic_xyz) else mic_xyz,
        dtype=np.float64,
    )
    if xyz.ndim != 2 or xyz.shape[1] < 2:
        raise ValueError('mic_xyz must have shape [microphone, coordinate]')
    if mic_mask is None:
        valid = np.arange(xyz.shape[0])
    else:
        mask = np.asarray(
            mic_mask.detach().cpu() if torch.is_tensor(mic_mask) else mic_mask
        ).reshape(-1)
        valid = np.flatnonzero(mask > 0.5)
    valid = valid[valid < xyz.shape[0]]
    return xyz[valid], valid.tolist()


def horizontal_geometry_rank(mic_xyz, mic_mask=None, relative_tol=1e-3):
    """Numerical rank of centered microphone coordinates in the xy plane."""
    xyz, _ = _valid_geometry(mic_xyz, mic_mask)
    if xyz.shape[0] < 2:
        return 0
    centered = xyz[:, :2] - xyz[:, :2].mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] <= 1e-10:
        return 0
    tolerance = max(1e-8, float(singular_values[0]) * float(relative_tol))
    return int(np.sum(singular_values > tolerance))


def _rank_one_axis(mic_xyz, mic_mask=None):
    xyz, _ = _valid_geometry(mic_xyz, mic_mask)
    centered = xyz[:, :2] - xyz[:, :2].mean(axis=0, keepdims=True)
    if centered.shape[0] < 2:
        raise ValueError('At least two distinct microphones are required')
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 1e-10:
        raise ValueError('Microphone geometry has zero horizontal aperture')
    axis = vh[0]
    return axis / np.linalg.norm(axis)


def reflect_angle_across_rank_one_array(angle_deg, mic_xyz, mic_mask=None):
    """Reflect a source direction across a rank-one array axis."""
    axis = _rank_one_axis(mic_xyz, mic_mask)
    radians = math.radians(float(angle_deg))
    direction = np.asarray([math.sin(radians), math.cos(radians)])
    reflected = 2.0 * axis * float(np.dot(axis, direction)) - direction
    return math.degrees(math.atan2(reflected[0], reflected[1])) % 360.0


def geometry_aware_error_deg(estimate, reference, mic_xyz, mic_mask=None):
    """Return DOA error and the ambiguity rule used for this array."""
    rank = horizontal_geometry_rank(mic_xyz, mic_mask)
    if rank < 1:
        return float('nan'), 'invalid'
    direct = circular_error_deg(estimate, reference)
    if rank >= 2:
        return direct, 'circular'
    reflected = reflect_angle_across_rank_one_array(
        estimate, mic_xyz, mic_mask)
    return min(direct, circular_error_deg(reflected, reference)), 'rank1_resolved'


def _next_power_of_two(value):
    return 1 if value <= 1 else 2 ** int(value - 1).bit_length()


def estimate_doa_gcc_grid(audio, mic_xyz, sample_rate, speed_of_sound=343.0,
                          angle_grid_step=1.0, interpolation=16, sign=1.0,
                          mic_mask=None):
    """Pairwise GCC-PHAT TDOAs followed by geometry-aware azimuth search."""
    audio_tensor = (
        audio.detach().cpu().float() if torch.is_tensor(audio)
        else torch.as_tensor(audio, dtype=torch.float32)
    )
    xyz, valid = _valid_geometry(mic_xyz, mic_mask)
    valid = [index for index in valid if index < audio_tensor.shape[0]]
    if len(valid) < 2:
        return {'valid_estimation': False,
                'failure_reason': 'too_few_valid_microphones'}
    signals = audio_tensor[valid].numpy().astype(np.float64, copy=False)
    signals = signals - signals.mean(axis=1, keepdims=True)
    source_rms = float(np.sqrt(np.mean(signals ** 2)))
    source_peak = float(np.max(np.abs(signals)))
    if not np.isfinite(source_rms) or source_rms < 1e-10:
        return {
            'valid_estimation': False,
            'failure_reason': 'silent_or_nonfinite',
            'source_rms': source_rms,
            'source_peak': source_peak,
        }

    n_fft = _next_power_of_two(signals.shape[-1] * 2)
    spectra = np.fft.rfft(signals, n=n_fft, axis=-1)
    n_correlation = n_fft * max(1, int(interpolation))
    eps = np.finfo(np.float64).eps
    pair_data = []
    for left in range(len(valid)):
        for right in range(left + 1, len(valid)):
            baseline = xyz[right] - xyz[left]
            baseline_length = float(np.linalg.norm(baseline))
            if baseline_length <= 0.0:
                continue
            cross = spectra[left] * np.conj(spectra[right])
            cross /= np.maximum(np.abs(cross), eps)
            correlation = np.fft.irfft(cross, n=n_correlation)
            max_tau = 1.25 * baseline_length / float(speed_of_sound)
            max_tau += 1.0 / float(sample_rate)
            max_shift = int(math.ceil(
                max_tau * float(sample_rate) * float(interpolation)))
            max_shift = max(1, min(max_shift, correlation.size // 2 - 1))
            window = np.concatenate(
                (correlation[-max_shift:], correlation[:max_shift + 1]))
            peak_index = int(np.argmax(window))
            shift = peak_index - max_shift
            pair_data.append({
                'baseline': baseline,
                'baseline_length': baseline_length,
                'tau': shift / (float(sample_rate) * float(interpolation)),
                'peak': float(window[peak_index]),
            })
    if not pair_data:
        return {
            'valid_estimation': False,
            'failure_reason': 'no_valid_pairs',
            'source_rms': source_rms,
            'source_peak': source_peak,
        }

    angles = np.arange(0.0, 360.0, float(angle_grid_step), dtype=np.float64)
    radians = np.deg2rad(angles)
    directions = np.stack(
        [np.sin(radians), np.cos(radians), np.zeros_like(radians)], axis=1)
    observed = np.asarray([item['tau'] for item in pair_data])
    modeled = np.stack([
        directions.dot(item['baseline']) / float(speed_of_sound)
        for item in pair_data
    ])
    residuals = observed[:, None] - float(sign) * modeled
    absolute = np.abs(residuals)
    huber_delta = 0.25 / float(sample_rate)
    losses = np.where(
        absolute <= huber_delta,
        0.5 * residuals ** 2,
        huber_delta * (absolute - 0.5 * huber_delta),
    )
    weights = np.asarray([
        max(item['baseline_length'], 1e-6) * max(item['peak'], 1e-6)
        for item in pair_data
    ])
    weights /= max(float(weights.sum()), eps)
    cost = np.sum(weights[:, None] * losses, axis=0)
    best_index = int(np.argmin(cost))
    best_residuals = residuals[:, best_index]
    return {
        'valid_estimation': True,
        'failure_reason': '',
        'est_azim_deg': float(angles[best_index] % 360.0),
        'gcc_grid_cost': float(cost[best_index]),
        'mean_abs_tdoa_residual_us': float(
            np.mean(np.abs(best_residuals)) * 1e6),
        'source_rms': source_rms,
        'source_peak': source_peak,
        'n_pairs': len(pair_data),
    }


def _frame_audio(audio, frame_length, hop_length):
    """Return zero-padded overlapping frames with shape [mic, frame, time]."""
    signals = np.asarray(audio, dtype=np.float64)
    if signals.ndim != 2:
        raise ValueError('audio must have shape [microphone, time]')
    n_samples = signals.shape[-1]
    if n_samples <= frame_length:
        n_frames = 1
    else:
        n_frames = int(math.ceil(
            (n_samples - frame_length) / float(hop_length))) + 1
    total = (n_frames - 1) * hop_length + frame_length
    if total > n_samples:
        signals = np.pad(signals, ((0, 0), (0, total - n_samples)))
    frames = np.lib.stride_tricks.sliding_window_view(
        signals, frame_length, axis=-1)
    return np.asarray(frames[:, ::hop_length, :], dtype=np.float64)


def _parabolic_peak_offset(left, center, right):
    """Sub-grid offset of a sampled maximum, clipped to one grid cell."""
    denominator = float(left) - 2.0 * float(center) + float(right)
    if abs(denominator) <= 1e-12:
        return 0.0
    offset = 0.5 * (float(left) - float(right)) / denominator
    return float(np.clip(offset, -1.0, 1.0))


def estimate_doa_srp_phat(
        audio, mic_xyz, sample_rate, speed_of_sound=343.0,
        angle_grid_step=1.0, sign=1.0, mic_mask=None,
        activity_audio=None, frame_length=1024, hop_length=256,
        frequency_min=200.0, frequency_max=3500.0,
        activity_threshold_db=-40.0, sidelobe_exclusion_deg=10.0):
    """Frame-aggregated broadband SRP-PHAT over all valid microphone pairs.

    The activity mask is derived from ``activity_audio`` and then applied to
    ``audio``.  During model evaluation the former is always the multichannel
    label, which makes the evaluated time support identical for every system.
    """
    audio_tensor = (
        audio.detach().cpu().float() if torch.is_tensor(audio)
        else torch.as_tensor(audio, dtype=torch.float32)
    )
    activity_tensor = (
        activity_audio.detach().cpu().float()
        if torch.is_tensor(activity_audio)
        else torch.as_tensor(
            audio if activity_audio is None else activity_audio,
            dtype=torch.float32)
    )
    xyz, valid = _valid_geometry(mic_xyz, mic_mask)
    valid = [
        index for index in valid
        if index < audio_tensor.shape[0]
        and index < activity_tensor.shape[0]
    ]
    if len(valid) < 2:
        return {'valid_estimation': False,
                'failure_reason': 'too_few_valid_microphones'}
    signals = audio_tensor[valid].numpy().astype(np.float64, copy=False)
    activity = activity_tensor[valid].numpy().astype(
        np.float64, copy=False)
    source_rms = float(np.sqrt(np.mean(signals ** 2)))
    source_peak = float(np.max(np.abs(signals)))
    if not np.isfinite(source_rms) or source_rms < 1e-10:
        return {
            'valid_estimation': False,
            'failure_reason': 'silent_or_nonfinite',
            'source_rms': source_rms,
            'source_peak': source_peak,
        }

    frame_length = int(frame_length)
    hop_length = int(hop_length)
    if frame_length < 16 or hop_length < 1:
        raise ValueError('Invalid SRP frame/hop length')
    signal_frames = _frame_audio(signals, frame_length, hop_length)
    activity_frames = _frame_audio(activity, frame_length, hop_length)
    if signal_frames.shape[1] != activity_frames.shape[1]:
        raise RuntimeError('Signal and activity frame counts differ')
    activity_energy = np.mean(activity_frames ** 2, axis=(0, 2))
    finite_energy = np.isfinite(activity_energy)
    peak_energy = float(np.max(activity_energy[finite_energy])) \
        if finite_energy.any() else 0.0
    if peak_energy <= 0.0:
        active = finite_energy
    else:
        relative = 10.0 ** (float(activity_threshold_db) / 10.0)
        active = finite_energy & (activity_energy >= peak_energy * relative)
    if not active.any():
        active[int(np.nanargmax(activity_energy))] = True

    window = np.hanning(frame_length).astype(np.float64)
    centered = signal_frames - signal_frames.mean(axis=-1, keepdims=True)
    spectra = np.fft.rfft(centered * window[None, None, :], axis=-1)
    frequencies = np.fft.rfftfreq(frame_length, d=1.0 / float(sample_rate))
    band = ((frequencies >= float(frequency_min))
            & (frequencies <= min(
                float(frequency_max), float(sample_rate) / 2.0)))
    if int(band.sum()) < 2:
        raise ValueError('SRP frequency band contains fewer than two bins')
    frequencies = frequencies[band]

    angles = np.arange(0.0, 360.0, float(angle_grid_step),
                       dtype=np.float64)
    radians = np.deg2rad(angles)
    directions = np.stack(
        [np.sin(radians), np.cos(radians), np.zeros_like(radians)], axis=1)
    response = np.zeros(angles.shape, dtype=np.float64)
    pair_count = 0
    eps = np.finfo(np.float64).eps
    for left in range(len(valid)):
        for right in range(left + 1, len(valid)):
            baseline = xyz[right] - xyz[left]
            if float(np.linalg.norm(baseline)) <= 0.0:
                continue
            cross = (spectra[left, active][:, band]
                     * np.conj(spectra[right, active][:, band]))
            magnitude = np.abs(cross)
            usable = magnitude > eps
            phat = np.zeros_like(cross)
            phat[usable] = cross[usable] / magnitude[usable]
            counts = usable.sum(axis=0)
            cross_mean = np.divide(
                phat.sum(axis=0), counts,
                out=np.zeros(phat.shape[1], dtype=np.complex128),
                where=counts > 0)
            modeled_tau = (directions.dot(baseline)
                           / float(speed_of_sound)) * float(sign)
            steering = np.exp(
                2j * np.pi * frequencies[:, None] * modeled_tau[None, :])
            pair_response = np.real(cross_mean @ steering)
            pair_response /= max(1, int((counts > 0).sum()))
            response += pair_response
            pair_count += 1
    if pair_count == 0 or not np.isfinite(response).any():
        return {
            'valid_estimation': False,
            'failure_reason': 'no_valid_pairs',
            'source_rms': source_rms,
            'source_peak': source_peak,
        }
    response /= float(pair_count)
    best_index = int(np.nanargmax(response))
    offset = _parabolic_peak_offset(
        response[(best_index - 1) % len(response)],
        response[best_index],
        response[(best_index + 1) % len(response)])
    estimate = (angles[best_index] + offset * float(angle_grid_step)) % 360.0

    distance = np.abs(
        (angles - estimate + 180.0) % 360.0 - 180.0)
    excluded = distance <= float(sidelobe_exclusion_deg)
    rank = horizontal_geometry_rank(xyz)
    if rank == 1:
        reflected = reflect_angle_across_rank_one_array(estimate, xyz)
        reflected_distance = np.abs(
            (angles - reflected + 180.0) % 360.0 - 180.0)
        excluded |= reflected_distance <= float(sidelobe_exclusion_deg)
    sidelobes = response[~excluded]
    second_peak = float(np.max(sidelobes)) if sidelobes.size else float('nan')
    scale = float(np.std(response))
    peak = float(response[best_index])
    confidence = (
        (peak - second_peak) / max(scale, eps)
        if np.isfinite(second_peak) else float('nan'))
    return {
        'valid_estimation': True,
        'failure_reason': '',
        'est_azim_deg': float(estimate),
        'srp_peak': peak,
        'srp_peak_to_sidelobe_std': float(confidence),
        'srp_peak_prominence_std': float(
            (peak - np.median(response)) / max(scale, eps)),
        'source_rms': source_rms,
        'source_peak': source_peak,
        'n_pairs': pair_count,
        'n_active_frames': int(active.sum()),
        'active_frame_fraction': float(active.mean()),
    }


def _atomic_write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix='.%s.' % path.name, suffix='.tmp', dir=str(path.parent))
    os.close(descriptor)
    try:
        pd.DataFrame(rows).to_csv(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _atomic_write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix='.%s.' % path.name, suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True,
                      allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _dataset_kwargs(test_data, saved_path, num_samples, max_n_mics=None):
    return {
        'saved_path': str(saved_path),
        'num_samples': num_samples,
        'max_n_mics': int(max_n_mics or test_data.get(
            'max_n_mics', test_data.get('n_mics', 8))),
        'azim_type': test_data.get('azim_type', 'cycpos'),
        'd_model': test_data.get('d_model', 40),
        'alpha': test_data.get('alpha', 80),
        'append_doa_unit_vector': test_data.get(
            'append_doa_unit_vector', False),
    }


def _best_checkpoint(experiment_dir, base_metric):
    checkpoints = sorted(
        glob.glob(str(Path(experiment_dir) / '*.pt')),
        key=lambda path: int(Path(path).stem),
    )
    if not checkpoints:
        raise FileNotFoundError('No integer-named checkpoints in %s' % experiment_dir)
    latest = torch.load(checkpoints[-1], map_location='cpu')
    history = np.asarray(latest['val_metrics'][base_metric], dtype=np.float64)
    if not np.isfinite(history).any():
        raise RuntimeError('No finite validation values for %s' % base_metric)
    epoch = int(np.nanargmax(history))
    checkpoint = Path(experiment_dir) / ('%d.pt' % epoch)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint.resolve(), epoch, float(history[epoch])


def _resolve_checkpoint(experiment_dir, base_metric, pretrain_path):
    if pretrain_path == 'best':
        return _best_checkpoint(experiment_dir, base_metric)
    checkpoint = Path(pretrain_path)
    if not checkpoint.is_absolute():
        candidate = Path(experiment_dir) / checkpoint
        checkpoint = candidate if candidate.is_file() else checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu')
    epoch = int(payload.get('epoch', int(checkpoint.stem)))
    history = np.asarray(
        payload.get('val_metrics', {}).get(base_metric, []),
        dtype=np.float64,
    )
    value = (
        float(history[epoch])
        if 0 <= epoch < history.size and np.isfinite(history[epoch])
        else None
    )
    return checkpoint, epoch, value


def _estimator_kwargs(args):
    common = {
        'sample_rate': args.sample_rate,
        'speed_of_sound': args.speed_of_sound,
        'angle_grid_step': args.angle_grid_step,
        'sign': args.sign,
    }
    if args.estimator == 'gcc_peak_grid':
        common['interpolation'] = args.gcc_interpolation
    else:
        common.update({
            'frame_length': args.srp_frame_length,
            'hop_length': args.srp_hop_length,
            'frequency_min': args.srp_frequency_min,
            'frequency_max': args.srp_frequency_max,
            'activity_threshold_db': args.srp_activity_threshold_db,
            'sidelobe_exclusion_deg': args.srp_sidelobe_exclusion_deg,
        })
    return common


def _source_estimate_row(audio, mic_xyz, mic_mask, metadata, args,
                         source_kind, activity_audio=None):
    if args.estimator == 'gcc_peak_grid':
        estimate = estimate_doa_gcc_grid(
            audio, mic_xyz, mic_mask=mic_mask, **_estimator_kwargs(args))
    else:
        estimate = estimate_doa_srp_phat(
            audio, mic_xyz, mic_mask=mic_mask,
            activity_audio=activity_audio, **_estimator_kwargs(args))
    row = {
        'sample_key': metadata['sample_key'],
        'sample_id': metadata['sample_id'],
        'subset': metadata['subset'],
        'source_kind': source_kind,
        'true_azim_deg': float(metadata['azim_deg']),
        'n_valid_mics': int(metadata['n_valid_mics']),
        'geometry_type': metadata.get('geometry_type', ''),
        'array_rank_xy': horizontal_geometry_rank(mic_xyz, mic_mask),
        **estimate,
    }
    if estimate.get('valid_estimation', False):
        error, mode = geometry_aware_error_deg(
            estimate['est_azim_deg'], metadata['azim_deg'], mic_xyz, mic_mask)
    else:
        error, mode = float('nan'), 'invalid'
    row['doa_error_mode'] = mode
    row['doa_error_deg'] = error
    return row


def build_label_cache(args):
    """Estimate DOA from each saved multichannel target exactly once."""
    dataset = SavedTSETestset(
        saved_path=args.testset,
        num_samples=args.num_samples,
        max_n_mics=args.max_n_mics,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=array_agnostic_collate_fn,
    )
    rows = []
    with ThreadPoolExecutor(max_workers=args.doa_workers) as doa_pool, tqdm(
            total=len(dataset), desc='Label DOA', ncols=110) as progress:
        for batch in loader:
            _, target, _, _, mic_mask, mic_xyz, metadata = unpack_batch(batch)
            def estimate_index(index):
                return _source_estimate_row(
                    target[index], mic_xyz[index], mic_mask[index],
                    metadata[index], args, 'label',
                    activity_audio=target[index])
            batch_rows = doa_pool.map(estimate_index, range(target.shape[0]))
            for row in batch_rows:
                rows.append(row)
                progress.update(1)
    if len(rows) != len(dataset):
        raise RuntimeError('Expected %d label rows, got %d' %
                           (len(dataset), len(rows)))
    _atomic_write_csv(rows, args.label_cache)
    _atomic_write_json({
        'testset': str(Path(args.testset).resolve()),
        'num_samples': len(dataset),
        'estimator': args.estimator,
        'sample_rate': args.sample_rate,
        'speed_of_sound': args.speed_of_sound,
        'angle_grid_step': args.angle_grid_step,
        'gcc_interpolation': args.gcc_interpolation,
        'srp_frame_length': args.srp_frame_length,
        'srp_hop_length': args.srp_hop_length,
        'srp_frequency_min': args.srp_frequency_min,
        'srp_frequency_max': args.srp_frequency_max,
        'srp_activity_threshold_db': args.srp_activity_threshold_db,
        'srp_sidelobe_exclusion_deg': args.srp_sidelobe_exclusion_deg,
        'sign': args.sign,
        'rank_one_error': 'reflection_resolved',
        'rank_two_error': 'circular',
    }, Path(args.label_cache).with_suffix('.json'))
    return pd.DataFrame(rows)


def _load_label_cache(path, expected_count, args=None):
    frame = pd.read_csv(path)
    required = {
        'sample_key', 'true_azim_deg', 'est_azim_deg', 'doa_error_deg',
        'valid_estimation', 'doa_error_mode',
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError('Label cache is missing columns: %s' % sorted(missing))
    if len(frame) != expected_count:
        raise ValueError('Label cache has %d rows; expected %d' %
                         (len(frame), expected_count))
    if frame['sample_key'].duplicated().any():
        raise ValueError('Label cache contains duplicate sample_key values')
    if args is not None:
        manifest_path = Path(path).with_suffix('.json')
        if not manifest_path.is_file():
            raise FileNotFoundError(
                'Missing label-cache manifest: %s' % manifest_path)
        with manifest_path.open(encoding='utf-8') as stream:
            manifest = json.load(stream)
        expected = {
            'testset': str(Path(args.testset).resolve()),
            'num_samples': int(expected_count),
            'estimator': args.estimator,
            'sample_rate': args.sample_rate,
            'speed_of_sound': args.speed_of_sound,
            'angle_grid_step': args.angle_grid_step,
            'srp_frame_length': args.srp_frame_length,
            'srp_hop_length': args.srp_hop_length,
            'srp_frequency_min': args.srp_frequency_min,
            'srp_frequency_max': args.srp_frequency_max,
            'srp_activity_threshold_db': args.srp_activity_threshold_db,
            'srp_sidelobe_exclusion_deg': args.srp_sidelobe_exclusion_deg,
            'sign': args.sign,
        }
        mismatches = []
        for key, expected_value in expected.items():
            actual_value = manifest.get(key)
            if isinstance(expected_value, float):
                matches = (
                    actual_value is not None
                    and np.isclose(float(actual_value), expected_value))
            else:
                matches = actual_value == expected_value
            if not matches:
                mismatches.append(
                    '%s=%r (expected %r)' %
                    (key, actual_value, expected_value))
        if mismatches:
            raise ValueError(
                'Label-cache manifest mismatch: ' + '; '.join(mismatches))
    return frame.set_index('sample_key', drop=False)


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {'true', '1', 'yes'}
    return bool(value)


def _bootstrap_mean_ci(values, seed, n_resamples=10000):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_resamples), dtype=np.float64)
    chunk = 1000
    for start in range(0, int(n_resamples), chunk):
        stop = min(start + chunk, int(n_resamples))
        indices = rng.integers(0, finite.size, size=(stop - start, finite.size))
        means[start:stop] = finite[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def summarize_rows(rows, bootstrap_seed):
    frame = pd.DataFrame(rows)
    summaries = []
    for offset, group in frame.groupby('doa_offset_deg', sort=True):
        valid = group[
            group['label_valid_estimation'].astype(bool)
            & group['output_valid_estimation'].astype(bool)
        ]
        excess = valid['excess_doa_error_deg'].to_numpy(dtype=np.float64)
        ci_low, ci_high = _bootstrap_mean_ci(
            excess, bootstrap_seed + int(round(float(offset) * 10.0)))
        drift = valid['doa_drift_deg'].to_numpy(dtype=np.float64)
        drift_ci_low, drift_ci_high = _bootstrap_mean_ci(
            drift, bootstrap_seed + 100000
            + int(round(float(offset) * 10.0)))
        reliable = valid[valid['label_doa_error_deg'] <= 5.0]
        summaries.append({
            'doa_offset_deg': float(offset),
            'n_samples': int(len(group)),
            'n_valid': int(len(valid)),
            'reference_scale_invariant_signal_noise_ratio_i': group[
                'reference_scale_invariant_signal_noise_ratio_i'].mean(),
            'label_mean_doa_error_deg': valid['label_doa_error_deg'].mean(),
            'label_median_doa_error_deg': valid['label_doa_error_deg'].median(),
            'label_p90_doa_error_deg': valid['label_doa_error_deg'].quantile(0.90),
            'output_mean_doa_error_deg': valid['output_doa_error_deg'].mean(),
            'output_median_doa_error_deg': valid['output_doa_error_deg'].median(),
            'output_p90_doa_error_deg': valid['output_doa_error_deg'].quantile(0.90),
            'mean_excess_doa_error_deg': valid['excess_doa_error_deg'].mean(),
            'median_excess_doa_error_deg': valid['excess_doa_error_deg'].median(),
            'p90_excess_doa_error_deg': valid['excess_doa_error_deg'].quantile(0.90),
            'mean_excess_doa_error_ci95_low_deg': ci_low,
            'mean_excess_doa_error_ci95_high_deg': ci_high,
            'fraction_output_worse_than_label': float(
                (valid['excess_doa_error_deg'] > 0.0).mean()),
            'mean_doa_drift_deg': valid['doa_drift_deg'].mean(),
            'median_doa_drift_deg': valid['doa_drift_deg'].median(),
            'p90_doa_drift_deg': valid['doa_drift_deg'].quantile(0.90),
            'mean_doa_drift_ci95_low_deg': drift_ci_low,
            'mean_doa_drift_ci95_high_deg': drift_ci_high,
            'pres_at_5deg': float((valid['doa_drift_deg'] <= 5.0).mean()),
            'pres_at_10deg': float((valid['doa_drift_deg'] <= 10.0).mean()),
            'n_label_reliable_at_5deg': int(len(reliable)),
            'label_reliable_coverage_at_5deg': float(
                len(reliable) / max(1, len(valid))),
            'reliable_output_mean_doa_error_deg': (
                reliable['output_doa_error_deg'].mean()),
            'reliable_output_loc_at_5deg': float(
                (reliable['output_doa_error_deg'] <= 5.0).mean())
            if len(reliable) else float('nan'),
        })
    return summaries


def evaluate_experiment(experiment, args):
    experiment_dir = Path(args.project_root) / 'experiments' / experiment
    params = utils.Params(str(experiment_dir / 'config.json'))
    amp_dtype_name = str(args.amp_dtype).lower()
    if amp_dtype_name == 'auto':
        amp_dtype_name = str(
            getattr(params, 'amp_dtype', 'float16')).lower()
    if amp_dtype_name in {'bfloat16', 'bf16'}:
        amp_dtype = torch.bfloat16
        amp_dtype_name = 'bfloat16'
    elif amp_dtype_name in {'float16', 'fp16'}:
        amp_dtype = torch.float16
        amp_dtype_name = 'float16'
    else:
        raise ValueError('Unsupported AMP dtype: %s' % args.amp_dtype)
    dataset = SavedTSETestset(**_dataset_kwargs(
        params.test_data, args.testset, args.num_samples, args.max_n_mics))
    label_frame = _load_label_cache(args.label_cache, len(dataset), args)

    if args.network_source:
        source = Path(args.network_source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        # The external DeFTAN tree also uses the top-level package name
        # ``src``.  This evaluator has already imported the local ``src``, so
        # importing the external network directly would otherwise resolve its
        # helper import to this project's incompatible module.  Load the
        # sibling helper explicitly and expose it only while executing the
        # external network module; the network keeps direct references to the
        # imported functions afterward.
        external_utils_source = source.parent.parent / 'helpers' / 'utils.py'
        if not external_utils_source.is_file():
            raise FileNotFoundError(external_utils_source)
        helper_spec = importlib.util.spec_from_file_location(
            'external_doa_mismatch_utils', external_utils_source)
        if helper_spec is None or helper_spec.loader is None:
            raise ImportError(
                'Unable to load external helpers %s' % external_utils_source)
        external_utils = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(external_utils)
        spec = importlib.util.spec_from_file_location(
            'external_doa_mismatch_network', source)
        if spec is None or spec.loader is None:
            raise ImportError('Unable to load network source %s' % source)
        network = importlib.util.module_from_spec(spec)
        module_key = 'src.helpers.utils'
        local_utils_module = sys.modules.get(module_key)
        sys.modules[module_key] = external_utils
        try:
            spec.loader.exec_module(network)
        finally:
            if local_utils_module is None:
                sys.modules.pop(module_key, None)
            else:
                sys.modules[module_key] = local_utils_module
    else:
        network = importlib.import_module(params.model)
    model = network.Net(**params.model_params).to(args.device)
    checkpoint, checkpoint_epoch, best_value = _resolve_checkpoint(
        experiment_dir, params.base_metric, args.pretrain_path)
    utils.load_checkpoint(str(checkpoint), model, data_parallel=False)
    model.eval()

    rows = []
    with ThreadPoolExecutor(max_workers=args.doa_workers) as doa_pool:
      for offset in args.offsets:
        offset_dataset = DOAOffsetDataset(dataset, offset)
        loader = torch.utils.data.DataLoader(
            offset_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
            pin_memory=args.device.type == 'cuda',
            collate_fn=array_agnostic_collate_fn,
        )
        description = '%s offset=%+g' % (Path(experiment).name, offset)
        with torch.no_grad(), tqdm(
                total=len(dataset), desc=description, ncols=120) as progress:
            for batch in loader:
                (mixture, target, enrollment, azim_vec, mic_mask, mic_xyz,
                 metadata) = unpack_batch(batch, args.device)
                auxiliary = model_auxiliary_kwargs(batch, args.device)
                with autocast(
                        enabled=args.use_amp and args.device.type == 'cuda',
                        dtype=amp_dtype):
                    if args.network_source:
                        output, _ = model(
                            mixture, target, enrollment, azim_vec)
                    else:
                        output, _ = model(
                            mixture, target, enrollment, azim_vec,
                            mic_mask=mic_mask, mic_xyz=mic_xyz, **auxiliary)
                output = output.detach().cpu()
                mixture_cpu = mixture.detach().cpu()
                target_cpu = target.detach().cpu()
                mic_mask_cpu = mic_mask.detach().cpu()
                mic_xyz_cpu = mic_xyz.detach().cpu()
                output_si_snr = utils.scale_invariant_signal_noise_ratio(
                    output[:, :1].float(), target_cpu[:, :1].float())
                mixture_si_snr = utils.scale_invariant_signal_noise_ratio(
                    mixture_cpu[:, :1].float(), target_cpu[:, :1].float())
                reference_si_snri = output_si_snr - mixture_si_snr
                def estimate_index(index):
                    return _source_estimate_row(
                        output[index], mic_xyz_cpu[index],
                        mic_mask_cpu[index], metadata[index], args, 'output',
                        activity_audio=target_cpu[index])
                output_rows = doa_pool.map(
                    estimate_index, range(output.shape[0]))
                for index, output_row in enumerate(output_rows):
                    sample_key = metadata[index]['sample_key']
                    if sample_key not in label_frame.index:
                        raise KeyError('Missing label estimate for %s' % sample_key)
                    label = label_frame.loc[sample_key]
                    label_valid = _as_bool(label['valid_estimation'])
                    output_valid = bool(output_row['valid_estimation'])
                    if label_valid and output_valid:
                        drift, _ = geometry_aware_error_deg(
                            output_row['est_azim_deg'], label['est_azim_deg'],
                            mic_xyz_cpu[index], mic_mask_cpu[index])
                    else:
                        drift = float('nan')
                    rows.append({
                        'experiment': experiment,
                        'checkpoint': str(checkpoint),
                        'checkpoint_epoch': checkpoint_epoch,
                        'doa_offset_deg': float(offset),
                        'sample_key': sample_key,
                        'sample_id': metadata[index]['sample_id'],
                        'subset': metadata[index]['subset'],
                        'true_azim_deg': float(metadata[index]['azim_deg']),
                        'n_valid_mics': int(metadata[index]['n_valid_mics']),
                        'geometry_type': metadata[index].get('geometry_type', ''),
                        'array_rank_xy': output_row['array_rank_xy'],
                        'doa_error_mode': output_row['doa_error_mode'],
                        'label_valid_estimation': label_valid,
                        'label_est_azim_deg': float(label['est_azim_deg']),
                        'label_doa_error_deg': float(label['doa_error_deg']),
                        'output_valid_estimation': output_valid,
                        'output_est_azim_deg': output_row.get(
                            'est_azim_deg', np.nan),
                        'output_doa_error_deg': output_row['doa_error_deg'],
                        'reference_scale_invariant_signal_noise_ratio_i': (
                            float(reference_si_snri[index].item())),
                        'excess_doa_error_deg': (
                            output_row['doa_error_deg']
                            - float(label['doa_error_deg'])),
                        'doa_drift_deg': drift,
                        'label_mean_abs_tdoa_residual_us': label.get(
                            'mean_abs_tdoa_residual_us', np.nan),
                        'output_mean_abs_tdoa_residual_us': output_row.get(
                            'mean_abs_tdoa_residual_us', np.nan),
                        'label_srp_confidence': label.get(
                            'srp_peak_to_sidelobe_std', np.nan),
                        'output_srp_confidence': output_row.get(
                            'srp_peak_to_sidelobe_std', np.nan),
                    })
                    progress.update(1)

    expected = len(dataset) * len(args.offsets)
    if len(rows) != expected:
        raise RuntimeError('Expected %d output rows, got %d' %
                           (expected, len(rows)))
    summaries = summarize_rows(rows, args.bootstrap_seed)
    for item in summaries:
        item.update({
            'experiment': experiment,
            'checkpoint': str(checkpoint),
            'checkpoint_epoch': checkpoint_epoch,
            'best_validation_metric': params.base_metric,
            'best_validation_value': best_value,
            'subset': Path(args.testset).name,
        })
    output_dir = Path(args.output_root) / experiment / Path(args.testset).name
    _atomic_write_csv(rows, output_dir / 'per_sample.csv')
    _atomic_write_csv(summaries, output_dir / 'summary.csv')
    _atomic_write_json({
        'experiment': experiment,
        'checkpoint': str(checkpoint),
        'checkpoint_epoch': checkpoint_epoch,
        'best_validation_metric': params.base_metric,
        'best_validation_value': best_value,
        'testset': str(Path(args.testset).resolve()),
        'label_cache': str(Path(args.label_cache).resolve()),
        'num_samples': len(dataset),
        'offsets_deg': [float(value) for value in args.offsets],
        'estimator': args.estimator,
        'sample_rate': args.sample_rate,
        'speed_of_sound': args.speed_of_sound,
        'angle_grid_step': args.angle_grid_step,
        'gcc_interpolation': args.gcc_interpolation,
        'srp_frame_length': args.srp_frame_length,
        'srp_hop_length': args.srp_hop_length,
        'srp_frequency_min': args.srp_frequency_min,
        'srp_frequency_max': args.srp_frequency_max,
        'srp_activity_threshold_db': args.srp_activity_threshold_db,
        'srp_sidelobe_exclusion_deg': args.srp_sidelobe_exclusion_deg,
        'sign': args.sign,
        'use_amp': args.use_amp,
        'amp_dtype': amp_dtype_name,
        'signal_metric_scope': 'reference_channel_0',
        'network_source': (
            str(Path(args.network_source).resolve())
            if args.network_source else None),
        'bootstrap_seed': args.bootstrap_seed,
        'bootstrap_resamples': 10000,
    }, output_dir / 'manifest.json')
    return summaries


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiments', nargs='*')
    parser.add_argument('--project_root', default='.')
    parser.add_argument('--testset', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--label_cache', required=True)
    parser.add_argument('--build_label_cache', action='store_true')
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--max_n_mics', type=int, default=8)
    parser.add_argument('--offsets', nargs='+', type=float, default=[0.0])
    parser.add_argument('--pretrain_path', default='best')
    parser.add_argument(
        '--network_source', default='',
        help='Optional Python source defining Net/metrics. Intended for a '
             'frozen checkpoint whose experiment lives outside project_root.')
    parser.add_argument('--sample_rate', type=int, default=8000)
    parser.add_argument('--speed_of_sound', type=float, default=343.0)
    parser.add_argument('--estimator', choices=['srp_phat', 'gcc_peak_grid'],
                        default='srp_phat')
    parser.add_argument('--angle_grid_step', type=float, default=1.0)
    parser.add_argument('--gcc_interpolation', type=int, default=16)
    parser.add_argument('--srp_frame_length', type=int, default=1024)
    parser.add_argument('--srp_hop_length', type=int, default=256)
    parser.add_argument('--srp_frequency_min', type=float, default=200.0)
    parser.add_argument('--srp_frequency_max', type=float, default=3500.0)
    parser.add_argument('--srp_activity_threshold_db', type=float,
                        default=-40.0)
    parser.add_argument('--srp_sidelobe_exclusion_deg', type=float,
                        default=10.0)
    parser.add_argument('--sign', type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--doa_workers', type=int, default=1,
                        help='CPU threads for per-sample DOA estimation')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument(
        '--amp_dtype', choices=['auto', 'float16', 'bfloat16'],
        default='auto',
        help='AMP dtype. "auto" follows the experiment config and falls '
             'back to float16 when the config does not specify one.')
    parser.add_argument('--bootstrap_seed', type=int, default=230)
    return parser.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args.project_root = str(Path(args.project_root).resolve())
    args.testset = str(Path(args.testset).resolve())
    if args.build_label_cache:
        build_label_cache(args)
    if not args.experiments:
        if args.build_label_cache:
            return
        raise ValueError('At least one experiment is required')
    if args.network_source and len(args.experiments) != 1:
        raise ValueError('--network_source requires exactly one experiment')
    if not Path(args.label_cache).is_file():
        raise FileNotFoundError(
            '%s; run once with --build_label_cache' % args.label_cache)
    args.device = torch.device(args.device)
    if args.device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_device(args.device)
    for experiment in args.experiments:
        summaries = evaluate_experiment(experiment, args)
        for row in summaries:
            logging.info(
                '%s %s offset=%+g: SI-SNRi=%.3f label=%.3f output=%.3f '
                'drift=%.3f deg Pres@5=%.3f',
                experiment, row['subset'], row['doa_offset_deg'],
                row['reference_scale_invariant_signal_noise_ratio_i'],
                row['label_mean_doa_error_deg'],
                row['output_mean_doa_error_deg'],
                row['mean_doa_drift_deg'], row['pres_at_5deg'])


if __name__ == '__main__':
    main()
