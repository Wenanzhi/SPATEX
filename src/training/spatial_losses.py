"""Stable array-agnostic spatial losses for variable-channel MIMO TSE.

Both losses use every valid unordered microphone pair and ignore padded
channels through ``mic_mask``. Spectral calculations run in float32 even under
AMP because phase and normalized covariance are unstable in float16 near
silent time-frequency bins.
"""

from contextlib import nullcontext

import torch
import torch.nn.functional as F


def _autocast_disabled(device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


def _validate_inputs(pred, target, mic_mask, n_fft, hop_length, eps):
    if pred.ndim != 3 or target.ndim != 3:
        raise ValueError(
            "pred and target must have shape [batch, microphones, time]")
    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shapes")
    if mic_mask is None:
        mic_mask = pred.new_ones(pred.shape[:2])
    if mic_mask.shape != pred.shape[:2]:
        raise ValueError("mic_mask must have shape [batch, microphones]")
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")
    return mic_mask.to(device=pred.device)


def multichannel_stft(signal, n_fft, hop_length):
    """Return a Hann-window STFT with shape [B, M, F, frames]."""
    batch, microphones, samples = signal.shape
    window = torch.hann_window(
        n_fft, device=signal.device, dtype=torch.float32)
    spectrum = torch.stft(
        signal.float().reshape(batch * microphones, samples),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    return spectrum.reshape(
        batch, microphones, spectrum.shape[-2], spectrum.shape[-1])


def _zero_loss(value):
    return value.real.float().sum() * 0.0


def _pair_indices(mic_mask, batch_index):
    valid = torch.nonzero(
        mic_mask[batch_index] > 0.5, as_tuple=False).flatten()
    if valid.numel() < 2:
        return None, None
    local_i, local_j = torch.triu_indices(
        valid.numel(), valid.numel(), offset=1, device=valid.device)
    return valid[local_i], valid[local_j]


def _target_energy_mask(pair_power, energy_floor_db, eps):
    """Keep bins within ``energy_floor_db`` of each target pair's peak."""
    peak = pair_power.amax(dim=(-2, -1), keepdim=True)
    relative_floor = 10.0 ** (float(energy_floor_db) / 10.0)
    return (peak > eps) & (pair_power >= peak * relative_floor)


def _masked_pair_mean(values, tf_mask):
    mask = tf_mask.to(values.dtype)
    counts = mask.sum(dim=(-2, -1))
    pair_values = (
        (values * mask).sum(dim=(-2, -1))
        / counts.clamp_min(1.0)
    )
    valid_pairs = counts > 0
    if valid_pairs.any():
        return pair_values[valid_pairs].mean()
    return values.sum() * 0.0


def _smooth_time(value, kernel_size):
    if kernel_size == 1:
        return value
    original_shape = value.shape
    flat = value.reshape(-1, 1, original_shape[-1])
    smoothed = F.avg_pool1d(
        flat, kernel_size, stride=1, padding=kernel_size // 2)
    return smoothed.reshape(original_shape)


def _smooth_complex_time(value, kernel_size):
    return torch.complex(
        _smooth_time(value.real, kernel_size),
        _smooth_time(value.imag, kernel_size),
    )


def _ipd_from_stft(pred_stft, target_stft, mic_mask,
                   energy_floor_db, eps):
    """Circular IPD distance averaged per sample, then across the batch."""
    sample_losses = []
    for batch_index in range(pred_stft.shape[0]):
        pair_i, pair_j = _pair_indices(mic_mask, batch_index)
        if pair_i is None:
            sample_losses.append(_zero_loss(pred_stft[batch_index]))
            continue

        pred_cross = (
            pred_stft[batch_index, pair_i]
            * pred_stft[batch_index, pair_j].conj()
        )
        target_i = target_stft[batch_index, pair_i]
        target_j = target_stft[batch_index, pair_j]
        target_cross = target_i * target_j.conj()

        pred_unit = pred_cross / pred_cross.abs().clamp_min(eps)
        target_unit = target_cross / target_cross.abs().clamp_min(eps)
        cosine_delta = (
            pred_unit * target_unit.conj()).real.clamp(-1.0, 1.0)
        circular_distance = 1.0 - cosine_delta

        target_pair_power = (
            target_i.abs().square()
            * target_j.abs().square()
        ).clamp_min(0.0).sqrt()
        tf_mask = _target_energy_mask(
            target_pair_power, energy_floor_db, eps)
        sample_losses.append(
            _masked_pair_mean(circular_distance, tf_mask))

    if not sample_losses:
        return _zero_loss(pred_stft)
    return torch.stack(sample_losses).mean()


def _coherence_from_stft(pred_stft, target_stft, mic_mask,
                         smoothing_frames, energy_floor_db, eps):
    """Complex normalized covariance/coherence distance."""
    if smoothing_frames <= 0 or smoothing_frames % 2 == 0:
        raise ValueError(
            "smoothing_frames must be a positive odd integer")

    sample_losses = []
    for batch_index in range(pred_stft.shape[0]):
        pair_i, pair_j = _pair_indices(mic_mask, batch_index)
        if pair_i is None:
            sample_losses.append(_zero_loss(pred_stft[batch_index]))
            continue

        pred_i = pred_stft[batch_index, pair_i]
        pred_j = pred_stft[batch_index, pair_j]
        target_i = target_stft[batch_index, pair_i]
        target_j = target_stft[batch_index, pair_j]

        pred_cross = _smooth_complex_time(
            pred_i * pred_j.conj(), smoothing_frames)
        target_cross = _smooth_complex_time(
            target_i * target_j.conj(), smoothing_frames)
        pred_power_i = _smooth_time(
            pred_i.abs().square(), smoothing_frames)
        pred_power_j = _smooth_time(
            pred_j.abs().square(), smoothing_frames)
        target_power_i = _smooth_time(
            target_i.abs().square(), smoothing_frames)
        target_power_j = _smooth_time(
            target_j.abs().square(), smoothing_frames)

        pred_power_product = (
            pred_power_i.clamp_min(0.0)
            * pred_power_j.clamp_min(0.0)
        )
        target_power_product = (
            target_power_i.clamp_min(0.0)
            * target_power_j.clamp_min(0.0)
        )
        # Clamp before sqrt to avoid the infinite derivative at zero.
        pred_norm = pred_power_product.clamp_min(eps * eps).sqrt()
        target_norm = target_power_product.clamp_min(eps * eps).sqrt()

        pred_coherence = pred_cross / pred_norm
        target_coherence = target_cross / target_norm
        complex_distance = (
            (pred_coherence.real - target_coherence.real).abs()
            + (pred_coherence.imag - target_coherence.imag).abs()
        )
        target_pair_power = target_power_product.sqrt()
        tf_mask = _target_energy_mask(
            target_pair_power, energy_floor_db, eps)
        sample_losses.append(
            _masked_pair_mean(complex_distance, tf_mask))

    if not sample_losses:
        return _zero_loss(pred_stft)
    return torch.stack(sample_losses).mean()


def spatial_fidelity_losses(pred, target, mic_mask, n_fft, hop_length,
                            compute_ipd=True, compute_coherence=True,
                            smoothing_frames=5, energy_floor_db=-40.0,
                            eps=1.0e-6):
    """Compute selected spatial losses using one shared STFT pair."""
    mic_mask = _validate_inputs(
        pred, target, mic_mask, n_fft, hop_length, eps)
    if not compute_ipd and not compute_coherence:
        return {}

    with _autocast_disabled(pred.device):
        pred_stft = multichannel_stft(
            pred, n_fft, hop_length)
        target_stft = multichannel_stft(
            target, n_fft, hop_length)
        losses = {}
        if compute_ipd:
            losses["ipd_loss"] = _ipd_from_stft(
                pred_stft, target_stft, mic_mask,
                energy_floor_db, eps)
        if compute_coherence:
            losses["coherence_loss"] = _coherence_from_stft(
                pred_stft, target_stft, mic_mask,
                smoothing_frames, energy_floor_db, eps)
        return losses


def masked_ipd_loss(pred, target, mic_mask, n_fft, hop_length,
                    energy_floor_db=-40.0, eps=1.0e-6):
    return spatial_fidelity_losses(
        pred, target, mic_mask, n_fft, hop_length,
        compute_ipd=True,
        compute_coherence=False,
        energy_floor_db=energy_floor_db,
        eps=eps,
    )["ipd_loss"]


def masked_complex_coherence_loss(
        pred, target, mic_mask, n_fft, hop_length,
        smoothing_frames=5, energy_floor_db=-40.0, eps=1.0e-6):
    return spatial_fidelity_losses(
        pred, target, mic_mask, n_fft, hop_length,
        compute_ipd=False,
        compute_coherence=True,
        smoothing_frames=smoothing_frames,
        energy_floor_db=energy_floor_db,
        eps=eps,
    )["coherence_loss"]
