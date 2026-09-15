"""Shared utilities for fixed-array, single-output external TSE baselines."""

import math

import torch

from src.helpers.utils import (
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
)


def doa_degrees_from_vector(azim_vec):
    """Recover azimuth degrees from the appended [sin, cos, 0] unit vector."""
    if azim_vec.shape[-1] < 3:
        raise ValueError("azim_vec must include the appended DOA unit vector")
    sin_azim = azim_vec[..., -3]
    cos_azim = azim_vec[..., -2]
    return torch.remainder(torch.atan2(sin_azim, cos_azim) * 180.0 / math.pi, 360.0)


def negative_snr(prediction, target, eps=1.0e-8):
    """Original COSPA-style negative SNR objective, averaged over the batch."""
    error = (prediction - target).pow(2).sum(dim=-1)
    signal = target.pow(2).sum(dim=-1)
    return (10.0 * torch.log10((error + eps) / (signal + eps))).mean()


def negative_si_snr(prediction, target, eps=1.0e-8):
    """Numerically stable negative SI-SNR for a [B, T] waveform batch."""
    prediction = prediction - prediction.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    scale = (prediction * target).sum(dim=-1, keepdim=True)
    scale = scale / (target.pow(2).sum(dim=-1, keepdim=True) + eps)
    projected = scale * target
    noise = prediction - projected
    ratio = projected.pow(2).sum(dim=-1) / (noise.pow(2).sum(dim=-1) + eps)
    return (-10.0 * torch.log10(ratio + eps)).mean()


def reference_channel_metrics(mixed, output, gt, **_kwargs):
    """Signal metrics for MC-to-SC systems, evaluated at reference channel 0."""
    prediction = output[:, 0]
    target = gt[:, 0]
    mixture = mixed[:, 0]
    result = {}
    for metric in (snr, si_snr):
        values = []
        improvements = []
        for source_i, prediction_i, target_i in zip(mixture, prediction, target):
            prediction_i = prediction_i.unsqueeze(0)
            target_i = target_i.unsqueeze(0)
            source_i = source_i.unsqueeze(0)
            estimate_value = metric(prediction_i, target_i)
            values.append(float(estimate_value.detach().cpu().item()))
            improvements.append(float((estimate_value - metric(source_i, target_i)).detach().cpu().item()))
        result[metric.__name__] = values
        result[metric.__name__ + '_i'] = improvements
    return result


def reference_channel_paper_metrics(mixed, output, gt, sample_rate=8000):
    """Per-sample reference-channel metrics used by the paper tables.

    Signal metrics use explicit ``reference_`` prefixes so they cannot be
    confused with the existing joint-MIMO metrics. PESQ failures are surfaced
    instead of silently replacing invalid samples with a fabricated score.
    """
    from pesq import pesq  # pylint: disable=import-outside-toplevel
    from pystoi import stoi  # pylint: disable=import-outside-toplevel

    prediction = output[:, 0].detach().float().cpu()
    target = gt[:, 0].detach().float().cpu()
    mixture = mixed[:, 0].detach().float().cpu()
    result = {
        'reference_signal_noise_ratio': [],
        'reference_signal_noise_ratio_i': [],
        'reference_scale_invariant_signal_noise_ratio': [],
        'reference_scale_invariant_signal_noise_ratio_i': [],
        'pesq_nb': [],
        'stoi': [],
    }
    for mixture_i, prediction_i, target_i in zip(
            mixture, prediction, target):
        prediction_batch = prediction_i.unsqueeze(0)
        target_batch = target_i.unsqueeze(0)
        mixture_batch = mixture_i.unsqueeze(0)
        for metric in (snr, si_snr):
            estimate_value = metric(prediction_batch, target_batch)
            mixture_value = metric(mixture_batch, target_batch)
            prefix = 'reference_' + metric.__name__
            result[prefix].append(float(estimate_value.item()))
            result[prefix + '_i'].append(
                float((estimate_value - mixture_value).item()))

        prediction_np = prediction_i.numpy()
        target_np = target_i.numpy()
        result['pesq_nb'].append(
            float(pesq(sample_rate, target_np, prediction_np, 'nb')))
        result['stoi'].append(
            float(stoi(target_np, prediction_np, sample_rate, extended=False)))
    return result
