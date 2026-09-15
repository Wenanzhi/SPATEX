"""Public inference interface using the original checkpoint parameter names."""
import json
import math
from pathlib import Path

import torch

from src.training.network_TACDeFTAN2 import Net


def encode_azimuth(azimuth_deg, dimension=40, alpha=80.0):
    """Match the dataset's cyclic encoding, including rounding to integer degrees.

    Azimuth is measured from +y toward +x; coordinates are in metres.
    Returns [dimension] for a scalar or [..., dimension] for a tensor.
    """
    if dimension <= 0 or dimension % 2:
        raise ValueError('The cyclic encoding dimension must be positive and even.')
    angles = torch.as_tensor(azimuth_deg, dtype=torch.float32)
    if not torch.isfinite(angles).all():
        raise ValueError('Azimuth must be finite.')
    # Construct the original table before indexing, to preserve floating-point
    # arithmetic and the original integer-degree convention.
    phi = torch.arange(360, device=angles.device).unsqueeze(1) * (math.pi / 180)
    div = torch.exp(torch.arange(0, dimension, 2, device=angles.device) *
                    -(torch.log(torch.tensor([10000.0], device=angles.device)) / dimension))
    table = torch.zeros(360, dimension, device=angles.device)
    table[:, 0::2] = torch.sin(torch.sin(phi) * alpha * div)
    table[:, 1::2] = torch.sin(torch.cos(phi) * alpha * div)
    table = table / torch.linalg.vector_norm(table, dim=-1, keepdim=True)
    return table[angles.round().long().remainder(360)]


def load_model(config_path, checkpoint_path, device='cpu'):
    """Load a PACT/Independent state dict strictly, without changing architecture.

    Accepts a weights-only state dict or the original training checkpoint.
    Checkpoint deserialization uses PyTorch's restricted weights-only loader.
    """
    config = json.loads(Path(config_path).read_text())
    if config.get('model') != 'src.training.network_TACDeFTAN2':
        raise ValueError('This interface supports the PACT backbone configs only.')
    if config['model_params'].get('cue_mode') != 'doa_only':
        raise ValueError('Enrollment-free inference requires cue_mode="doa_only".')
    model = Net(**config['model_params'])
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state = payload.get('model_state_dict', payload)
    if state and all(key.startswith('module.') for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.inference_azimuth_alpha = config.get('test_data', {}).get('alpha', 80.0)
    return model.to(device).eval()


@torch.inference_mode()
def extract(model, mixture, mic_xyz, azimuth_deg, mic_mask=None):
    """Extract [B,M,T] target waveforms at 8 kHz; no enrollment is required.

    Coordinates have shape [B,M,3], and mic_mask (optional) has shape [B,M].
    Channel order must agree across all three inputs. Invalid outputs are zero.
    """
    if mixture.ndim != 3 or mixture.shape[-1] < model.win:
        raise ValueError('mixture must be [B,M,T] with T >= the STFT window.')
    batch, microphones, _ = mixture.shape
    if microphones < 2 or microphones > model.max_n_mics:
        raise ValueError('Microphone slots must be between 2 and the configured maximum.')
    if mic_xyz.shape != (batch, microphones, 3):
        raise ValueError('mic_xyz must have shape [B,M,3].')
    device = next(model.parameters()).device
    mixture = mixture.to(device=device, dtype=torch.float32)
    mic_xyz = mic_xyz.to(device=device, dtype=torch.float32)
    if mic_mask is None:
        mic_mask = mixture.new_ones(batch, microphones)
    mic_mask = mic_mask.to(device=device, dtype=torch.float32)
    if mic_mask.shape != (batch, microphones) or not ((mic_mask == 0) | (mic_mask == 1)).all():
        raise ValueError('mic_mask must be binary with shape [B,M].')
    if (mic_mask.sum(dim=1) < 2).any():
        raise ValueError('At least two valid microphones per scene are required.')
    if not torch.isfinite(mixture).all() or not torch.isfinite(mic_xyz).all():
        raise ValueError('Waveforms and coordinates must be finite, including padded slots.')
    azimuth = torch.as_tensor(azimuth_deg, device=device, dtype=torch.float32)
    if azimuth.ndim == 0:
        azimuth = azimuth.expand(batch)
    if azimuth.shape != (batch,):
        raise ValueError('Provide one azimuth per scene, or one scalar for the batch.')
    dimension = model.azim_embedding[0][0].in_features
    clue = encode_azimuth(azimuth, dimension, getattr(model, 'inference_azimuth_alpha', 80.0))
    model.eval()
    return model(mixture, azim_vec=clue, mic_mask=mic_mask, mic_xyz=mic_xyz)
