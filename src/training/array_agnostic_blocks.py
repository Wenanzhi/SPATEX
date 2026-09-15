import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_as(x, mic_mask):
    shape = [mic_mask.shape[0], mic_mask.shape[1]] + [1] * (x.ndim - 2)
    return mic_mask.to(dtype=x.dtype, device=x.device).view(*shape)


def masked_mean(x, mic_mask, dim=1, keepdim=True, eps=1e-8):
    mask = mask_as(x, mic_mask)
    denom = mask.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
    return (x * mask).sum(dim=dim, keepdim=keepdim) / denom


def normalize_mic_xyz(mic_xyz, mic_mask, eps=1e-6, preserve_scale=False):
    if mic_xyz is None:
        return None
    mask = mic_mask.to(dtype=mic_xyz.dtype, device=mic_xyz.device).unsqueeze(-1)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    centroid = (mic_xyz * mask).sum(dim=1, keepdim=True) / denom
    rel = (mic_xyz - centroid) * mask
    radius = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
    scale = radius.amax(dim=1, keepdim=True).clamp_min(eps)
    normalized = rel / scale
    if not preserve_scale:
        return normalized
    scale_feature = scale.expand_as(radius)
    return torch.cat([rel, normalized, scale_feature], dim=-1) * mask


class GeometryEmbedding(nn.Module):
    def __init__(self, emb_dim, hidden_dim=None, preserve_scale=False):
        super().__init__()
        hidden_dim = hidden_dim or emb_dim
        self.preserve_scale = preserve_scale
        input_dim = 7 if preserve_scale else 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, mic_xyz, mic_mask):
        xyz = normalize_mic_xyz(mic_xyz, mic_mask, preserve_scale=self.preserve_scale)
        if xyz is None:
            return None
        return self.net(xyz) * mic_mask.to(xyz.dtype).unsqueeze(-1)


class DOAMicrophonePositionEncoding(nn.Module):
    """Jointly encode target direction and microphone geometry.

    ``azim_vec`` can contain any learned/cyclic direction representation.  If
    ``direction_dim`` is non-zero, its final entries must additionally contain
    an explicit Cartesian unit direction.  The explicit direction is used to
    compute a physically meaningful projected propagation delay for each mic.
    """

    def __init__(self, angle_dim, emb_dim, hidden_dim=None, direction_dim=3,
                 preserve_scale=True, sample_rate=8000, speed_of_sound=343.0,
                 delay_bands=4):
        super().__init__()
        if direction_dim not in (0, 2, 3):
            raise ValueError("direction_dim must be 0, 2, or 3")
        hidden_dim = hidden_dim or max(emb_dim, 32)
        self.angle_dim = angle_dim
        self.direction_dim = direction_dim
        self.preserve_scale = preserve_scale
        self.sample_rate = float(sample_rate)
        self.speed_of_sound = float(speed_of_sound)
        self.delay_bands = int(delay_bands)
        position_dim = 7 if preserve_scale else 3
        delay_dim = 2 + 2 * self.delay_bands if direction_dim > 0 else 0
        self.net = nn.Sequential(
            nn.Linear(position_dim + angle_dim + delay_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, mic_xyz, mic_mask, azim_vec):
        position = normalize_mic_xyz(
            mic_xyz, mic_mask, preserve_scale=self.preserve_scale)
        if position is None:
            return None
        b, m, _ = position.shape
        if azim_vec is None:
            azim_vec = position.new_zeros(b, self.angle_dim)
        if azim_vec.shape[-1] != self.angle_dim:
            raise ValueError(
                f"Expected azim_vec dimension {self.angle_dim}, got {azim_vec.shape[-1]}")
        azim_vec = azim_vec.to(device=position.device, dtype=position.dtype)
        angle = azim_vec[:, None, :].expand(-1, m, -1)
        parts = [position, angle]

        if self.direction_dim > 0:
            direction = azim_vec[:, -self.direction_dim:]
            if self.direction_dim == 2:
                direction = torch.cat([direction, direction.new_zeros(b, 1)], dim=-1)
            direction = F.normalize(direction, dim=-1, eps=1e-6)
            rel = (normalize_mic_xyz(mic_xyz, mic_mask, preserve_scale=True)[..., :3]
                   .to(dtype=position.dtype))
            projection_m = torch.einsum('bmd,bd->bm', rel, direction)
            delay_samples = projection_m * self.sample_rate / self.speed_of_sound
            bands = torch.pow(
                delay_samples.new_tensor(2.0),
                torch.arange(self.delay_bands, device=delay_samples.device,
                             dtype=delay_samples.dtype))
            phase = math.pi * delay_samples[..., None] * bands
            delay_features = torch.cat([
                projection_m[..., None],
                delay_samples[..., None],
                torch.sin(phase),
                torch.cos(phase),
            ], dim=-1)
            parts.append(delay_features)

        encoded = self.net(torch.cat(parts, dim=-1))
        return encoded * mic_mask.to(encoded.dtype).unsqueeze(-1)


class GeometryFiLMLayer(nn.Module):
    """Per-microphone FiLM modulation initialized as an identity mapping."""

    def __init__(self, channels, condition_dim, hidden_dim=None,
                 init_zero=True, max_scale=1.0):
        super().__init__()
        hidden_dim = hidden_dim or max(channels, condition_dim)
        self.max_scale = float(max_scale)
        self.net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, channels * 2),
        )
        if init_zero:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, condition, mic_mask):
        if condition is None:
            return z * mask_as(z, mic_mask)
        scale, shift = self.net(condition).chunk(2, dim=-1)
        scale = self.max_scale * torch.tanh(scale)
        scale = scale[:, :, :, None, None]
        shift = shift[:, :, :, None, None]
        return (z * (1.0 + scale) + shift) * mask_as(z, mic_mask)


class MaskedTACBlock(nn.Module):
    def __init__(self, channels, hidden_channels=None, geom_dim=0, dropout=0.0):
        super().__init__()
        hidden_channels = hidden_channels or channels
        self.proj = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(1, hidden_channels),
            nn.PReLU(),
        )
        self.update = nn.Sequential(
            nn.Conv2d(hidden_channels * 2 + geom_dim, channels, kernel_size=1),
            nn.GroupNorm(1, channels),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.geom_dim = geom_dim

    def forward(self, z, mic_mask, geom_emb=None):
        b, m, c, t, f = z.shape
        z = z * mask_as(z, mic_mask)
        f_local = self.proj(z.reshape(b * m, c, t, f)).reshape(b, m, -1, t, f)
        f_bar = masked_mean(f_local, mic_mask, dim=1, keepdim=True).expand(-1, m, -1, -1, -1)
        parts = [f_local, f_bar]
        if self.geom_dim > 0:
            if geom_emb is None:
                geom = z.new_zeros(b, m, self.geom_dim)
            else:
                geom = geom_emb.to(dtype=z.dtype, device=z.device)
            parts.append(geom[:, :, :, None, None].expand(-1, -1, -1, t, f))
        update_in = torch.cat(parts, dim=2).reshape(b * m, -1, t, f)
        delta = self.update(update_in).reshape(b, m, c, t, f)
        return (z + delta) * mask_as(z, mic_mask)


class MaskedTAttCBlock(nn.Module):
    """Transform-Attend-Concatenate communication over microphones.

    Attention is evaluated independently at every time-frequency bin.  Invalid
    microphones are excluded as keys and are zeroed as queries/outputs.  No
    microphone-index positional embedding is used, so the block is channel
    permutation equivariant.
    """

    def __init__(self, channels, hidden_channels=None, geom_dim=0, n_heads=4,
                 dropout=0.0):
        super().__init__()
        hidden_channels = hidden_channels or channels
        if hidden_channels % n_heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by n_heads ({n_heads})")
        self.hidden_channels = hidden_channels
        self.n_heads = n_heads
        self.head_dim = hidden_channels // n_heads
        self.scale = self.head_dim ** -0.5
        self.geom_dim = geom_dim
        self.dropout = float(dropout)
        self.local_proj = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(1, hidden_channels),
            nn.PReLU(),
        )
        self.to_qkv = nn.Conv2d(hidden_channels, hidden_channels * 3, kernel_size=1,
                                bias=False)
        self.attn_out = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1,
                                  bias=False)
        self.update = nn.Sequential(
            nn.Conv2d(hidden_channels * 2 + geom_dim, channels, kernel_size=1),
            nn.GroupNorm(1, channels),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def _channel_attention(self, local, mic_mask):
        b, m, hdim, t, f = local.shape
        qkv = self.to_qkv(local.reshape(b * m, hdim, t, f))
        qkv = qkv.reshape(b, m, 3, self.n_heads, self.head_dim, t, f)
        q, k, v = qkv.unbind(dim=2)
        # [B, T, F, H, M, D]
        q = q.permute(0, 4, 5, 2, 1, 3)
        k = k.permute(0, 4, 5, 2, 1, 3)
        v = v.permute(0, 4, 5, 2, 1, 3)
        logits = torch.einsum('btfhmd,btfhnd->btfhmn', q.float(), k.float()) * self.scale
        key_mask = ~mic_mask.to(device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(key_mask[:, None, None, None, None, :],
                                    torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training).to(v.dtype)
        attended = torch.einsum('btfhmn,btfhnd->btfhmd', weights, v)
        attended = attended.permute(0, 4, 3, 5, 1, 2).reshape(b, m, hdim, t, f)
        attended = self.attn_out(attended.reshape(b * m, hdim, t, f))
        return attended.reshape(b, m, hdim, t, f)

    def forward(self, z, mic_mask, geom_emb=None):
        # Keep the full TAttC path in FP32 even when the surrounding network
        # uses AMP. Casting q/k to float after ``to_qkv`` is too late: the
        # projection itself can already overflow in FP16 as its weights grow,
        # turning the subsequent softmax into NaN. TAC, Co-Attention and
        # geometry-FiLM do not use this path and are intentionally untouched.
        input_dtype = z.dtype
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            b, m, c, t, f = z.shape
            z = z * mask_as(z, mic_mask)
            local = self.local_proj(z.reshape(b * m, c, t, f)).reshape(
                b, m, self.hidden_channels, t, f)
            attended = self._channel_attention(local, mic_mask)
            parts = [local, attended]
            if self.geom_dim > 0:
                if geom_emb is None:
                    geom = z.new_zeros(b, m, self.geom_dim)
                else:
                    geom = geom_emb.to(dtype=z.dtype, device=z.device)
                parts.append(geom[:, :, :, None, None].expand(-1, -1, -1, t, f))
            update_in = torch.cat(parts, dim=2).reshape(b * m, -1, t, f)
            delta = self.update(update_in).reshape(b, m, c, t, f)
            output = (z + delta) * mask_as(z, mic_mask)
        return output.to(dtype=input_dtype)
