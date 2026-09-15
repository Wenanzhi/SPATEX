import math
import os
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from packaging.version import parse as V
from torch.nn import init
from torch.nn.parameter import Parameter

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from src.helpers.utils import (signal_noise_ratio as snr, 
                               scale_invariant_signal_noise_ratio as si_snr,
                               phase_constrained_magnitude as pcm,
                               delta_ILD as ild, delta_IPD as ipd, delta_ITD_cc as itd_cc, delta_ITD_gccphat as itd_gccphat)

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
except ImportError:
    flash_attn_qkvpacked_func = None
    flash_attn_func = None


# ============================================================================
# Speaker Encoder
# ============================================================================

class SimpleSpeakerEncoder(nn.Module):
    """Lightweight CNN speaker encoder trainable from scratch."""

    def __init__(self, spk_emb_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, spk_emb_dim)

    def forward(self, x):
        """
        Args:
            x: [B, T] mono enrollment waveform
        Returns:
            [B, spk_emb_dim] speaker embedding (L2-normalized)
        """
        h = self.encoder(x.unsqueeze(1)).squeeze(-1)
        emb = self.fc(h)
        return F.normalize(emb, p=2, dim=-1)


class EcapaSpeakerEncoder(nn.Module):
    """Wrapper to load a pretrained ECAPA-TDNN checkpoint.

    If *checkpoint_path* is provided, the state dict is loaded directly.
    Otherwise, the module falls back to ``SimpleSpeakerEncoder`` so that
    training can start without pretrained weights.

    To use a SpeechBrain ECAPA checkpoint::

        1. Download the model files (e.g. ``classifier.ckpt``, ``embedding_model.ckpt``)
        2. Set ``spk_encoder_ckpt`` in config.json to the path of the ``.ckpt`` file
    """

    def __init__(self, spk_emb_dim=192, checkpoint_path=None):
        super().__init__()
        self.spk_emb_dim = spk_emb_dim
        if checkpoint_path and os.path.isfile(checkpoint_path):
            state = torch.load(checkpoint_path, map_location='cpu')
            from collections import OrderedDict
            self.model = nn.Sequential(OrderedDict([
                ('body', nn.Identity())
            ]))
            self.model.load_state_dict(state, strict=False)
            logging.info("Loaded ECAPA checkpoint from %s", checkpoint_path)
        else:
            logging.warning(
                "ECAPA checkpoint not found (%s); using SimpleSpeakerEncoder "
                "as fallback.", checkpoint_path)
            self.model = SimpleSpeakerEncoder(spk_emb_dim)

    def forward(self, x):
        return self.model(x)


def build_speaker_encoder(spk_encoder_type='simple', spk_emb_dim=256,
                          spk_encoder_ckpt=None):
    if spk_encoder_type == 'ecapa':
        return EcapaSpeakerEncoder(spk_emb_dim, spk_encoder_ckpt)
    return SimpleSpeakerEncoder(spk_emb_dim)


# ============================================================================
# Activity Head
# ============================================================================

class ActivityHead(nn.Module):
    """Predict frame-level target-speaker activity from encoder features
    and speaker embedding.

    Pipeline:
        1. Project spk embedding -> emb_dim, broadcast to [B, C, T, F]
        2. Multiply with encoder feature (speaker-conditioned feature)
        3. Pool over frequency -> [B, C, T]
        4. Conv1d stack -> [B, 1, T] -> sigmoid -> [B, T]
    """

    def __init__(self, emb_dim, spk_emb_dim):
        super().__init__()
        self.spk_proj = nn.Sequential(
            nn.Linear(spk_emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.PReLU(),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, kernel_size=5, padding=2),
            nn.GroupNorm(1, emb_dim, 1e-5),
            nn.PReLU(),
            nn.Conv1d(emb_dim, 1, kernel_size=1),
        )

    def forward(self, feat, spk_emb):
        """
        Args:
            feat:    [B, C, T, F]  encoder feature
            spk_emb: [B, spk_emb_dim]
        Returns:
            logits: [B, T] raw logits (apply sigmoid yourself for gating)
        """
        spk = self.spk_proj(spk_emb)                           # [B, C]
        conditioned = feat * rearrange(spk, 'b c -> b c 1 1')  # [B, C, T, F]
        pooled = conditioned.mean(dim=-1)                       # [B, C, T]
        logits = self.temporal_conv(pooled).squeeze(1)           # [B, T]
        return logits


# ============================================================================
# Main Network
# ============================================================================

class Net(nn.Module):
    def __init__(self, n_srcs=4, win=256, n_mics=4, n_layers=6,
                 att_dim=64, hidden_dim=256, n_head=4, emb_dim=64,
                 emb_ks=4, emb_hs=1, dropout=0.1, eps=1.0e-5,
                 angle_dim=40,
                 spk_emb_dim=256,
                 spk_encoder_type='simple',
                 spk_encoder_ckpt=None,
                 freeze_spk_encoder=False,
                 act_loss_weight=0.1,
                 cue_mode='both',
                 # legacy params (ignored)
                 clue_type=None, label_len=None, **kwargs):
        super(Net, self).__init__()

        self.n_srcs = n_srcs
        self.win = win
        self.hop = win // 2
        self.n_mics = n_mics
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.act_loss_weight = act_loss_weight
        self.cue_mode = cue_mode
        assert win % 2 == 0

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)

        # ---------- encoder ----------
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_mics, emb_dim * n_head, ks, padding=padding),
            nn.GroupNorm(1, emb_dim * n_head, eps=eps),
            InverseDenseBlock2d(emb_dim * n_head, emb_dim, n_head)
        )

        # ---------- DeFTAN blocks ----------
        self.mix_blocks = nn.ModuleList([])
        for idx in range(self.n_layers):
            self.mix_blocks.append(
                DeFTANblock(idx, emb_dim, emb_ks, emb_hs,
                            att_dim, hidden_dim, n_head, dropout, eps))

        # ---------- direction embedding (per-layer) ----------
        self.azim_embedding = nn.ModuleList([])
        for _ in range(self.n_layers):
            self.azim_embedding.append(nn.Sequential(
                nn.Linear(angle_dim, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.PReLU()
            ))

        # ---------- speaker encoder ----------
        self.spk_encoder = build_speaker_encoder(
            spk_encoder_type, spk_emb_dim, spk_encoder_ckpt)
        if freeze_spk_encoder:
            for p in self.spk_encoder.parameters():
                p.requires_grad = False

        # ---------- activity head ----------
        self.activity_head = ActivityHead(emb_dim, spk_emb_dim)

        # ---------- decoder ----------
        self.deconv = nn.Sequential(
            nn.Conv2d(emb_dim, 2 * n_srcs * n_head, ks, padding=padding),
            InverseDenseBlock2d(2 * n_srcs * n_head, 2 * n_srcs, n_head)
        )

    # -----------------------------------------------------------------

    def pad_signal(self, input):
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nchannel = input.size(1)
        nsample = input.size(2)

        rest = self.win - (self.hop + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, nchannel, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, nchannel, self.hop)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)
        return input, rest

    @torch.no_grad()
    def _compute_oracle_activity(self, gt, T_target):
        """Compute frame-level oracle activity from ground truth signal.

        Args:
            gt: [B, M, T_orig]  multichannel ground-truth target
            T_target: expected number of STFT frames (from encoder)
        Returns:
            activity: [B, T_target] binary activity in {0, 1}
        """
        gt_padded, _ = self.pad_signal(gt)
        B, M, N = gt_padded.size()
        ref = gt_padded[:, 0, :]  # reference channel [B, N]
        stft_gt = torch.stft(
            ref, n_fft=self.win, hop_length=self.hop,
            window=torch.hann_window(self.win).to(gt.device),
            return_complex=False)   # [B, F, T, 2]
        energy = (stft_gt[:, :, :, 0] ** 2
                  + stft_gt[:, :, :, 1] ** 2).sum(dim=1)  # [B, T]
        log_energy = 10 * torch.log10(energy.clamp(min=1e-10))
        max_log = log_energy.max(dim=1, keepdim=True)[0]
        activity = (log_energy > (max_log - 30)).float()

        # trim / pad to match encoder frame count
        if activity.shape[1] > T_target:
            activity = activity[:, :T_target]
        elif activity.shape[1] < T_target:
            activity = F.pad(activity, (0, T_target - activity.shape[1]))
        return activity

    # -----------------------------------------------------------------

    def forward(self, input, gt, enrollment, azim_vec):
        """
        Args:
            input:       [B, M, T]       multichannel mixture
            gt:          [B, M, T]       multichannel target (for loss)
            enrollment:  [B, T_enroll]   mono enrollment waveform
            azim_vec:    [B, angle_dim]  direction clue

        Returns:
            output:      [B, n_srcs, T]  multichannel estimate
            total_loss:  scalar
        """
        _input = input
        input, rest = self.pad_signal(input)
        B, M, N = input.size()
        mix_std_ = torch.std(input, dim=(1, 2), keepdim=True)  # [B, 1, 1]
        input = input / mix_std_

        # ---- STFT ----
        stft_input = torch.stft(
            input.view([-1, N]), n_fft=self.win, hop_length=self.hop,
            window=torch.hann_window(self.win).type(input.type()),
            return_complex=False)
        _, n_freqs, T, _ = stft_input.size()
        xi = stft_input.view([B, M, n_freqs, T, 2])
        xi = xi.permute(0, 1, 4, 3, 2).contiguous()   # [B, M, 2, T, n_freqs]
        batch = xi.view([B, M * 2, T, n_freqs])        # [B, 2M, T, n_freqs]

        batch = self.conv(batch)                        # [B, C, T, n_freqs]

        # ---- speaker embedding ----
        spk_emb = self.spk_encoder(enrollment)          # [B, spk_emb_dim]

        # ---- predict target-speaker activity ----
        activity_logits = self.activity_head(batch, spk_emb)    # [B, T]
        activity_hat = torch.sigmoid(activity_logits)            # [B, T]

        # ---- DeFTAN blocks with direction × activity gating ----
        for ii in range(self.n_layers):
            if self.cue_mode == 'doa_only':
                dir_emb = self.azim_embedding[ii](azim_vec)
                gate = rearrange(dir_emb, 'b c -> b c 1 1')
            elif self.cue_mode == 'spk_emb_only':
                gate = rearrange(activity_hat, 'b t -> b 1 t 1')
            else:  # 'both'
                dir_emb = self.azim_embedding[ii](azim_vec)
                gate = (rearrange(dir_emb, 'b c -> b c 1 1')
                        * rearrange(activity_hat, 'b t -> b 1 t 1'))
            batch = batch * gate                          # [B, C, T, n_freqs]
            batch = self.mix_blocks[ii](batch)            # [B, C, T, n_freqs]

        # ---- iSTFT decoder ----
        batch = self.deconv(batch)
        batch = batch.view([B, self.n_srcs, 2, T, n_freqs])
        batch = batch.view([B * self.n_srcs, 2, T, n_freqs])
        batch = batch.permute(0, 3, 2, 1).type(input.type())
        istft_input = torch.complex(batch[:, :, :, 0], batch[:, :, :, 1])
        istft_output = torch.istft(
            istft_input, n_fft=self.win, hop_length=self.hop,
            window=torch.hann_window(self.win).type(input.type()),
            return_complex=False)

        output = istft_output[:, self.hop:-(rest + self.hop)].unsqueeze(1)
        output = output.view([B, self.n_srcs, -1])
        output = output * mix_std_

        # ---- loss ----
        pcm_loss = loss(_input, output, gt, self.win, self.hop)

        oracle_activity = self._compute_oracle_activity(gt, T)
        act_loss = F.binary_cross_entropy_with_logits(
            activity_logits, oracle_activity)

        total_loss = pcm_loss + self.act_loss_weight * act_loss

        return output, total_loss


# ============================================================================
# Building blocks (unchanged from original)
# ============================================================================

class InverseDenseBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        assert in_channels // out_channels == groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.blocks = nn.ModuleList([])
        for idx in range(groups):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(out_channels * ((idx > 0) + 1), out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(1, out_channels, 1e-5),
                nn.PReLU(out_channels)
            ))

    def forward(self, x):
        B, C, L = x.size()
        g = self.groups
        x = x.view(B, g, C//g, L).transpose(1, 2).reshape(B, C, L)
        skip = x[:, ::g, :]
        for idx in range(g):
            output = self.blocks[idx](skip)
            skip = torch.cat([output, x[:, idx+1::g, :]], dim=1)
        return output


class InverseDenseBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        assert in_channels // out_channels == groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.blocks = nn.ModuleList([])
        for idx in range(groups):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(out_channels * ((idx > 0) + 1), out_channels, kernel_size=(3, 3), padding=(1, 1)),
                nn.GroupNorm(1, out_channels, 1e-5),
                nn.PReLU(out_channels)
            ))

    def forward(self, x):
        B, C, T, Q = x.size()
        g = self.groups
        x = x.view(B, g, C//g, T, Q).transpose(1, 2).reshape(B, C, T, Q)
        skip = x[:, ::g, :, :]
        for idx in range(g):
            output = self.blocks[idx](skip)
            skip = torch.cat([output, x[:, idx+1::g, :, :]], dim=1)
        return output


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


COATTENTION_AGGREGATIONS = {'sqrt_count', 'mean'}


def aggregate_coattention_logits(logits, mic_mask, aggregation):
    """Aggregate per-microphone logits without including padded channels."""
    if aggregation not in COATTENTION_AGGREGATIONS:
        raise ValueError(
            "coattention aggregation must be one of: %s" %
            ', '.join(sorted(COATTENTION_AGGREGATIONS)))
    mask = mic_mask.to(device=logits.device, dtype=logits.dtype)
    masked_logits = logits * mask[:, :, None, None, None, None]
    valid_count = mask.sum(dim=1).clamp_min(1.0)
    denominator = (
        valid_count.sqrt() if aggregation == 'sqrt_count' else valid_count
    )
    return masked_logits.sum(dim=1) / denominator[:, None, None, None, None]


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout,
                 coattention_aggregation='sqrt_count'):
        super().__init__()
        if coattention_aggregation not in COATTENTION_AGGREGATIONS:
            raise ValueError(
                "coattention_aggregation must be one of: %s" %
                ', '.join(sorted(COATTENTION_AGGREGATIONS)))
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.coattention_aggregation = coattention_aggregation

        self.cv_qk = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=3, padding=1, bias=False),
            nn.GLU(dim=1))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.p_drop = dropout

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, coattention_context=None):
        qk = self.cv_qk(x.transpose(1, 2)).transpose(1, 2)
        q = rearrange(self.to_q(qk), 'b n (h d) -> b n h d', h=self.heads)
        q = q.to(dtype=torch.float16)
        k = rearrange(self.to_k(qk), 'b n (h d) -> b n h d', h=self.heads)
        k = k.to(dtype=torch.float16)
        v = rearrange(self.to_v(x), 'b n (h d) -> b n h d', h=self.heads)
        v = v.to(dtype=torch.float16)

        if coattention_context is not None:
            batch_size = int(coattention_context['batch_size'])
            n_mics = int(coattention_context['n_mics'])
            n_groups = int(coattention_context['n_groups'])
            mic_mask = coattention_context['mic_mask']
            if q.shape[0] != batch_size * n_mics * n_groups:
                raise ValueError(
                    "Co-attention context does not match the flattened sequence batch: "
                    f"{q.shape[0]} != {batch_size}*{n_mics}*{n_groups}")
            n_tokens, head_dim = q.shape[1], q.shape[-1]
            q = q.float().reshape(batch_size, n_mics, n_groups,
                                  n_tokens, self.heads, head_dim)
            k = k.float().reshape(batch_size, n_mics, n_groups,
                                  n_tokens, self.heads, head_dim)
            v = v.float().reshape(batch_size, n_mics, n_groups,
                                  n_tokens, self.heads, head_dim)
            # Per-microphone temporal affinity, then masked aggregation over M.
            logits = torch.einsum('bmqihd,bmqjhd->bmqhij', q, k) * self.scale
            shared_logits = aggregate_coattention_logits(
                logits, mic_mask, self.coattention_aggregation)
            attn = torch.softmax(shared_logits, dim=-1)
            attn = F.dropout(attn, p=self.p_drop, training=self.training)
            out = torch.einsum('bqhij,bmqjhd->bmqihd', attn, v)
            mask = mic_mask.to(device=out.device, dtype=out.dtype)
            out = out * mask[:, :, None, None, None, None]
            out = out.reshape(batch_size * n_mics * n_groups,
                              n_tokens, self.heads, head_dim)
        elif flash_attn_func is not None and q.is_cuda:
            out = flash_attn_func(q, k, v, dropout_p=self.p_drop, softmax_scale=self.scale)
            out = out.to(dtype=torch.float32)
        else:
            q, k, v = q.float(), k.float(), v.float()
            attn = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-1, -2)) * self.scale
            attn = torch.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.p_drop, training=self.training)
            out = torch.matmul(attn, v.transpose(1, 2)).transpose(1, 2)

        out = rearrange(out, 'b n h d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, idx, dropout):
        super().__init__()
        self.PW1 = nn.Sequential(
            nn.Linear(dim, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.PW2 = nn.Sequential(
            nn.Linear(dim, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.DW_Conv = nn.Sequential(
            nn.Conv1d(hidden_dim//2, hidden_dim//2, kernel_size=5, dilation=2**idx, padding='same'),
            nn.GroupNorm(1, hidden_dim//2, 1e-5),
            nn.PReLU(hidden_dim//2)
        )
        self.PW3 = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        ffw_out = self.PW1(x)
        dw_out = self.DW_Conv(self.PW2(x).transpose(1, 2)).transpose(1, 2)
        out = self.PW3(torch.cat((ffw_out, dw_out), dim=2))
        return out


class DeFTANblock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(self, idx, emb_dim, emb_ks, emb_hs, att_dim, hidden_dim,
                 n_head, dropout, eps,
                 coattention_aggregation='sqrt_count'):
        super().__init__()
        in_channels = emb_dim * emb_ks
        self.intra_norm = LayerNormalization4D(emb_dim, eps)
        self.intra_inv = InverseDenseBlock1d(in_channels, emb_dim, emb_ks)
        self.intra_att = PreNorm(emb_dim, Attention(emb_dim, n_head, att_dim, dropout))
        self.intra_ffw = PreNorm(emb_dim, FeedForward(emb_dim, hidden_dim, idx, dropout))
        self.intra_linear = nn.ConvTranspose1d(emb_dim, emb_dim, emb_ks, stride=emb_hs)

        self.inter_norm = LayerNormalization4D(emb_dim, eps)
        self.inter_inv = InverseDenseBlock1d(in_channels, emb_dim, emb_ks)
        self.inter_att = PreNorm(
            emb_dim,
            Attention(
                emb_dim,
                n_head,
                att_dim,
                dropout,
                coattention_aggregation=coattention_aggregation,
            ),
        )
        self.inter_ffw = PreNorm(emb_dim, FeedForward(emb_dim, hidden_dim, idx, dropout))
        self.inter_linear = nn.ConvTranspose1d(emb_dim, emb_dim, emb_ks, stride=emb_hs)

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        B, C, old_T, old_Q = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))

        # F-transformer
        input_ = x
        intra_rnn = self.intra_norm(input_)
        intra_rnn = intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q)
        intra_rnn = F.unfold(intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))
        intra_rnn = self.intra_inv(intra_rnn)

        intra_rnn = intra_rnn.transpose(1, 2)
        intra_rnn = self.intra_att(intra_rnn) + intra_rnn
        intra_rnn = self.intra_ffw(intra_rnn) + intra_rnn
        intra_rnn = intra_rnn.transpose(1, 2)

        intra_rnn = self.intra_linear(intra_rnn)
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()
        intra_rnn = intra_rnn + input_

        # T-transformer
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
        inter_rnn = F.unfold(inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))
        inter_rnn = self.inter_inv(inter_rnn)

        inter_rnn = inter_rnn.transpose(1, 2)
        inter_rnn = self.inter_att(inter_rnn) + inter_rnn
        inter_rnn = self.inter_ffw(inter_rnn) + inter_rnn
        inter_rnn = inter_rnn.transpose(1, 2)

        inter_rnn = self.inter_linear(inter_rnn)
        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()
        inter_rnn = inter_rnn + input_

        return inter_rnn

    def forward_multichannel(self, x, mic_mask, coattention=True):
        """Run a DeFTAN block while preserving the microphone axis.

        Frequency attention remains independent per microphone.  Temporal
        attention can share its attention map across all valid microphones.
        """
        if x.ndim != 5:
            raise ValueError(f"Expected [B, M, C, T, F], got shape {tuple(x.shape)}")
        B, M, C, old_T, old_Q = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        mask = mic_mask.to(device=x.device, dtype=x.dtype)[:, :, None, None, None]
        x = F.pad(x * mask, (0, Q - old_Q, 0, T - old_T))

        # F-transformer: independent for each microphone.
        input_ = x
        intra_rnn = self.intra_norm(x.reshape(B * M, C, T, Q))
        intra_rnn = intra_rnn.transpose(1, 2).contiguous().view(B * M * T, C, Q)
        intra_rnn = F.unfold(intra_rnn[..., None], (self.emb_ks, 1),
                             stride=(self.emb_hs, 1))
        intra_rnn = self.intra_inv(intra_rnn)
        intra_rnn = intra_rnn.transpose(1, 2)
        intra_rnn = self.intra_att(intra_rnn) + intra_rnn
        intra_rnn = self.intra_ffw(intra_rnn) + intra_rnn
        intra_rnn = self.intra_linear(intra_rnn.transpose(1, 2))
        intra_rnn = intra_rnn.view(B, M, T, C, Q).permute(0, 1, 3, 2, 4).contiguous()
        intra_rnn = (intra_rnn + input_) * mask

        # T-transformer: share temporal affinity across valid microphones.
        input_ = intra_rnn
        inter_rnn = self.inter_norm(
            intra_rnn.reshape(B * M, C, T, Q))
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * M * Q, C, T)
        inter_rnn = F.unfold(inter_rnn[..., None], (self.emb_ks, 1),
                             stride=(self.emb_hs, 1))
        inter_rnn = self.inter_inv(inter_rnn)
        inter_rnn = inter_rnn.transpose(1, 2)
        context = None
        if coattention:
            context = {
                'batch_size': B,
                'n_mics': M,
                'n_groups': Q,
                'mic_mask': mic_mask,
            }
        inter_rnn = self.inter_att(inter_rnn, coattention_context=context) + inter_rnn
        inter_rnn = self.inter_ffw(inter_rnn) + inter_rnn
        inter_rnn = self.inter_linear(inter_rnn.transpose(1, 2))
        inter_rnn = inter_rnn.view(B, M, Q, C, T).permute(0, 1, 3, 4, 2).contiguous()
        inter_rnn = (inter_rnn + input_) * mask
        return inter_rnn[:, :, :, :old_T, :old_Q]


class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat



# ============================================================================
# Optimizer, loss, metrics
# ============================================================================

def optimizer(model, data_parallel=False, **kwargs):
    import torch.optim as optim
    return optim.Adam(model.parameters(), **kwargs)

def loss(mix, pred, tgt, win, hop):
    pcm_loss = (pcm(mix, pred, tgt, win, hop)).mean()
    return pcm_loss

def metrics(mixed, output, gt):
    """ Function to compute metrics """
    metrics = {}

    def _metric(metric, pred, tgt):
        _vals = []
        for t, p in zip(tgt, pred):
            _vals.append((metric(p, t)).cpu().item())
        return _vals

    def metric_i(metric, src, pred, tgt):
        _vals = []
        for s, t, p in zip(src, tgt, pred):
            _vals.append((metric(p, t) - metric(s, t)).cpu().item())
        return _vals

    def metric_ild(metric, pred, tgt, idx1, idx2):
        _vals = []
        for t, p in zip(tgt, pred):
            _vals.append((metric(p, t, idx1, idx2)).cpu().item())
        return _vals

    def metric_ipd(metric, pred, tgt, idx1, idx2, win=256, hop=128):
        _vals = []
        for t, p in zip(tgt, pred):
            _vals.append((metric(p, t, win, hop, idx1, idx2)).cpu().item())
        return _vals

    def metric_itd(metric, pred, tgt, idx1, idx2, fs=8000):
        _vals = []
        for t, p in zip(tgt, pred):
            _vals.append((metric(p, t, fs, idx1, idx2)).cpu().item())
        return _vals

    for m_fn in [snr, si_snr]:
        metrics[m_fn.__name__] = _metric(m_fn, output, gt)
        metrics[m_fn.__name__+'_i'] = metric_i(m_fn, mixed, output, gt)

    pair_list = ['_12', '_13', '_14', '_23', '_24', '_34']
    for pair in pair_list:
        idx1, idx2 = int(pair[1]) - 1, int(pair[2]) - 1
        metrics[ild.__name__+pair] = metric_ild(ild, output, gt, idx1, idx2)
        metrics[ipd.__name__+pair] = metric_ipd(ipd, output, gt, idx1, idx2, win=256, hop=128)
        metrics[itd_gccphat.__name__+pair] = metric_itd(itd_gccphat, output, gt, idx1, idx2, fs=8000)

    return metrics
