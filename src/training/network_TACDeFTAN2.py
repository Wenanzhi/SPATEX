import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange

from src.helpers.utils import (signal_noise_ratio as snr,
                               scale_invariant_signal_noise_ratio as si_snr,
                               phase_constrained_magnitude as pcm,
                               delta_ILD as ild,
                               delta_IPD as ipd,
                               delta_ITD_gccphat as itd_gccphat)
from src.training.array_agnostic_blocks import (DOAMicrophonePositionEncoding,
                                                 GeometryEmbedding,
                                                 GeometryFiLMLayer,
                                                 MaskedTACBlock,
                                                 MaskedTAttCBlock,
                                                 mask_as, masked_mean)
from src.training.network_DeFTAN2 import (ActivityHead, DeFTANblock, InverseDenseBlock2d,
                                          build_speaker_encoder)
from src.training.spatial_losses import spatial_fidelity_losses


class GlobalGainHead(nn.Module):
    def __init__(self, emb_dim, hidden_dim=None, max_log_scale=1.0, init_zero=True):
        super().__init__()
        hidden_dim = hidden_dim or emb_dim
        self.max_log_scale = max_log_scale
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1),
        )
        if init_zero:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, mic_mask):
        pooled = masked_mean(x, mic_mask, dim=1, keepdim=False).mean(dim=(2, 3))
        log_gain = self.max_log_scale * torch.tanh(self.net(pooled))
        return torch.exp(log_gain).view(x.shape[0], 1, 1)


class OutputRefinementHead(nn.Module):
    def __init__(self, hidden_dim=32, n_layers=2, residual_scale=1.0, init_zero=True, eps=1.0e-5):
        super().__init__()
        hidden_dim = hidden_dim or 32
        layers = [
            nn.Conv2d(2, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim, eps=eps),
            nn.PReLU(),
        ]
        for _ in range(max(0, n_layers - 1)):
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(1, hidden_dim, eps=eps),
                nn.PReLU(),
            ])
        final = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        if init_zero:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)
        self.residual_scale = residual_scale

    def forward(self, spec):
        return spec + self.residual_scale * self.net(spec)


class SpatialFeatureBranch(nn.Module):
    def __init__(self, emb_dim, hidden_dim=None, init_zero=True, eps=1.0e-8):
        super().__init__()
        hidden_dim = hidden_dim or emb_dim
        self.eps = eps
        final = nn.Conv2d(hidden_dim, emb_dim, 3, padding=1)
        if init_zero:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.PReLU(),
            final,
        )

    def forward(self, stft_complex, mic_mask):
        mask = mic_mask[:, :, None, None].to(dtype=stft_complex.real.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        ref = (stft_complex * mask).sum(dim=1, keepdim=True) / denom
        cross = stft_complex * ref.conj()
        phase_diff = torch.angle(cross)
        log_mag_ratio = torch.log(stft_complex.abs().clamp_min(self.eps)) - torch.log(ref.abs().clamp_min(self.eps))
        feat = torch.stack([torch.sin(phase_diff), torch.cos(phase_diff), log_mag_ratio], dim=2)
        b, m, _, frames, n_freqs = feat.shape
        delta = self.net(feat.reshape(b * m, 3, frames, n_freqs)).reshape(b, m, -1, frames, n_freqs)
        return delta * mask_as(delta, mic_mask)


class Net(nn.Module):
    def __init__(self, n_srcs=None, win=256, n_mics=4, max_n_mics=None, n_layers=6,
                 att_dim=64, hidden_dim=256, n_head=4, emb_dim=64,
                 emb_ks=4, emb_hs=1, dropout=0.1, eps=1.0e-5, angle_dim=40,
                 spk_emb_dim=256, spk_encoder_type='simple', spk_encoder_ckpt=None,
                 freeze_spk_encoder=False, act_loss_weight=0.1, cue_mode='both',
                 use_masked_tac=True, tac_insert='every_block', use_geometry_embedding=True,
                 preserve_geometry_scale=True, geom_emb_dim=16, tac_hidden_dim=None,
                 channel_comm_type=None, channel_attention_heads=4,
                 deftan_attention_type='independent',
                 coattention_aggregation='sqrt_count',
                 geometry_embedding_type=None,
                 doa_direction_dim=0, doa_mpe_hidden_dim=None, sample_rate=8000,
                 speed_of_sound=343.0, delay_bands=4, use_geometry_film=False,
                 geometry_film_at_encoder=True, geometry_film_layers=None,
                 geometry_film_hidden_dim=None, geometry_film_init_zero=True,
                 geometry_film_max_scale=1.0,
                 disable_geometry_conditioning=False,
                 use_clue_gate=True, use_activity_head=True, si_snr_loss_weight=0.0,
                 waveform_loss_weight=0.0, spatial_loss_weight=0.0,
                 ipd_loss_weight=0.0, coherence_loss_weight=0.0,
                 spatial_energy_floor_db=-40.0,
                 coherence_smoothing_frames=5, spatial_loss_eps=1.0e-6,
                 use_global_gain_head=False, global_gain_hidden_dim=None,
                 global_gain_init_zero=True, global_gain_max_log_scale=1.0,
                 use_output_refinement_head=False, output_refinement_hidden_dim=None,
                 output_refinement_layers=2, output_refinement_init_zero=True,
                 output_refinement_residual_scale=1.0,
                 use_spatial_feature_branch=False, spatial_feature_hidden_dim=None,
                 spatial_feature_init_zero=True,
                 clue_type=None, label_len=None, **kwargs):
        super().__init__()
        self.win = win
        self.hop = win // 2
        self.max_n_mics = max_n_mics or n_mics
        self.n_mics = self.max_n_mics
        self.n_srcs = self.max_n_mics if n_srcs is None else n_srcs
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.emb_hs = emb_hs
        self.act_loss_weight = act_loss_weight
        self.si_snr_loss_weight = si_snr_loss_weight
        self.waveform_loss_weight = waveform_loss_weight
        if float(spatial_loss_weight) != 0.0:
            raise ValueError(
                "spatial_loss_weight is the discarded legacy ILD loss; "
                "keep it at zero and use ipd_loss_weight and/or "
                "coherence_loss_weight")
        if ipd_loss_weight < 0 or coherence_loss_weight < 0:
            raise ValueError("spatial loss weights must be non-negative")
        if coherence_smoothing_frames <= 0 or coherence_smoothing_frames % 2 == 0:
            raise ValueError(
                "coherence_smoothing_frames must be a positive odd integer")
        if spatial_loss_eps <= 0:
            raise ValueError("spatial_loss_eps must be positive")
        self.ipd_loss_weight = float(ipd_loss_weight)
        self.coherence_loss_weight = float(coherence_loss_weight)
        self.spatial_energy_floor_db = float(spatial_energy_floor_db)
        self.coherence_smoothing_frames = int(
            coherence_smoothing_frames)
        self.spatial_loss_eps = float(spatial_loss_eps)
        self.cue_mode = cue_mode
        if channel_comm_type is None:
            channel_comm_type = 'tac' if use_masked_tac else 'none'
        if channel_comm_type not in {'none', 'tac', 'tattc'}:
            raise ValueError("channel_comm_type must be one of: none, tac, tattc")
        if deftan_attention_type not in {'independent', 'coattention'}:
            raise ValueError(
                "deftan_attention_type must be one of: independent, coattention")
        if coattention_aggregation not in {'sqrt_count', 'mean'}:
            raise ValueError(
                "coattention_aggregation must be one of: mean, sqrt_count")
        if geometry_embedding_type is None:
            geometry_embedding_type = 'static' if use_geometry_embedding else 'none'
        if geometry_embedding_type not in {'none', 'static', 'doa_mpe'}:
            raise ValueError(
                "geometry_embedding_type must be one of: none, static, doa_mpe")
        self.channel_comm_type = channel_comm_type
        self.use_channel_comm = channel_comm_type != 'none'
        # Backward-compatible public attribute used by older scripts/tests.
        self.use_masked_tac = self.use_channel_comm
        self.tac_insert = tac_insert
        self.deftan_attention_type = deftan_attention_type
        self.coattention_aggregation = coattention_aggregation
        self.geometry_embedding_type = geometry_embedding_type
        self.use_geometry_embedding = geometry_embedding_type != 'none'
        self.geom_emb_dim = geom_emb_dim if self.use_geometry_embedding else 0
        self.disable_geometry_conditioning = bool(
            disable_geometry_conditioning)
        self.use_geometry_film = bool(use_geometry_film)
        self.geometry_film_at_encoder = bool(geometry_film_at_encoder)
        self.geometry_film_layers = sorted(set(geometry_film_layers or []))
        invalid_film_layers = [i for i in self.geometry_film_layers
                               if i < 0 or i >= n_layers]
        if invalid_film_layers:
            raise ValueError(
                f"geometry_film_layers contains invalid indices: {invalid_film_layers}")
        if self.use_geometry_film and not self.use_geometry_embedding:
            raise ValueError("Geometry FiLM requires a geometry embedding")
        self.use_clue_gate = use_clue_gate
        self.use_activity_head = use_activity_head
        assert win % 2 == 0

        ks, padding = (3, 3), (1, 1)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, emb_dim * n_head, ks, padding=padding),
            nn.GroupNorm(1, emb_dim * n_head, eps=eps),
            InverseDenseBlock2d(emb_dim * n_head, emb_dim, n_head),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(emb_dim, 2 * n_head, ks, padding=padding),
            InverseDenseBlock2d(2 * n_head, 2, n_head),
        )

        self.mix_blocks = nn.ModuleList([
            DeFTANblock(
                i,
                emb_dim,
                emb_ks,
                emb_hs,
                att_dim,
                hidden_dim,
                n_head,
                dropout,
                eps,
                coattention_aggregation=coattention_aggregation,
            )
            for i in range(n_layers)
        ])
        tac_hidden_dim = tac_hidden_dim or emb_dim

        def make_channel_comm():
            if self.channel_comm_type == 'tac':
                return MaskedTACBlock(
                    emb_dim, tac_hidden_dim, self.geom_emb_dim, dropout)
            if self.channel_comm_type == 'tattc':
                return MaskedTAttCBlock(
                    emb_dim, tac_hidden_dim, self.geom_emb_dim,
                    channel_attention_heads, dropout)
            return None

        self.encoder_tac = make_channel_comm()
        self.block_tacs = nn.ModuleList([
            make_channel_comm() for _ in range(n_layers)
        ]) if self.use_channel_comm else nn.ModuleList()
        if self.geometry_embedding_type == 'static':
            self.geom_embedding = GeometryEmbedding(
                self.geom_emb_dim, preserve_scale=preserve_geometry_scale)
        elif self.geometry_embedding_type == 'doa_mpe':
            self.geom_embedding = DOAMicrophonePositionEncoding(
                angle_dim, self.geom_emb_dim, hidden_dim=doa_mpe_hidden_dim,
                direction_dim=doa_direction_dim,
                preserve_scale=preserve_geometry_scale,
                sample_rate=sample_rate, speed_of_sound=speed_of_sound,
                delay_bands=delay_bands)
        else:
            self.geom_embedding = None
        self.encoder_geometry_film = (
            GeometryFiLMLayer(
                emb_dim, self.geom_emb_dim, geometry_film_hidden_dim,
                geometry_film_init_zero, geometry_film_max_scale)
            if self.use_geometry_film and self.geometry_film_at_encoder else None
        )
        self.block_geometry_films = nn.ModuleDict({
            str(i): GeometryFiLMLayer(
                emb_dim, self.geom_emb_dim, geometry_film_hidden_dim,
                geometry_film_init_zero, geometry_film_max_scale)
            for i in self.geometry_film_layers
        }) if self.use_geometry_film else nn.ModuleDict()
        self.azim_embedding = nn.ModuleList([
            nn.Sequential(nn.Linear(angle_dim, emb_dim), nn.LayerNorm(emb_dim), nn.PReLU())
            for _ in range(n_layers)
        ])
        self.spk_encoder = build_speaker_encoder(spk_encoder_type, spk_emb_dim, spk_encoder_ckpt)
        if freeze_spk_encoder:
            for p in self.spk_encoder.parameters():
                p.requires_grad = False
        self.activity_head = ActivityHead(emb_dim, spk_emb_dim)
        self.global_gain_head = (
            GlobalGainHead(emb_dim, global_gain_hidden_dim, global_gain_max_log_scale, global_gain_init_zero)
            if use_global_gain_head else None
        )
        self.output_refinement_head = (
            OutputRefinementHead(output_refinement_hidden_dim, output_refinement_layers,
                                 output_refinement_residual_scale, output_refinement_init_zero, eps)
            if use_output_refinement_head else None
        )
        self.spatial_feature_branch = (
            SpatialFeatureBranch(emb_dim, spatial_feature_hidden_dim, spatial_feature_init_zero)
            if use_spatial_feature_branch else None
        )

    def pad_signal(self, input):
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        if input.dim() == 2:
            input = input.unsqueeze(1)
        b, m, nsample = input.size()
        rest = self.win - (self.hop + nsample % self.win) % self.win
        if rest > 0:
            input = torch.cat([input, Variable(torch.zeros(b, m, rest)).type(input.type())], 2)
        pad_aux = Variable(torch.zeros(b, m, self.hop)).type(input.type())
        return torch.cat([pad_aux, input, pad_aux], 2), rest

    def _default_mask(self, x, mic_mask):
        if mic_mask is None:
            mic_mask = x.new_ones(x.shape[0], x.shape[1])
        return mic_mask.to(device=x.device, dtype=x.dtype)

    def _masked_std(self, x, mic_mask):
        mask = mic_mask[:, :, None]
        denom = (mask.sum(dim=(1, 2), keepdim=True) * x.shape[-1]).clamp_min(1.0)
        mean = (x * mask).sum(dim=(1, 2), keepdim=True) / denom
        var = (((x - mean) * mask) ** 2).sum(dim=(1, 2), keepdim=True) / denom
        return var.sqrt().clamp_min(1e-6)

    @torch.no_grad()
    def _compute_oracle_activity(self, gt, mic_mask, T_target):
        gt_padded, _ = self.pad_signal(gt * mic_mask[:, :, None])
        energy_ref = (gt_padded ** 2).sum(dim=-1)
        ref_idx = (energy_ref * mic_mask).argmax(dim=1)
        ref = gt_padded[torch.arange(gt_padded.shape[0], device=gt.device), ref_idx]
        stft_gt = torch.stft(ref, n_fft=self.win, hop_length=self.hop,
                             window=torch.hann_window(self.win).to(gt.device), return_complex=False)
        energy = (stft_gt[..., 0] ** 2 + stft_gt[..., 1] ** 2).sum(dim=1)
        log_energy = 10 * torch.log10(energy.clamp(min=1e-10))
        activity = (log_energy > (log_energy.max(dim=1, keepdim=True)[0] - 30)).float()
        if activity.shape[1] > T_target:
            activity = activity[:, :T_target]
        elif activity.shape[1] < T_target:
            activity = F.pad(activity, (0, T_target - activity.shape[1]))
        return activity

    def _gate(self, layer_idx, azim_vec, activity_hat, batch):
        if not self.use_clue_gate:
            return batch
        if azim_vec is None:
            dir_gate = batch.new_ones(batch.shape[0], self.emb_dim)
        else:
            dir_gate = self.azim_embedding[layer_idx](azim_vec)
        if activity_hat is None:
            act_gate = batch.new_ones(batch.shape[0], batch.shape[3])
        else:
            act_gate = activity_hat
        if self.cue_mode == 'doa_only':
            gate = rearrange(dir_gate, 'b c -> b 1 c 1 1')
        elif self.cue_mode == 'spk_emb_only':
            gate = rearrange(act_gate, 'b t -> b 1 1 t 1')
        else:
            gate = rearrange(dir_gate, 'b c -> b 1 c 1 1') * rearrange(act_gate, 'b t -> b 1 1 t 1')
        return batch * gate

    def _geometry_embedding(self, mic_xyz, mic_mask, azim_vec):
        if self.geom_embedding is None or mic_xyz is None:
            return None
        if self.geometry_embedding_type == 'doa_mpe':
            return self.geom_embedding(mic_xyz, mic_mask, azim_vec)
        return self.geom_embedding(mic_xyz, mic_mask)

    def forward(self, input, gt=None, enrollment=None, azim_vec=None,
                mic_mask=None, mic_xyz=None, timestamp_or_activity=None):
        raw_input = input
        mic_mask = self._default_mask(input, mic_mask)
        input = input * mic_mask[:, :, None]
        input, rest = self.pad_signal(input)
        b, m, n = input.size()
        mix_std = self._masked_std(input, mic_mask)
        input = input / mix_std

        stft_input = torch.stft(input.reshape(-1, n), n_fft=self.win, hop_length=self.hop,
                                window=torch.hann_window(self.win).type(input.type()), return_complex=False)
        _, n_freqs, frames, _ = stft_input.size()
        stft_view = stft_input.view(b, m, n_freqs, frames, 2)
        encoder_in = stft_view.permute(0, 1, 4, 3, 2).contiguous()
        x = self.encoder(encoder_in.reshape(b * m, 2, frames, n_freqs)).reshape(b, m, self.emb_dim, frames, n_freqs)
        x = x * mask_as(x, mic_mask)
        if self.spatial_feature_branch is not None:
            stft_complex = torch.complex(stft_view[..., 0], stft_view[..., 1]).permute(0, 1, 3, 2).contiguous()
            x = x + self.spatial_feature_branch(stft_complex, mic_mask)
            x = x * mask_as(x, mic_mask)

        geom_emb = self._geometry_embedding(mic_xyz, mic_mask, azim_vec)
        if self.disable_geometry_conditioning and geom_emb is not None:
            # Keep the full geometry branch and all downstream channel sizes in
            # the checkpoint, but prevent any explicit coordinate-derived
            # feature from reaching TAC or geometry FiLM.
            geom_emb = torch.zeros_like(geom_emb)
        if self.encoder_geometry_film is not None:
            x = self.encoder_geometry_film(x, geom_emb, mic_mask)
        if self.use_channel_comm and self.tac_insert in ['encoder_only', 'every_block']:
            x = self.encoder_tac(x, mic_mask, geom_emb)

        if enrollment is None:
            spk_emb = x.new_zeros(b, self.activity_head.spk_proj[0].in_features)
        else:
            spk_emb = self.spk_encoder(enrollment)
        pooled = masked_mean(x, mic_mask, dim=1, keepdim=False)
        activity_logits = self.activity_head(pooled, spk_emb) if self.use_activity_head else None
        activity_hat = torch.sigmoid(activity_logits) if activity_logits is not None else None
        if timestamp_or_activity is not None:
            activity_hat = timestamp_or_activity.to(x.device, x.dtype)
            if activity_hat.shape[-1] != frames:
                activity_hat = F.interpolate(activity_hat[:, None], size=frames, mode='nearest').squeeze(1)

        valid_flat = mic_mask.reshape(b * m).bool()
        for ii, block in enumerate(self.mix_blocks):
            x = self._gate(ii, azim_vec, activity_hat, x)
            film_key = str(ii)
            if film_key in self.block_geometry_films:
                x = self.block_geometry_films[film_key](x, geom_emb, mic_mask)
            if self.deftan_attention_type == 'coattention':
                x = block.forward_multichannel(x, mic_mask, coattention=True)
            else:
                x_flat = x.reshape(b * m, self.emb_dim, frames, n_freqs)
                y_flat = x_flat.new_zeros(x_flat.shape)
                if valid_flat.any():
                    y_valid = block(x_flat[valid_flat])
                    y_flat[valid_flat] = y_valid[:, :, :frames, :n_freqs]
                x = y_flat.reshape(b, m, self.emb_dim, frames, n_freqs) * mask_as(x, mic_mask)
            if self.use_channel_comm and self.tac_insert == 'every_block':
                x = self.block_tacs[ii](x, mic_mask, geom_emb)

        spec = self.decoder(x.reshape(b * m, self.emb_dim, frames, n_freqs))
        if self.output_refinement_head is not None:
            spec = self.output_refinement_head(spec)
            spec_bm = spec.reshape(b, m, 2, frames, n_freqs)
            spec_bm = spec_bm * mask_as(spec_bm, mic_mask)
            spec = spec_bm.reshape(b * m, 2, frames, n_freqs)
        spec = spec.permute(0, 3, 2, 1).type(input.type())
        istft_input = torch.complex(spec[..., 0], spec[..., 1])
        wav = torch.istft(istft_input, n_fft=self.win, hop_length=self.hop,
                          window=torch.hann_window(self.win).type(input.type()), return_complex=False)
        output = wav[:, self.hop:-(rest + self.hop)].view(b, m, -1) * mix_std
        if self.global_gain_head is not None:
            output = output * self.global_gain_head(x, mic_mask)
        output = output * mic_mask[:, :, None]

        if gt is None:
            return output
        total_loss = masked_pcm_loss(raw_input, output, gt, mic_mask, self.win, self.hop)
        if self.si_snr_loss_weight > 0:
            total_loss = total_loss - self.si_snr_loss_weight * masked_si_snr(output, gt, mic_mask).mean()
        if self.waveform_loss_weight > 0:
            total_loss = total_loss + self.waveform_loss_weight * masked_waveform_l1(output, gt, mic_mask)
        if activity_logits is not None and self.act_loss_weight > 0:
            oracle_activity = self._compute_oracle_activity(gt, mic_mask, frames)
            total_loss = total_loss + self.act_loss_weight * F.binary_cross_entropy_with_logits(activity_logits, oracle_activity)
        if self.ipd_loss_weight > 0 or self.coherence_loss_weight > 0:
            spatial_terms = spatial_fidelity_losses(
                output,
                gt,
                mic_mask,
                self.win,
                self.hop,
                compute_ipd=self.ipd_loss_weight > 0,
                compute_coherence=self.coherence_loss_weight > 0,
                smoothing_frames=self.coherence_smoothing_frames,
                energy_floor_db=self.spatial_energy_floor_db,
                eps=self.spatial_loss_eps,
            )
            if self.ipd_loss_weight > 0:
                total_loss = (
                    total_loss
                    + self.ipd_loss_weight * spatial_terms["ipd_loss"]
                )
            if self.coherence_loss_weight > 0:
                total_loss = (
                    total_loss
                    + self.coherence_loss_weight
                    * spatial_terms["coherence_loss"]
                )
        return output, total_loss


def optimizer(model, data_parallel=False, **kwargs):
    import torch.optim as optim
    return optim.Adam(model.parameters(), **kwargs)


def _valid_mask(mic_mask, b):
    valid = mic_mask[b].bool()
    if valid.any():
        return valid
    valid = mic_mask[b].new_zeros(mic_mask.shape[1], dtype=torch.bool)
    valid[0] = True
    return valid


def masked_pcm_loss(mix, pred, tgt, mic_mask, win, hop):
    vals = []
    for b in range(pred.shape[0]):
        valid = _valid_mask(mic_mask, b)
        vals.append(pcm(mix[b:b + 1, valid], pred[b:b + 1, valid], tgt[b:b + 1, valid], win, hop))
    return torch.cat(vals, dim=0).mean()


def masked_si_snr(pred, tgt, mic_mask):
    pred = pred.float()
    tgt = tgt.float()
    vals = []
    for b in range(pred.shape[0]):
        valid = _valid_mask(mic_mask, b)
        vals.append(si_snr(pred[b:b + 1, valid], tgt[b:b + 1, valid]))
    out = torch.cat(vals, dim=0)
    out = torch.nan_to_num(out, nan=0.0, posinf=50.0, neginf=-50.0)
    return out.clamp(min=-50.0, max=50.0)


def masked_waveform_l1(pred, tgt, mic_mask):
    per_mic = (pred - tgt).abs().mean(dim=-1)
    mask = mic_mask.to(per_mic.dtype)
    return (per_mic * mask).sum() / mask.sum().clamp_min(1.0)


def loss(mix, pred, tgt, win, hop, mic_mask=None):
    if mic_mask is None:
        mic_mask = mix.new_ones(mix.shape[:2])
    return masked_pcm_loss(mix, pred, tgt, mic_mask, win, hop)


def _metric_values(metric, pred, tgt, mic_mask=None):
    vals = []
    for b in range(pred.shape[0]):
        if mic_mask is None:
            vals.append(metric(pred[b:b + 1], tgt[b:b + 1]).detach().cpu().item())
        else:
            valid = _valid_mask(mic_mask, b)
            vals.append(metric(pred[b:b + 1, valid], tgt[b:b + 1, valid]).detach().cpu().item())
    return vals


def _metric_improvement(metric, mix, pred, tgt, mic_mask=None):
    vals = []
    for b in range(pred.shape[0]):
        if mic_mask is None:
            vals.append((metric(pred[b:b + 1], tgt[b:b + 1]) - metric(mix[b:b + 1], tgt[b:b + 1])).detach().cpu().item())
        else:
            valid = _valid_mask(mic_mask, b)
            vals.append((metric(pred[b:b + 1, valid], tgt[b:b + 1, valid]) - metric(mix[b:b + 1, valid], tgt[b:b + 1, valid])).detach().cpu().item())
    return vals


def _spatial_metric_values(metric, pred, tgt, mic_mask, *args):
    vals = []
    for b in range(pred.shape[0]):
        valid = (torch.arange(pred.shape[1], device=pred.device) if mic_mask is None
                 else torch.nonzero(mic_mask[b] > 0.5, as_tuple=False).flatten())
        pair_vals = []
        for i in range(valid.numel()):
            for j in range(i + 1, valid.numel()):
                idx1, idx2 = int(valid[i]), int(valid[j])
                pair_vals.append(metric(pred[b:b + 1], tgt[b:b + 1], *args, idx1, idx2).detach())
        if pair_vals:
            vals.append(torch.stack(pair_vals).mean().cpu().item())
        else:
            vals.append(0.0)
    return vals


def metrics(mixed, output, gt, mic_mask=None, mic_xyz=None, metadata=None):
    out = {}
    for m_fn in [snr, si_snr]:
        out[m_fn.__name__] = _metric_values(m_fn, output, gt, mic_mask)
        out[m_fn.__name__ + '_i'] = _metric_improvement(m_fn, mixed, output, gt, mic_mask)
    out['delta_ILD_mean'] = _spatial_metric_values(ild, output, gt, mic_mask)
    out['delta_IPD_mean'] = _spatial_metric_values(ipd, output, gt, mic_mask, 256, 128)
    out['delta_ITD_gccphat_mean'] = _spatial_metric_values(itd_gccphat, output, gt, mic_mask, 8000)
    if mic_mask is not None:
        out['n_valid_mics'] = mic_mask.detach().sum(dim=1).cpu().numpy().astype(float).tolist()
    else:
        out['n_valid_mics'] = [float(output.shape[1])] * output.shape[0]
    return out
