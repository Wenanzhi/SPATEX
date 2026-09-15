"""
LibriSpeech-based target speaker extraction dataset with optional array-agnostic
microphone sampling.
"""

import os
import math
import random
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


_MAX_SAMPLE_SEED = (1 << 63) - 1


def _compose_sample_seed(base_seed, epoch, index):
    return (
        int(base_seed) * 6364136223846793005
        + int(epoch) * 1442695040888963407
        + int(index) * 22695477
    ) % _MAX_SAMPLE_SEED


# ---------------------------------------------------------------------------
# LibriSpeech scanning / audio I/O
# ---------------------------------------------------------------------------

def _scan_librispeech(root_dir, subsets):
    speaker_utts = {}
    for subset in subsets:
        subset_dir = os.path.join(root_dir, subset)
        if not os.path.isdir(subset_dir):
            logging.warning("LibriSpeech subset dir not found: %s", subset_dir)
            continue
        for spk_id in sorted(os.listdir(subset_dir)):
            spk_dir = os.path.join(subset_dir, spk_id)
            if not os.path.isdir(spk_dir):
                continue
            for chap_id in sorted(os.listdir(spk_dir)):
                chap_dir = os.path.join(spk_dir, chap_id)
                if not os.path.isdir(chap_dir):
                    continue
                for fname in sorted(os.listdir(chap_dir)):
                    if fname.endswith(('.wav', '.flac')):
                        speaker_utts.setdefault(spk_id, []).append(os.path.join(chap_dir, fname))
    return {k: v for k, v in speaker_utts.items() if len(v) >= 2}


def _load_wav(path, sr, target_samples, rng=None):
    wav, orig_sr = torchaudio.load(path)
    wav = wav.mean(dim=0)
    if orig_sr != sr:
        wav = torchaudio.transforms.Resample(orig_sr, sr)(wav.unsqueeze(0)).squeeze(0)
    if wav.shape[0] > target_samples:
        max_start = wav.shape[0] - target_samples
        start = (rng or random).randint(0, max_start) if max_start > 0 else 0
        wav = wav[start:start + target_samples]
    elif wav.shape[0] < target_samples:
        wav = F.pad(wav, (0, target_samples - wav.shape[0]))
    return wav


# ---------------------------------------------------------------------------
# Geometry and spatialization
# ---------------------------------------------------------------------------

def _azimuth_unit_vector(azimuth_deg):
    azimuth_rad = azimuth_deg * math.pi / 180.0
    return torch.tensor([math.sin(azimuth_rad), math.cos(azimuth_rad), 0.0], dtype=torch.float32)


def _generate_room_rirs(azimuths_deg, n_mics=4, mic_spacing=0.04, sr=8000,
                        room_dim=None, rt60=0.4, source_distance=2.0,
                        mic_height=1.5, rng=None, mic_xyz=None,
                        return_direct=False):
    import pyroomacoustics as pra

    if rng is None:
        rng = random.Random()
    if room_dim is None:
        room_dim = [rng.uniform(5.0, 10.0), rng.uniform(4.0, 8.0), rng.uniform(2.5, 4.0)]

    for _retry in range(10):
        try:
            e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
            break
        except ValueError:
            rt60 = rt60 * 1.5
    else:
        e_absorption, max_order = 0.5, 10
    max_order = min(max_order, 50)

    room = pra.ShoeBox(room_dim, fs=sr, materials=pra.Material(e_absorption), max_order=max_order)
    cx, cy, cz = room_dim[0] / 2.0, room_dim[1] / 2.0, mic_height
    if mic_xyz is None:
        mic_locs = np.zeros((3, n_mics))
        for m in range(n_mics):
            mic_locs[0, m] = cx + (m - (n_mics - 1) / 2.0) * mic_spacing
            mic_locs[1, m] = cy
            mic_locs[2, m] = cz
    else:
        xyz = mic_xyz.detach().cpu().numpy().astype(np.float64)
        mic_locs = np.stack([cx + xyz[:, 0], cy + xyz[:, 1], cz + xyz[:, 2]], axis=0)
    room.add_microphone_array(mic_locs)

    margin = 0.05
    source_positions = []
    for az_deg in azimuths_deg:
        az_rad = az_deg * np.pi / 180.0
        sx = float(np.clip(cx + source_distance * np.sin(az_rad), margin, room_dim[0] - margin))
        sy = float(np.clip(cy + source_distance * np.cos(az_rad), margin, room_dim[1] - margin))
        sz = float(np.clip(cz, margin, room_dim[2] - margin))
        source_position = [sx, sy, sz]
        source_positions.append(source_position)
        room.add_source(source_position)

    room.compute_rir()
    def collect_rirs(source_room):
        collected = []
        for src_idx in range(len(azimuths_deg)):
            mic_rirs = [
                source_room.rir[mic_idx][src_idx]
                for mic_idx in range(n_mics)
            ]
            max_len = max(r.shape[0] for r in mic_rirs)
            padded = np.zeros((n_mics, max_len), dtype=np.float32)
            for mic_idx, rir in enumerate(mic_rirs):
                padded[mic_idx, :rir.shape[0]] = rir.astype(np.float32)
            collected.append(padded)
        return collected

    rirs = collect_rirs(room)
    if not return_direct:
        return rirs

    # SoundCompass supervises a direct-path (called ``anechoic`` in the
    # released code) decoder and a residual-reverberation decoder separately.
    # Generate the direct component with the exact same room, microphones and
    # source locations but image-source order zero.  Consequently
    # direct + (full - direct) reconstructs the common reverberant target.
    direct_room = pra.ShoeBox(room_dim, fs=sr, max_order=0)
    direct_room.add_microphone_array(mic_locs.copy())
    for source_position in source_positions:
        direct_room.add_source(source_position)
    direct_room.compute_rir()
    return rirs, collect_rirs(direct_room)


def spatialize_source(mono, azimuth_deg, n_mics=4, mic_spacing=0.04,
                      sr=8000, c=343.0, rir=None, mic_xyz=None):
    T = mono.shape[0]
    if rir is not None:
        rir_t = torch.from_numpy(rir).float() if isinstance(rir, np.ndarray) else rir.float()
        n_fft = T + rir_t.shape[1] - 1
        mono_fft = torch.fft.rfft(mono.float(), n=n_fft)
        rir_fft = torch.fft.rfft(rir_t, n=n_fft)
        return torch.fft.irfft(mono_fft.unsqueeze(0) * rir_fft, n=n_fft)[:, :T].float()

    n_fft = T
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr)
    mono_fft = torch.fft.rfft(mono, n=n_fft)
    if mic_xyz is None:
        azimuth_rad = azimuth_deg * math.pi / 180.0
        delays = torch.tensor([m * mic_spacing * math.sin(azimuth_rad) / c for m in range(n_mics)], dtype=torch.float32)
    else:
        direction = _azimuth_unit_vector(azimuth_deg)
        rel = mic_xyz.float() - mic_xyz.float().mean(dim=0, keepdim=True)
        delays = rel.matmul(direction) / c
    phase_shifts = torch.exp(-2j * math.pi * delays.unsqueeze(1).to(torch.complex64) * freqs.unsqueeze(0).to(torch.complex64))
    return torch.fft.irfft(mono_fft.unsqueeze(0).to(torch.complex64) * phase_shifts, n=n_fft)[:, :T].float()


def array_agnostic_collate_fn(batch):
    if not isinstance(batch[0], dict):
        return torch.utils.data.default_collate(batch)
    out = {}
    for key in [
            'mixture', 'target', 'enrollment', 'azim_vec', 'mic_mask',
            'mic_xyz', 'source_components', 'source_directions',
            'target_anechoic', 'target_reverb']:
        if key in batch[0] and batch[0][key] is not None:
            out[key] = torch.stack([item[key] for item in batch], dim=0)
    out['metadata'] = [item.get('metadata', {}) for item in batch]
    return out


def unpack_batch(batch, device=None):
    if isinstance(batch, dict):
        mixed = batch['mixture']
        gt = batch.get('target', batch.get('target_mc'))
        enrollment = batch['enrollment']
        azim_vec = batch['azim_vec']
        mic_mask = batch.get('mic_mask')
        mic_xyz = batch.get('mic_xyz')
        metadata = batch.get('metadata')
    else:
        mixed, gt, enrollment, azim_vec = batch
        mic_mask, mic_xyz, metadata = None, None, None
    if device is not None:
        mixed = mixed.to(device)
        gt = gt.to(device)
        enrollment = enrollment.to(device)
        azim_vec = azim_vec.to(device)
        if mic_mask is not None:
            mic_mask = mic_mask.to(device)
        if mic_xyz is not None:
            mic_xyz = mic_xyz.to(device)
    return mixed, gt, enrollment, azim_vec, mic_mask, mic_xyz, metadata


def model_auxiliary_kwargs(batch, device=None):
    """Return optional tensor supervision carried by dictionary datasets.

    These tensors are deliberately kept out of :func:`unpack_batch` so the
    public seven-item return contract remains compatible with existing
    training and evaluation utilities.  Models that do not request this
    supervision continue to receive no additional keyword arguments.
    """
    if not isinstance(batch, dict):
        return {}
    auxiliary = {}
    for key in (
            'source_components', 'source_directions',
            'target_anechoic', 'target_reverb'):
        value = batch.get(key)
        if value is not None:
            auxiliary[key] = value.to(device) if device is not None else value
    return auxiliary


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LibriSpeechTSEDataset(torch.utils.data.Dataset):
    _DEFAULT_SUBSETS = {
        'train': ['train-clean-100'],
        'val': ['dev-clean'],
        'test': ['test-clean'],
    }

    def __init__(self, input_dir, dset='train', sr=8000, win=256,
                 subsets=None, n_mics=4, n_interferers=2,
                 segment_len=6.0, enrollment_len=3.0,
                 mic_spacing=0.04, azim_type='cycpos', d_model=40, alpha=80,
                 num_samples=10000, min_angle_gap=15.0, use_reverb=False,
                 rt60_range=(0.15, 0.7), room_dim_range=None,
                 source_distance_range=(1.5, 3.0), mic_height=1.5,
                 target_anechoic=False, random_seed=None, **kwargs):
        super().__init__()
        self.input_dir = input_dir
        assert dset in ['train', 'val', 'test']
        self.dset = dset
        self.sr = sr
        self.win = win
        self.hop = win // 2
        self.n_mics = n_mics
        self.max_n_mics = int(kwargs.get('max_n_mics', n_mics))
        self.min_n_mics = int(kwargs.get('min_n_mics', n_mics))
        self.variable_n_mics = bool(kwargs.get('variable_n_mics', False))
        self.return_array_metadata = bool(kwargs.get('return_array_metadata', False))
        self.return_source_components = bool(
            kwargs.get('return_source_components', False))
        self.return_target_components = bool(
            kwargs.get('return_target_components', False))
        self.append_doa_unit_vector = bool(kwargs.get('append_doa_unit_vector', False))
        self.geometry_types = list(kwargs.get('geometry_types', ['linear']))
        self.seen_geometry_types = list(kwargs.get('seen_geometry_types', self.geometry_types))
        self.unseen_geometry_types = list(kwargs.get('unseen_geometry_types', []))
        self.geometry_split = kwargs.get('geometry_split', 'all')
        self.linear_spacing_range = tuple(kwargs.get('linear_spacing_range', (mic_spacing, mic_spacing)))
        self.planar_spacing_range = tuple(kwargs.get('planar_spacing_range', (mic_spacing, mic_spacing)))
        self.random_aperture_range = tuple(kwargs.get('random_aperture_range', (0.05, 0.25)))
        self.sensor_position_error_std = float(kwargs.get('sensor_position_error_std', 0.0))
        self.n_interferers = n_interferers
        self.segment_samples = int(segment_len * sr)
        self.enrollment_samples = int(enrollment_len * sr)
        self.mic_spacing = mic_spacing
        self.azim_type = azim_type
        self.d_model = d_model
        self.alpha = alpha
        self.num_samples = num_samples
        self.min_angle_gap = min_angle_gap
        self.use_reverb = use_reverb
        self.rt60_range = tuple(rt60_range)
        self.room_dim_range = room_dim_range or {'length': (5.0, 10.0), 'width': (4.0, 8.0), 'height': (2.5, 4.0)}
        self.source_distance_range = tuple(source_distance_range)
        self.mic_height = mic_height
        self.target_anechoic = target_anechoic
        if self.return_target_components and self.target_anechoic:
            raise ValueError(
                "return_target_components requires the primary target to be "
                "the full reverberant signal (target_anechoic=False)")
        if random_seed is None:
            random_seed = 0 if dset == 'train' else 42
        self.random_seed = int(random_seed)
        self.epoch = 0

        if subsets is None:
            subsets = self._DEFAULT_SUBSETS.get(dset, ['train-clean-100'])
        logging.info("Scanning LibriSpeech subsets %s from %s for dset=%s", subsets, input_dir, dset)
        self.speaker_utts = _scan_librispeech(input_dir, subsets)
        self.speaker_ids = sorted(self.speaker_utts.keys())
        assert len(self.speaker_ids) >= 2, "Need at least 2 speakers, found %d" % len(self.speaker_ids)
        logging.info("Found %d speakers, %d utterances", len(self.speaker_ids), sum(len(v) for v in self.speaker_utts.values()))

    def _get_azim_vector_onehot(self, angle, resolution=1):
        num_positions = int(360 / resolution)
        vector = torch.zeros(num_positions)
        if round(angle / resolution) == num_positions:
            angle = 0
        vector[int(round(angle / resolution))] = 1
        return vector

    def _get_azim_vector_cycpos(self, angle):
        max_len = 360
        pe = torch.zeros(max_len, self.d_model)
        phi = torch.arange(0, max_len).unsqueeze(1) * (math.pi / 180)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * -(torch.log(torch.tensor([10000.0])) / self.d_model))
        pe[:, 0::2] = torch.sin(torch.sin(phi) * self.alpha * div_term)
        pe[:, 1::2] = torch.sin(torch.cos(phi) * self.alpha * div_term)
        for a in range(max_len):
            pe[a] = pe[a] / torch.norm(pe[a])
        return pe[int(round(angle)) % 360]

    def _get_azim_vector(self, angle):
        if self.azim_type == 'onehot':
            encoded = self._get_azim_vector_onehot(angle)
        else:
            encoded = self._get_azim_vector_cycpos(angle)
        if self.append_doa_unit_vector:
            encoded = torch.cat([encoded, _azimuth_unit_vector(angle)], dim=0)
        return encoded

    def _sample_azimuths(self, n_sources, rng):
        azimuths = []
        for _ in range(n_sources):
            for _attempt in range(100):
                a = rng.uniform(0, 360)
                if all(min(abs(a - b), 360 - abs(a - b)) >= self.min_angle_gap for b in azimuths):
                    break
            azimuths.append(a)
        return azimuths

    def _geometry_pool(self):
        if self.geometry_split == 'seen':
            return self.seen_geometry_types or self.geometry_types
        if self.geometry_split == 'unseen':
            return self.unseen_geometry_types or self.geometry_types
        return self.geometry_types

    def _sample_array(self, rng, torch_generator=None):
        if torch_generator is None:
            torch_generator = torch.Generator()
            torch_generator.manual_seed(rng.randrange(_MAX_SAMPLE_SEED))
        n_valid = rng.randint(self.min_n_mics, self.max_n_mics) if self.variable_n_mics else self.n_mics
        geom = rng.choice(self._geometry_pool())
        if geom == 'linear':
            spacing = rng.uniform(*self.linear_spacing_range)
            xs = torch.arange(n_valid, dtype=torch.float32) - (n_valid - 1) / 2.0
            xyz = torch.stack([xs * spacing, torch.zeros(n_valid), torch.zeros(n_valid)], dim=-1)
        elif geom == 'planar':
            spacing = rng.uniform(*self.planar_spacing_range)
            side = math.ceil(math.sqrt(n_valid))
            pts = []
            for i in range(side):
                for j in range(side):
                    pts.append([(i - (side - 1) / 2.0) * spacing, (j - (side - 1) / 2.0) * spacing, 0.0])
            xyz = torch.tensor(pts[:n_valid], dtype=torch.float32)
        elif geom == 'sparse':
            spacing = rng.uniform(*self.planar_spacing_range)
            coords = [(i * spacing, j * spacing, 0.0) for i in range(-2, 3) for j in range(-2, 3)]
            xyz = torch.tensor(rng.sample(coords, n_valid), dtype=torch.float32)
            xyz = xyz - xyz.mean(dim=0, keepdim=True)
        elif geom == 'random':
            aperture = rng.uniform(*self.random_aperture_range)
            xyz = (torch.rand(n_valid, 3, generator=torch_generator) - 0.5) * aperture
            xyz[:, 2] *= 0.2
            xyz = xyz - xyz.mean(dim=0, keepdim=True)
        else:
            raise ValueError("Unsupported geometry_type: %s" % geom)
        if self.sensor_position_error_std > 0:
            position_noise = torch.randn(
                xyz.shape,
                dtype=xyz.dtype,
                device=xyz.device,
                generator=torch_generator,
            )
            xyz = xyz + position_noise * self.sensor_position_error_std
        return n_valid, geom, xyz.float()

    def _pad_array(self, x, mic_xyz):
        n_valid = x.shape[0]
        if n_valid == self.max_n_mics:
            padded = x
            xyz_padded = mic_xyz
        else:
            padded = F.pad(x, (0, 0, 0, self.max_n_mics - n_valid))
            xyz_padded = F.pad(mic_xyz, (0, 0, 0, self.max_n_mics - n_valid))
        mask = torch.zeros(self.max_n_mics, dtype=torch.float32)
        mask[:n_valid] = 1.0
        return padded, xyz_padded, mask

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def _rng_for_index(self, idx):
        sample_epoch = self.epoch if self.dset == 'train' else 0
        sample_seed = _compose_sample_seed(
            self.random_seed, sample_epoch, int(idx))
        rng = random.Random(sample_seed)
        torch_generator = torch.Generator()
        torch_generator.manual_seed(sample_seed)
        return rng, torch_generator, sample_seed

    def __getitem__(self, idx):
        rng, torch_generator, sample_seed = self._rng_for_index(idx)
        target_spk = rng.choice(self.speaker_ids)
        other_spks = [s for s in self.speaker_ids if s != target_spk]
        interferer_spks = rng.sample(other_spks, min(self.n_interferers, len(other_spks)))
        target_utt, enroll_utt = rng.sample(self.speaker_utts[target_spk], 2)
        interferer_utts = [rng.choice(self.speaker_utts[s]) for s in interferer_spks]
        target_wav = _load_wav(target_utt, self.sr, self.segment_samples, rng)
        enroll_wav = _load_wav(enroll_utt, self.sr, self.enrollment_samples, rng)
        interferer_wavs = [_load_wav(u, self.sr, self.segment_samples, rng) for u in interferer_utts]
        azimuths = self._sample_azimuths(1 + len(interferer_wavs), rng)
        target_azim = azimuths[0]
        n_valid, geometry_type, mic_xyz = self._sample_array(
            rng, torch_generator)

        source_directions = torch.stack([
            _azimuth_unit_vector(azimuth) for azimuth in azimuths], dim=0)
        metadata = {'n_valid_mics': n_valid, 'geometry_type': geometry_type,
                    'seen_unseen_geometry': 'unseen' if geometry_type in self.unseen_geometry_types else 'seen',
                    'min_angle_gap': self.min_angle_gap,
                    'sensor_position_error_std': self.sensor_position_error_std,
                    'sample_seed': int(sample_seed),
                    'dataset_epoch': self.epoch if self.dset == 'train' else 0,
                    'azim_deg': float(target_azim),
                    'source_azimuths_deg': [float(value) for value in azimuths],
                    'mic_xyz': mic_xyz.tolist()}
        if self.use_reverb:
            target_mc, mixture, reverb_meta, source_components, \
                target_components = \
                self._spatialize_reverb(
                    target_wav, interferer_wavs, azimuths, rng, mic_xyz)
            metadata.update(reverb_meta)
        else:
            target_mc, mixture, sir_db, source_components, \
                target_components = \
                self._spatialize_freefield(
                    target_wav, interferer_wavs, azimuths, rng, mic_xyz)
            metadata['sir_db'] = sir_db
        azim_vec = self._get_azim_vector(target_azim)

        if not self.return_array_metadata:
            return mixture, target_mc, enroll_wav, azim_vec
        mixture, mic_xyz_padded, mic_mask = self._pad_array(mixture, mic_xyz)
        target_mc, _, _ = self._pad_array(target_mc, mic_xyz)
        item = {
            'mixture': mixture, 'target': target_mc,
            'enrollment': enroll_wav, 'azim_vec': azim_vec,
            'mic_mask': mic_mask, 'mic_xyz': mic_xyz_padded,
            'metadata': metadata,
        }
        if self.return_source_components:
            item['source_components'] = F.pad(
                source_components,
                (0, 0, 0, self.max_n_mics - n_valid))
            item['source_directions'] = source_directions
        if target_components is not None:
            for key, component in target_components.items():
                item[key], _, _ = self._pad_array(component, mic_xyz)
        return item

    def _spatialize_freefield(self, target_wav, interferer_wavs, azimuths, rng, mic_xyz=None):
        n_mics = self.n_mics if mic_xyz is None else mic_xyz.shape[0]
        target_mc = spatialize_source(target_wav, azimuths[0], n_mics, self.mic_spacing, self.sr, mic_xyz=mic_xyz)
        mixture = target_mc.clone()
        source_components = [target_mc]
        last_sir = 0.0
        for i, iw in enumerate(interferer_wavs):
            imc = spatialize_source(iw, azimuths[1 + i], n_mics, self.mic_spacing, self.sr, mic_xyz=mic_xyz)
            target_power = (target_mc ** 2).mean().clamp(min=1e-10)
            int_power = (imc ** 2).mean().clamp(min=1e-10)
            last_sir = rng.uniform(-5, 5)
            scale = torch.sqrt(target_power / int_power) * (10 ** (-last_sir / 20))
            scaled_interferer = imc * scale
            mixture = mixture + scaled_interferer
            source_components.append(scaled_interferer)
        target_components = None
        if self.return_target_components:
            target_components = {
                'target_anechoic': target_mc,
                'target_reverb': torch.zeros_like(target_mc),
            }
        return target_mc, mixture, last_sir, torch.stack(
            source_components, dim=0), target_components

    def _spatialize_reverb(self, target_wav, interferer_wavs, azimuths, rng, mic_xyz=None):
        rt60 = rng.uniform(*self.rt60_range)
        rdim = self.room_dim_range
        room_dim = [rng.uniform(*rdim['length']), rng.uniform(*rdim['width']), rng.uniform(*rdim['height'])]
        src_dist = rng.uniform(*self.source_distance_range)
        n_mics = self.n_mics if mic_xyz is None else mic_xyz.shape[0]
        generated_rirs = _generate_room_rirs(
            azimuths, n_mics=n_mics, mic_spacing=self.mic_spacing,
            sr=self.sr, room_dim=room_dim, rt60=rt60,
            source_distance=src_dist, mic_height=self.mic_height, rng=rng,
            mic_xyz=mic_xyz, return_direct=self.return_target_components)
        if self.return_target_components:
            rirs, direct_rirs = generated_rirs
        else:
            rirs = generated_rirs
            direct_rirs = None
        if self.target_anechoic:
            target_mc = spatialize_source(target_wav, azimuths[0], n_mics, self.mic_spacing, self.sr, mic_xyz=mic_xyz)
        else:
            target_mc = spatialize_source(target_wav, azimuths[0], n_mics, self.mic_spacing, self.sr, rir=rirs[0])
        target_mc_reverb = spatialize_source(target_wav, azimuths[0], n_mics, self.mic_spacing, self.sr, rir=rirs[0])
        mixture = target_mc_reverb.clone()
        source_components = [target_mc_reverb]
        last_sir = 0.0
        for i, iw in enumerate(interferer_wavs):
            imc = spatialize_source(iw, azimuths[1 + i], n_mics, self.mic_spacing, self.sr, rir=rirs[1 + i])
            target_power = (target_mc_reverb ** 2).mean().clamp(min=1e-10)
            int_power = (imc ** 2).mean().clamp(min=1e-10)
            last_sir = rng.uniform(-5, 5)
            scale = torch.sqrt(target_power / int_power) * (10 ** (-last_sir / 20))
            scaled_interferer = imc * scale
            mixture = mixture + scaled_interferer
            source_components.append(scaled_interferer)
        target_components = None
        if direct_rirs is not None:
            target_direct = spatialize_source(
                target_wav, azimuths[0], n_mics, self.mic_spacing,
                self.sr, rir=direct_rirs[0])
            target_components = {
                'target_anechoic': target_direct,
                'target_reverb': target_mc_reverb - target_direct,
            }
        return target_mc, mixture, {'rt60': rt60, 'sir_db': last_sir}, \
            torch.stack(source_components, dim=0), target_components


# ---------------------------------------------------------------------------
# Tensorboard helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Saved (pre-generated) testset
# ---------------------------------------------------------------------------

class SavedTSETestset(torch.utils.data.Dataset):
    """Load a pre-generated testset previously dumped by ``build_testsets``.

    Disk layout (per subset)::

        mixture/<idx>.wav      [n_valid_mics, T] 16-bit PCM
        target/<idx>.wav       [n_valid_mics, T]
        enrollment/<idx>.wav   mono
        meta/<idx>.json        {azim_deg, mic_xyz, n_valid_mics, ...}
        meta.json              subset config

    The loader re-pads channels to ``max_n_mics`` and recomputes ``azim_vec``
    from ``azim_deg`` using *this* dataset's direction encoder so the same
    saved testset can serve models with different direction encoding hparams.

    Args:
        saved_path: subset directory.
        max_n_mics: zero-pad / slice channels to this size. Defaults to the
            ``max_n_mics`` recorded in the subset's ``meta.json`` (or
            ``n_valid_mics`` when the subset is fixed-array).
        num_samples: cap the sample count (default: all).
        azim_type, d_model, alpha: direction encoding hparams. Defaults pull
            from the subset config in ``meta.json`` so existing experiments
            do not need to set them.
        return_dict: True (default) returns the array-agnostic dict shape;
            False returns the legacy 4-tuple ``(mixture, target, enrollment,
            azim_vec)`` (channels are then exactly ``max_n_mics``).
    """

    def __init__(self, saved_path, num_samples=None, max_n_mics=None,
                 azim_type=None, d_model=None, alpha=None,
                 return_dict=True, **kwargs):
        super().__init__()
        self.saved_path = saved_path
        self.return_dict = return_dict

        meta_path = os.path.join(saved_path, 'meta.json')
        if os.path.exists(meta_path):
            import json
            with open(meta_path) as fh:
                self.subset_meta = json.load(fh)
            cfg = self.subset_meta.get('config', {})
        else:
            self.subset_meta, cfg = None, {}

        self.max_n_mics = int(max_n_mics if max_n_mics is not None
                              else cfg.get('max_n_mics', cfg.get('n_mics', 4)))
        self.azim_type = azim_type or cfg.get('azim_type', 'cycpos')
        self.d_model = int(d_model if d_model is not None else cfg.get('d_model', 40))
        self.alpha = float(alpha if alpha is not None else cfg.get('alpha', 80))
        self.append_doa_unit_vector = bool(
            kwargs.get('append_doa_unit_vector', cfg.get('append_doa_unit_vector', False)))

        mix_dir = os.path.join(saved_path, 'mixture')
        files = sorted(f for f in os.listdir(mix_dir) if f.endswith('.wav'))
        if not files:
            raise RuntimeError(f"No .wav samples found under {mix_dir}")
        if num_samples is not None:
            files = files[:num_samples]
        self.indices = [os.path.splitext(f)[0] for f in files]
        logging.info("SavedTSETestset: %d samples from %s (max_n_mics=%d, azim=%s/d_model=%d)",
                     len(self.indices), saved_path, self.max_n_mics, self.azim_type, self.d_model)

    def __len__(self):
        return len(self.indices)

    # azim encoding helpers (mirror LibriSpeechTSEDataset)
    def _get_azim_vector_onehot(self, angle, resolution=1):
        num_positions = int(360 / resolution)
        vector = torch.zeros(num_positions)
        if round(angle / resolution) == num_positions:
            angle = 0
        vector[int(round(angle / resolution))] = 1
        return vector

    def _get_azim_vector_cycpos(self, angle):
        max_len = 360
        pe = torch.zeros(max_len, self.d_model)
        phi = torch.arange(0, max_len).unsqueeze(1) * (math.pi / 180)
        div_term = torch.exp(torch.arange(0, self.d_model, 2)
                             * -(torch.log(torch.tensor([10000.0])) / self.d_model))
        pe[:, 0::2] = torch.sin(torch.sin(phi) * self.alpha * div_term)
        pe[:, 1::2] = torch.sin(torch.cos(phi) * self.alpha * div_term)
        for a in range(max_len):
            pe[a] = pe[a] / torch.norm(pe[a])
        return pe[int(round(angle)) % 360]

    def _get_azim_vector(self, angle):
        if self.azim_type == 'onehot':
            encoded = self._get_azim_vector_onehot(angle)
        else:
            encoded = self._get_azim_vector_cycpos(angle)
        if self.append_doa_unit_vector:
            encoded = torch.cat([encoded, _azimuth_unit_vector(angle)], dim=0)
        return encoded

    def _load_audio(self, path):
        wav, _ = torchaudio.load(path)  # [C, T] float in [-1, 1]
        return wav.float()

    def __getitem__(self, idx):
        import json
        stem = self.indices[idx]
        mixture = self._load_audio(os.path.join(self.saved_path, 'mixture', stem + '.wav'))
        target = self._load_audio(os.path.join(self.saved_path, 'target', stem + '.wav'))
        enrollment = self._load_audio(os.path.join(self.saved_path, 'enrollment', stem + '.wav')).squeeze(0)
        with open(os.path.join(self.saved_path, 'meta', stem + '.json')) as fh:
            meta = json.load(fh)
        subset = os.path.basename(os.path.normpath(self.saved_path))
        meta['sample_id'] = stem
        meta['scene_id'] = stem
        meta['subset'] = subset
        meta['sample_key'] = '%s/%s' % (subset, stem)
        n_valid = int(meta['n_valid_mics'])
        mic_xyz = torch.tensor(meta['mic_xyz'], dtype=torch.float32)
        # mixture / target are saved with exactly n_valid channels.
        assert mixture.shape[0] == n_valid, (
            "mixture channels %d != n_valid_mics %d" % (mixture.shape[0], n_valid))

        azim_vec = self._get_azim_vector(meta['azim_deg'])

        # Re-pad to max_n_mics (or slice if a smaller budget is requested).
        target_max = self.max_n_mics
        mic_mask = torch.zeros(target_max, dtype=torch.float32)
        mic_mask[:min(n_valid, target_max)] = 1.0
        if n_valid < target_max:
            pad = target_max - n_valid
            mixture = F.pad(mixture, (0, 0, 0, pad))
            target = F.pad(target, (0, 0, 0, pad))
            mic_xyz = F.pad(mic_xyz, (0, 0, 0, pad))
        elif n_valid > target_max:
            mixture = mixture[:target_max]
            target = target[:target_max]
            mic_xyz = mic_xyz[:target_max]

        if not self.return_dict:
            return mixture, target, enrollment, azim_vec
        return {'mixture': mixture, 'target': target, 'enrollment': enrollment,
                'azim_vec': azim_vec, 'mic_mask': mic_mask, 'mic_xyz': mic_xyz,
                'metadata': meta}


def tensorboard_add_sample(writer, tag, sample, step, params):
    sr = params.get('sr', 8000)
    import matplotlib.pyplot as plt
    m, gt, o = sample
    m, gt, o = m.cpu(), gt.cpu(), o.cpu()

    def _add_audio(a, audio_tag, axis, plt_title):
        for i, ch in enumerate(a):
            axis.plot(ch, label='mic %d' % i)
            writer.add_audio('%s/mic %d' % (audio_tag, i), ch.type(torch.float64), step, sr)
        axis.set_title(plt_title)
        axis.legend()

    for b in range(m.shape[0]):
        fig = plt.figure(figsize=(10, 6))
        axes = fig.subplots(3, 1, sharex=True)
        _add_audio(m[b], '%s/sample_%d/0_input' % (tag, b), axes[0], "Mixed")
        _add_audio(o[b], '%s/sample_%d/1_output' % (tag, b), axes[1], "Output")
        _add_audio(gt[b], '%s/sample_%d/2_gt' % (tag, b), axes[2], "GT")
        writer.add_figure('%s/sample_%d/waveform' % (tag, b), fig, step)


def tensorboard_add_metrics(writer, tag, metrics, step):
    vals = np.asarray(metrics.get('scale_invariant_signal_noise_ratio', [0.0]), dtype='float')
    vals = np.nan_to_num(vals, nan=0.0, posinf=1.0, neginf=-1.0)
    writer.add_histogram('%s/%s' % (tag, 'SI-SNR'), vals, step)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ds = LibriSpeechTSEDataset(
        input_dir='data/LibriSpeech', dset='test',
        subsets=['test-clean'], n_mics=4, max_n_mics=8, min_n_mics=2,
        variable_n_mics=True, return_array_metadata=True, num_samples=2,
        geometry_types=['linear', 'planar', 'sparse', 'random'])
    loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=array_agnostic_collate_fn)
    batch = next(iter(loader))
    print(batch['mixture'].shape, batch['target'].shape, batch['mic_mask'], batch['mic_xyz'].shape)
