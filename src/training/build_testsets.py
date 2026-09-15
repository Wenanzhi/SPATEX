"""
Build and save the pre-generated TSE test sets to disk.

Layout produced under ``--out_root``::

    seg_2s/
        1_4ch_fixed/
            mixture/000000.wav ... 000499.wav     # [n_valid_mics, T]
            target/000000.wav ...                 # [n_valid_mics, T]
            enrollment/000000.wav ...             # mono
            meta/000000.json ...                  # per-sample metadata
            meta.json                             # subset config
        2_4ch_unfixed/  ...
        3_var_match/    ...
        4_var_unmatch/  ...
        5_8ch_fixed/    ...
        6_8ch_unfixed/  ...
    seg_6s/
        ... (same subsets)

Per-sample ``meta/<idx>.json`` stores the *physical* sample description::

    {
        'azim_deg': float,                 # target speaker azimuth (degrees)
        'n_valid_mics': int,               # number of valid microphones
        'mic_xyz': [[x, y, z], ...],       # array geometry, n_valid_mics x 3
        'geometry_type': str,              # 'linear' / 'planar' / 'sparse' / 'random'
        'seen_unseen_geometry': str,
        'sensor_position_error_std': float,
        'rt60': float (only when use_reverb=True),
        'sir_db': float,
    }

The companion subset-level ``meta.json`` records the full dataset config so
the saved set is fully self-describing and reproducible.

We deliberately do NOT store the direction-encoded ``azim_vec`` — the encoding
(``azim_type``, ``d_model``, ``alpha``) is a model hyper-parameter and the
loader recomputes it from ``azim_deg`` so the same testset can serve models
with different direction encoders.

Audio is stored as 16-bit PCM WAV at the dataset's sample rate, with only
the valid microphone channels (no zero-padding) so multi-mic geometries with
fewer channels stay compact.

Usage
-----
    python -m src.training.build_testsets \
        --out_root data/Testset \
        --num_samples 500
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from src.training.dataset import LibriSpeechTSEDataset


LIBRISPEECH_ROOT = 'data/LibriSpeech'

# Common parameters shared across all subsets / segment lengths.
COMMON = dict(
    input_dir=LIBRISPEECH_ROOT,
    dset='test',
    sr=8000,
    win=256,
    subsets=['test-clean'],
    n_interferers=2,
    enrollment_len=3.0,
    azim_type='cycpos',
    d_model=40,
    alpha=80,
    min_angle_gap=15.0,
    use_reverb=True,
    rt60_range=[0.15, 0.7],
    room_dim_range={'length': [5.0, 10.0], 'width': [4.0, 8.0], 'height': [2.5, 4.0]},
    source_distance_range=[1.5, 3.0],
    mic_height=1.5,
    target_anechoic=False,
    return_array_metadata=True,
)


def _subset_configs():
    """Return ``{subset_name: extra_kwargs_dict}`` for saved test sets."""
    return {
        # 1) Four-channel fixed linear array (matches M2M-SPKTSE's current testset)
        '1_4ch_fixed': dict(
            n_mics=4, max_n_mics=4, min_n_mics=4, variable_n_mics=False,
            mic_spacing=0.04,
            geometry_types=['linear'],
            seen_geometry_types=['linear'],
            unseen_geometry_types=[],
            geometry_split='seen',
            linear_spacing_range=[0.04, 0.04],
            planar_spacing_range=[0.04, 0.04],
            random_aperture_range=[0.04, 0.04],
            sensor_position_error_std=0.0,
        ),
        # 2) Four-channel but non-fixed: probe geometric generalisation while
        #    keeping the channel count identical to subset 1.
        '2_4ch_unfixed': dict(
            n_mics=4, max_n_mics=4, min_n_mics=4, variable_n_mics=False,
            mic_spacing=0.04,
            geometry_types=['linear', 'sparse', 'random'],
            seen_geometry_types=['linear'],
            unseen_geometry_types=['sparse', 'random'],
            geometry_split='all',
            # Wider OOD spacings: well outside the [0.025, 0.08] m training band.
            linear_spacing_range=[0.10, 0.20],
            planar_spacing_range=[0.10, 0.20],
            random_aperture_range=[0.30, 0.50],
            sensor_position_error_std=0.0,
        ),
        # 3) Variable channel count, geometry MATCHES training (linear+planar,
        #    in-distribution spacings) -> upper-bound for array-agnostic models.
        '3_var_match': dict(
            n_mics=8, max_n_mics=8, min_n_mics=2, variable_n_mics=True,
            mic_spacing=0.04,
            geometry_types=['linear', 'planar'],
            seen_geometry_types=['linear', 'planar'],
            unseen_geometry_types=[],
            geometry_split='seen',
            linear_spacing_range=[0.025, 0.08],
            planar_spacing_range=[0.025, 0.08],
            random_aperture_range=[0.05, 0.25],
            sensor_position_error_std=0.0,
        ),
        # 4) Variable channel count, geometry DOES NOT match training
        #    (sparse+random + sensor jitter) -> matches the current TAC-M2M
        #    test_data configuration.
        '4_var_unmatch': dict(
            n_mics=8, max_n_mics=8, min_n_mics=2, variable_n_mics=True,
            mic_spacing=0.04,
            geometry_types=['linear', 'planar', 'sparse', 'random'],
            seen_geometry_types=['linear', 'planar'],
            unseen_geometry_types=['sparse', 'random'],
            geometry_split='unseen',
            linear_spacing_range=[0.025, 0.08],
            planar_spacing_range=[0.025, 0.08],
            random_aperture_range=[0.05, 0.25],
            sensor_position_error_std=0.01,
        ),
        # 5) Eight-channel fixed linear array, matching the existing saved
        #    ``5_8ch_fixed`` testset under Testset/seg_2s.
        '5_8ch_fixed': dict(
            n_mics=8, max_n_mics=8, min_n_mics=8, variable_n_mics=False,
            mic_spacing=0.04,
            geometry_types=['linear'],
            seen_geometry_types=['linear'],
            unseen_geometry_types=[],
            geometry_split='seen',
            linear_spacing_range=[0.04, 0.04],
            planar_spacing_range=[0.04, 0.04],
            random_aperture_range=[0.04, 0.04],
            sensor_position_error_std=0.0,
        ),
        # 6) Eight-channel but non-fixed: same OOD geometry stress test as
        #    ``2_4ch_unfixed`` while keeping all eight microphones valid.
        '6_8ch_unfixed': dict(
            n_mics=8, max_n_mics=8, min_n_mics=8, variable_n_mics=False,
            mic_spacing=0.04,
            geometry_types=['linear', 'sparse', 'random'],
            seen_geometry_types=['linear'],
            unseen_geometry_types=['sparse', 'random'],
            geometry_split='all',
            linear_spacing_range=[0.10, 0.20],
            planar_spacing_range=[0.10, 0.20],
            random_aperture_range=[0.30, 0.50],
            sensor_position_error_std=0.0,
        ),
        # 7) Variable channel count with strictly unseen geometry families,
        #    but no sensor-position perturbation.  This isolates geometry OOD
        #    from the 1 cm physical array jitter used by ``4_var_unmatch``.
        '7_var_unmatch_clean': dict(
            n_mics=8, max_n_mics=8, min_n_mics=2, variable_n_mics=True,
            random_seed=42,
            mic_spacing=0.04,
            geometry_types=['linear', 'planar', 'sparse', 'random'],
            seen_geometry_types=['linear', 'planar'],
            unseen_geometry_types=['sparse', 'random'],
            geometry_split='unseen',
            linear_spacing_range=[0.025, 0.08],
            planar_spacing_range=[0.025, 0.08],
            random_aperture_range=[0.05, 0.25],
            sensor_position_error_std=0.0,
        ),
    }


def _make_dataset(segment_len, num_samples, subset_kwargs):
    cfg = dict(COMMON)
    cfg.update(subset_kwargs)
    cfg['segment_len'] = segment_len
    cfg['num_samples'] = num_samples
    return LibriSpeechTSEDataset(**cfg), cfg


def _to_int16(x):
    """Float [-1,1] tensor -> int16 tensor (clipping to int16 range)."""
    return (x.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16)


def _save_audio_int16(path, x, sr):
    """Save a [C, T] or [T] float tensor as 16-bit PCM WAV."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    torchaudio.save(str(path), _to_int16(x), sr,
                    encoding='PCM_S', bits_per_sample=16)


def build_one_subset(out_dir, segment_len, num_samples, subset_kwargs,
                     overwrite=False, n_workers=0):
    out_dir = Path(out_dir)
    (out_dir / 'mixture').mkdir(parents=True, exist_ok=True)
    (out_dir / 'target').mkdir(parents=True, exist_ok=True)
    (out_dir / 'enrollment').mkdir(parents=True, exist_ok=True)
    (out_dir / 'meta').mkdir(parents=True, exist_ok=True)

    ds, cfg = _make_dataset(segment_len, num_samples, subset_kwargs)
    sr = cfg['sr']

    # Persist the exact subset config so the saved testset is reproducible.
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump({'config': cfg, 'num_samples': num_samples,
                   'segment_len': segment_len}, f, indent=2)

    pad = max(6, len(str(num_samples - 1)))
    pending_indices = [
        idx for idx in range(num_samples)
        if overwrite or not (
            out_dir / 'mixture' / f'{idx:0{pad}d}.wav').exists()
    ]
    pending_dataset = torch.utils.data.Subset(ds, pending_indices)
    loader = torch.utils.data.DataLoader(
        pending_dataset, batch_size=None, num_workers=n_workers,
        persistent_workers=n_workers > 0)
    iterator = zip(pending_indices, loader)
    for idx, sample in tqdm(iterator, total=len(pending_indices),
                            desc=str(out_dir.name), ncols=120):
        mix_path = out_dir / 'mixture' / f'{idx:0{pad}d}.wav'
        meta = sample['metadata']
        n_valid = int(meta['n_valid_mics'])

        # Strip zero-padded mics — audio only carries valid channels; the
        # loader re-pads to max_n_mics from mic_xyz length.
        mixture = sample['mixture'][:n_valid]
        target = sample['target'][:n_valid]
        enrollment = sample['enrollment']

        _save_audio_int16(mix_path, mixture, sr)
        _save_audio_int16(out_dir / 'target' / f'{idx:0{pad}d}.wav', target, sr)
        _save_audio_int16(out_dir / 'enrollment' / f'{idx:0{pad}d}.wav', enrollment, sr)
        with open(out_dir / 'meta' / f'{idx:0{pad}d}.json', 'w') as f:
            json.dump(meta, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_root', type=str,
                        default='data/Testset')
    parser.add_argument('--librispeech_root', default='data/LibriSpeech')
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--segments', nargs='+', type=float, default=[2.0, 6.0],
                        help='Segment lengths (seconds) to build.')
    parser.add_argument('--subsets', nargs='+', type=str, default=None,
                        help='Subset names to build (default: all subsets).')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--n_workers', type=int, default=0,
                        help='Parallel dataset-generation workers per subset.')
    args = parser.parse_args()
    COMMON['input_dir'] = args.librispeech_root

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    all_subsets = _subset_configs()
    selected = args.subsets or list(all_subsets.keys())
    for s in selected:
        if s not in all_subsets:
            raise ValueError(f"Unknown subset: {s}. Available: {list(all_subsets)}")

    for seg_len in args.segments:
        seg_tag = ('seg_%gs' % seg_len).replace('.0s', 's')  # 2.0 -> seg_2s
        for subset_name in selected:
            out_dir = Path(args.out_root) / seg_tag / subset_name
            logging.info("Building %s (segment_len=%gs) -> %s",
                         subset_name, seg_len, out_dir)
            build_one_subset(out_dir, seg_len, args.num_samples,
                             all_subsets[subset_name],
                             overwrite=args.overwrite,
                             n_workers=args.n_workers)
    logging.info("All requested testsets built.")


if __name__ == '__main__':
    main()
