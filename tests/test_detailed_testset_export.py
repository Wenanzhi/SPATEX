import json
from types import SimpleNamespace

import pandas as pd
import torch

from src.training.dataset import SavedTSETestset
from src.training import eval as eval_mod
from src.training.eval import (_append_per_sample_rows,
                               _atomic_write_per_sample_csv)
from src.training.run_testsets import (_build_arg_parser,
                                       _detailed_per_sample_path,
                                       _update_detailed_manifest)


def test_saved_testset_injects_stable_sample_metadata(tmp_path, monkeypatch):
    subset_path = tmp_path / 'seg_2s' / '7_var_unmatch_clean'
    for directory in ('mixture', 'target', 'enrollment', 'meta'):
        (subset_path / directory).mkdir(parents=True, exist_ok=True)
    for directory in ('mixture', 'target', 'enrollment'):
        (subset_path / directory / '000017.wav').touch()
    (subset_path / 'meta' / '000017.json').write_text(json.dumps({
        'n_valid_mics': 2,
        'azim_deg': 30.0,
        'mic_xyz': [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0]],
    }))
    (subset_path / 'meta.json').write_text(json.dumps({
        'config': {'max_n_mics': 2, 'd_model': 4},
    }))

    dataset = SavedTSETestset(str(subset_path))
    monkeypatch.setattr(
        dataset, '_load_audio',
        lambda path: (torch.ones(1, 16) if 'enrollment' in path
                      else torch.ones(2, 16)))
    metadata = dataset[0]['metadata']

    assert metadata['sample_id'] == '000017'
    assert metadata['scene_id'] == '000017'
    assert metadata['subset'] == '7_var_unmatch_clean'
    assert metadata['sample_key'] == '7_var_unmatch_clean/000017'


def test_per_sample_rows_keep_only_aligned_metrics_and_no_batch_loss():
    rows = []
    metadata = [
        {
            'sample_id': '000001', 'scene_id': '000001',
            'sample_key': '3_var_match/000001', 'subset': '3_var_match',
            'geometry_type': 'linear', 'seen_unseen_geometry': 'seen',
            'sensor_position_error_std': 0.0,
        },
        {
            'sample_id': '000002', 'scene_id': '000002',
            'sample_key': '3_var_match/000002', 'subset': '3_var_match',
            'geometry_type': 'planar', 'seen_unseen_geometry': 'seen',
            'sensor_position_error_std': 0.0,
        },
    ]
    _append_per_sample_rows(
        rows,
        {
            'scale_invariant_signal_noise_ratio_i': [1.25, 2.5],
            'delta_IPD_mean': torch.tensor([0.1, 0.2]),
            'batch_only_metric': [99.0],
            'loss': [3.0],
        },
        metadata, torch.tensor([[1.0, 1.0], [1.0, 0.0]]), 2, 0,
        {'experiment': 'model', 'segment': 'seg_2s'})

    assert [row['sample_id'] for row in rows] == ['000001', '000002']
    assert [row['n_valid_mics'] for row in rows] == [2, 1]
    assert [row['scale_invariant_signal_noise_ratio_i'] for row in rows] == [1.25, 2.5]
    assert all('loss' not in row for row in rows)
    assert all('batch_only_metric' not in row for row in rows)


def test_per_sample_csv_and_manifest_are_replaceable(tmp_path):
    csv_path = tmp_path / 'test_detailed' / 'seg_2s' / 'subset' / 'model' / 'per_sample.csv'
    _atomic_write_per_sample_csv([{'sample_id': 'old', 'metric': 1.0}], csv_path)
    _atomic_write_per_sample_csv([{'sample_id': 'new', 'metric': 2.0}], csv_path)
    frame = pd.read_csv(csv_path)
    assert frame.to_dict('records') == [{'sample_id': 'new', 'metric': 2.0}]

    manifest_path = tmp_path / 'test_detailed' / 'manifest.json'
    first = {
        'experiment': 'model', 'checkpoint': '/checkpoints/1.pt',
        'segment': 'seg_2s', 'subset': '1_4ch_fixed',
    }
    second = dict(first, checkpoint='/checkpoints/2.pt')
    _update_detailed_manifest(manifest_path, first)
    _update_detailed_manifest(manifest_path, second)
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest['entries']) == 1
    assert manifest['entries'][0]['checkpoint'] == '/checkpoints/2.pt'


def test_evaluate_optionally_writes_per_sample_csv(tmp_path, monkeypatch):
    samples = []
    for sample_id in ('000001', '000002'):
        samples.append({
            'mixture': torch.ones(2, 16),
            'target': torch.ones(2, 16),
            'enrollment': torch.ones(8),
            'azim_vec': torch.ones(4),
            'mic_mask': torch.ones(2),
            'mic_xyz': torch.zeros(2, 3),
            'metadata': {
                'sample_id': sample_id, 'scene_id': sample_id,
                'sample_key': f'1_4ch_fixed/{sample_id}',
                'subset': '1_4ch_fixed', 'n_valid_mics': 2,
                'geometry_type': 'linear',
                'seen_unseen_geometry': 'seen',
                'sensor_position_error_std': 0.0,
            },
        })

    class DummyModel(torch.nn.Module):
        def forward(self, mixed, target, enrollment, azim_vec,
                    mic_mask=None, mic_xyz=None):
            return mixed, mixed.new_tensor(4.0)

    def metrics(mixed, output, target, **kwargs):
        return {
            'scale_invariant_signal_noise_ratio_i': [1.0, 2.0],
            'delta_IPD_mean': [0.2, 0.3],
        }

    monkeypatch.setattr(eval_mod, 'SavedTSETestset', lambda **kwargs: samples)
    per_sample_path = tmp_path / 'per_sample.csv'
    args = SimpleNamespace(
        test_data={'saved_path': '/unused'}, use_cuda=False, gpu_ids=None,
        n_workers=0, eval_batch_size=2, model_params={}, pretrain_path='',
        exp_dir=str(tmp_path), base_metric='loss', n_items=None,
        profiling=False, per_sample_path=str(per_sample_path),
        per_sample_context={'experiment': 'dummy', 'segment': 'seg_2s'},
    )

    result = eval_mod.evaluate(
        SimpleNamespace(Net=DummyModel, metrics=metrics), args)
    frame = pd.read_csv(per_sample_path)

    assert result['loss'] == 4.0
    assert frame['sample_id'].tolist() == [1, 2]
    assert frame['scale_invariant_signal_noise_ratio_i'].tolist() == [1.0, 2.0]
    assert 'loss' not in frame.columns


def test_cli_accepts_new_subsets_and_detailed_output_path():
    args = _build_arg_parser().parse_args([
        'tac_deftan2_mimo_tse/method2_coatt_tac', '--detailed',
        '--subsets', '7_var_unmatch_clean',
    ])
    assert args.detailed is True
    assert args.subsets == ['7_var_unmatch_clean']
    path = _detailed_per_sample_path(
        '/tmp/test_detailed', 'seg_2s', args.subsets[0],
        args.experiments[0])
    assert str(path).endswith(
        'seg_2s/7_var_unmatch_clean/tac_deftan2_mimo_tse/'
        'method2_coatt_tac/per_sample.csv')
