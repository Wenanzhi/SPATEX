import json
from pathlib import Path

import pytest
import torch

from pact import encode_azimuth, extract, load_model
from src.training.dataset import LibriSpeechTSEDataset
from src.training.network_TACDeFTAN2 import Net
from src.training.network_DeFTAN2 import aggregate_coattention_logits


def small_config():
    config = json.loads((Path(__file__).parents[1] / 'configs/archived/pact_full.json').read_text())
    config['model_params'].update(n_layers=1, emb_dim=8, att_dim=8, hidden_dim=16,
                                  n_head=2, dropout=0.0, tac_hidden_dim=8)
    return config


def test_public_encoding_matches_saved_dataset_for_all_angles():
    dataset = LibriSpeechTSEDataset.__new__(LibriSpeechTSEDataset)
    dataset.d_model, dataset.alpha = 40, 80
    for angle in list(range(360)) + [-721.3, -0.5, 360.5, 719.6]:
        expected = dataset._get_azim_vector_cycpos(angle)
        torch.testing.assert_close(encode_azimuth(angle), expected, atol=2e-6, rtol=2e-6)


def test_mean_consensus_ignores_invalid_microphone_logits():
    logits = torch.randn(2, 4, 1, 2, 5, 5)
    mask = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 1.]])
    expected = torch.stack([logits[0, :2].mean(0), logits[1].mean(0)])
    actual = aggregate_coattention_logits(logits, mask, 'mean')
    torch.testing.assert_close(actual, expected)
    logits[0, 2:] = 1e5
    torch.testing.assert_close(aggregate_coattention_logits(logits, mask, 'mean'), expected)


def test_checkpoint_and_enrollment_free_extraction_preserve_channel_contract(tmp_path):
    torch.manual_seed(4)
    config = small_config()
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    original = Net(**config['model_params']).eval()
    checkpoint = tmp_path / 'weights.pt'
    torch.save({'model_state_dict': original.state_dict()}, checkpoint)
    model = load_model(config_path, checkpoint)
    mixture = torch.randn(1, 4, 1024)
    xyz = torch.randn(1, 4, 3) * 0.04
    mask = torch.tensor([[1., 1., 0., 0.]])
    azimuth = torch.tensor([30.0])
    clue = encode_azimuth(azimuth)
    with torch.no_grad():
        expected = original(mixture, enrollment=torch.randn(1, 2048),
                            azim_vec=clue, mic_xyz=xyz, mic_mask=mask)
    actual = extract(model, mixture, xyz, azimuth, mask)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == mixture.shape
    assert actual[:, 2:].count_nonzero() == 0
    permutation = torch.tensor([2, 1, 3, 0])
    permuted = extract(model, mixture[:, permutation], xyz[:, permutation], azimuth, mask[:, permutation])
    torch.testing.assert_close(permuted, actual[:, permutation], atol=3e-5, rtol=3e-5)
    mixture[:, 2:] *= 1000
    xyz[:, 2:] += 1000
    torch.testing.assert_close(extract(model, mixture, xyz, azimuth, mask), actual, atol=3e-5, rtol=3e-5)
    # A removed parameter must fail strict loading rather than silently randomize it.
    broken = dict(original.state_dict())
    broken.pop(next(iter(broken)))
    torch.save(broken, checkpoint)
    with pytest.raises(RuntimeError):
        load_model(config_path, checkpoint)


@pytest.mark.parametrize('bad_mask', [[[0., 0.]], [[1., 0.]], [[1., 0.5]]])
def test_extract_rejects_unsupported_masks(bad_mask):
    model = Net(**small_config()['model_params']).eval()
    with pytest.raises(ValueError):
        extract(model, torch.randn(1, 2, 512), torch.zeros(1, 2, 3), 0., torch.tensor(bad_mask))
