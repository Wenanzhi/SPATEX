"""Regression test for AMP GradScaler checkpoint persistence."""

import torch

from src.helpers import utils


class _StatefulScaler:
    def __init__(self, scale):
        self.scale = float(scale)

    def state_dict(self):
        return {"scale": self.scale, "growth_tracker": 17}

    def load_state_dict(self, state_dict):
        self.scale = float(state_dict["scale"])


def test_checkpoint_round_trip_restores_grad_scaler(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    source_model = torch.nn.Linear(2, 1)
    source_scaler = _StatefulScaler(scale=512.0)

    utils.save_checkpoint(
        str(checkpoint), 3, source_model,
        train_metrics={}, val_metrics={}, scaler=source_scaler)

    target_model = torch.nn.Linear(2, 1)
    target_scaler = _StatefulScaler(scale=65536.0)
    epoch, train_metrics, val_metrics = utils.load_checkpoint(
        str(checkpoint), target_model, scaler=target_scaler)

    assert epoch == 3
    assert train_metrics == {}
    assert val_metrics == {}
    assert target_scaler.scale == 512.0
