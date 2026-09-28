"""Named release checkpoints work with the paper DOA evaluation entrypoint."""
import torch

from src.training.eval_doa_preservation import _resolve_checkpoint


def test_named_release_checkpoint_uses_stored_epoch(tmp_path):
    checkpoint = tmp_path / 'spatex_variable.pt'
    torch.save({'epoch': 85, 'model_state_dict': {}, 'val_metrics': {}}, checkpoint)
    path, epoch, value = _resolve_checkpoint(tmp_path, 'si_snri', str(checkpoint))
    assert path == checkpoint.resolve()
    assert epoch == 85
    assert value is None


def test_named_checkpoint_without_epoch_does_not_require_numeric_filename(tmp_path):
    checkpoint = tmp_path / 'weights.pt'
    torch.save({'model_state_dict': {}}, checkpoint)
    path, epoch, value = _resolve_checkpoint(tmp_path, 'si_snri', str(checkpoint))
    assert path == checkpoint.resolve()
    assert epoch == -1
    assert value is None
