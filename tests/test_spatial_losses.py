import math

import pytest
import torch

from src.training.network_TACDeFTAN2 import Net
from src.training.spatial_losses import (
    _ipd_from_stft,
    masked_complex_coherence_loss,
    masked_ipd_loss,
    spatial_fidelity_losses,
)


def test_identity_losses_are_zero_and_coherence_is_scale_invariant():
    torch.manual_seed(20)
    target = torch.randn(2, 4, 512)
    mic_mask = torch.tensor([
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ])

    ipd = masked_ipd_loss(
        target, target, mic_mask, 64, 32)
    coherence = masked_complex_coherence_loss(
        3.0 * target, target, mic_mask, 64, 32,
        smoothing_frames=5)

    assert ipd.item() < 1.0e-6
    assert coherence.item() < 1.0e-5


def test_ipd_wraparound_is_continuous():
    delta = 1.0e-3
    target_phase = math.pi - delta
    pred_phase = -math.pi + delta
    target = torch.ones(1, 2, 1, 1, dtype=torch.complex64)
    pred = torch.ones(1, 2, 1, 1, dtype=torch.complex64)
    target[0, 0, 0, 0] = torch.polar(
        torch.tensor(1.0), torch.tensor(target_phase))
    pred[0, 0, 0, 0] = torch.polar(
        torch.tensor(1.0), torch.tensor(pred_phase))

    loss = _ipd_from_stft(
        pred, target, torch.ones(1, 2),
        energy_floor_db=-40.0, eps=1.0e-6)

    assert loss.item() < 1.0e-5


def test_silence_and_near_silence_have_finite_gradients():
    torch.manual_seed(21)
    pred = (torch.randn(2, 4, 256) * 1.0e-12).requires_grad_()
    target = torch.zeros_like(pred)
    mic_mask = torch.tensor([
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ])

    terms = spatial_fidelity_losses(
        pred, target, mic_mask, 64, 32,
        compute_ipd=True,
        compute_coherence=True,
        smoothing_frames=5,
    )
    loss = terms["ipd_loss"] + terms["coherence_loss"]
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(pred.grad).all()


def test_padding_and_permutation_do_not_change_losses():
    torch.manual_seed(22)
    pred = torch.randn(1, 5, 512)
    target = torch.randn(1, 5, 512)
    mic_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
    noisy_pred = pred.clone()
    noisy_target = target.clone()
    noisy_pred[:, 3:] = torch.randn_like(noisy_pred[:, 3:]) * 1000
    noisy_target[:, 3:] = torch.randn_like(noisy_target[:, 3:]) * 1000
    permutation = torch.tensor([2, 4, 0, 3, 1])

    base = spatial_fidelity_losses(
        pred, target, mic_mask, 64, 32,
        smoothing_frames=5)
    padded = spatial_fidelity_losses(
        noisy_pred, noisy_target, mic_mask, 64, 32,
        smoothing_frames=5)
    permuted = spatial_fidelity_losses(
        noisy_pred[:, permutation],
        noisy_target[:, permutation],
        mic_mask[:, permutation],
        64,
        32,
        smoothing_frames=5,
    )

    for name in ("ipd_loss", "coherence_loss"):
        assert torch.allclose(base[name], padded[name], atol=1.0e-6)
        assert torch.allclose(base[name], permuted[name], atol=1.0e-6)


def test_single_valid_microphone_returns_differentiable_zero():
    pred = torch.randn(1, 4, 256, requires_grad=True)
    target = torch.randn_like(pred)
    mic_mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    terms = spatial_fidelity_losses(
        pred, target, mic_mask, 64, 32)
    loss = terms["ipd_loss"] + terms["coherence_loss"]
    loss.backward()

    assert loss.item() == 0.0
    assert torch.isfinite(pred.grad).all()


def test_invalid_coherence_smoothing_is_rejected():
    signal = torch.randn(1, 2, 256)
    with pytest.raises(ValueError, match="positive odd"):
        masked_complex_coherence_loss(
            signal, signal, torch.ones(1, 2),
            64, 32, smoothing_frames=4)


def test_network_combines_both_losses_with_finite_backward():
    torch.manual_seed(24)
    model = Net(
        max_n_mics=3,
        n_srcs=3,
        win=64,
        n_layers=1,
        hidden_dim=16,
        n_head=1,
        emb_dim=8,
        att_dim=8,
        angle_dim=40,
        spk_emb_dim=16,
        geom_emb_dim=4,
        tac_hidden_dim=8,
        act_loss_weight=0.0,
        dropout=0.0,
        cue_mode="doa_only",
        ipd_loss_weight=0.15,
        coherence_loss_weight=0.03,
    )
    mixture = torch.randn(1, 3, 256)
    target = torch.randn_like(mixture)
    mic_mask = torch.tensor([[1.0, 1.0, 0.0]])
    mic_xyz = torch.randn(1, 3, 3) * mic_mask[:, :, None]
    azim_vec = torch.randn(1, 40)
    mixture = mixture * mic_mask[:, :, None]
    target = target * mic_mask[:, :, None]

    output, loss = model(
        mixture,
        target,
        enrollment=None,
        azim_vec=azim_vec,
        mic_mask=mic_mask,
        mic_xyz=mic_xyz,
    )
    loss.backward()

    assert output.shape == target.shape
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_legacy_ild_weight_is_rejected():
    with pytest.raises(ValueError, match="discarded legacy ILD"):
        Net(
            max_n_mics=2,
            n_srcs=2,
            win=64,
            n_layers=1,
            hidden_dim=16,
            n_head=1,
            emb_dim=8,
            att_dim=8,
            angle_dim=40,
            spk_emb_dim=16,
            spatial_loss_weight=0.1,
        )
