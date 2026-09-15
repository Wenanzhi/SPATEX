import torch

from src.training.network_TACDeFTAN2 import Net, metrics


def tiny_model():
    return Net(max_n_mics=4, n_srcs=4, win=64, n_layers=1, hidden_dim=16,
               n_head=1, emb_dim=8, att_dim=8, angle_dim=40, spk_emb_dim=16,
               use_masked_tac=True, tac_insert='every_block',
               use_geometry_embedding=True, preserve_geometry_scale=True,
               geom_emb_dim=4, tac_hidden_dim=8, act_loss_weight=0.0).eval()


def test_tac_deftan2_variable_mic_forward_loss_metrics():
    torch.manual_seed(0)
    model = tiny_model()
    batch, mmax, t = 2, 4, 256
    mixture = torch.randn(batch, mmax, t)
    target = torch.randn(batch, mmax, t)
    mic_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    mixture = mixture * mic_mask[:, :, None]
    target = target * mic_mask[:, :, None]
    mic_xyz = torch.randn(batch, mmax, 3) * mic_mask[:, :, None]
    enrollment = torch.randn(batch, 512)
    azim_vec = torch.randn(batch, 40)

    with torch.no_grad():
        output, loss = model(mixture, target, enrollment, azim_vec,
                             mic_mask=mic_mask, mic_xyz=mic_xyz)
    assert output.shape == (batch, mmax, t)
    assert torch.isfinite(loss)
    assert torch.allclose(output * (1.0 - mic_mask[:, :, None]), torch.zeros_like(output), atol=1e-6)
    metric_batch = metrics(mixture, output, target, mic_mask=mic_mask, mic_xyz=mic_xyz)
    assert 'scale_invariant_signal_noise_ratio_i' in metric_batch
    assert metric_batch['n_valid_mics'] == [2.0, 4.0]
    assert len(metric_batch['delta_ILD_mean']) == batch


def test_tac_deftan2_padding_invariance_for_invalid_channels():
    torch.manual_seed(1)
    model = tiny_model()
    b, mmax, valid, t = 1, 4, 2, 256
    mixture = torch.zeros(b, mmax, t)
    target = torch.zeros(b, mmax, t)
    mic_xyz = torch.zeros(b, mmax, 3)
    mic_mask = torch.zeros(b, mmax)
    mic_mask[:, :valid] = 1.0
    mixture[:, :valid] = torch.randn(b, valid, t)
    target[:, :valid] = torch.randn(b, valid, t)
    mic_xyz[:, :valid] = torch.randn(b, valid, 3)
    noisy_pad = mixture.clone()
    noisy_pad[:, valid:] = torch.randn(b, mmax - valid, t) * 100.0
    noisy_xyz = mic_xyz.clone()
    noisy_xyz[:, valid:] = torch.randn(b, mmax - valid, 3) * 100.0
    enrollment = torch.randn(b, 512)
    azim_vec = torch.randn(b, 40)

    with torch.no_grad():
        out, _ = model(mixture, target, enrollment, azim_vec, mic_mask=mic_mask, mic_xyz=mic_xyz)
        out_pad, _ = model(noisy_pad, target, enrollment, azim_vec, mic_mask=mic_mask, mic_xyz=noisy_xyz)
    assert torch.allclose(out[:, :valid], out_pad[:, :valid], atol=1e-4, rtol=1e-4)


def test_tac_deftan2_permutation_equivariance_eval_mode():
    torch.manual_seed(2)
    model = tiny_model()
    b, m, t = 1, 4, 256
    mixture = torch.randn(b, m, t)
    target = torch.randn(b, m, t)
    mic_mask = torch.ones(b, m)
    mic_xyz = torch.randn(b, m, 3)
    enrollment = torch.randn(b, 512)
    azim_vec = torch.randn(b, 40)
    perm = torch.tensor([2, 0, 3, 1])

    with torch.no_grad():
        out, _ = model(mixture, target, enrollment, azim_vec, mic_mask=mic_mask, mic_xyz=mic_xyz)
        out_p, _ = model(mixture[:, perm], target[:, perm], enrollment, azim_vec,
                         mic_mask=mic_mask[:, perm], mic_xyz=mic_xyz[:, perm])
    assert torch.allclose(out[:, perm], out_p, atol=1e-4, rtol=1e-4)
