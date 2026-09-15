"""A collection of useful helper functions"""

import os
import logging
import json

import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import pandas as pd
# from torchmetrics.functional import(
#     scale_invariant_signal_noise_ratio as si_snr,
#     signal_noise_ratio as snr,
#     signal_distortion_ratio as sdr,
#     scale_invariant_signal_distortion_ratio as si_sdr)
try:
    from torchmetrics.functional import mean_squared_error as mse
except ImportError:
    def mse(preds, target):
        return torch.mean((preds - target) ** 2)
import matplotlib.pyplot as plt

from torch import Tensor
import torch.fft as fft



class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

def save_graph(train_metrics, test_metrics, save_dir, epochs):
    metrics = [signal_noise_ratio, scale_invariant_signal_noise_ratio]
    results = {'train_loss': train_metrics['loss'],
               'test_loss' : test_metrics['loss']}

    for m_fn in metrics:
        name = m_fn.__name__
        if name in train_metrics:
            results["train_" + name] = train_metrics[name]
        if name in test_metrics:
            results["test_" + name] = test_metrics[name]

    results_pd = pd.DataFrame(results)

    results_pd.to_csv(os.path.join(save_dir, 'results.csv'))

    fig, temp_ax = plt.subplots(1, 3, figsize=(15, 5))
    axs = []
    for i in temp_ax:
        axs.append(i)

    x = range(len(train_metrics['loss']))
    axs[0].plot(x, train_metrics['loss'], label='train')
    axs[0].plot(x, test_metrics['loss'], label='test')
    axs[0].set(ylabel='Loss')
    axs[0].set(xlabel='Epoch')
    axs[0].set_title('loss',fontweight='bold')
    axs[0].legend()
    axs[0].set_xlim([0, epochs])

    for i in range(len(metrics)):
        name = metrics[i].__name__
        if name in train_metrics:
            axs[i+1].plot(x, train_metrics[name], label='train')
        if name in test_metrics:
            axs[i+1].plot(x, test_metrics[name], label='test')
        axs[i+1].set(xlabel='Epoch')
        axs[i+1].set_title(name, fontweight='bold')
        axs[i+1].legend()
        axs[i+1].set_xlim([0, epochs])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'results.png'))
    plt.close(fig)

def set_logger(log_path):

    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

def load_checkpoint(checkpoint, model, optim=None, lr_sched=None,
                    data_parallel=False, scaler=None):
    """Loads model parameters (state_dict) from file_path.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        data_parallel: (bool) if the model is a data parallel model
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))

    state_dict = torch.load(checkpoint, map_location='cpu')

    if data_parallel:
        state_dict['model_state_dict'] = {
            'module.' + k: state_dict['model_state_dict'][k]
            for k in state_dict['model_state_dict'].keys()}
    model.load_state_dict(state_dict['model_state_dict'])

    if optim is not None:
        optim.load_state_dict(state_dict['optim_state_dict'])

    if lr_sched is not None:
        lr_sched.load_state_dict(state_dict['lr_sched_state_dict'])

    scaler_state_dict = state_dict.get('scaler_state_dict')
    if scaler is not None and scaler_state_dict is not None:
        scaler.load_state_dict(scaler_state_dict)

    return state_dict['epoch'], state_dict['train_metrics'], \
           state_dict['val_metrics']

def save_checkpoint(checkpoint, epoch, model, optim=None, lr_sched=None,
                    train_metrics=None, val_metrics=None, data_parallel=False,
                    scaler=None):
    """Saves model parameters (state_dict) to file_path.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        data_parallel: (bool) if the model is a data parallel model
    """
    if os.path.exists(checkpoint):
        raise("File already exists {}".format(checkpoint))

    model_state_dict = model.state_dict()
    if data_parallel:
        model_state_dict = {
            k.partition('module.')[2]:
            model_state_dict[k] for k in model_state_dict.keys()}

    optim_state_dict = None if not optim else optim.state_dict()
    lr_sched_state_dict = None if not lr_sched else lr_sched.state_dict()
    scaler_state_dict = None if scaler is None else scaler.state_dict()

    state_dict = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optim_state_dict': optim_state_dict,
        'lr_sched_state_dict': lr_sched_state_dict,
        'scaler_state_dict': scaler_state_dict,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }

    torch.save(state_dict, checkpoint)

def model_size(model):
    """
    Returns size of the `model` in millions of parameters.
    """
    num_train_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    return num_train_params / 1e6

def run_time(model, inputs, profiling=False):
    """
    Returns runtime of a model in ms.
    """
    # Warmup
    for _ in range(100):
        output = model(*inputs)

    with profile(activities=[ProfilerActivity.CPU],
                 record_shapes=True) as prof:
        with record_function("model_inference"):
            output = model(*inputs)

    # Print profiling results
    if profiling:
        print(prof.key_averages().table(sort_by="self_cpu_time_total",
                                        row_limit=20))

    # Return runtime in ms
    return prof.profiler.self_cpu_time_total / 1000

def format_lr_info(optimizer):
    lr_info = ""
    for i, pg in enumerate(optimizer.param_groups):
        lr_info += " {group %d: params=%.5fM lr=%.1E}" % (
            i, sum([p.numel() for p in pg['params']]) / (1024 ** 2), pg['lr'])
    return lr_info



def signal_noise_ratio(preds: Tensor, target: Tensor, zero_mean: bool = False) -> Tensor:
    """
    Calculates `Signal-to-noise ratio`_ (SNR_) metric 
    for evaluating quality of audio. It is defined as:

    .. math::
        \text{SNR} = \frac{P_{signal}}{P_{noise}}

    where  :math:`P` denotes the power of each signal. The SNR metric 
    compares the level of the desired signal to the level of background noise. 
    Therefore, a high value of SNR means that the audio is clear.

    The number of microphones is also considered for multi-channel inputs.

    Args:
        preds: float tensor with shape ``(batch, num_mic, time)``
        target: float tensor with shape ``(batch, num_mic, time)``
        zero_mean: if to zero mean target and preds or not

    Returns:
        Float tensor with shape ``(batch,)`` of SNR values per sample
    """
    if preds.ndim == 2: preds = preds.unsqueeze(0)
    if target.ndim == 2: target = target.unsqueeze(0)

    eps = torch.finfo(preds.dtype).eps

    if (preds.shape[-2] > 1) and (target.shape[-2] == 1):
        preds = preds[:, 0].unsqueeze(-2) # reference channel = 1st channel

    if zero_mean:
        target = target - torch.mean(target, dim=(-1, -2), keepdim=True)
        preds = preds - torch.mean(preds, dim=(-1, -2), keepdim=True)

    noise = target - preds

    numer = torch.sum(torch.sum(target**2, dim=-1), dim=-1) + eps
    denom = torch.sum(torch.sum(noise**2, dim=-1), dim=-1) + eps

    snr_value = 10 * torch.log10(numer / denom)

    return snr_value

def scale_invariant_signal_noise_ratio(preds: Tensor, target: Tensor, zero_mean: bool = False) -> Tensor:
    """
    `Scale-invariant signal-to-distortion ratio`_ (SI-SDR_ = SI-SNR_) 
    The SI-SDR value is in general considered an overall
    measure of how good a source sound.

    The number of microphones is also considered for multi-channel inputs.

    Args:
        preds: float tensor with shape ``(batch, num_mic, time)``
        target: float tensor with shape ``(batch, num_mic, time)``
        zero_mean: If to zero mean target and preds or not

    Returns:
        Float tensor with shape ``(batch,)`` of SDR values per sample
    """
    if preds.ndim == 2: preds = preds.unsqueeze(0)
    if target.ndim == 2: target = target.unsqueeze(0)

    M = preds.shape[1]

    eps = torch.finfo(preds.dtype).eps

    if (preds.shape[-2] > 1) and (target.shape[-2] == 1):
        preds = preds[:, 0].unsqueeze(-2) # reference channel = 1st channel

    if zero_mean:
        target = target - torch.mean(target, dim=(-1, -2), keepdim=True)
        preds = preds - torch.mean(preds, dim=(-1, -2), keepdim=True)
    
    preds_list =  [preds[:, i] for i in range(M)];  preds_list = torch.cat(preds_list, dim=-1)
    target_list = [target[:, i] for i in range(M)]; target_list = torch.cat(target_list, dim=-1)

    alpha = (torch.sum(preds_list * target_list, dim=-1, keepdim=True) + eps) / (
          torch.sum(target_list**2, dim=-1, keepdim=True) + eps)
    target_scaled = alpha.unsqueeze(-1) * target

    noise = target_scaled - preds

    numer = torch.sum(torch.sum(target_scaled**2, dim=-1), dim=-1) + eps
    denom = torch.sum(torch.sum(noise**2, dim=-1), dim=-1) + eps

    si_snr_value = 10 * torch.log10(numer / denom)

    return si_snr_value

def phase_constrained_magnitude(mixture: Tensor, preds: Tensor, target: Tensor, win, hop) -> Tensor:
    """
    Calculates the phase magnitude loss averaged by the number of microphones

    Args:
        mixture: float tensor with shape ``(batch, num_mic, time)``
        preds: float tensor with shape ``(batch, num_mic, time)``
        target: float tensor with shape ``(batch, num_mic, time)``
        win, hop: any values for STFT

    Returns:
        Float tensor with shape ``(batch,)`` of PCM values per sample
    """
    if mixture.ndim == 2: mixture = mixture.unsqueeze(0)
    if preds.ndim == 2: preds = preds.unsqueeze(0)
    if target.ndim == 2: target = target.unsqueeze(0)

    def sm_loss(pred, tgt):
        pred_real, pred_imag = pred[:, :, :, 0], pred[:, :, :, 1]
        tgt_real, tgt_imag = tgt[:, :, :, 0], tgt[:, :, :, 1]

        pred_value = torch.abs(pred_real) + torch.abs(pred_imag)
        tgt_value = torch.abs(tgt_real) + torch.abs(tgt_imag)

        value = torch.mean((torch.abs(tgt_value - pred_value)), dim=(-1, -2))

        return value

    B, M, T = mixture.shape
    if M > 1 and target.shape[1] == 1:
        mixture = mixture[:, 0, :].unsqueeze(-2)

    stft_mixture = torch.stft(mixture.view([-1, T]), 
                              n_fft=win, hop_length=hop, return_complex=False) # B*M, F, L, 2
    stft_preds = torch.stft(preds.view([-1, T]), 
                            n_fft=win, hop_length=hop, return_complex=False)   # B*M, F, L, 2
    stft_target = torch.stft(target.view([-1, T]), 
                             n_fft=win, hop_length=hop, return_complex=False)  # B*M, F, L, 2

    signal_value = sm_loss(stft_preds, stft_target)
    noise_value = sm_loss(stft_mixture - stft_preds, stft_mixture - stft_target)
    pcm_value = 0.5 * signal_value + 0.5 * noise_value
    if M > 1 and target.shape[1] == 1:
        pcm_value = pcm_value
    else:
        pcm_value = pcm_value.view(B, M).mean(dim=-1)

    return pcm_value #.mean() # averaged by batch_size



def compute_ild(sig1, sig2):
    """
    Compute the Interaural Level Difference (ILD) between two signals.

    Args:
        sig1: float tensor with shape ``(batch, time)``
        sig2: float tensor with shape ``(batch, time)``

    Returns:
        Float tensor with shape ``(batch,)`` of ILD values per sample
    """
    eps = torch.finfo(sig1.dtype).eps

    sig1_energy = torch.norm(sig1, dim=-1) ** 2
    sig2_energy = torch.norm(sig2, dim=-1) ** 2
    ild = 10 * torch.log10(sig1_energy / (sig2_energy + eps))
    return ild

def compute_ipd(sig1, sig2, win, hop):
    """
    Compute the Interaural Phase Difference (IPD) between two signals.

    Args:
        sig1: float tensor with shape ``(batch, time)``
        sig2: float tensor with shape ``(batch, time)``
        win, hop: window size and hop size for STFT

    Returns:
        Float tensor with shape ``(batch, FREQ, TIME)`` of IPD values per sample
    """
    stft1 = torch.stft(sig1, n_fft=win, hop_length=hop, return_complex=True)
    stft2 = torch.stft(sig2, n_fft=win, hop_length=hop, return_complex=True)
    ipd = stft1 * torch.conj(stft2)
    return ipd

def compute_itd_cc(sig1, sig2, fs, max_delay_ms=1.0):
    """
    Compute the Interaural Time Difference (ITD) between two signals using cross-correlation directly.
    Limiting the delay to a maximum of max_delay_ms milliseconds.

    Args:
        sig1: float tensor with shape ``(batch, time)``
        sig2: float tensor with shape ``(batch, time)``
        fs: sampling frequency
        max_delay_ms: a scalar limiting the predicted delay to be in a given range

    Returns:
        Float tensor with shape ``(batch,)`` of ITD values per sample
    """
    n = sig1.shape[-1]
    
    tau_list = []
    
    for b in range(sig1.shape[0]):
        cc = F.conv1d(sig1[b].unsqueeze(0).unsqueeze(1), sig2[b].unsqueeze(0).unsqueeze(1), padding=n).squeeze(0).squeeze(1)
        
        max_shift = int(fs * max_delay_ms / 1e3)
        
        mid_point = cc.shape[-1] // 2
        cc = cc[:, mid_point - max_shift: mid_point + max_shift + 1]
        
        shift = torch.argmax(cc, dim=-1) - max_shift
        tau = shift / float(fs)

        tau_list.append(tau * 1e6)
    
    return torch.stack(tau_list, dim=0).squeeze(1)

def compute_itd_gccphat(sig1, sig2, fs, max_delay_ms=1.0):
    """
    Compute the Interaural Time Difference (ITD) between two signals using GCC-PHAT.
    Limiting the delay to a maximum of max_delay_ms milliseconds.

    Args:
        sig1: float tensor with shape ``(batch, time)``
        sig2: float tensor with shape ``(batch, time)``
        fs: sampling frequency
        max_delay_ms: a scalar limiting the predicted delay to be in a given range

    Returns:
        Float tensor with shape ``(batch,)`` of ITD values per sample
    """
    def next_power_of_two(x):
        return 1 if x == 0 else 2**(x - 1).bit_length()
    
    eps = torch.finfo(sig1.dtype).eps

    n = sig1.shape[-1] + sig2.shape[-1]
    n_padded = next_power_of_two(n)
    
    SIG1 = torch.fft.fft(sig1, n=n_padded, dim=-1)
    SIG2 = torch.fft.fft(sig2, n=n_padded, dim=-1)
    
    R = SIG1 * torch.conj(SIG2)
    R = R / (torch.abs(R) + eps)
    
    cc = torch.fft.irfft(R, n=n_padded, dim=-1)
    
    max_shift = int(fs * max_delay_ms / 1e3)
    
    cc = torch.cat((cc[:, -max_shift:], cc[:, :max_shift+1]), dim=-1)
    
    shift = torch.argmax(cc, dim=-1) - max_shift
    tau = shift / float(fs)
    
    return tau * 1e6


def delta_ILD(preds, target, idx1, idx2):
    """
    Compute ILD for all pairs of 4-channel signals and compute the changes
    between reference (target) and extracted (preds) signals.

    Args:
        preds: float tensor with shape ``(batch, channel, time)``
        target: float tensor with shape ``(batch, channel, time)``
        idx1, idx2: channel indexes of a pair

    Returns:
        Float tensor with shape ``(batch,)`` of ILD values per sample
    """
    if preds.ndim == 2: preds = preds.unsqueeze(0)
    if target.ndim == 2: target = target.unsqueeze(0)

    def delta_ild(preds1, preds2, target1, target2):
        ild_pred = compute_ild(preds1, preds2)
        ild_tgt = compute_ild(target1, target2)
        return torch.abs(ild_pred - ild_tgt)

    delta_ild_list = [delta_ild(preds[:, idx1], preds[:, idx2], target[:, idx1], target[:, idx2]),
                      delta_ild(preds[:, idx2], preds[:, idx1], target[:, idx2], target[:, idx1])]
    
    return torch.mean(torch.stack(delta_ild_list, dim=-1), dim=-1)

def delta_IPD(preds, target, win, hop, idx1, idx2):
    """
    Compute IPD for all pairs of 4-channel signals and compute the changes
    between reference (target) and extracted (preds) signals.

    Args:
        preds: float tensor with shape ``(batch, channel, time)``
        target: float tensor with shape ``(batch, channel, time)``

    Returns:
        Float tensor with shape ``(batch,)`` of IPD values per sample
    """
    if preds.ndim == 2: preds = preds.unsqueeze(0)
    if target.ndim == 2: target = target.unsqueeze(0)

    def delta_ipd(preds1, preds2, target1, target2):
        ipd_pred = compute_ipd(preds1, preds2, win, hop)
        ipd_tgt = compute_ipd(target1, target2, win, hop)
        return torch.mean(torch.abs(torch.angle(ipd_pred * torch.conj(ipd_tgt))), dim=(-1, -2))

    return delta_ipd(preds[:, idx1], preds[:, idx2], target[:, idx1], target[:, idx2])

def delta_ITD_cc(preds, target, fs, idx1, idx2):
    """
    Compute ITD for all pairs of 4-channel signals and compute the changes
    between reference (target) and extracted (preds) signals using simple cross-correlation.

    Args:
        preds: float tensor with shape ``(batch, channel, time)``
        target: float tensor with shape ``(batch, channel, time)``
        idx1, idx2: channel indexes of a pair

    Returns:
        Float tensor with shape ``(batch,)`` of ITD values per sample
    """
    if preds.ndim == 2: preds = preds.unsqueeze(0)
    if target.ndim == 2: target = target.unsqueeze(0)

    def delta_itd_cc(preds1, preds2, target1, target2):
        itd_pred = compute_itd_cc(preds1, preds2, fs)
        itd_tgt = compute_itd_cc(target1, target2, fs)
        return torch.abs(itd_pred - itd_tgt)

    return delta_itd_cc(preds[:, idx1], preds[:, idx2], target[:, idx1], target[:, idx2])

def delta_ITD_gccphat(preds, target, fs, idx1, idx2):
    """
    Compute ITD for all pairs of 4-channel signals and compute the changes
    between reference (target) and extracted (preds) signals using GCC-PHAT.

    Args:
        preds: float tensor with shape ``(batch, channel, time)``
        target: float tensor with shape ``(batch, channel, time)``
        idx1, idx2: channel indexes of a pair

    Returns:
        Float tensor with shape ``(batch,)`` of ITD values per sample
    """
    if preds.ndim == 2: preds = preds.unsqueeze(0)
    if target.ndim == 2: target = target.unsqueeze(0)

    def delta_itd_gccphat(preds1, preds2, target1, target2):
        itd_pred = compute_itd_gccphat(preds1, preds2, fs)
        itd_tgt = compute_itd_gccphat(target1, target2, fs)
        return torch.abs(itd_pred - itd_tgt)

    return delta_itd_gccphat(preds[:, idx1], preds[:, idx2], target[:, idx1], target[:, idx2])
