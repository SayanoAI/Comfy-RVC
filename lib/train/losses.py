from collections import namedtuple
import functools
from typing import Callable, List, Optional
import typing
from librosa.filters import mel as librosa_mel_fn
import numpy as np
from scipy import signal
import torch
import torchaudio.transforms as T
import torch.nn.functional as F

from ..utils import gc_collect
from ..infer_pack.commons import compute_correlation, median_pool1d, minmax_scale

class LossBalancer:
    model: torch.nn.Module

    def __init__(self, model: torch.nn.Module, initial_weights: dict={}, historical_losses: dict={}, ema_weights: dict={}, epsilon=1e-8, weights_decay=0., loss_decay=.0, active=True, use_pareto=True, use_norm=False):
        """
        Initializes the LossBalancer with optional initial weights, partial gradient computation, and EMA.
        
        Args:
            model (torch.nn.Module): The model for which gradients will be computed.
            initial_weights (dict): Initial weights for each loss term.
            epsilon (float): A small value to avoid division by zero.
            partial (bool): Whether to compute gradients only for the output layer.
        """
        self.model = model
        self.epsilon = epsilon
        self.weights_decay = weights_decay
        self.loss_decay = loss_decay
        self.initial_weights = initial_weights
        self.ema_weights = ema_weights
        self.historical_losses = historical_losses
        self.active = active
        self.use_pareto = use_pareto # apply the 20/80 rule to loss balancing
        self.use_norm = use_norm

    def to_dict(self):
        return dict(
            epsilon = self.epsilon,
            weights_decay = self.weights_decay,
            loss_decay = self.loss_decay,
            ema_weights = self.ema_weights,
            initial_weights = self.initial_weights,
            historical_losses = self.historical_losses,
            active = self.active,
            use_pareto = self.use_pareto, # apply the 20/80 rule to loss balancing
            use_norm = self.use_norm
        )

    def update_ema_weights(self, new_weights: dict):
        """Updates the EMA of the weights."""
        if not self.ema_weights: self.ema_weights = dict(**new_weights)
        else: self.ema_weights = {
                k: self.weights_decay * self.ema_weights.get(k, 1.) + (1 - self.weights_decay) * new_weights[k]
                for k in new_weights
            }
        return dict(**self.ema_weights)
    
    def update_historical_losses(self, new_losses: dict):
        """Updates the EMA of the weights."""
        if not self.historical_losses: self.historical_losses = dict(**new_losses)
        else: 
            for k,v in new_losses.items():
                if np.nan_to_num(v,nan=-1)==-1: print(f"{k=} {v=}")
                self.historical_losses[k]=self.loss_decay*self.historical_losses.get(k, v) + (1 - self.loss_decay)*v
        return dict(**self.historical_losses)

    def calculate_loss_slope(self, key: str, current_loss: torch.Tensor):
        """Calculates the slope of the loss using the current loss and its EMA."""
        ema_loss = self.historical_losses.get(key, current_loss)+self.epsilon
        slope = (current_loss-ema_loss)/ema_loss  # relative loss change
        return slope
    
    def calculate_gradients(self, key: str, current_loss: torch.Tensor, input: torch.Tensor):
        """Calculates the gradient norm of the current loss wrt the model params."""
        self.model.zero_grad()  # Clear previous gradients
        input.requires_grad_(True)

        # Backward pass to output layer
        output_params = torch.autograd.grad(current_loss, [input], retain_graph=True, allow_unused=True, materialize_grads=True)[0]

        # Compute L2 gradient norm
        # grad_norm = output_params.view(output_params.size(0),-1).norm(2, dim=-1).mean()
        if output_params.ndim>1: grad_norm = output_params.view(output_params.size(0), -1).norm(2, dim=-1).mean()
        else: grad_norm = output_params.norm(2)

        slope = self.calculate_loss_slope(key, current_loss)
        return grad_norm * slope

    def pareto_normalizer(self, data: dict):
        # Sort the data in descending order based on the values
        sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate the number of elements in the top 20%
        top_20_count = max(int(0.2 * len(sorted_data)),1)
        
        # Calculate the weights
        weights = np.zeros(len(sorted_data))
        total_weight = 1.0
        top_20_weight = 0.8 * total_weight
        remaining_weight = total_weight - top_20_weight
        
        # Assign weights to the top 20%
        for i in range(top_20_count):
            weights[i] = top_20_weight / top_20_count
        
        # Assign weights to the remaining 80%
        for i in range(top_20_count, len(sorted_data)):
            weights[i] = remaining_weight / (len(sorted_data) - top_20_count)
        
        # Normalize the weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Combine the keys with their normalized weights
        normalized_data = {sorted_data[i][0]: normalized_weights[i]*len(data) for i in range(len(sorted_data))}
        
        return normalized_data

    def on_train_batch_start(self, losses: dict, input: Optional[torch.Tensor]=None):
        """
        Balances the loss terms based on their gradients before each training batch.
        
        Args:
            losses (dict): Dictionary of individual loss terms (scalar tensors).
        
        Returns:
            balanced_loss (torch.Tensor): Weighted sum of losses.
            new_weights (dict): Updated weights for each loss term.
        """
        
        if len(losses)==0: return 0. # no losses to balance
        if not self.initial_weights: self.initial_weights = {k: 1. for k in losses} # initialize weights
        if not self.ema_weights: self.ema_weights = {k: 1. for k in losses} # initialize ema weights
        if not self.active:
            self.update_historical_losses({k: v.item()*self.initial_weights.get(k,1.) for k,v in losses.items() if v>0})
            return sum(v*self.initial_weights.get(k,1.) for k,v in losses.items())

        gradients = {}
        valid_losses = {}

        # Process each loss, skip if the corresponding weight is 0
        for key, loss in losses.items():
            weight = self.initial_weights.get(key, 1.)
            if weight == 0 or loss == 0 or not isinstance(loss,torch.Tensor): continue  # Skip loss with weight 0 or constants
            weighted_loss = loss * weight

            if self.use_norm and input is not None:
                # Compute L2 gradient norm
                grad_norm = self.calculate_gradients(key, weighted_loss, input)
                gradients[key] = max(grad_norm.item(), self.epsilon)
            else:
                # Use loss plateau detection with EMA
                loss_slope = self.calculate_loss_slope(key, weighted_loss)
                gradients[key] = max(loss_slope.item(), self.epsilon)
            valid_losses[key] = weighted_loss.nan_to_num(self.epsilon)

        if not valid_losses: return torch.tensor(0.0)  # If all losses are skipped

        # Calculate loss weights based on gradient magnitudes
        if self.use_pareto: #20% hardest tasks gets 80% weight
            normalized_weights = self.pareto_normalizer(gradients)
        else:
            total_gradient = sum(gradients.values()) + self.epsilon
            normalized_weights = {k: w/total_gradient*len(gradients) for k, w in gradients.items()}

        # Update EMA weights
        self.update_ema_weights(normalized_weights)
        balanced_loss = 0
        for k, loss in valid_losses.items():
            balanced_loss += self.ema_weights.get(k, 1.) * loss
            
        # update historical losses
        self.update_historical_losses({k: loss.item() for k, loss in valid_losses.items()})
        
        return balanced_loss

    def on_epoch_end(self, weights_decay=None, loss_decay=None):
        """
        Optional method to call at the end of each epoch for logging or further adjustments.
        """
        if weights_decay is not None: self.weights_decay=weights_decay
        if loss_decay is not None: self.loss_decay=loss_decay
        weights = dict(sorted(self.ema_weights.items(),key=lambda x:x[1],reverse=True))
        losses = dict(sorted(self.historical_losses.items(),key=lambda x:x[1],reverse=True))
        print(f"|| EMA weights: {weights} ({self.weights_decay=})")
        print(f"|| EMA losses: {losses} ({self.loss_decay=})")
        print(f"===> Weighted EMA loss: {self.weighted_ema_loss:.3f} <====")
        gc_collect()

    @property
    def weighted_ema_loss(self):
        return sum(v*self.ema_weights.get(k,1.) for k,v in self.historical_losses.items())

def compute_tsi_loss(original_log_magnitude: torch.Tensor, generated_log_magnitude: torch.Tensor, dim=-1, eps=1e-8):
    """
    Computes the correlation loss between the original and generated log-magnitude spectrograms.
    :param original_envelope: Original log magnitude spectrogram (batch, time)
    :param generated_envelope: Generated log magnitude spectrogram (batch, time)
    :return: Correlation loss.
    """
    original_envelope = compute_envelope(original_log_magnitude, eps=eps, dim=dim)
    generated_envelope = compute_envelope(generated_log_magnitude, eps=eps, dim=dim)

    # Normalize the envelope
    original_envelope = minmax_scale(original_envelope, eps=eps)
    generated_envelope = minmax_scale(generated_envelope, eps=eps)
    
    # Compute the correlation
    correlation = compute_correlation(original_envelope, generated_envelope, eps=eps)
    
    # Compute the loss as negative correlation
    loss = 1-correlation.abs()
    
    return loss.mean()

def compute_envelope(log_magnitude: torch.Tensor, dim=-1, kernel_size=3, eps=1e-8):
    """
    Compute the envelope of the log-magnitude spectrum of an audio signal.

    Args:
        log_magnitude (torch.Tensor): The log-magnitude spectrum of the audio signal.
        dim (int): The dimension along which to average. Default is -1.
        kernel_size (int): The size of the max pooling kernel. Default is 3.

    Returns:
        torch.Tensor: The computed envelope of the log-magnitude spectrum.
    """

    # Normalize the pooled magnitude
    log_magnitude = F.normalize(log_magnitude, dim=dim, eps=eps)

    # Use max pooling to get the peaks
    max_filtered_tensor = F.max_pool1d(log_magnitude, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    # Replace NaNs with zeros and sum along the second-to-last dimension
    return max_filtered_tensor.nan_to_num(eps).sum(dim)

def compute_tefs(audio_signal: torch.Tensor, eps=1e-8):
    """
    Computes the Hilbert transform of a batched mono audio signal to extract the envelope.
    
    Args:
        audio_signal (torch.Tensor): Batched mono audio signal of shape (batch_size, signal_length).
        
    Returns:
        torch.Tensor: The envelope of the audio signal of shape (batch_size, signal_length).
    """
    # Perform FFT on the input signal
    audio_fft = torch.fft.fft(audio_signal.float(), dim=-1)

    # Create a filter to zero out negative frequencies
    signal_length = audio_signal.shape[-1]
    h = torch.zeros(signal_length, device=audio_signal.device)
    
    # First component is left untouched
    h[0] = 1
    
    if signal_length % 2 == 0:
        # For even signal length
        h[1:signal_length // 2] = 2
        h[signal_length // 2] = 1
    else:
        # For odd signal length
        h[1:(signal_length + 1) // 2] = 2

    # Apply the filter in the frequency domain
    hilbert_fft = audio_fft * h

    # Perform inverse FFT to obtain the analytic signal
    analytic_signal = torch.fft.ifft(hilbert_fft, dim=-1)

    # Extract the envelope by taking the magnitude of the analytic signal
    envelope = torch.abs(analytic_signal)

    # Normalize the envelope
    envelope = minmax_scale(envelope, eps=eps)

    # Compute the instantaneous phase from the analytic signal
    phase = torch.angle(analytic_signal).cos()

    return envelope.nan_to_num(eps), phase.nan_to_num(eps)

def compute_harmonics(mag: torch.Tensor, harmonic_kernel_sizes=[11,17,23], percussive_kernel_sizes=[3,7,13], eps=1e-8):
    # Calculate log-magnitude spectrograms

    harmonic_list = []
    percussive_list = []

    for kernel_size in harmonic_kernel_sizes:
        harmonic_list.append(median_pool1d(mag, kernel_size=kernel_size, dim=-1).nan_to_num(eps).view(mag.size(0),-1))

    for kernel_size in percussive_kernel_sizes:
        percussive_list.append(median_pool1d(mag, kernel_size=kernel_size, dim=-2).nan_to_num(eps).view(mag.size(0),-1))

    # Concatenate the results
    harmonic = torch.cat(harmonic_list, dim=-1)
    percussive = torch.cat(percussive_list, dim=-1)

    # normalize values
    harmonic = minmax_scale(harmonic, eps=eps)
    percussive = minmax_scale(percussive, eps=eps)

    return harmonic.nan_to_num(eps), percussive.nan_to_num(eps)

def combined_aux_loss(
        original_audio: torch.Tensor, generated_audio: torch.Tensor,
        c_tefs=1., c_hd=1., c_tsi=1., n_mels=128, sample_rate=40000,
        n_fft=1024, hop_length=320, win_length=1024,
        eps=None):

    kernel_size = n_fft//hop_length+1
    if kernel_size % 2 == 0: kernel_size+=1 #enforce odd kernel size
    if eps is None: eps = torch.finfo(original_audio.dtype).eps

    # Compute STFT once
    if c_hd+c_tsi>0:
        
        hann_window = torch.hann_window(win_length).to(
            dtype=original_audio.dtype, device=original_audio.device
        )
        generated_stft = torch.stft(
            generated_audio.view(-1,generated_audio.size(-1)),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=hann_window,
            return_complex=True,
            onesided=True,
            center=False)
        original_stft = torch.stft(
            original_audio.view(-1,original_audio.size(-1)),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=hann_window,
            return_complex=True,
            onesided=True,
            center=False)
        
        MelScaler = T.MelScale(n_mels=n_mels, sample_rate=sample_rate, n_stft=n_fft // 2 + 1).to(original_audio.device)
        org_mag = MelScaler(original_stft.abs()+eps)
        gen_mag = MelScaler(generated_stft.abs()+eps)
    
    # Harmonic Loss
    if c_hd>0:
        original_harmonics, original_percussives = compute_harmonics(org_mag, eps=eps)
        generated_harmonics, generated_percussives = compute_harmonics(gen_mag, eps=eps)
        # Define loss terms
        harmonic_loss = F.l1_loss(generated_harmonics, original_harmonics)
        harmonic_loss += F.l1_loss(generated_percussives, original_percussives)
    else: harmonic_loss = 0

    # temporal invariant phase
    if c_tsi>0:
        freq_tsi = compute_tsi_loss(org_mag,gen_mag,dim=-1, eps=eps)
        temp_tsi = compute_tsi_loss(org_mag,gen_mag,dim=-2, eps=eps)
        tsi_loss = (freq_tsi+temp_tsi)
    else: tsi_loss = 0

    # Temperol Envelope and Fine Structure Loss
    if c_tefs>0:
        # temporal envelope
        gen_te, gen_tfs = compute_tefs(generated_audio, eps=eps)
        org_te, org_tfs = compute_tefs(original_audio, eps=eps)
        correlation = 1-compute_correlation(gen_tfs,org_tfs, eps=eps).abs()

        tefs_loss = F.l1_loss(gen_te, org_te) + correlation.mean()
    else: tefs_loss = 0
    
    return harmonic_loss, tefs_loss, tsi_loss

def gradient_norm_loss(original_audio: torch.Tensor, generated_audio: torch.Tensor, net_d: torch.nn.Module, eps=1e-8):
    loss=0
    # Compute the gradient penalty
    # Randomly interpolate between real and generated data
    size = [1]*original_audio.ndim
    size[0] = original_audio.size(0)
    alpha = torch.rand(*size, device=original_audio.device)
    interpolated = alpha * original_audio + (1 - alpha) * generated_audio
    interpolated.requires_grad_(True)
    # Get the discriminator output for the interpolated data
    _, disc_interpolated_output, _, _ = net_d(original_audio, interpolated)
    # Compute gradients of discriminator output w.r.t. interpolated data
    for output in disc_interpolated_output:
        net_d.zero_grad()
        gradients = torch.autograd.grad(
            outputs=output,
            inputs=interpolated,
            grad_outputs=torch.ones_like(output, device=original_audio.device),
            retain_graph=True,
            allow_unused=True,
            materialize_grads=True
        )[0]
        if gradients.ndim>1: grad_norm = gradients.view(gradients.size(0), -1).norm(2, dim=-1).mean()
        else: grad_norm = gradients.norm(2)
        loss += torch.log1p((grad_norm - 1) ** 2)
    return loss/len(disc_interpolated_output)

# Adapted from https://github.com/NVIDIA/BigVGAN/blob/main/loss.py
# LICENSE: https://github.com/NVIDIA/BigVGAN/blob/main/LICENSE
class MultiScaleMelSpectrogramLoss(torch.nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [5, 10, 20, 40, 80, 160, 320],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [32, 64, 128, 256, 512, 1024, 2048]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 0.0 (no ampliciation on mag part)
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 1.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    Additional code copied and modified from https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
    """

    def __init__(
        self,
        sampling_rate: int,
        n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320],
        window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
        loss_fn: Callable = torch.nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 0.0,
        log_weight: float = 1.0,
        pow: float = 1.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = [0, 0, 0, 0, 0, 0, 0],
        mel_fmax: List[float] = [None, None, None, None, None, None, None],
        window_type: str = "hann",
    ):
        super().__init__()
        self.sampling_rate = sampling_rate

        STFTParams = namedtuple(
            "STFTParams",
            ["window_length", "hop_length", "window_type", "match_stride"],
        )

        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    @staticmethod
    @functools.lru_cache(None)
    def get_window(
        window_type,
        window_length,
    ):
        return signal.get_window(window_type, window_length)

    @staticmethod
    @functools.lru_cache(None)
    def get_mel_filters(sr, n_fft, n_mels, fmin, fmax):
        return librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def mel_spectrogram(
        self,
        wav,
        n_mels,
        fmin,
        fmax,
        window_length,
        hop_length,
        match_stride,
        window_type,
    ):
        """
        Mirrors AudioSignal.mel_spectrogram used by BigVGAN-v2 training from: 
        https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
        """
        B, C, T = wav.shape

        if match_stride:
            assert (
                hop_length == window_length // 4
            ), "For match_stride, hop must equal n_fft // 4"
            right_pad = np.ceil(T / hop_length) * hop_length - T
            pad = (window_length - hop_length) // 2
        else:
            right_pad = 0
            pad = 0

        wav = torch.nn.functional.pad(wav, (pad, pad + right_pad), mode="reflect")

        window = self.get_window(window_type, window_length)
        window = torch.from_numpy(window).to(wav.device).float()

        stft = torch.stft(
            wav.reshape(-1, T),
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        _, nf, nt = stft.shape
        stft = stft.reshape(B, C, nf, nt)
        if match_stride:
            """
            Drop first two and last two frames, which are added, because of padding. Now num_frames * hop_length = num_samples.
            """
            stft = stft[..., 2:-2]
        magnitude = torch.abs(stft)

        nf = magnitude.shape[2]
        mel_basis = self.get_mel_filters(
            self.sampling_rate, 2 * (nf - 1), n_mels, fmin, fmax
        )
        mel_basis = torch.from_numpy(mel_basis).to(wav.device)
        mel_spectrogram = magnitude.transpose(2, -1) @ mel_basis.T
        mel_spectrogram = mel_spectrogram.transpose(-1, 2)

        return mel_spectrogram

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : torch.Tensor
            Estimate signal
        y : torch.Tensor
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """

        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "n_mels": n_mels,
                "fmin": fmin,
                "fmax": fmax,
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "match_stride": s.match_stride,
                "window_type": s.window_type,
            }

            x_mels = self.mel_spectrogram(x, **kwargs)
            y_mels = self.mel_spectrogram(y, **kwargs)
            x_logmels = torch.log10(x_mels.pow(self.pow)+self.clamp_eps).nan_to_num(self.clamp_eps)
            y_logmels = torch.log10(y_mels.pow(self.pow)+self.clamp_eps).nan_to_num(self.clamp_eps)

            if self.log_weight!=0: loss += self.log_weight * self.loss_fn(x_logmels, y_logmels)
            if self.mag_weight!=0: loss += self.mag_weight * self.loss_fn(x_mels, y_mels)

        return loss
    
def feature_loss(fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss

def discriminator_loss(
        disc_real_outputs: List[torch.Tensor],
        disc_generated_outputs: List[torch.Tensor]
        ):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(disc_outputs: List[torch.Tensor]):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l
