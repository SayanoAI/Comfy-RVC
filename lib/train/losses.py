import torch
import torchaudio
import torch.nn.functional as F
from ..infer_pack.commons import median_pool1d

def log1p_autocorrelation(log_magnitude, max_lag=None):
    """
    Computes the autocorrelation of log-magnitude spectrogram.
    :param log_magnitude: Log magnitude spectrogram (batch, channels, time)
    :param max_lag: Maximum lag for which to compute autocorrelation.
    :return: Autocorrelation of the log-magnitude spectrogram.
    """
    if max_lag is None:
        max_lag = log_magnitude.size(-1) // 2  # Default to half the signal length

    # Ensure the signal is zero-mean
    log_magnitude = log_magnitude - log_magnitude.mean(dim=-1, keepdim=True)
    
    # Compute autocorrelation using FFT-based convolution
    padded_log_magnitude = F.pad(log_magnitude, (0, max_lag))
    # autocorr = torch.corrcoef(padded_log_magnitude)
    autocorr = F.conv1d(padded_log_magnitude, log_magnitude.flip(dims=[-1]))
    
    # Take only the relevant part and normalize
    autocorr = F.normalize(autocorr[:, :max_lag + 1],p=2,dim=-1)
    
    return autocorr.nan_to_num(0)

def compute_phase_loss(original_log_magnitude, generated_log_magnitude, max_lag=None):
    """
    Computes a time-shift invariant loss based on autocorrelation for log-magnitude data.
    :param original_log_magnitude: Original log-magnitude spectrogram (batch, channels, time)
    :param generated_log_magnitude: Generated log-magnitude spectrogram (batch, channels, time)
    :param max_lag: Maximum lag for which to compute autocorrelation.
    :return: Loss value.
    """
    # Compute autocorrelations
    original_autocorr = log1p_autocorrelation(original_log_magnitude, max_lag)
    generated_autocorr = log1p_autocorrelation(generated_log_magnitude, max_lag)

    # Compute the loss between autocorrelations
    loss = F.smooth_l1_loss(generated_autocorr, original_autocorr)
    
    return loss
    
def compute_temporal_envelope(log_magnitude: torch.Tensor, kernel_size=31):
    """Compute the temporal envelope loss using median pooling and second-order differences."""
    
    # Apply median pooling to the log-magnitude spectrogram
    pooled_stft = median_pool1d(log_magnitude, kernel_size, stride=kernel_size // 2 + 1)
    
    # Compute second-order differences on the median-pooled audio
    diff = torch.diff(pooled_stft, dim=-1, n=2)
    
    return diff.nan_to_num(0)

def compute_cepstrals(log_magnitude: torch.Tensor, kernel_size=31):

    pooled_magnitude = median_pool1d(log_magnitude, kernel_size, stride=kernel_size//2+1)

    # Compute the cepstrum
    cepstral_coefficients = torch.fft.ifft(pooled_magnitude, dim=-2).abs()
    
    return cepstral_coefficients.sum(dim=-2).nan_to_num(0)

def compute_sts_loss(gen_stft: torch.Tensor, org_stft: torch.Tensor, kernel_size=31):
    
    # B?N?T
    org_mag = torch.log1p(org_stft.abs())
    gen_mag = torch.log1p(gen_stft.abs())
    org_temp = torch.log1p(org_stft.abs().sum(-2,keepdim=True))
    gen_temp = torch.log1p(gen_stft.abs().sum(-2,keepdim=True))

    # temporal invariant phase
    phase_loss = compute_phase_loss(org_temp,gen_temp)

    # temporal envelope
    gen_temp = compute_temporal_envelope(gen_mag, kernel_size=kernel_size)
    org_temp = compute_temporal_envelope(org_mag, kernel_size=kernel_size)
    temporal_loss = F.smooth_l1_loss(gen_temp, org_temp)

    # spectral
    gen_spec = compute_cepstrals(gen_mag,kernel_size=kernel_size)
    org_spec = compute_cepstrals(org_mag,kernel_size=kernel_size)
    spectral_loss = F.smooth_l1_loss(gen_spec, org_spec)

    return temporal_loss, spectral_loss, phase_loss

def compute_harmonics(stft_matrix: torch.Tensor, kernel_size=31):
    # Calculate log-magnitude spectrograms
    mag = torch.log1p(stft_matrix.abs())
    
    # Apply median filtering to separate harmonics and percussives
    harmonic = median_pool1d(mag, kernel_size, stride=kernel_size//2+1)
    percussive = median_pool1d(mag.transpose(-2,-1), kernel_size, stride=kernel_size//2+1).transpose(-2,-1)
    
    return harmonic.nan_to_num(0), percussive.nan_to_num(0)

def combined_aux_loss(
        original_audio: torch.Tensor, generated_audio: torch.Tensor, sample_rate: int,
        c_mfcc=1., c_lfcc=1., c_hd=1., c_sts=1.,
        n_fft=1024, hop_length=320, win_length=1024,
        n_mfcc=13, n_lfcc=13, n_filter=80,
        f_min=0., f_max=None):

    kernel_size = n_fft//hop_length+1
    if kernel_size % 2 == 0: kernel_size+=1

    # Compute STFT once
    if c_hd+c_sts>0:
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
    
    # Harmonic Loss
    if c_hd>0:
        original_harmonics, original_percussives = compute_harmonics(original_stft,kernel_size)
        generated_harmonics, generated_percussives = compute_harmonics(generated_stft,kernel_size)
        # Define loss terms
        harmonic_loss = F.smooth_l1_loss(generated_harmonics, original_harmonics)
        harmonic_loss += F.smooth_l1_loss(generated_percussives, original_percussives)
        harmonic_loss *= c_hd
        harmonic_loss
    else: harmonic_loss = 0

    # Spectral Temporal Smoothness Loss
    if c_sts>0:
        temporal_loss, spectral_loss, phase_loss = compute_sts_loss(generated_stft,original_stft,kernel_size)
        temporal_loss *= c_sts
        spectral_loss *= c_sts
        phase_loss *= c_sts
    else: spectral_loss = temporal_loss = phase_loss = 0

    # MFCC Loss
    if c_mfcc>0:
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=n_mfcc, log_mels=True,
            melkwargs=dict(n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_filter, f_min=f_min, f_max=f_max)
        ).to(original_audio.device)
        original_mfcc = mfcc_transform(original_audio)
        generated_mfcc = mfcc_transform(generated_audio)
        mfcc_loss = F.smooth_l1_loss(generated_mfcc, original_mfcc) * c_mfcc
    else: mfcc_loss = 0

    # LFCC Loss
    if c_lfcc>0:
        lfcc_transform = torchaudio.transforms.LFCC(
            sample_rate=sample_rate, n_lfcc=n_lfcc, n_filter=n_filter, f_min=f_min, f_max=f_max,log_lf=True,
            speckwargs=dict(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        ).to(original_audio.device)
        original_lfcc = lfcc_transform(original_audio)
        generated_lfcc = lfcc_transform(generated_audio)
        lfcc_loss = F.smooth_l1_loss(generated_lfcc, original_lfcc) * c_lfcc
    else: lfcc_loss = 0
    
    return harmonic_loss, temporal_loss, spectral_loss, mfcc_loss, lfcc_loss, phase_loss

def gradient_norm_loss(original_audio: torch.Tensor, generated_audio: torch.Tensor, net_d):
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
        gradient = torch.autograd.grad(
            outputs=output,
            inputs=interpolated,
            grad_outputs=torch.ones(output.size(), device=original_audio.device),
            create_graph=True,
            only_inputs=True
        )[0]
        gradient = gradient.view(gradient.size(0), -1)
        loss += torch.log(gradient.norm(2,dim=-1)).abs().mean() # force gradnorm=1
    return loss/len(disc_interpolated_output)

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    # loss /= len(disc_generated_outputs) #average
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    # loss /= len(disc_outputs) #average
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
