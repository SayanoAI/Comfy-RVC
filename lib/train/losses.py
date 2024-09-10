import torch
import torchaudio.transforms as T
import torch.nn.functional as F

from ..utils import gc_collect
from ..infer_pack.commons import median_pool1d, minmax_scale

class LossBalancer:
    model: torch.nn.Module

    def __init__(self, model, initial_weights={}, historical_losses={}, ema_weights={}, epsilon=1e-8, weights_decay=0., loss_decay=.0, active=True, use_pareto=True, use_norm=False):
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

    def update_ema_weights(self, new_weights):
        """Updates the EMA of the weights."""
        if not self.ema_weights: self.ema_weights = new_weights
        else: self.ema_weights = {
                k: self.weights_decay * self.ema_weights.get(k, 0.) + (1 - self.weights_decay) * new_weights[k]
                for k in new_weights
            }
    
    def update_historical_losses(self, new_losses: dict):
        """Updates the EMA of the weights."""
        if not self.historical_losses: self.historical_losses = new_losses
        else: 
            for k in new_losses:
                self.historical_losses[k]=self.loss_decay*self.historical_losses.get(k, 0.) + (1 - self.loss_decay)*new_losses[k]
                

    def calculate_loss_slope(self, key, current_loss):
        """Calculates the slope of the loss using the current loss and its EMA."""
        ema_loss = self.historical_losses.get(key, current_loss)+self.epsilon
        slope = abs(current_loss-ema_loss)/ema_loss  # relative loss change
        return slope
    
    def calculate_gradients(self, key, current_loss):
        """Calculates the gradient norm of the current loss wrt the model params."""
        self.model.zero_grad()  # Clear previous gradients

        # Backward pass to output layer
        model_params = list(self.model.parameters())[-1]
        output_params = torch.autograd.grad(current_loss, model_params, grad_outputs=torch.ones_like(current_loss, device=model_params.device), retain_graph=True, only_inputs=True)[0]

        # Compute L2 gradient norm
        grad_norm = output_params.view(output_params.size(0),-1).norm(2, dim=-1)
        ema_loss = self.historical_losses.get(key, current_loss)+self.epsilon
        return grad_norm/ema_loss

    def pareto_normalizer(self, data: dict):
        # Sort the data in descending order based on the values
        sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate the number of elements in the top 20%
        top_20_count = max(int(0.2 * len(sorted_data)),1)
        
        # Calculate the weights
        weights = [0] * len(sorted_data)
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
        normalized_data = {sorted_data[i][0]: normalized_weights[i] for i in range(len(sorted_data))}
        
        return normalized_data

    def on_train_batch_start(self, losses: dict):
        """
        Balances the loss terms based on their gradients before each training batch.
        
        Args:
            losses (dict): Dictionary of individual loss terms (scalar tensors).
        
        Returns:
            balanced_loss (torch.Tensor): Weighted sum of losses.
            new_weights (dict): Updated weights for each loss term.
        """
        
        if not self.initial_weights: self.initial_weights = {k: 1.0 for k in losses} # initialize weights
        if not self.active: return sum(v*self.initial_weights[k] if k in self.initial_weights else v for k,v in losses)

        gradients = {}
        valid_losses = {}

        # Process each loss, skip if the corresponding weight is 0
        for key, loss in losses.items():
            weight = self.initial_weights[key] if key in self.initial_weights else 1.
            if weight == 0 or loss == 0 or not isinstance(loss,torch.Tensor): continue  # Skip loss with weight 0 or constants

            if self.use_norm:
                # Compute L2 gradient norm
                grad_norm = self.calculate_gradients(key, loss)
                gradients[key] = grad_norm.item()
            else:
                # Use loss plateau detection with EMA
                loss_slope = self.calculate_loss_slope(key, loss)
                gradients[key] = loss_slope.item()  # Smaller slope -> plateau -> lower priority
            valid_losses[key] = loss

        if not valid_losses: return torch.tensor(0.0)  # If all losses are skipped

        # Calculate loss weights based on gradient magnitudes
        if self.use_pareto: #20% hardest tasks gets 80% weight
            normalized_weights = self.pareto_normalizer(gradients)
        else:
            total_gradient = sum(gradients.values()) + self.epsilon
            normalized_weights = {k: w/total_gradient for k, w in gradients.items()}

        # Update EMA weights
        self.update_ema_weights(normalized_weights)
        balanced_loss = 0
        for k, loss in valid_losses.items():
            balanced_loss += self.ema_weights[k] * loss
            
            # update historical losses
            self.update_historical_losses({k: loss.item()})
        
        return balanced_loss

    def on_epoch_end(self, weights_decay=None, loss_decay=None):
        """
        Optional method to call at the end of each epoch for logging or further adjustments.
        """
        if weights_decay is not None: self.weights_decay=weights_decay
        if loss_decay is not None: self.loss_decay=loss_decay
        print(f"|| EMA loss weights: {self.ema_weights} ({self.weights_decay=})")
        print(f"|| EMA losses: {self.historical_losses} ({self.loss_decay=})")
        weighted_loss = sum(v*self.ema_weights.get(k,0.) for k,v in self.historical_losses.items())
        print(f"===> EMA loss: {weighted_loss:.3f} <====")
        gc_collect()


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
    numerator = (original_envelope * generated_envelope).sum(dim=-1)
    denominator = torch.sqrt((original_envelope ** 2).sum(dim=-1) * (generated_envelope ** 2).sum(dim=-1)+eps)
    correlation = numerator / denominator
    
    # Compute the loss as negative correlation
    loss = 1-correlation.mean()
    
    return loss

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
    return max_filtered_tensor.nan_to_num(0).sum(dim)

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
    phase = torch.angle(analytic_signal)

    return envelope, phase

def compute_harmonics(mag: torch.Tensor, harmonic_kernel_sizes=[11,17,23], percussive_kernel_sizes=[3,7,13], eps=1e-8):
    # Calculate log-magnitude spectrograms

    harmonic_list = []
    percussive_list = []

    for kernel_size in harmonic_kernel_sizes:
        harmonic_list.append(median_pool1d(mag, kernel_size=kernel_size, dim=-1).nan_to_num(0).view(mag.size(0),-1))

    for kernel_size in percussive_kernel_sizes:
        percussive_list.append(median_pool1d(mag, kernel_size=kernel_size, dim=-2).nan_to_num(0).view(mag.size(0),-1))

    # Concatenate the results
    harmonic = torch.cat(harmonic_list, dim=-1)
    percussive = torch.cat(percussive_list, dim=-1)

    # normalize values
    harmonic = minmax_scale(harmonic, eps=eps)
    percussive = minmax_scale(percussive, eps=eps)

    return harmonic, percussive

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
        harmonic_loss *= c_hd
    else: harmonic_loss = 0

    # temporal invariant phase
    if c_tsi>0:
        freq_tsi = compute_tsi_loss(org_mag,gen_mag,dim=-1, eps=eps)
        temp_tsi = compute_tsi_loss(org_mag,gen_mag,dim=-2, eps=eps)
        tsi_loss = (freq_tsi+temp_tsi) * c_tsi
    else: tsi_loss = 0

    # Temperol Envelope and Fine Structure Loss
    if c_tefs>0:
        # temporal envelope
        gen_te, gen_tfs = compute_tefs(generated_audio, eps=eps)
        org_te, org_tfs = compute_tefs(original_audio, eps=eps)
        phase_diff = torch.remainder(gen_tfs - org_tfs, 2 * torch.pi)

        tefs_loss = F.l1_loss(gen_te, org_te) +  phase_diff.sin().abs().mean() #F.l1_loss(gen_tfs, org_tfs)
        tefs_loss*=c_tefs
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
            only_inputs=True
        )[0]
        grad_norm = gradients.view(gradients.size(0), -1).norm(2, dim=-1)+eps
        loss += torch.log1p((grad_norm - 1) ** 2).mean()
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
