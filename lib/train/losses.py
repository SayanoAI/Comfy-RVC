import torch
import torchaudio

def mfcc_loss(original_audio, generated_audio, sample_rate, n_mfcc=13, **kwargs):
    """
    Computes the loss between the original and generated audio based on MFCCs.
    
    Parameters:
    original_audio (Tensor): The original audio waveform.
    generated_audio (Tensor): The generated audio waveform.
    sample_rate (int): The sample rate of the audio.
    n_mfcc (int): Number of MFCC coefficients to compute (default is 13).
    
    Returns:
    Tensor: The computed MFCC loss.
    """
    
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate,n_mfcc=n_mfcc,melkwargs=kwargs)
    mfcc_transform.to(original_audio.device)

    # Compute MFCCs for original and generated audio
    original_mfcc = mfcc_transform(original_audio)
    generated_mfcc = mfcc_transform(generated_audio)
    
    # Compute the L2 loss between the MFCCs of the original and generated audio
    loss = torch.nn.functional.smooth_l1_loss(generated_mfcc, original_mfcc)
    
    return loss

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

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
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
