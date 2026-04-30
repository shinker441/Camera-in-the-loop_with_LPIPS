"""
Neural Holography - CITL with MSE + LPIPS:

Utility functions. Modified from the original neural-holography repository to use
MSE + λ × LPIPS as the combined optimisation loss, and LPIPS/SSIM/PSNR as evaluation metrics.

Original paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training.
ACM TOG (SIGGRAPH Asia), 2020.

LPIPS paper:
R. Zhang, P. Isola, A. Efros, E. Shechtman, O. Wang. The Unreasonable Effectiveness of Deep Features
as a Perceptual Metric. CVPR, 2018.

This code is released under CC BY-NC 4.0. Non-commercial use only.
"""

import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.modules.loss as ll

import lpips
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------------------------
# Combined loss: MSE + λ × LPIPS
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """Combined MSE + λ × LPIPS loss for hologram phase optimisation.

    MSE provides stable pixel-level gradients that guarantee convergence.
    LPIPS guides the optimisation toward perceptually better quality.
    Setting lambda_lpips=0 recovers pure MSE (useful for ablation studies).

    Accepts amplitude tensors in [0, 1] range with shape (N, C, H, W).
    Single-channel inputs are repeated to 3 channels for LPIPS.
    LPIPS network parameters are frozen (requires_grad=False).

    After each forward() call, the component values are available as:
        self.last_mse   — MSE term (detached scalar tensor)
        self.last_lpips — LPIPS term (detached scalar tensor)

    Args:
        net:          LPIPS backbone. 'vgg' recommended for training loss;
                      'alex' is faster and better for evaluation only.
        lambda_lpips: Weight for the LPIPS term (default 0.1).
                      Set to 0 for pure MSE.
    """

    def __init__(self, net='vgg', lambda_lpips=0.1):
        super().__init__()
        self.lambda_lpips = lambda_lpips
        self.mse_loss = nn.MSELoss()
        self.lpips_net = lpips.LPIPS(net=net)
        for param in self.lpips_net.parameters():
            param.requires_grad = False
        self.last_mse = None
        self.last_lpips = None

    def forward(self, input, target):
        mse_val = self.mse_loss(input, target)

        if self.lambda_lpips > 0:
            input_3ch = self._to_3ch(input)
            target_3ch = self._to_3ch(target)
            lpips_val = self.lpips_net(input_3ch, target_3ch).mean()
        else:
            lpips_val = torch.zeros(1, device=input.device)

        self.last_mse = mse_val.detach()
        self.last_lpips = lpips_val.detach()

        return mse_val + self.lambda_lpips * lpips_val

    @staticmethod
    def _to_3ch(amp):
        """Scale [0,1] → [-1,1] and repeat to 3 channels for LPIPS."""
        x = amp.clamp(0.0, 1.0) * 2.0 - 1.0
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x


def _prep_amp_for_lpips(amp):
    """Ensure a tensor is (N, C, H, W), scale [0,1]→[-1,1], expand to 3 ch."""
    x = amp.clone()
    while x.dim() < 4:
        x = x.unsqueeze(0)
    x = x.clamp(0.0, 1.0) * 2.0 - 1.0
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    return x


def _numpy_to_lpips_tensor(arr, device):
    """Convert a numpy amplitude array to a 4-D LPIPS-ready tensor on *device*."""
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    if t.ndim == 2:                       # (H, W)
        t = t.unsqueeze(0).unsqueeze(0)   # → (1, 1, H, W)
    elif t.ndim == 3:                     # (H, W, C)
        t = t.permute(2, 0, 1).unsqueeze(0)  # → (1, C, H, W)
    t = t.to(device)
    t = t.clamp(0.0, 1.0) * 2.0 - 1.0
    if t.shape[1] == 1:
        t = t.expand(-1, 3, -1, -1)
    return t


# ---------------------------------------------------------------------------
# Complex-field utilities (unchanged from original)
# ---------------------------------------------------------------------------

def mul_complex(t1, t2):
    a, b = t1.split(1, 4)
    c, d = t2.split(1, 4)
    return torch.cat((a * c - b * d, b * c + a * d), 4)


def div_complex(t1, t2):
    (a, b) = t1.split(1, 4)
    (c, d) = t2.split(1, 4)
    mag = torch.mul(c, c) + torch.mul(d, d)
    return torch.cat(((a * c + b * d) / mag, (b * c - a * d) / mag), 4)


def reciprocal_complex(t):
    (a, b) = t.split(1, 4)
    mag = torch.mul(a, a) + torch.mul(b, b)
    return torch.cat((a / mag, -(b / mag)), 4)


def rect_to_polar(real, imag):
    mag = torch.pow(real**2 + imag**2, 0.5)
    ang = torch.atan2(imag, real)
    return mag, ang


def polar_to_rect(mag, ang):
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag


def replace_amplitude(field, amplitude):
    real, imag = polar_to_rect(amplitude, field.angle())
    return torch.complex(real, imag)


def ifftshift(tensor):
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, -math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, -math.floor(size[3] / 2.0), 3)
    return tensor_shifted


def fftshift(tensor):
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, math.floor(size[3] / 2.0), 3)
    return tensor_shifted


def ifft2(tensor_re, tensor_im, shift=False):
    tensor_out = torch.stack((tensor_re, tensor_im), 4)
    if shift:
        tensor_out = ifftshift(tensor_out)
    (tensor_out_re, tensor_out_im) = torch.ifft(tensor_out, 2, True).split(1, 4)
    tensor_out_re = tensor_out_re.squeeze(4)
    tensor_out_im = tensor_out_im.squeeze(4)
    return tensor_out_re, tensor_out_im


def fft2(tensor_re, tensor_im, shift=False):
    (tensor_out_re, tensor_out_im) = torch.fft(torch.stack((tensor_re, tensor_im), 4), 2, True).split(1, 4)
    tensor_out_re = tensor_out_re.squeeze(4)
    tensor_out_im = tensor_out_im.squeeze(4)
    if shift:
        tensor_out_re = fftshift(tensor_out_re)
        tensor_out_im = fftshift(tensor_out_im)
    return tensor_out_re, tensor_out_im


def roll_torch(tensor, shift, axis):
    if shift == 0:
        return tensor
    if axis < 0:
        axis += tensor.dim()
    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)
    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)


def pad_stacked_complex(field, pad_width, padval=0, mode='constant'):
    if padval == 0:
        pad_width = (0, 0, *pad_width)
        return nn.functional.pad(field, pad_width, mode=mode)
    else:
        if isinstance(padval, torch.Tensor):
            padval = padval.item()
        real, imag = field[..., 0], field[..., 1]
        real = nn.functional.pad(real, pad_width, mode=mode, value=padval)
        imag = nn.functional.pad(imag, pad_width, mode=mode, value=0)
        return torch.stack((real, imag), -1)


def pad_image(field, target_shape, pytorch=True, stacked_complex=True, padval=0, mode='constant'):
    if pytorch:
        if stacked_complex:
            size_diff = np.array(target_shape) - np.array(field.shape[-3:-1])
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(target_shape) - np.array(field.shape[-2:])
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(target_shape) - np.array(field.shape[-2:])
        odd_dim = np.array(field.shape[-2:]) % 2

    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        if pytorch:
            pad_axes = [int(p)
                        for tple in zip(pad_front[::-1], pad_end[::-1])
                        for p in tple]
            if stacked_complex:
                return pad_stacked_complex(field, pad_axes, mode=mode, padval=padval)
            else:
                return nn.functional.pad(field, pad_axes, mode=mode, value=padval)
        else:
            leading_dims = field.ndim - 2
            if leading_dims > 0:
                pad_front = np.concatenate(([0] * leading_dims, pad_front))
                pad_end = np.concatenate(([0] * leading_dims, pad_end))
            return np.pad(field, tuple(zip(pad_front, pad_end)), mode,
                          constant_values=padval)
    else:
        return field


def crop_image(field, target_shape, pytorch=True, stacked_complex=True):
    if target_shape is None:
        return field

    if pytorch:
        if stacked_complex:
            size_diff = np.array(field.shape[-3:-1]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
        odd_dim = np.array(field.shape[-2:]) % 2

    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        if pytorch and stacked_complex:
            return field[(..., *crop_slices, slice(None))]
        else:
            return field[(..., *crop_slices)]
    else:
        return field


# ---------------------------------------------------------------------------
# Color-space helpers (unchanged from original)
# ---------------------------------------------------------------------------

def srgb_gamma2lin(im_in):
    thresh = 0.04045
    im_out = np.where(im_in <= thresh, im_in / 12.92, ((im_in + 0.055) / 1.055)**(2.4))
    return im_out


def srgb_lin2gamma(im_in):
    thresh = 0.0031308
    im_out = np.where(im_in <= thresh, 12.92 * im_in, 1.055 * (im_in**(1 / 2.4)) - 0.055)
    return im_out


# ---------------------------------------------------------------------------
# Misc utilities (unchanged from original)
# ---------------------------------------------------------------------------

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def phasemap_8bit(phasemap, inverted=True):
    output_phase = ((phasemap + np.pi) % (2 * np.pi)) / (2 * np.pi)
    if inverted:
        phase_out_8bit = ((1 - output_phase) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8)
    else:
        phase_out_8bit = ((output_phase) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8)
    return phase_out_8bit


def burst_img_processor(img_burst_list):
    img_tensor = np.stack(img_burst_list, axis=0)
    img_avg = np.mean(img_tensor, axis=0)
    return im2float(img_avg)


def im2float(im, dtype=np.float32):
    if issubclass(im.dtype.type, np.floating):
        return im.astype(dtype)
    elif issubclass(im.dtype.type, np.integer):
        return im / dtype(np.iinfo(im.dtype).max)
    else:
        raise ValueError(f'Unsupported data type {im.dtype}')


def propagate_field(input_field, propagator, prop_dist=0.2, wavelength=520e-9, feature_size=(6.4e-6, 6.4e-6),
                    prop_model='ASM', dtype=torch.float32, precomputed_H=None):
    if prop_model == 'ASM':
        output_field = propagator(u_in=input_field, z=prop_dist, feature_size=feature_size, wavelength=wavelength,
                                  dtype=dtype, precomped_H=precomputed_H)
    elif 'MODEL' in prop_model.upper():
        _, input_phase = rect_to_polar(input_field.real, input_field.imag)
        output_field = propagator(input_phase)
    elif prop_model == 'CAMERA':
        _, input_phase = rect_to_polar(input_field.real, input_field.imag)
        output_field = propagator(input_phase)
    else:
        raise ValueError('Unexpected prop_model value')
    return output_field


# ---------------------------------------------------------------------------
# TensorBoard summary functions — PSNR replaced with LPIPS
# ---------------------------------------------------------------------------

def write_sgd_summary(slm_phase, out_amp, target_amp, k,
                      writer=None, path=None, s=0., prefix='test', lpips_fn=None):
    """TensorBoard summary for SGD. Uses LPIPS instead of PSNR.

    Args:
        slm_phase: SLM phase tensor (unused here, kept for API compatibility).
        out_amp:   Reconstructed amplitude tensor.
        target_amp: Target amplitude tensor.
        k:         Current iteration index.
        writer:    SummaryWriter instance.
        path:      Path for saving images (optional).
        s:         Current amplitude scale factor.
        prefix:    Tag prefix for TensorBoard.
        lpips_fn:  Pre-initialised lpips.LPIPS instance. Created internally if None.
    """
    device = out_amp.device

    if lpips_fn is None:
        lpips_fn = lpips.LPIPS(net='alex').to(device)

    loss_mse = nn.MSELoss().to(device)
    loss_value = loss_mse(s * out_amp, target_amp)

    ssim_value = ssim(
        target_amp.squeeze().cpu().detach().numpy(),
        (s * out_amp).squeeze().cpu().detach().numpy(),
        data_range=1.0
    )

    # LPIPS — current scale s
    with torch.no_grad():
        lpips_value = lpips_fn(
            _prep_amp_for_lpips(s * out_amp),
            _prep_amp_for_lpips(target_amp)
        ).mean().item()

    # LPIPS — least-squares optimal scale s_min
    s_min = (target_amp * out_amp).mean() / (out_amp ** 2).mean()
    ssim_value_min = ssim(
        target_amp.squeeze().cpu().detach().numpy(),
        (s_min * out_amp).squeeze().cpu().detach().numpy(),
        data_range=1.0
    )
    with torch.no_grad():
        lpips_value_min = lpips_fn(
            _prep_amp_for_lpips(s_min * out_amp),
            _prep_amp_for_lpips(target_amp)
        ).mean().item()

    if writer is not None:
        writer.add_image(f'{prefix}_Recon/amp', (s * out_amp).squeeze(0), k)
        writer.add_scalar(f'{prefix}_loss', loss_value, k)
        writer.add_scalar(f'{prefix}_lpips', lpips_value, k)
        writer.add_scalar(f'{prefix}_ssim', ssim_value, k)
        writer.add_scalar(f'{prefix}_lpips/scaled', lpips_value_min, k)
        writer.add_scalar(f'{prefix}_ssim/scaled', ssim_value_min, k)
        writer.add_scalar(f'{prefix}_scalar', s, k)


def write_gs_summary(slm_field, recon_field, target_amp, k, writer,
                     roi=(880, 1600), prefix='test', lpips_fn=None):
    """TensorBoard summary for GS. Uses LPIPS instead of PSNR."""
    recon_amp = recon_field.abs()
    device = recon_amp.device

    if lpips_fn is None:
        lpips_fn = lpips.LPIPS(net='alex').to(device)

    loss_mse = nn.MSELoss().to(device)

    recon_amp = crop_image(recon_amp, target_shape=roi, stacked_complex=False)
    target_amp = crop_image(target_amp, target_shape=roi, stacked_complex=False)

    recon_amp *= (torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True)
                  / torch.sum(recon_amp * recon_amp, (-2, -1), keepdim=True))

    loss_value = loss_mse(recon_amp, target_amp)

    ssim_value = ssim(
        target_amp.squeeze().cpu().detach().numpy(),
        recon_amp.squeeze().cpu().detach().numpy(),
        data_range=1.0
    )

    with torch.no_grad():
        lpips_value = lpips_fn(
            _prep_amp_for_lpips(recon_amp),
            _prep_amp_for_lpips(target_amp)
        ).mean().item()

    if writer is not None:
        writer.add_image(f'{prefix}_Recon/amp', recon_amp.squeeze(0), k)
        writer.add_scalar(f'{prefix}_loss', loss_value, k)
        writer.add_scalar(f'{prefix}_lpips', lpips_value, k)
        writer.add_scalar(f'{prefix}_ssim', ssim_value, k)


# ---------------------------------------------------------------------------
# Evaluation metric: LPIPS + SSIM across three colour-space domains
# ---------------------------------------------------------------------------

def get_lpips_ssim(recon_amp, target_amp, lpips_fn=None, device='cpu', multichannel=False):
    """Compute LPIPS and SSIM across amplitude, linear, and sRGB domains.

    Replaces the original get_psnr_ssim(). LPIPS is a *distance* — lower is better.

    Args:
        recon_amp:   Numpy array, reconstructed amplitude in [0, 1].
        target_amp:  Numpy array, reference amplitude in [0, 1].
        lpips_fn:    Pre-initialised lpips.LPIPS instance (optional).
        device:      Torch device string, used when lpips_fn is None.
        multichannel: Passed to skimage SSIM for RGB images.

    Returns:
        lpips_vals: dict with keys 'amp', 'lin', 'srgb'  (lower = better)
        ssims:      dict with keys 'amp', 'lin', 'srgb'  (higher = better)
    """
    if lpips_fn is None:
        lpips_fn = lpips.LPIPS(net='alex').to(device)

    lpips_vals, ssims = {}, {}

    def _lpips(tgt_np, rec_np):
        t = _numpy_to_lpips_tensor(tgt_np, device)
        r = _numpy_to_lpips_tensor(rec_np, device)
        with torch.no_grad():
            return lpips_fn(r, t).mean().item()

    # Amplitude domain
    lpips_vals['amp'] = _lpips(target_amp, recon_amp)
    ssims['amp'] = ssim(target_amp, recon_amp, multichannel=multichannel, data_range=1.0
)

    # Linear (intensity) domain
    target_linear = target_amp ** 2
    recon_linear = recon_amp ** 2
    lpips_vals['lin'] = _lpips(target_linear, recon_linear)
    ssims['lin'] = ssim(target_linear, recon_linear, multichannel=multichannel, data_range=1.0
)

    # sRGB (gamma-corrected) domain
    target_srgb = srgb_lin2gamma(np.clip(target_linear, 0.0, 1.0))
    recon_srgb = srgb_lin2gamma(np.clip(recon_linear, 0.0, 1.0))
    lpips_vals['srgb'] = _lpips(target_srgb, recon_srgb)
    ssims['srgb'] = ssim(target_srgb, recon_srgb, multichannel=multichannel, data_range=1.0
)

    return lpips_vals, ssims


# ---------------------------------------------------------------------------
# Remaining utilities (unchanged from original)
# ---------------------------------------------------------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def make_kernel_gaussian(sigma, kernel_size):
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2
    variance = sigma ** 2

    gaussian_kernel = ((1 / (2 * math.pi * variance))
                       * torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1)
                                   / (2 * variance)))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    return gaussian_kernel


def quantized_phase(phasemap):
    phasemap = (phasemap + np.pi) / (2 * np.pi)
    phasemap = torch.round(255 * phasemap)
    phasemap = phasemap / 255 * 2 * np.pi - np.pi
    return phasemap
