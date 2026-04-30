"""
Neural Holography - CITL with LPIPS:

Evaluation script. Modified from the original neural-holography repository to use
LPIPS (Learned Perceptual Image Patch Similarity) instead of PSNR as the primary
quality metric.  SSIM is retained alongside LPIPS.

Note: LPIPS is a *distance* metric — lower values indicate better perceptual quality,
      which is the opposite direction from PSNR/SSIM (higher = better).

Original paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop
Training. ACM TOG (SIGGRAPH Asia), 2020.

LPIPS paper:
R. Zhang, P. Isola, A. Efros, E. Shechtman, O. Wang. The Unreasonable Effectiveness of
Deep Features as a Perceptual Metric. CVPR, 2018.

This code is released under CC BY-NC 4.0. Non-commercial use only.

-----

$ python eval.py --channel=[0 or 1 or 2 or 3] --root_path=[some path]
"""

import imageio
import os
import skimage.io
import scipy.io as sio
import sys
import torch
import numpy as np
import configargparse
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage

from propagation_ASM import propagation_ASM
from utils.augmented_image_loader import ImageLoader
import utils.utils as utils
from utils.modules import PhysicalProp
from propagation_model import ModelPropagate

# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--channel', type=int, default=1, help='red:0, green:1, blue:2, rgb:3')
p.add_argument('--prop_model', type=str, default='ASM',
               help='Type of propagation model for reconstruction: ASM / MODEL / CAMERA')
p.add_argument('--root_path', type=str, default='./phases',
               help='Directory where test phases are being stored.')
p.add_argument('--prop_model_dir', type=str, default='./calibrated_models/',
               help='Directory for the CITL-calibrated wave propagation models.')
p.add_argument('--calibration_path', type=str, default='./calibration',
               help='Directory where calibration phases are being stored.')
p.add_argument('--lpips_net', type=str, default='alex',
               help='Backbone for LPIPS evaluation: alex (fast) or vgg (reference quality).')

opt = p.parse_args()
channel = opt.channel
chs = range(channel) if channel == 3 else [channel]
run_id = f'{opt.root_path.split("/")[-1]}_{opt.prop_model}'

# ---------------------------------------------------------------------------
# Physical / optical parameters
# ---------------------------------------------------------------------------
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
chan_strs = ('red', 'green', 'blue', 'rgb')
prop_dists = (20 * cm, 20 * cm, 20 * cm)
wavelengths = (638 * nm, 520 * nm, 450 * nm)
feature_size = (6.4 * um, 6.4 * um)

slm_res = (1080, 1920)
if 'HOLONET' in run_id.upper():
    slm_res = (1072, 1920)
elif 'UNET' in run_id.upper():
    slm_res = (1024, 2048)

image_res = (1080, 1920)
roi_res = (880, 1600)
dtype = torch.float32
device = torch.device('cuda')

# ---------------------------------------------------------------------------
# Propagation model setup
# ---------------------------------------------------------------------------
precomputed_H = [None] * 3
if opt.prop_model == 'ASM':
    propagator = propagation_ASM
    for c in chs:
        precomputed_H[c] = propagator(torch.empty(1, 1, *slm_res, 2), feature_size,
                                      wavelengths[c], prop_dists[c], return_H=True).to(device)

elif opt.prop_model.upper() == 'CAMERA':
    propagator = PhysicalProp(channel, laser_arduino=True, roi_res=(roi_res[1], roi_res[0]),
                              slm_settle_time=0.15,
                              range_row=(220, 1000), range_col=(300, 1630),
                              patterns_path=opt.calibration_path,
                              show_preview=True)

elif opt.prop_model.upper() == 'MODEL':
    blur = utils.make_kernel_gaussian(0.85, 3)
    propagators = {}
    for c in chs:
        propagator = ModelPropagate(distance=prop_dists[c],
                                    feature_size=feature_size,
                                    wavelength=wavelengths[c],
                                    blur=blur).to(device)
        propagator.load_state_dict(
            torch.load(os.path.join(opt.prop_model_dir, f'{chan_strs[c]}.pth'), map_location=device))
        propagator.eval()
        propagators[c] = propagator

print(f'  - reconstruction with {opt.prop_model}...')

# ---------------------------------------------------------------------------
# LPIPS model — initialised once and reused for all images
# ---------------------------------------------------------------------------
lpips_fn = lpips.LPIPS(net=opt.lpips_net).to(device)
print(f'  - LPIPS evaluation using {opt.lpips_net} backbone (lower = better)')

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
data_path = './data'
recon_path = './recon'

image_loader = ImageLoader(data_path, channel=channel if channel < 3 else None,
                           image_res=image_res, homography_res=roi_res,
                           crop_to_homography=True,
                           shuffle=False, vertical_flips=False, horizontal_flips=False)

# Metric accumulators
lpips_vals = {'amp': [], 'lin': [], 'srgb': []}
ssims      = {'amp': [], 'lin': [], 'srgb': []}
psnrs      = {'amp': [], 'lin': [], 'srgb': []}
idxs = []

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
for k, target in enumerate(image_loader):
    target_amp, target_res, target_filename = target
    target_path, target_filename = os.path.split(target_filename[0])
    target_idx = target_filename.split('_')[-1]
    target_amp = target_amp.to(device)

    print(f'    - running for img_{target_idx}...')

    target_amp = utils.crop_image(target_amp, target_shape=roi_res, stacked_complex=False).to(device)

    recon_amp = []

    for c in chs:
        # Load and invert phase (SLM convention)
        phase_filename = os.path.join(opt.root_path, chan_strs[c], f'{target_idx}.png')
        slm_phase = skimage.io.imread(phase_filename) / 255.
        slm_phase = torch.tensor((1 - slm_phase) * 2 * np.pi - np.pi,
                                 dtype=dtype).reshape(1, 1, *slm_res).to(device)

        # Build complex field at SLM plane
        real, imag = utils.polar_to_rect(torch.ones_like(slm_phase), slm_phase)
        slm_field = torch.complex(real, imag)

        if opt.prop_model.upper() == 'MODEL':
            propagator = propagators[c]
        recon_field = utils.propagate_field(slm_field, propagator, prop_dists[c], wavelengths[c],
                                            feature_size, opt.prop_model, dtype)

        recon_amp_c = recon_field.abs()
        recon_amp_c = utils.crop_image(recon_amp_c, target_shape=roi_res, stacked_complex=False)
        recon_amp.append(recon_amp_c)

    # Concatenate channels and apply least-squares amplitude scaling
    recon_amp = torch.cat(recon_amp, dim=1)
    recon_amp *= (torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True)
                  / torch.sum(recon_amp * recon_amp, (-2, -1), keepdim=True))

    # Convert to numpy for metric computation
    recon_np  = recon_amp.squeeze().cpu().detach().numpy()
    target_np = target_amp.squeeze().cpu().detach().numpy()

    if channel == 3:
        recon_np  = recon_np.transpose(1, 2, 0)
        target_np = target_np.transpose(1, 2, 0)

    # Compute LPIPS + SSIM across three domains
    lpips_val, ssim_val = utils.get_lpips_ssim(
        recon_np, target_np,
        lpips_fn=lpips_fn,
        device=str(device),
        multichannel=(channel == 3)
    )

    # Compute PSNR across three domains
    target_linear = target_np ** 2
    recon_linear  = recon_np  ** 2
    target_srgb   = utils.srgb_lin2gamma(np.clip(target_linear, 0.0, 1.0))
    recon_srgb    = utils.srgb_lin2gamma(np.clip(recon_linear,  0.0, 1.0))
    psnr_val = {
        'amp':  psnr_skimage(target_np,     recon_np,     data_range=1.0),
        'lin':  psnr_skimage(target_linear, recon_linear, data_range=1.0),
        'srgb': psnr_skimage(target_srgb,   recon_srgb,   data_range=1.0),
    }

    idxs.append(target_idx)
    for domain in ['amp', 'lin', 'srgb']:
        lpips_vals[domain].append(lpips_val[domain])
        ssims[domain].append(ssim_val[domain])
        psnrs[domain].append(psnr_val[domain])
        print(f'  LPIPS({domain}): {lpips_val[domain]:.4f}  '
              f'SSIM({domain}): {ssim_val[domain]:.4f}  '
              f'PSNR({domain}): {psnr_val[domain]:.2f} dB')

    # Save reconstructed sRGB image
    recon_srgb = utils.srgb_lin2gamma(np.clip(recon_np ** 2, 0.0, 1.0))
    utils.cond_mkdir(recon_path)
    imageio.imwrite(
        os.path.join(recon_path, f'{target_idx}_{run_id}_{chan_strs[channel]}.png'),
        (recon_srgb * np.iinfo(np.uint8).max).round().astype(np.uint8)
    )

# ---------------------------------------------------------------------------
# Save metrics to .mat
# ---------------------------------------------------------------------------
data_dict = {'img_idx': idxs}
for domain in ['amp', 'lin', 'srgb']:
    data_dict[f'ssims_{domain}']  = ssims[domain]
    data_dict[f'lpips_{domain}']  = lpips_vals[domain]
    data_dict[f'psnrs_{domain}']  = psnrs[domain]

sio.savemat(
    os.path.join(recon_path, f'metrics_{run_id}_{chan_strs[channel]}.mat'),
    data_dict
)

print(f'\n  Results saved to {recon_path}')
print(f'  Mean LPIPS (sRGB, lower=better):  {np.mean(lpips_vals["srgb"]):.4f}')
print(f'  Mean SSIM  (sRGB, higher=better): {np.mean(ssims["srgb"]):.4f}')
print(f'  Mean PSNR  (sRGB, higher=better): {np.mean(psnrs["srgb"]):.2f} dB')
