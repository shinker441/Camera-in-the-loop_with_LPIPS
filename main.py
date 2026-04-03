"""
Neural Holography - CITL with LPIPS:

Phase generation using HoloNet/UNET or iterative optimisation (GS/DPAC/SGD)
with optional Camera-in-the-Loop (CITL).

Modified from the original neural-holography repository:
  - MSELoss replaced with LPIPSLoss (perceptual loss) for SGD phase optimisation.
  - TensorBoard summaries now log LPIPS instead of PSNR.

Original paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop
Training. ACM TOG (SIGGRAPH Asia), 2020.

This code is released under CC BY-NC 4.0. Non-commercial use only.

-----

$ python main.py --channel=0 --algorithm=HOLONET --root_path=./phases --generator_dir=./pretrained_models
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import configargparse
from torch.utils.tensorboard import SummaryWriter

import utils.utils as utils
from utils.augmented_image_loader import ImageLoader
from propagation_model import ModelPropagate
from utils.modules import SGD, GS, DPAC, PhysicalProp
from holonet import HoloNet, InitialPhaseUnet, FinalPhaseOnlyUnet, PhaseOnlyUnet
from propagation_ASM import propagation_ASM

# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--channel', type=int, default=1, help='Red:0, green:1, blue:2')
p.add_argument('--method', type=str, default='SGD',
               help='Type of algorithm: GS / SGD / DPAC / HOLONET / UNET')
p.add_argument('--prop_model', type=str, default='ASM',
               help='Type of propagation model: ASM or MODEL')
p.add_argument('--root_path', type=str, default='./phases',
               help='Directory where optimised phases will be saved.')
p.add_argument('--data_path', type=str, default='./data',
               help='Directory for the dataset.')
p.add_argument('--generator_dir', type=str, default='./pretrained_networks',
               help='Directory for the pretrained HoloNet/UNet network.')
p.add_argument('--prop_model_dir', type=str, default='./calibrated_models',
               help='Directory for the CITL-calibrated wave propagation models.')
p.add_argument('--citl', type=utils.str2bool, default=False,
               help='Use Camera-in-the-Loop optimisation with SGD.')
p.add_argument('--experiment', type=str, default='', help='Name of experiment.')
p.add_argument('--lr', type=float, default=8e-3,
               help='Learning rate for phase variables (SGD).')
p.add_argument('--lr_s', type=float, default=2e-3,
               help='Learning rate for learnable scale (SGD).')
p.add_argument('--num_iters', type=int, default=500,
               help='Number of iterations (GS, SGD).')
p.add_argument('--lpips_net', type=str, default='vgg',
               help='LPIPS backbone for the optimisation loss: vgg (default) or alex.')

opt = p.parse_args()

run_id = f'{opt.experiment}_{opt.method}_{opt.prop_model}'
if opt.citl:
    run_id = f'{run_id}_citl'

channel = opt.channel
chan_str = ('red', 'green', 'blue')[channel]

print(f'   - optimising phase with {opt.method}/{opt.prop_model} ...')
if opt.citl:
    print(f'     with camera-in-the-loop ...')
print(f'   - optimisation loss: LPIPS ({opt.lpips_net} backbone)')

# ---------------------------------------------------------------------------
# Physical / optical parameters
# ---------------------------------------------------------------------------
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
prop_dist    = (20 * cm, 20 * cm, 20 * cm)[channel]
wavelength   = (638 * nm, 520 * nm, 450 * nm)[channel]
feature_size = (6.4 * um, 6.4 * um)
slm_res      = (1080, 1920)
image_res    = (1080, 1920)
roi_res      = (880, 1600)
dtype        = torch.float32
device       = torch.device('cuda')

# ---------------------------------------------------------------------------
# Loss function: LPIPS (perceptual) — replaces MSELoss
# ---------------------------------------------------------------------------
loss = utils.LPIPSLoss(net=opt.lpips_net).to(device)

s0 = 1.0  # initial amplitude scale

root_path = os.path.join(opt.root_path, run_id, chan_str)

# TensorBoard writer
summaries_dir = os.path.join(root_path, 'summaries')
utils.cond_mkdir(summaries_dir)
writer = SummaryWriter(summaries_dir)

# ---------------------------------------------------------------------------
# Hardware setup for CITL
# ---------------------------------------------------------------------------
if opt.citl:
    camera_prop = PhysicalProp(channel, laser_arduino=True,
                               roi_res=(roi_res[1], roi_res[0]),
                               slm_settle_time=0.12,
                               range_row=(220, 1000), range_col=(300, 1630),
                               patterns_path='F:/citl/calibration',
                               show_preview=True)
else:
    camera_prop = None

# ---------------------------------------------------------------------------
# Propagation model
# ---------------------------------------------------------------------------
if opt.prop_model == 'ASM':
    propagator = propagation_ASM

elif opt.prop_model.upper() == 'MODEL':
    blur = utils.make_kernel_gaussian(0.85, 3)
    propagator = ModelPropagate(distance=prop_dist,
                                feature_size=feature_size,
                                wavelength=wavelength,
                                blur=blur).to(device)
    propagator.load_state_dict(
        torch.load(f'{opt.prop_model_dir}/{chan_str}.pth', map_location=device))
    propagator.eval()

# ---------------------------------------------------------------------------
# Phase-generation algorithm
# ---------------------------------------------------------------------------
if opt.method == 'SGD':
    phase_only_algorithm = SGD(prop_dist, wavelength, feature_size, opt.num_iters, roi_res, root_path,
                               opt.prop_model, propagator, loss, opt.lr, opt.lr_s, s0,
                               opt.citl, camera_prop, writer, device)
elif opt.method == 'GS':
    phase_only_algorithm = GS(prop_dist, wavelength, feature_size, opt.num_iters, root_path,
                              opt.prop_model, propagator, writer, device)
elif opt.method == 'DPAC':
    phase_only_algorithm = DPAC(prop_dist, wavelength, feature_size,
                                opt.prop_model, propagator, device)
elif opt.method == 'HOLONET':
    phase_only_algorithm = HoloNet(prop_dist, wavelength, feature_size,
                                   initial_phase=InitialPhaseUnet(4, 16),
                                   final_phase_only=FinalPhaseOnlyUnet(4, 16, num_in=2)).to(device)
    model_path = os.path.join(opt.generator_dir, f'holonet20_{chan_str}.pth')
    image_res = (1072, 1920)
elif opt.method == 'UNET':
    phase_only_algorithm = PhaseOnlyUnet(num_features_init=32).to(device)
    model_path = os.path.join(opt.generator_dir, f'unet20_{chan_str}.pth')
    image_res = (1024, 2048)

if 'NET' in opt.method:
    checkpoint = torch.load(model_path)
    phase_only_algorithm.load_state_dict(checkpoint)
    phase_only_algorithm.eval()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
image_loader = ImageLoader(opt.data_path, channel=channel,
                           image_res=image_res, homography_res=roi_res,
                           crop_to_homography=True,
                           shuffle=False, vertical_flips=False, horizontal_flips=False)

# ---------------------------------------------------------------------------
# Main loop over dataset
# ---------------------------------------------------------------------------
for k, target in enumerate(image_loader):
    target_amp, target_res, target_filename = target
    target_path, target_filename = os.path.split(target_filename[0])
    target_idx = target_filename.split('_')[-1]
    target_amp = target_amp.to(device)
    print(target_idx)

    phase_only_algorithm.init_scale = s0 * utils.crop_image(target_amp, roi_res, stacked_complex=False).mean()
    phase_only_algorithm.phase_path = os.path.join(root_path)

    if opt.method in ['DPAC', 'HOLONET', 'UNET']:
        _, final_phase = phase_only_algorithm(target_amp)
    else:
        init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *slm_res)).to(device)
        final_phase = phase_only_algorithm(target_amp, init_phase)

    print(final_phase.shape)

    phase_out_8bit = utils.phasemap_8bit(final_phase.cpu().detach(), inverted=True)
    utils.cond_mkdir(root_path)
    cv2.imwrite(os.path.join(root_path, f'{target_idx}.png'), phase_out_8bit)

print(f'    - Done, result: --root_path={root_path}')
