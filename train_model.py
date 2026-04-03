"""
Neural Holography - CITL with LPIPS:

Training script for the parameterised wave propagation model with
Camera-in-the-Loop (CITL).

Modified from the original neural-holography repository:
  - MSELoss for phase optimisation (loss_phase) replaced with LPIPSLoss.
  - MSELoss for model training (loss_model) replaced with LPIPSLoss.
  - TensorBoard summaries log LPIPS instead of PSNR.

Original paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop
Training. ACM TOG (SIGGRAPH Asia), 2020.

This code is released under CC BY-NC 4.0. Non-commercial use only.

-----

$ python train_model.py --channel=1 --experiment=test
"""

import os
import cv2
import sys
import time
import torch
import numpy as np
import configargparse
import skimage.util
import torch.nn as nn
import torch.optim as optim

import utils.utils as utils
from utils.modules import PhysicalProp
from propagation_model import ModelPropagate
from utils.augmented_image_loader import ImageLoader
from utils.utils_tensorboard import SummaryModelWriter

# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--channel', type=int, default=1, help='red:0, green:1, blue:2, rgb:3')
p.add_argument('--pretrained_path', type=str, default='',
               help='Path of pretrained checkpoint to start from.')
p.add_argument('--model_path', type=str, default='./models',
               help='Directory for saving checkpoints.')
p.add_argument('--phase_path', type=str, default='./precomputed_phases',
               help='Directory for pre-calculated phases.')
p.add_argument('--calibration_path', type=str, default='./calibration',
               help='Directory where calibration phases are stored.')
p.add_argument('--lr_model', type=float, default=3e-3,
               help='Learning rate for model parameters.')
p.add_argument('--lr_phase', type=float, default=5e-3,
               help='Learning rate for phase.')
p.add_argument('--num_epochs', type=int, default=15, help='Number of epochs.')
p.add_argument('--batch_size', type=int, default=2, help='Mini-batch size.')
p.add_argument('--step_lr', type=utils.str2bool, default=True,
               help='Use LR scheduler.')
p.add_argument('--experiment', type=str, default='', help='Name of experiment.')
p.add_argument('--lpips_net', type=str, default='vgg',
               help='LPIPS backbone for training losses: vgg (default) or alex.')

opt = p.parse_args()

channel = opt.channel
chan_str = ('red', 'green', 'blue')[channel]
run_id = f'{chan_str}_{opt.experiment}_lr{opt.lr_model}_batchsize{opt.batch_size}'

print(f'   - training parameterised wave propagation model...')
print(f'   - loss functions: LPIPSLoss ({opt.lpips_net} backbone)')

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
# Loss functions
# ---------------------------------------------------------------------------
lr_s_phase = opt.lr_phase / 200

# LPIPSLoss for phase optimisation: comparing simulated amplitude vs target
loss_phase = utils.LPIPSLoss(net=opt.lpips_net).to(device)

# LPIPSLoss for model training: comparing model output amplitude vs camera capture
loss_model = utils.LPIPSLoss(net=opt.lpips_net).to(device)

# MSELoss kept for reference reporting (camera vs target)
loss_mse = nn.MSELoss().to(device)

s0_phase = 1.0
s0_model = 1.0
sa = torch.tensor(s0_phase, device=device, requires_grad=True)
sb = torch.tensor(0.3,      device=device, requires_grad=True)

num_iters_model_update = 1
num_iters_phase_update = 1

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
model_path = opt.model_path
utils.cond_mkdir(model_path)
phase_path = opt.phase_path
data_path  = './data/train1080'

# ---------------------------------------------------------------------------
# Hardware setup
# ---------------------------------------------------------------------------
camera_prop = PhysicalProp(channel, laser_arduino=True,
                           roi_res=(roi_res[1], roi_res[0]),
                           slm_settle_time=0.15,
                           range_row=(220, 1000), range_col=(300, 1630),
                           patterns_path=opt.calibration_path,
                           show_preview=True)

# ---------------------------------------------------------------------------
# Propagation model
# ---------------------------------------------------------------------------
blur = utils.make_kernel_gaussian(0.85, 3)
model = ModelPropagate(distance=prop_dist,
                       feature_size=feature_size,
                       wavelength=wavelength,
                       blur=blur).to(device)

if opt.pretrained_path != '':
    print(f'   - starting from pre-trained model: {opt.pretrained_path}')
    model.load_state_dict(torch.load(opt.pretrained_path))
model = model.train()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
image_loader = ImageLoader(data_path,
                           channel=channel,
                           batch_size=opt.batch_size,
                           image_res=image_res,
                           homography_res=roi_res,
                           crop_to_homography=False,
                           shuffle=True,
                           vertical_flips=False,
                           horizontal_flips=False)

# ---------------------------------------------------------------------------
# Optimisers
# ---------------------------------------------------------------------------
optimizer_model = optim.Adam(
    [{'params': [p for name, p in model.named_parameters()
                 if 'source_amp' not in name and 'process_phase' not in name]},
     {'params': model.source_amp.parameters(),   'lr': opt.lr_model * 1},
     {'params': model.process_phase.parameters(), 'lr': opt.lr_model * 1}],
    lr=opt.lr_model
)
optimizer_phase_scale = optim.Adam([sa, sb], lr=lr_s_phase)

if opt.step_lr:
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer_model, step_size=5, gamma=0.2)

# TensorBoard writer
summaries_dir = os.path.join('runs', run_id)
utils.cond_mkdir(summaries_dir)
writer = SummaryModelWriter(model, summaries_dir, ch=channel)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
i_acc = 0
for e in range(opt.num_epochs):
    print(f'   - Epoch {e + 1} ...')

    with torch.no_grad():
        writer.visualize_model(e)

    for i, target in enumerate(image_loader):
        target_amp, _, target_filenames = target

        # Extract image indices
        idxs = []
        for name in target_filenames:
            _, fname = os.path.split(name)
            idxs.append(fname.split('_')[-1])

        target_amp = utils.crop_image(target_amp, target_shape=roi_res,
                                      stacked_complex=False).to(device)

        # ── Load pre-computed phases ─────────────────────────────────────────
        slm_phases = []
        for k, idx in enumerate(idxs):
            if e > 0:
                phase_filename = os.path.join(phase_path, f'{chan_str}', f'{idx}.png')
            else:
                phase_filename = os.path.join(phase_path, f'{chan_str}',
                                              f'{idx}_{channel}', 'phasemaps_1000.png')
            slm_phase = skimage.io.imread(phase_filename) / np.iinfo(np.uint8).max
            slm_phase = torch.tensor((1 - slm_phase) * 2 * np.pi - np.pi,
                                     dtype=dtype).reshape(1, 1, *slm_res).to(device)
            slm_phases.append(slm_phase)
        slm_phases = torch.cat(slm_phases, 0).detach().requires_grad_(True)

        optimizer_phase = optim.Adam([slm_phases], lr=opt.lr_phase)

        # ── Stage 1: Phase optimisation (simulation) ─────────────────────────
        model = model.eval()
        for j in range(max(e * num_iters_phase_update, 1)):
            optimizer_phase.zero_grad()

            recon_field = model(slm_phases)
            recon_amp   = recon_field.abs()
            model_amp   = utils.crop_image(recon_amp, target_shape=roi_res,
                                           pytorch=True, stacked_complex=False)

            with torch.no_grad():
                scale_phase = (
                    (model_amp * target_amp).mean(dim=[-2, -1], keepdim=True)
                    / (model_amp ** 2).mean(dim=[-2, -1], keepdim=True)
                )

            # LPIPS loss: scaled simulated amplitude vs target
            loss_value_phase = loss_phase(scale_phase * model_amp, target_amp)
            loss_value_phase.backward()
            optimizer_phase.step()
            optimizer_phase_scale.step()

        # Write updated phases to disk (phase pool)
        with torch.no_grad():
            for k, idx in enumerate(idxs):
                phase_out_8bit = utils.phasemap_8bit(
                    slm_phases[k, np.newaxis, ...].cpu().detach(), inverted=True)
                cv2.imwrite(os.path.join(phase_path, f'{idx}.png'), phase_out_8bit)

        # Quantise phases to 8-bit as displayed on SLM
        slm_phases = utils.quantized_phase(slm_phases)

        # ── Stage 2: Physical display & capture ──────────────────────────────
        camera_amp = []
        with torch.no_grad():
            for k, idx in enumerate(idxs):
                slm_phase = slm_phases[k, np.newaxis, ...]
                camera_amp.append(camera_prop(slm_phase))
            camera_amp = torch.cat(camera_amp, 0)

        # ── Stage 3: Model update ─────────────────────────────────────────────
        model = model.train()
        for j in range(num_iters_model_update):
            optimizer_model.zero_grad()

            recon_field = model(slm_phases)
            recon_amp   = recon_field.abs()
            model_amp   = utils.crop_image(recon_amp, target_shape=roi_res,
                                           pytorch=True, stacked_complex=False)

            # LPIPS loss: model output amplitude vs physically captured amplitude
            loss_value_model = loss_model(model_amp, camera_amp)
            loss_value_model.backward()
            optimizer_model.step()

        # ── TensorBoard logging ───────────────────────────────────────────────
        with torch.no_grad():
            if i % 50 == 0:
                writer.add_scalar('Scale/sa', sa, i_acc)
                writer.add_scalar('Scale/sb', sb, i_acc)
                for idx_s in range(opt.batch_size):
                    writer.add_scalar(f'Scale/model_vs_target_{idx_s}', scale_phase[idx_s], i_acc)
                writer.add_scalar('Loss/phase_lpips', loss_value_phase, i_acc)
                writer.add_scalar('Loss/model_lpips', loss_value_model, i_acc)
                # Reference MSE between camera and target (for monitoring)
                writer.add_scalar(
                    'Loss/camera_vs_target_mse',
                    loss_mse(camera_amp * target_amp.mean() / camera_amp.mean(), target_amp),
                    i_acc
                )

            if i % 50 == 0:
                recon    = model_amp[0, ...]
                captured = camera_amp[0, ...]
                gt       = target_amp[0, ...] / scale_phase[0, ...]
                max_amp  = max(recon.max(), captured.max(), gt.max())
                writer.add_image('Amp/recon',    recon    / max_amp, i_acc)
                writer.add_image('Amp/captured', captured / max_amp, i_acc)
                writer.add_image('Amp/target',   gt       / max_amp, i_acc)

            i_acc += 1

    # Save checkpoint after each epoch
    torch.save(model.state_dict(),
               os.path.join(model_path, f'{run_id}_{e}epoch.pth'))

    if opt.step_lr:
        lr_scheduler.step()

# ---------------------------------------------------------------------------
# Clean up hardware connections
# ---------------------------------------------------------------------------
if camera_prop is not None:
    camera_prop.disconnect()
    camera_prop.alc.disconnect()
