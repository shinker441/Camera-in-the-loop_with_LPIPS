"""
Neural Holography - CITL with MSE + LPIPS:

Phase generation using HoloNet/UNET or iterative optimisation (GS/DPAC/SGD)
with optional Camera-in-the-Loop (CITL).

Modified from the original neural-holography repository:
  - MSE + λ × LPIPS combined loss used for SGD phase optimisation.
  - TensorBoard summaries log MSE, LPIPS, and combined loss separately.
  - Main settings can be controlled directly at the top of this file.

Original paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop
Training. ACM TOG (SIGGRAPH Asia), 2020.

This code is released under CC BY-NC 4.0. Non-commercial use only.
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

# ===========================================================================
# HOW TO USE EACH MODE
#
# まず基本:
#   - 普段は CODE_SETTINGS だけ編集する
#   - 実行は基本 `python .\main.py` だけでよい
#   - USE_CODE_SETTINGS = False にすると、元のCLI引数方式に戻る
#
# ---------------------------------------------------------------------------
# 1) SGD + ASM + 非CITL（まず最初のソフト確認用）
# ---------------------------------------------------------------------------
# 用途:
#   - ハードウェアを使わず、ソフトだけで位相最適化が通るか確認する
# 設定:
#   method      = 'SGD'
#   prop_model  = 'ASM'
#   citl        = False
#   data_path   = './data/test'
#   num_iters   = 3 ～ 10 くらい
#   lpips_net   = 'alex'   # 軽く確認したいとき
#   lambda_lpips= 0.1      # MSE + 0.1*LPIPS
#
# 例:
#   'method': 'SGD',
#   'prop_model': 'ASM',
#   'citl': False,
#   'experiment': 'smoke',
#   'num_iters': 3,
#
# ---------------------------------------------------------------------------
# 2) SGD + ASM + CITL（実機を使う本命モード）
# ---------------------------------------------------------------------------
# 用途:
#   - SLM + Basler + homography を使って camera-in-the-loop を回す
# 設定:
#   method           = 'SGD'
#   prop_model       = 'ASM'
#   citl             = True
#   homography_file  = './calibration/homography.npy'
#   monitor_index    = 1         # SLM側モニタ
#   camera_index     = 0         # Basler 1台目
#   pixel_format     = 'Mono8'   # 今の環境ならこれ
#   slm_settle_time  = 0.3       # 必要なら 0.4～0.6 に増やす
#   num_iters        = 本番では 50 / 100 / 500 など
#
# 注意:
#   - citl=True は基本的に SGD 用。
#   - GS/DPAC/HOLONET/UNET にしても、今の main.py では
#     camera_prop は作られるが、実際に citl フラグを受け取るのは SGD だけ。
#
# 例:
#   'method': 'SGD',
#   'prop_model': 'ASM',
#   'citl': True,
#   'experiment': 'hwcheck',
#   'num_iters': 1,
#   'homography_file': './calibration/homography.npy',
#   'monitor_index': 1,
#   'camera_index': 0,
#   'pixel_format': 'Mono8',
#
# ---------------------------------------------------------------------------
# 3) SGD + MODEL（学習済み伝搬モデルを使う）
# ---------------------------------------------------------------------------
# 用途:
#   - ASM の代わりに calibrated propagation model を使う
# 設定:
#   method          = 'SGD'
#   prop_model      = 'MODEL'
#   prop_model_dir  = './calibrated_models'
#   citl            = False でも True でも可
#
# 必要ファイル:
#   - prop_model_dir の下に
#       red.pth / green.pth / blue.pth
#     のような伝搬モデル重みが必要
#
# 例:
#   'method': 'SGD',
#   'prop_model': 'MODEL',
#   'prop_model_dir': './calibrated_models',
#   'citl': False,
## ---------------------------------------------------------------------------
# 6) HOLONET
# ---------------------------------------------------------------------------
# 用途:
#   - 学習済み HoloNet で高速に phase を出したいとき
# 設定:
#   method         = 'HOLONET'
#   generator_dir  = './pretrained_networks'
#   citl           = False 推奨
#
# 必要ファイル:
#   - generator_dir の下に
#       holonet20_red.pth
#       holonet20_green.pth
#       holonet20_blue.pth
#     の対応する重みが必要
#
# 注意:
#   - HOLONET を選ぶと image_res が自動で (1072, 1920) になる
#   - prop_model はコード上では残るが、主役は generator 側
#
# 例:
#   'method': 'HOLONET',
#   'generator_dir': './pretrained_networks',
#   'citl': False,
#
# ---------------------------------------------------------------------------
# 7) UNET
# ---------------------------------------------------------------------------
# 用途:
#   - 学習済み UNET で phase を出したいとき
# 設定:
#   method         = 'UNET'
#   generator_dir  = './pretrained_networks'
#   citl           = False 推奨
#
# 必要ファイル:
#   - generator_dir の下に
#       unet20_red.pth
#       unet20_green.pth
#       unet20_blue.pth
#     の対応する重みが必要
#
# 注意:
#   - UNET を選ぶと image_res が自動で (1024, 2048) になる
#
# 例:
#   'method': 'UNET',
#   'generator_dir': './pretrained_networks',
#   'citl': False,
#
# ---------------------------------------------------------------------------
# 8) 色チャネル設定
# ---------------------------------------------------------------------------
# channel:
#   0 = red
#   1 = green
#   2 = blue
#
# これに応じて自動で変わるもの:
#   - chan_str
#   - wavelength
#   - 保存先フォルダ名 (red/green/blue)
#   - MODEL/HOLONET/UNET の読み込みファイル名
#
# 通常は今の環境なら green=1 を使う。
#
# ---------------------------------------------------------------------------
# 9) ハードウェア項目の意味
# ---------------------------------------------------------------------------
# homography_file:
#   - CITL のときに使う camera -> target plane の 3x3 行列
#   - 例: './calibration/homography.npy'
#
# monitor_index:
#   - SLM を表示するモニタ番号
#   - 今の環境では 1
#
# camera_index:
#   - Basler のデバイス番号
#   - 今の環境では 0
#
# pixel_format:
#   - Basler の撮像フォーマット
#   - 今の環境では 'Mono8' 推奨
#
# slm_flip_udlr:
#   - SLMの表示を180°反転したいとき True
#
# slm_settle_time:
#   - SLM表示更新後、撮影前に待つ秒数
#   - 撮影が不安定なら大きくする
#
# pixel_pitch:
#   - SLM ピクセルピッチ [m]
#   - 今の FHD SLM なら 6.4e-6
#
# ---------------------------------------------------------------------------

# ===========================================================================
# USER SETTINGS
# Edit only this section for normal use.
# If USE_CODE_SETTINGS = True, these values override command-line arguments.
# ===========================================================================
USE_CODE_SETTINGS = True

CODE_SETTINGS = {
    # -----------------------------------------------------------------------
    # Run / dataset
    # -----------------------------------------------------------------------
    'channel': 1,                         # red:0, green:1, blue:2
    'method': 'SGD',                      # GS / SGD / DPAC / HOLONET / UNET
    'prop_model': 'ASM',                  # ASM / Fresnel/Model
    'root_path': './phases/test',
    'data_path': './data/test',
    'generator_dir': './pretrained_networks',
    'prop_model_dir': './calibrated_models',
    'citl': True,
    'experiment': 'hwcheck',

    # -----------------------------------------------------------------------
    # Optimisation
    # -----------------------------------------------------------------------
    'lr': 8e-3,
    'lr_s': 2e-3,
    'num_iters': 1,
    'lpips_net': 'alex',                  # vgg / alex
    'lambda_lpips': 0.1,

    # -----------------------------------------------------------------------
    # Hardware (CITL)
    # -----------------------------------------------------------------------
    'slm_settle_time': 0.3,
    'homography_file': './calibration/homography.npy',
    'slm_flip_udlr': True,
    'camera_index': 0,
    'monitor_index': 1,
    'pixel_format': 'Mono8',              # RGB8 / BGR8 / Mono8
    'pixel_pitch': 6.4e-6,
}

# ---------------------------------------------------------------------------
# Physical / optical constants
# ---------------------------------------------------------------------------
PROP_DIST_BY_CHANNEL = (0.20, 1.0, 0.20)          # metres
WAVELENGTH_BY_CHANNEL = (638e-9, 520e-9, 450e-9)   # metres
SLM_RES = (1080, 1920)
IMAGE_RES_DEFAULT = (1080, 1920)
ROI_RES = (880, 1600)
DTYPE = torch.float32
DEVICE = 'cuda'
S0 = 1.0
SHOW_PREVIEW = True


# ---------------------------------------------------------------------------
# Command-line arguments
# These are kept for compatibility, but can be overridden by CODE_SETTINGS.
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
p.add_argument('--lambda_lpips', type=float, default=0.1,
               help='Weight for the LPIPS term in the combined MSE + λ × LPIPS loss. '
                    'Set to 0 for pure MSE (ablation).')

# ---------------------------------------------------------------------------
# Hardware arguments (CITL: SLM + Basler camera)
# ---------------------------------------------------------------------------
p.add_argument('--slm_settle_time', type=float, default=0.3,
               help='Seconds to wait after SLM update before camera capture.')
p.add_argument('--homography_file', type=str, default='',
               help='Path to .npy file containing pre-computed 3×3 homography '
                    'matrix H (camera → target plane). Leave empty to skip.')
p.add_argument('--slm_flip_udlr', type=utils.str2bool, default=True,
               help='Flip SLM image 180° before display (for upside-down mounting).')
p.add_argument('--camera_index', type=int, default=0,
               help='Basler camera device index (0 = first found).')
p.add_argument('--monitor_index', type=int, default=1,
               help='Monitor index for SLM window (1 = second monitor, 0 = primary).')
p.add_argument('--pixel_format', type=str, default='RGB8',
               help="Basler pixel format: 'RGB8' (default), 'BGR8', or 'Mono8'.")
p.add_argument('--pixel_pitch', type=float, default=6.4e-6,
               help='SLM pixel pitch in metres (default 6.4 μm).')

opt = p.parse_args()

# ---------------------------------------------------------------------------
# Override CLI/config settings with in-code settings
# ---------------------------------------------------------------------------
if USE_CODE_SETTINGS:
    for key, value in CODE_SETTINGS.items():
        setattr(opt, key, value)

run_id = f'{opt.experiment}_{opt.method}_{opt.prop_model}'
if opt.citl:
    run_id = f'{run_id}_citl'

channel = opt.channel
chan_str = ('red', 'green', 'blue')[channel]

print(f'   - optimising phase with {opt.method}/{opt.prop_model} ...')
if opt.citl:
    print(f'     with camera-in-the-loop ...')
print(f'   - optimisation loss: MSE + {opt.lambda_lpips} × LPIPS ({opt.lpips_net} backbone)')

# ---------------------------------------------------------------------------
# Physical / optical parameters
# ---------------------------------------------------------------------------
prop_dist = PROP_DIST_BY_CHANNEL[channel]
wavelength = WAVELENGTH_BY_CHANNEL[channel]
feature_size = (opt.pixel_pitch, opt.pixel_pitch)
slm_res = SLM_RES
image_res = IMAGE_RES_DEFAULT
roi_res = ROI_RES
dtype = DTYPE
device = torch.device(DEVICE)

# ---------------------------------------------------------------------------
# Loss function: MSE + λ × LPIPS combined loss
# ---------------------------------------------------------------------------
loss = utils.CombinedLoss(net=opt.lpips_net, lambda_lpips=opt.lambda_lpips).to(device)

s0 = S0  # initial amplitude scale

root_path = os.path.join(opt.root_path, run_id, chan_str)

# TensorBoard writer
summaries_dir = os.path.join(root_path, 'summaries')
utils.cond_mkdir(summaries_dir)
writer = SummaryWriter(summaries_dir)

# ---------------------------------------------------------------------------
# Hardware setup for CITL (SLM + Basler camera)
# ---------------------------------------------------------------------------
if opt.citl:
    camera_prop = PhysicalProp(
        channel,
        slm_settle_time=opt.slm_settle_time,
        roi_res=(roi_res[1], roi_res[0]),   # (W, H)
        homography_file=opt.homography_file,
        slm_flip_udlr=opt.slm_flip_udlr,
        show_preview=SHOW_PREVIEW,
        camera_index=opt.camera_index,
        pixel_format=opt.pixel_format,
        monitor_index=opt.monitor_index,
    )
else:
    camera_prop = None

# ---------------------------------------------------------------------------
# Propagation model
# ---------------------------------------------------------------------------
if opt.prop_model == 'ASM':
    propagator = propagation_ASM

elif opt.prop_model.upper() == 'MODEL':
    blur = utils.make_kernel_gaussian(0.85, 3)
    propagator = ModelPropagate(
        distance=prop_dist,
        feature_size=feature_size,
        wavelength=wavelength,
        blur=blur
    ).to(device)
    propagator.load_state_dict(
        torch.load(f'{opt.prop_model_dir}/{chan_str}.pth', map_location=device)
    )
    propagator.eval()

else:
    raise ValueError(f'Unexpected prop_model: {opt.prop_model}')

# ---------------------------------------------------------------------------
# Phase-generation algorithm
# ---------------------------------------------------------------------------
if opt.method == 'SGD':
    phase_only_algorithm = SGD(
        prop_dist, wavelength, feature_size, opt.num_iters, roi_res, root_path,
        opt.prop_model, propagator, loss, opt.lr, opt.lr_s, s0,
        opt.citl, camera_prop, writer, device
    )
elif opt.method == 'GS':
    phase_only_algorithm = GS(
        prop_dist, wavelength, feature_size, opt.num_iters, root_path,
        opt.prop_model, propagator, writer, device
    )
elif opt.method == 'DPAC':
    phase_only_algorithm = DPAC(
        prop_dist, wavelength, feature_size,
        opt.prop_model, propagator, device
    )
elif opt.method == 'HOLONET':
    phase_only_algorithm = HoloNet(
        prop_dist, wavelength, feature_size,
        initial_phase=InitialPhaseUnet(4, 16),
        final_phase_only=FinalPhaseOnlyUnet(4, 16, num_in=2)
    ).to(device)
    model_path = os.path.join(opt.generator_dir, f'holonet20_{chan_str}.pth')
    image_res = (1072, 1920)
elif opt.method == 'UNET':
    phase_only_algorithm = PhaseOnlyUnet(num_features_init=32).to(device)
    model_path = os.path.join(opt.generator_dir, f'unet20_{chan_str}.pth')
    image_res = (1024, 2048)
else:
    raise ValueError(f'Unexpected method: {opt.method}')

if 'NET' in opt.method:
    checkpoint = torch.load(model_path, map_location=device)
    phase_only_algorithm.load_state_dict(checkpoint)
    phase_only_algorithm.eval()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
image_loader = ImageLoader(
    opt.data_path,
    channel=channel,
    image_res=image_res,
    homography_res=roi_res,
    crop_to_homography=True,
    shuffle=False,
    vertical_flips=False,
    horizontal_flips=False
)

# ---------------------------------------------------------------------------
# Main loop over dataset
# ---------------------------------------------------------------------------
for k, target in enumerate(image_loader):
    target_amp, target_res, target_filename = target
    target_path, target_filename = os.path.split(target_filename[0])
    target_idx = target_filename.split('_')[-1]
    target_amp = target_amp.to(device)
    print(target_idx)

    phase_only_algorithm.init_scale = s0 * utils.crop_image(
        target_amp, roi_res, stacked_complex=False
    ).mean()
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

# ---------------------------------------------------------------------------
# Cleanup hardware
# ---------------------------------------------------------------------------
if camera_prop is not None:
    camera_prop.disconnect()
    camera_prop.alc.disconnect()