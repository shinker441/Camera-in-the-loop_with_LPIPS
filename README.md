# Camera-in-the-Loop with MSE + LPIPS

Neural Holography の Camera-in-the-Loop (CITL) 実装を、最適化損失に **MSE + λ × LPIPS** の併用方式を採用し、評価指標に **LPIPS / SSIM / PSNR** を用いるように改修したコードです。

元論文: [Neural Holography with Camera-in-the-loop Training (Peng et al., SIGGRAPH Asia 2020)](https://github.com/computational-imaging/neural-holography)

---

## 目次

1. [概要](#概要)
2. [LPIPSについて](#lpipsについて)
3. [動作環境](#動作環境)
4. [セットアップ](#セットアップ)
5. [ディレクトリ構成](#ディレクトリ構成)
6. [使い方](#使い方)
   - [位相生成 (main.py)](#位相生成-mainpy)
   - [伝播モデル学習 (train_model.py)](#伝播モデル学習-train_modelpy)
   - [評価 (eval.py)](#評価-evalpy)
7. [主な変更点](#主な変更点)
8. [ライセンス](#ライセンス)

---

## 概要

このリポジトリは以下を行います：

- **SGD による位相最適化**：`MSE + λ × LPIPS` の併用損失を用いて、安定した収束と知覚的品質向上を両立した位相パターンを生成します。
- **CITL 伝播モデル学習**：実カメラ撮影画像と SLM 表示画像の `MSE + λ × LPIPS` 距離を最小化するよう、波面伝播モデルを学習します。
- **評価**：振幅 (amp) / 線形強度 (lin) / sRGB の3ドメインで LPIPS・SSIM・PSNR を計算します。

### 損失関数の設計方針

MSEは最適化の安定的な収束を保証する土台として残し、LPIPSは知覚的品質向上の方向付けとして併用します。

| 役割 | 損失 | 理由 |
|------|------|------|
| 収束の安定 | MSE | ピクセル単位の勾配を安定供給し、発散を防ぐ |
| 知覚品質の向上 | λ × LPIPS | 人間の知覚に近い方向へ最適化を誘導する |

LPIPS 単独使用はほぼなく、画像生成・超解像の分野でも MSE + LPIPS の併用が標準的です。

> **注意**: このリポジトリは元の neural-holography リポジトリの変更ファイルのみを含みます。動作させるには元リポジトリのファイルと合わせて使用してください（[セットアップ](#セットアップ) 参照）。

---

## LPIPSについて

LPIPS (Zhang et al., CVPR 2018) は深層特徴量を用いた知覚的類似度指標です。

| 指標 | 方向 | 特徴 |
|------|------|------|
| PSNR | 高いほど良い | ピクセル単位の誤差を測る |
| SSIM | 高いほど良い | 輝度・コントラスト・構造を考慮 |
| **LPIPS** | **低いほど良い** | 人間の知覚に近い評価が可能 |

損失ネットワークの選択：

| `--lpips_net` | 用途 |
|---|---|
| `vgg` (デフォルト) | 学習損失として推奨（品質重視） |
| `alex` | 評価・高速計算に推奨 |

---

## 動作環境

| 項目 | バージョン |
|------|-----------|
| Python | 3.8 以上 |
| PyTorch | 1.9 以上 |
| CUDA | 10.2 以上 (GPU 必須) |

### ハードウェア（CITL 使用時）

| 機器 | 要件 |
|------|------|
| SLM | セカンドモニタとして接続した FHD (1920×1080) SLM |
| カメラ | Basler カメラ（pypylon でアクセス可能なもの） |

---

## セットアップ

### 1. 元リポジトリをクローン

```bash
git clone https://github.com/computational-imaging/neural-holography
cd neural-holography
```

### 2. このリポジトリの変更ファイルを上書きコピー

**重要**: `modules.py` は SGD/GS/DPAC を元ファイルから再インポートするため、
コピー前に元ファイルを `_modules_orig.py` にリネームしてください。

```bash
# 元の modules.py を退避（SGD/GS/DPAC の実装を保持するため必須）
cp utils/modules.py utils/_modules_orig.py

# このリポジトリをクローン
git clone https://github.com/shinker441/Camera-in-the-loop_with_LPIPS
cd Camera-in-the-loop_with_LPIPS

# neural-holography ディレクトリへファイルをコピー
cp main.py                        ../neural-holography/
cp train_model.py                  ../neural-holography/
cp eval.py                         ../neural-holography/
cp utils/utils.py                  ../neural-holography/utils/
cp utils/modules.py                ../neural-holography/utils/
cp utils/slm_display_module.py     ../neural-holography/utils/
cp utils/camera_capture_module.py  ../neural-holography/utils/
```

### 3. 依存ライブラリのインストール

```bash
cd ../neural-holography
pip install -r requirements.txt

# 追加ライブラリ
pip install lpips screeninfo pypylon-pylon
```

### 4. ホモグラフィ行列の準備（CITL 使用時）

SLMに表示した既知パターンをカメラで撮影し、OpenCV の `findHomography` で
3×3 ホモグラフィ行列 H を計算して `.npy` 形式で保存してください。

```python
import numpy as np, cv2
# src_pts: カメラ上の対応点  (N,2) float32
# dst_pts: ターゲット平面の対応点 (N,2) float32
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
np.save('./calibration/homography.npy', H)
```

実行時に `--homography_file=./calibration/homography.npy` で指定します。

### 5. データセットの配置

```
neural-holography/
└── data/
    ├── train1080/      # 学習用画像 (train_model.py)
    └── test/           # 評価用画像 (eval.py, main.py)
```

---

## ディレクトリ構成

```
neural-holography/          ← 元リポジトリ
├── main.py                 ← 【変更】位相生成スクリプト
├── train_model.py          ← 【変更】CITL モデル学習スクリプト
├── eval.py                 ← 【変更】評価スクリプト
├── algorithms.py           ← GS / SGD / DPAC アルゴリズム
├── propagation_ASM.py      ← 角スペクトル法による伝播
├── propagation_model.py    ← パラメトリック伝播モデル
├── holonet.py              ← HoloNet / U-Net ネットワーク
├── utils/
│   ├── utils.py            ← 【変更】ユーティリティ関数 (MSE+LPIPS 損失)
│   ├── modules.py          ← 【変更】PhysicalProp (Basler + OpenCV SLM)
│   ├── _modules_orig.py    ← 元の modules.py をリネームしたもの (SGD/GS/DPAC)
│   ├── slm_display_module.py     ← 【新規】OpenCV SLM 表示モジュール
│   ├── camera_capture_module.py  ← 【新規】Basler カメラモジュール (pypylon)
│   ├── augmented_image_loader.py
│   └── calibration_module.py
├── data/
│   ├── train1080/
│   └── test/
├── phases/                 ← 最適化済み位相パターン (出力)
├── recon/                  ← 再構成画像 (出力)
└── models/                 ← 学習済みモデル (出力)
```

---

## 使い方

### 位相生成 (main.py)

SGD・GS・DPAC・HoloNet・UNet を用いて位相パターンを生成します。  
SGD では LPIPS を最適化損失として使用します。

#### 基本実行（SGD + ASM シミュレーション）

```bash
python main.py \
  --channel=1 \
  --method=SGD \
  --prop_model=ASM \
  --root_path=./phases \
  --data_path=./data \
  --num_iters=500 \
  --lr=8e-3 \
  --lpips_net=vgg
```

#### CITL（Basler カメラ + OpenCV SLM を使用した最適化）

```bash
python main.py \
  --channel=1 \
  --method=SGD \
  --prop_model=ASM \
  --citl=True \
  --root_path=./phases \
  --data_path=./data \
  --num_iters=500 \
  --homography_file=./calibration/homography.npy \
  --slm_settle_time=0.3 \
  --slm_flip_udlr=True \
  --monitor_index=1 \
  --camera_index=0 \
  --pixel_format=RGB8 \
  --pixel_pitch=6.4e-6
```

#### CITL 校正済みモデルを使用

```bash
python main.py \
  --channel=1 \
  --method=SGD \
  --prop_model=MODEL \
  --prop_model_dir=./calibrated_models \
  --root_path=./phases \
  --data_path=./data
```

#### HoloNet / UNet（学習済みモデルによる直接生成）

```bash
python main.py \
  --channel=1 \
  --method=HOLONET \
  --generator_dir=./pretrained_networks \
  --root_path=./phases
```

#### 主なオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--channel` | `1` | 波長チャンネル (赤:0 / 緑:1 / 青:2) |
| `--method` | `SGD` | アルゴリズム: `SGD` / `GS` / `DPAC` / `HOLONET` / `UNET` |
| `--prop_model` | `ASM` | 伝播モデル: `ASM` / `MODEL` |
| `--citl` | `False` | カメラインザループ最適化を有効化 |
| `--num_iters` | `500` | 反復回数 |
| `--lr` | `8e-3` | 位相の学習率 |
| `--lpips_net` | `vgg` | LPIPS バックボーン (`vgg` / `alex`) |
| `--lambda_lpips` | `0.1` | 損失式 `MSE + λ × LPIPS` の λ。`0` で純粋な MSE（アブレーション用） |
| `--slm_settle_time` | `0.3` | SLM 表示後、カメラ撮影までの待機時間 [秒] |
| `--homography_file` | `''` | ホモグラフィ行列 `.npy` ファイルのパス（空欄でリサイズのみ） |
| `--slm_flip_udlr` | `True` | SLM 画像を 180° 反転して表示（逆さ設置対応） |
| `--monitor_index` | `1` | SLM ウィンドウを表示するモニタ番号（1=セカンド） |
| `--camera_index` | `0` | Basler カメラのデバイス番号 |
| `--pixel_format` | `RGB8` | Basler ピクセルフォーマット (`RGB8` / `BGR8` / `Mono8`) |
| `--pixel_pitch` | `6.4e-6` | SLM ピクセルピッチ [m] |

---

### 伝播モデル学習 (train_model.py)

CITL によってパラメトリック波面伝播モデルを学習します。  
位相最適化損失 (`loss_phase`) とモデル学習損失 (`loss_model`) の両方に `MSE + λ × LPIPS` を使用します。

> **注意**: 実機の SLM（セカンドモニタ）と Basler カメラが接続されている必要があります。

#### 基本実行

```bash
python train_model.py \
  --channel=1 \
  --experiment=lpips_test \
  --model_path=./models \
  --phase_path=./precomputed_phases \
  --num_epochs=15 \
  --batch_size=2 \
  --lr_model=3e-3 \
  --lr_phase=5e-3 \
  --lpips_net=vgg \
  --homography_file=./calibration/homography.npy \
  --slm_settle_time=0.3 \
  --slm_flip_udlr=True \
  --monitor_index=1 \
  --camera_index=0 \
  --pixel_format=RGB8 \
  --pixel_pitch=6.4e-6
```

#### 事前学習済みモデルから再開

```bash
python train_model.py \
  --channel=1 \
  --pretrained_path=./models/green_test_lr0.003_batchsize2_10epoch.pth \
  --experiment=lpips_finetune
```

#### 主なオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--channel` | `1` | 波長チャンネル |
| `--num_epochs` | `15` | エポック数 |
| `--batch_size` | `2` | バッチサイズ |
| `--lr_model` | `3e-3` | モデルパラメータの学習率 |
| `--lr_phase` | `5e-3` | 位相の学習率 |
| `--lpips_net` | `vgg` | LPIPS バックボーン |
| `--step_lr` | `True` | StepLR スケジューラを使用 |
| `--pretrained_path` | `''` | 事前学習済みモデルのパス |
| `--phase_path` | `./precomputed_phases` | 事前計算済み位相プールのパス |
| `--lambda_lpips_phase` | `0.1` | 位相最適化の LPIPS 重み λ（`MSE + λ × LPIPS`） |
| `--lambda_lpips_model` | `0.05` | モデル学習の LPIPS 重み λ（物理的忠実度優先のため小さめ） |
| `--slm_settle_time` | `0.3` | SLM 表示後、撮影までの待機時間 [秒] |
| `--homography_file` | `''` | ホモグラフィ行列 `.npy` ファイルのパス |
| `--slm_flip_udlr` | `True` | SLM 画像を 180° 反転して表示 |
| `--monitor_index` | `1` | SLM ウィンドウのモニタ番号 |
| `--camera_index` | `0` | Basler カメラのデバイス番号 |
| `--pixel_format` | `RGB8` | Basler ピクセルフォーマット |
| `--pixel_pitch` | `6.4e-6` | SLM ピクセルピッチ [m] |

#### 学習の流れ（1バッチあたり）

```
Stage 1: 位相最適化
  シミュレーション振幅 vs 目標画像 → MSE + λ_phase × LPIPS 最小化 → 位相を更新

Stage 2: 物理表示・撮影
  最適化した位相を SLM に表示 → カメラで撮影

Stage 3: モデル学習
  シミュレーション振幅 vs カメラ撮影 → MSE + λ_model × LPIPS 最小化 → モデルを更新
```

---

### 評価 (eval.py)

最適化済み位相パターンを再構成し、LPIPS と SSIM を3ドメインで計算します。

#### ASM シミュレーションで評価

```bash
python eval.py \
  --channel=1 \
  --prop_model=ASM \
  --root_path=./phases/SGD_ASM/green \
  --lpips_net=alex
```

#### CITL 校正済みモデルで評価

```bash
python eval.py \
  --channel=1 \
  --prop_model=MODEL \
  --root_path=./phases \
  --prop_model_dir=./calibrated_models \
  --lpips_net=alex
```

#### 実カメラで評価

```bash
python eval.py \
  --channel=1 \
  --prop_model=CAMERA \
  --root_path=./phases \
  --calibration_path=./calibration
```

#### 主なオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--channel` | `1` | 波長チャンネル (3 = RGB) |
| `--prop_model` | `ASM` | 評価モデル: `ASM` / `MODEL` / `CAMERA` |
| `--root_path` | `./phases` | 位相パターンの保存ディレクトリ |
| `--lpips_net` | `alex` | LPIPS バックボーン |

#### 出力

- `./recon/` — 再構成 sRGB 画像 (`.png`)
- `./recon/metrics_*.mat` — LPIPS・SSIM・PSNR の数値データ

**`.mat` ファイルの構造:**

```
img_idx          : 画像インデックスのリスト
lpips_amp        : LPIPS (振幅ドメイン)
lpips_lin        : LPIPS (線形強度ドメイン)
lpips_srgb       : LPIPS (sRGB ドメイン) ← 主評価指標
ssims_amp        : SSIM (振幅ドメイン)
ssims_lin        : SSIM (線形強度ドメイン)
ssims_srgb       : SSIM (sRGB ドメイン)
psnrs_amp        : PSNR (振幅ドメイン, dB)
psnrs_lin        : PSNR (線形強度ドメイン, dB)
psnrs_srgb       : PSNR (sRGB ドメイン, dB) ← 既存研究との比較用
```

---

## 主な変更点

元の neural-holography リポジトリからの差分です。

### `utils/utils.py`

- `LPIPSLoss` クラスを `CombinedLoss` クラスに変更（`MSE + λ × LPIPS` 損失）
  - `lambda_lpips=0` で純粋な MSE として動作（アブレーション実験対応）
  - LPIPS ネットワークのパラメータは `requires_grad=False` で固定
  - `forward()` 後に `self.last_mse` / `self.last_lpips` でコンポーネント値を参照可能
- `get_psnr_ssim()` → `get_lpips_ssim()` に置換
- `write_sgd_summary()` / `write_gs_summary()` の TensorBoard ログを PSNR → LPIPS に変更

### `eval.py`

- LPIPS / SSIM に加え **PSNR** を3ドメインで計算・出力（既存研究との比較用）
- `.mat` 出力に `psnrs_*` キーを追加

### `main.py`

```python
# 変更前
loss = utils.LPIPSLoss(net=opt.lpips_net).to(device)
# 変更後
loss = utils.CombinedLoss(net=opt.lpips_net, lambda_lpips=opt.lambda_lpips).to(device)
```

引数 `--lambda_lpips`（デフォルト `0.1`）および SLM/カメラ関連引数を追加。

### `train_model.py`

```python
# 変更前
loss_phase = utils.LPIPSLoss(net=opt.lpips_net).to(device)
loss_model = utils.LPIPSLoss(net=opt.lpips_net).to(device)
# 変更後
loss_phase = utils.CombinedLoss(net=opt.lpips_net, lambda_lpips=opt.lambda_lpips_phase).to(device)
loss_model = utils.CombinedLoss(net=opt.lpips_net, lambda_lpips=opt.lambda_lpips_model).to(device)
```

引数 `--lambda_lpips_phase`（デフォルト `0.1`）と `--lambda_lpips_model`（デフォルト `0.05`）および SLM/カメラ関連引数を追加。
モデル学習の λ を位相最適化より小さくしている理由：モデルはカメラ画像への物理的忠実度が重要なため。

TensorBoard ログでは各損失の MSE 値・LPIPS 値・合計値を個別に記録します。

### `utils/modules.py` (新規)

元の HOLOEYE SDK + FLIR PyCapture2 ベースの `PhysicalProp` を差し替え。

```
旧: HOLOEYE SDK で SLM 表示 + FLIR PyCapture2 でカメラ撮影
新: OpenCV (cv2.imshow) でセカンドモニタに SLM 表示 + pypylon で Basler カメラ撮影
```

- `SGD` / `GS` / `DPAC` は元の `_modules_orig.py` から再インポート（動作を維持）
- `PhysicalProp.forward()`: 位相テンソル (1,1,H,W) → 振幅テンソル (1,1,H,W)
- ホモグラフィ行列を `.npy` ファイルから読み込んで warpPerspective 適用
- `camera_prop.alc.disconnect()` 呼び出しは `_DummyALC` スタブで互換対応

### `utils/slm_display_module.py` (新規)

- `screeninfo.get_monitors()` でセカンドモニタを検出
- `cv2.namedWindow` + `cv2.setWindowProperty(FULLSCREEN)` でフルスクリーン表示
- `slm_flip_udlr=True` で 180° 反転（`cv2.flip(img, -1)`）

### `utils/camera_capture_module.py` (新規)

- `pypylon` で Basler カメラを初期化（RGB8 / BGR8 / Mono8 自動フォールバック）
- `grab_gray()`: フレームをグレースケール float32 \[0, 1\] で返す
- pypylon の RGB8 出力を `cv2.cvtColor(RGB→BGR)` で変換

---

## ライセンス

このコードは元の neural-holography リポジトリのライセンス [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) に従い、**非商用利用のみ**可能です。

このコードを利用・引用する場合は以下を引用してください：

```bibtex
@article{Peng:2020:NeuralHolography,
  author  = {Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein},
  title   = {{Neural Holography with Camera-in-the-loop Training}},
  journal = {ACM Trans. Graph. (SIGGRAPH Asia)},
  year    = {2020},
}

@inproceedings{zhang2018unreasonable,
  title     = {The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author    = {Zhang, Richard and Isola, Phillip and Efros, Alexei A and Shechtman, Eli and Wang, Oliver},
  booktitle = {CVPR},
  year      = {2018},
}
```
