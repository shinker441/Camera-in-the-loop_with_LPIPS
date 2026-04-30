# Neural Holography — CITL + LPIPS 改造版

元論文: [Neural Holography with Camera-in-the-loop Training](http://www.computationalimaging.org/publications/neuralholography/) (SIGGRAPH Asia 2020)  
著者: Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein (Stanford)

本リポジトリは上記の公式コードを実験環境に合わせて改造したものです。主な変更点を以下に示します。

---

## 主な改造内容

| 項目 | オリジナル | 本リポジトリ |
|------|-----------|-------------|
| 損失関数 | MSE のみ | MSE + λ × LPIPS (組み合わせ損失) |
| SLM 制御 | HOLOEYE SDK | OpenCV フルスクリーンウィンドウ |
| カメラ制御 | FLIR PyCapture2 | Basler pypylon |
| ホモグラフィ較正 | 付属ツールなし | `calibrate_homography_4pt.py` |
| 評価指標 | PSNR | LPIPS + SSIM (PSNR も保持) |

---

## 動作要件

- Python 3.8 以上
- PyTorch >= 1.7.0 (Complex64 テンソルが必要)
- CUDA 環境推奨

```
pip install torch torchvision
pip install lpips scikit-image opencv-python screeninfo configargparse tensorboard
pip install pypylon-pylon   # Basler カメラを使う場合のみ
```

または conda 環境を使う場合:

```
conda env create -f environment_windows.yml
conda activate neural-holography
```

---

## ファイル構成

### メインスクリプト

| ファイル | 説明 |
|---------|------|
| `main.py` | 位相パターン生成 (SGD / GS / DPAC / HoloNet / U-Net) |
| `eval.py` | 最適化済み位相パターンの評価 (ASM シミュレーション / カメラ実測) |
| `train_holonet.py` | HoloNet の学習 |
| `train_model.py` | CITL を用いた波動伝搬モデルの学習 |
| `algorithms.py` | GS / SGD / DPAC アルゴリズム実装 |
| `propagation_ASM.py` | 角スペクトル法 (ASM) 波動伝搬演算子 |
| `propagation_model.py` | パラメータ化波動伝搬モデル |
| `holonet.py` | HoloNet / U-Net モジュール |
| `calibrate_homography_4pt.py` | SLM-カメラ間のホモグラフィ較正ツール |

### utils/

| ファイル | 説明 |
|---------|------|
| `utils.py` | ユーティリティ関数 + `CombinedLoss` (MSE + λ × LPIPS) |
| `modules.py` | SGD / GS / DPAC / `PhysicalProp` (SLM + Basler) ラッパー |
| `slm_display_module.py` | OpenCV フルスクリーンウィンドウで SLM 制御 (`SLMDisplay`) |
| `camera_capture_module.py` | Basler カメラ pypylon インターフェース (`BaslerCamera`) |
| `calibration_module.py` | ホモグラフィ較正モジュール |
| `augmented_image_loader.py` | 画像データセットローダー |
| `perceptualloss.py` | VGG19 知覚損失 (旧版。現在は `CombinedLoss` 推奨) |
| `utils_tensorboard.py` | TensorBoard 可視化ユーティリティ |
| `_modules_orig.py` | オリジナルの modules.py バックアップ |

---

## 損失関数: MSE + λ × LPIPS

`utils/utils.py` の `CombinedLoss` クラスが本改造の核心です。

```
Loss = MSE(s·amp_recon, amp_target) + λ_lpips × LPIPS(s·amp_recon, amp_target)
```

- **MSE**: ピクセル単位の安定した勾配を提供し、収束を保証する
- **LPIPS**: 知覚的品質へ最適化を誘導する
- `lambda_lpips=0` に設定すると純粋な MSE に戻る (アブレーション用)
- TensorBoard に MSE / LPIPS / 合計損失が個別に記録される

---

## 実行方法

### 1) 位相最適化 (`main.py`)

#### コード内設定 (推奨)

`main.py` 冒頭の `CODE_SETTINGS` を直接編集して実行するだけでよい。

```python
USE_CODE_SETTINGS = True

CODE_SETTINGS = {
    'channel': 1,          # 0=赤, 1=緑, 2=青
    'method': 'SGD',       # GS / SGD / DPAC / HOLONET / UNET
    'prop_model': 'ASM',   # ASM / MODEL
    'citl': False,         # True でカメラ実機ループを使う
    'num_iters': 500,
    'lambda_lpips': 0.1,
    'lpips_net': 'vgg',    # vgg / alex
    ...
}
```

```
python main.py
```

#### CLI 引数での実行 (互換)

```bash
# SGD + ASM (ソフトウェアのみ)
python main.py --channel=1 --method=SGD --prop_model=ASM --root_path=./phases

# SGD + CITL (実機ループ)
python main.py --channel=1 --method=SGD --citl=True --homography_file=./calibration/homography.npy

# GS
python main.py --channel=1 --method=GS --root_path=./phases

# DPAC
python main.py --channel=1 --method=DPAC --root_path=./phases

# HoloNet (事前学習済み重みが必要)
python main.py --channel=1 --method=HOLONET --generator_dir=./pretrained_networks

# U-Net (事前学習済み重みが必要)
python main.py --channel=1 --method=UNET --generator_dir=./pretrained_networks
```

主なパラメータ:

| 引数 | 説明 | デフォルト |
|------|------|-----------|
| `--channel` | カラーチャネル (0=R, 1=G, 2=B) | 1 |
| `--method` | アルゴリズム | SGD |
| `--prop_model` | 伝搬モデル (ASM / MODEL) | ASM |
| `--citl` | カメラ in the loop 有効化 | False |
| `--num_iters` | 反復回数 | 500 |
| `--lr` | 位相変数の学習率 | 8e-3 |
| `--lr_s` | スケール学習率 | 2e-3 |
| `--lambda_lpips` | LPIPS の重み λ | 0.1 |
| `--lpips_net` | LPIPS バックボーン (vgg / alex) | vgg |

### 2) 評価 (`eval.py`)

```bash
# ASM シミュレーション
python eval.py --channel=1 --root_path=./phases/hwcheck_SGD_ASM_citl --prop_model=ASM

# CITL 較正済みモデルでシミュレーション
python eval.py --channel=1 --root_path=./phases/hwcheck_SGD_ASM_citl --prop_model=MODEL

# 実機カメラで評価
python eval.py --channel=1 --root_path=./phases/hwcheck_SGD_ASM_citl --prop_model=CAMERA
```

評価指標として LPIPS (低いほど良い) / SSIM / PSNR が出力される。

---

## ハードウェア設定 (CITL)

`citl=True` にする場合は、以下のハードウェアパラメータを `CODE_SETTINGS` または CLI 引数で指定する。

| パラメータ | 説明 | 現環境の値 |
|-----------|------|-----------|
| `monitor_index` | SLM 表示モニタ番号 | 1 (第2モニタ) |
| `camera_index` | Basler デバイス番号 | 0 |
| `pixel_format` | Basler 画素フォーマット | `Mono8` |
| `pixel_pitch` | SLM ピクセルピッチ [m] | 6.4e-6 |
| `slm_flip_udlr` | SLM を 180° 反転 | True |
| `slm_settle_time` | SLM 更新後の待機時間 [s] | 0.3 |
| `homography_file` | ホモグラフィ行列ファイルパス | `./calibration/homography.npy` |

#### 光学定数 (コード上部で直接編集)

```python
PROP_DIST_BY_CHANNEL  = (0.20, 1.0, 0.20)           # 伝搬距離 [m] (R, G, B)
WAVELENGTH_BY_CHANNEL = (638e-9, 520e-9, 450e-9)      # 波長 [m]
SLM_RES               = (1080, 1920)                  # SLM 解像度 (H, W)
ROI_RES               = (880, 1600)                   # 関心領域 (H, W)
```

---

## ホモグラフィ較正

`calibrate_homography_4pt.py` を使ってカメラと SLM 間のホモグラフィ行列を求める。

```
python calibrate_homography_4pt.py
```

1. SLM に較正パターン (4 隅の円マーカー) が表示される
2. Basler カメラで撮影した画像が別ウィンドウに表示される
3. **TL → TR → BR → BL の順** で 4 点をクリックする
4. `s` キーで保存、`r` キーでリセット、ESC でキャンセル
5. ホモグラフィ行列が `./calibration/homography.npy` に保存される

設定はファイル上部で変更できる:

```python
MONITOR_INDEX = 1        # SLM モニタ
CAMERA_INDEX  = 0        # Basler
PIXEL_FORMAT  = "Mono8"
MARGIN        = 120      # マーカー余白 [px]
```

---

## 学習

### 波動伝搬モデルの学習 (CITL)

```
python train_model.py --channel=1
```

### HoloNet の学習

```
# ASM を使った学習
python train_holonet.py --channel=1 --run_id=my_holonet --batch_size=4 --perfect_prop_model=True

# CITL 較正済みモデルを使った学習
python train_holonet.py --channel=1 --run_id=my_holonet --perfect_prop_model=False --model_path=./calibrated_models
```

---

## TensorBoard

最適化の経過は TensorBoard で確認できる。

```
tensorboard --logdir=./phases
```

SGD の場合、以下が記録される:
- `loss/mse` — MSE 損失
- `loss/lpips` — LPIPS 損失
- `loss/total` — 合計損失 (MSE + λ × LPIPS)

---

## ディレクトリ構成 (実行後)

```
./phases/           最適化した位相パターン (.png)
./recon/            シミュレーション再構成画像
./calibration/      ホモグラフィ行列 (homography.npy)
./pretrained_networks/  HoloNet / U-Net 事前学習済み重み
./calibrated_models/    CITL 較正済み伝搬モデル重み
./data/             入力画像
```

---

## 引用

本コードを利用する場合は、オリジナル論文を引用してください。

```bibtex
@article{Peng:2020:NeuralHolography,
  author  = {Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein},
  title   = {Neural Holography with Camera-in-the-loop Training},
  journal = {ACM Trans. Graph. (SIGGRAPH Asia)},
  volume  = {39},
  number  = {6},
  year    = {2020},
}
```

LPIPS を使用する場合は以下も引用してください。

```bibtex
@inproceedings{Zhang:2018:LPIPS,
  author    = {R. Zhang, P. Isola, A. Efros, E. Shechtman, O. Wang},
  title     = {The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  booktitle = {CVPR},
  year      = {2018},
}
```

---

## ライセンス

本リポジトリのコードは CC BY-NC 4.0 (非商用限定) でリリースされています。  
商用利用を希望する場合は Stanford University にお問い合わせください。

オリジナルコードの著作権は Stanford University (2020) に帰属します。
