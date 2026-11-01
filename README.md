# Synthetic Shape Segmentation with MiniUNet

A small PyTorch project that generates grayscale images containing circles, squares, and triangles, then trains a compact U-Net to segment each shape from its background.

This is **binary foreground/background segmentation**: mask pixels are `0` for background and `1` for the shape. Shape names are stored separately to group evaluation results; the network does not predict three shape classes.

## Quick start

Run these commands from the repository directory, preferably in a Python virtual environment:

```bash
python -m pip install -r requirements.txt
python generate_data.py
python train_and_eval.py
```

The training script automatically uses CUDA when `torch.cuda.is_available()` is true and otherwise runs on the CPU. Matplotlib uses the `Agg` backend, so figures are saved without opening a display window.

## Files

| File | Purpose |
| --- | --- |
| `generate_data.py` | Generate synthetic images, binary masks, and shape names. |
| `train_and_eval.py` | Train MiniUNet, print segmentation metrics, and save visualizations. |
| `requirements.txt` | Direct third-party dependencies. |

## Dataset and model

Each generated sample contains one bright shape on a dark, noisy `128 × 128` grayscale background. The generator cycles through circles, squares, and triangles. Squares and triangles have randomized orientations, and shape positions and sizes vary.

MiniUNet has two encoder stages, a bottleneck, and two decoder stages with skip connections. Each convolution block contains two `3 × 3` convolutions with ReLU activations. Max pooling downsamples the features, transposed convolutions upsample them, and a final sigmoid produces a foreground probability for every pixel.

## Defaults

| Setting | Value |
| --- | --- |
| Training / test samples | 1,000 / 200 |
| Image size | `128 × 128` |
| Base channel width | 32 (encoder: 32, 64; bottleneck: 128) |
| Epochs | 15 |
| Batch size | 16 |
| Optimizer | Adam, learning rate `0.001` |
| Loss | Binary cross-entropy (`BCELoss`) |
| Input normalization | Pixel values divided by 255 |
| Foreground threshold | Probability strictly greater than `0.5` |
| NumPy / PyTorch seed values | 2026 |

These settings and output paths are defined directly in the scripts; there are no command-line options. The test split is evaluated after every epoch and used for the test-loss curve. There is no separate validation split.

## Outputs

Running `generate_data.py` creates six NumPy arrays in `dataset/`:

- `train_images.npy`, `train_masks.npy`, `train_shapes.npy`
- `test_images.npy`, `test_masks.npy`, `test_shapes.npy`

Running `train_and_eval.py` prints training and test loss for each epoch, then pixel-level TP, FP, FN, TN, accuracy, and foreground intersection over union (IoU) for both splits. It also prints test accuracy and IoU separately for circles, squares, and triangles, grouped by each image's shape name.

The script saves two figures in the repository directory:

- `loss_curve.png`: training loss per optimizer update and test loss per epoch.
- `samples.png`: input image, ground-truth mask, and predicted mask for two test samples of each shape.

Model weights are not saved by the current script. Generated data, figures, and common checkpoint files are excluded by `.gitignore`.
