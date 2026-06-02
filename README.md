# Rethinking Multi-view Mammogram Representation Learning via Counterfactual Reasoning with Kolmogorov-Arnold Theorem

This project trains and evaluates a MammoCRKAN model for mammography classification. The main training entry is `train_ddsm_ours.py`; the core model is `GMIC/src/modeling/double_gmic.py`, where the final fusion classifier uses `KAN` from `models/KANLinear.py`.

## Environment

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The code expects PyTorch with CUDA when training on GPU. CPU execution is supported for debugging, but full training is intended for GPU.

## Data Layout

By default, the scripts look for:

```text
ddsm_csv/
  newtrain1.csv
  newvalid1.csv
  newtest1.csv
ddsm_images/
  all_imgs/
result_ckp/
```

Each CSV should contain these columns:

```text
patient_id,image_id,laterality,view,cancer
```

Images are matched with filenames in this format:

```text
{patient_id}_{laterality}_{view}_{image_id}
```

Common extensions such as `.png`, `.jpg`, `.jpeg`, `.tif`, and `.tiff` are supported.

## Training

Run training with the default paths:

```bash
python train_ddsm_ours.py
```

Example with custom paths:

```bash
python train_ddsm_ours.py \
  --train-csv /path/to/train.csv \
  --valid-csv /path/to/valid.csv \
  --test-csv /path/to/test.csv \
  --images-dir /path/to/images \
  --output-dir /path/to/output \
  --batch-size 8 \
  --epochs 80
```

Outputs are saved to `result_ckp/` by default:

```text
best_model.pth
last_model.pth
history.json
loss.png
```

## Evaluation

Evaluate a trained checkpoint:

```bash
python test_ddsm_ours.py \
  --checkpoint result_ckp/best_model.pth \
  --test-csv ddsm_csv/newtest1.csv \
  --images-dir ddsm_images/all_imgs
```

## Main Files

```text
train_ddsm_ours.py                 Training entry point
test_ddsm_ours.py                  Evaluation entry point
multi_dataset_ddsm.py              DDSM double-view dataset and dataloaders
GMIC/src/modeling/double_gmic.py   Double-view GMIC model
GMIC/src/modeling/modules.py       GMIC network modules
models/KANLinear.py                KAN and KANLinear implementation
requirements.txt                   Python dependencies
```

## Notes

- The model expects each sample to contain two views with tensor shape `[2, 1, H, W]`.
- Labels default to three classes: `benign_without_callbacks`, `benigns`, and `cancers`.
- Use command-line arguments to override paths, image size, crop size, learning rate, KAN settings, and OFU settings.
