# Foundation X+ — Change Log

This file documents all code changes made to the Foundation X+ codebase, with rationale and file-level details.

---

## Change 1 — CANDID-PTX Classification: FileNotFoundError Fix

### Problem

During classification training, the CANDID-PTX dataloader raised:
```
FileNotFoundError: Image not found for path
  '/scratch/sejong/class-dataset/CANDID-PTX/png/2.8.17...'
```

The split `.txt` files contain DICOM-style SOP UID filenames with no extension. The PNG directory had matching filenames with `.png` extension, but the classification dataloader was configured to look in `images` (PNG dir) rather than `images_dicom` (DICOM dir), where bare filenames would match correctly.

### Root cause

`config/dataset_locations_asu_sol.yml` had `use_dicom: false` for `classification.candid_ptx`. This caused `datasets_medical.py` to use the `images` key (PNG dir). The filenames in the split files had no extension, so no PNG file was found.

### Fix

**File:** `config/dataset_locations_asu_sol.yml`
```yaml
# Before:
candid_ptx:
  use_dicom: false

# After:
candid_ptx:
  use_dicom: true
  images_dicom: /scratch/sejong/class-dataset/CANDID-PTX/dataset
  images: /scratch/sejong/class-dataset/CANDID-PTX/png
```

**File:** `datasets_medical.py`

The inline `_load_dataset_locations()` and `_dataset_location()` helpers were extracted to a shared utility module (see Change 2). The CANDID-PTX classification dataloader now reads the `use_dicom` flag:
```python
use_dicom = _dataset_location("classification", "candid_ptx", "use_dicom", default=False)
images_key = "images_dicom" if use_dicom else "images"
root_dir = _dataset_location("classification", "candid_ptx", images_key, default=FALLBACK)
```

---

## Change 2 — Shared YAML Path Loader (`util/dataset_locations.py`)

### Problem

Both `datasets_medical.py` (classification / segmentation) and `datasets/coco.py` (localization) need to resolve dataset paths from the same YAML config. A direct import of `datasets_medical` from `datasets/coco.py` would create a circular import.

### Fix

**New file created:** `util/dataset_locations.py`

Extracted the path-resolution helpers into a neutral module under `util/`:

```python
def _load_dataset_locations() -> dict:
    """Load and cache the YAML config file."""
    ...

def _dataset_location(*keys: str, default=None):
    """Look up a nested key path in the loaded YAML, return default if missing."""
    ...
```

The YAML file path is resolved at runtime:
1. If `FOUNDATION_X_DATASET_LOCATIONS_YML` env var is set → use that path
2. Otherwise default to `config/dataset_locations_asu_sol.yml`
3. If relative, resolve relative to the project root (parent of `util/`)

**Files updated to import from the new module:**
- `datasets_medical.py`: replaced inline helpers with `from util.dataset_locations import _dataset_location, _load_dataset_locations`
- `datasets/coco.py`: added `from util.dataset_locations import _dataset_location`

---

## Change 3 — Localization Dataset Path Overrides in `datasets/coco.py`

### Problem

The `PATHS` dict in `datasets/coco.py` had hardcoded image directory and annotation JSON paths for all six medical localization datasets (TBX11K, NODE21, ChestX-Det, RSNA Pneumonia, SIIM-ACR PTX, CANDID-PTX). These paths pointed to the DFS cluster layout and would break on the ASU SOL scratch filesystem.

### Fix

**File:** `datasets/coco.py`

All six datasets' `PATHS` entries now call `_dataset_location("localization", ...)` with the old hardcoded `DATASETS_CONFIG` values as fallback:

```python
# Example pattern for TBX11K:
PATHS = {
    "tbx11k_train": (
        _dataset_location("localization", "tbx11k", "images",
                          default=DATASETS_CONFIG["tbx11k_catagnostic"]["img_path"]),
        _dataset_location("localization", "tbx11k", "splits", "train_json",
                          default=DATASETS_CONFIG["tbx11k_catagnostic"]["ann_path_train"]),
    ),
    ...
}
```

Datasets with split-specific image directories (ChestX-Det, SIIM-ACR PTX) use separate `images_train` / `images_test` / `images_val` YAML keys.

**File:** `config/dataset_locations_asu_sol.yml`

Added a top-level `localization:` section with entries for all six datasets:

```yaml
localization:
  tbx11k:
    images: /scratch/sejong/class-dataset/TBX11K/imgs
    splits:
      train_json: .../TBX11K/annotations/json/all_train.json
      test_json:  .../TBX11K/annotations/json/all_val.json

  node21:
    images: /scratch/sejong/class-dataset/NODE21/png_images
    splits:
      train_json: .../Node21_Nodule_Bbox_NAD_train_3.json
      test_json:  .../Node21_Nodule_Bbox_NAD_test_3.json

  chestx_det:
    images_train: /scratch/sejong/class-dataset/ChestX-Det/train
    images_test:  /scratch/sejong/class-dataset/ChestX-Det/test
    splits:
      train_json: .../ChestX_det_train_NAD_v2.json
      test_json:  .../ChestX_det_test_NAD_v2.json

  rsna_pneumonia:
    images: /scratch/sejong/class-dataset/rsna-pneumonia/stage_2_train_images_png
    splits:
      train_json/val_json/test_json: ...

  siim_acr_ptx:
    images_train: /scratch/sejong/class-dataset/SIIM-ACR-Pneumothorax/train_jpeg
    images_val:   /scratch/sejong/class-dataset/SIIM-ACR-Pneumothorax/val_jpeg
    images_test:  /scratch/sejong/class-dataset/SIIM-ACR-Pneumothorax/test_jpeg
    splits:
      train_json/val_json/test_json: ...

  candid_ptx:
    images: /scratch/sejong/class-dataset/CANDID-PTX/png
    splits:
      train_json/val_json/test_json: ...
```

---

## Change 4 — DN-Aware EMA Consistency Loss in `engine.py`

### Problem

During localization training with `--modelEMA True_Epoch` and `--use_dn True`, the training crashed with:

```
RuntimeError: The size of tensor a (1100) must match the size of tensor b (900)
at non-singleton dimension 2
```

This occurred in `criterionMSE(features_cons, features_cons_ema)`.

### Root cause

DINO denoising (CDN) is controlled by the model's `self.training` flag:
- **Student** (`model.train()`): `prepare_for_cdn` prepends `pad_size=200` denoising queries → decoder hidden states have shape `[B, num_layers, pad_size + num_queries, d_model]` = `[B, 6, 1100, 256]`
- **EMA teacher** (`model_ema.eval()`): `prepare_for_cdn` is skipped → `[B, 6, 900, 256]`

When both tensors are passed to `nn.MSELoss`, the shape mismatch at dimension 2 raises the error.

### Fix

**File:** `engine.py`, inside `train_one_epoch()`, after the EMA teacher forward pass:

```python
if model_ema is not None:
    model_ema.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                _, features_cons_ema, features_Encons_ema = model_ema(samples, targets)
            else:
                _, features_cons_ema, features_Encons_ema = model_ema(samples)

    # DN-aware slice: strip DN prefix from student decoder features.
    # The student ran in train mode, so prepare_for_cdn prepended pad_size
    # denoising queries (indices 0..pad_size-1).  The EMA teacher ran in
    # eval mode and got none, so its decoder output is num_queries tokens.
    # Slice off the DN prefix so both tensors have matching shape.
    dn_meta = outputs.get('dn_meta')
    if dn_meta is not None and dn_meta.get('pad_size', 0) > 0:
        pad_size = dn_meta['pad_size']
        features_cons = features_cons[:, :, pad_size:, :]

    loss_cons = criterionMSE(features_cons, features_cons_ema)
    loss_cons_2 = criterionMSE(features_Encons, features_Encons_ema)
    losses = (1-coff)*losses + coff*((loss_cons + loss_cons_2)/2)
```

Note: `features_Encons` (encoder features) is not affected because DN tokens are added only to the decoder input, not the encoder. The encoder features from student and teacher remain the same shape.

---

## Summary Table

| # | File(s) Modified | Change | Reason |
|---|-----------------|--------|--------|
| 1 | `config/dataset_locations_asu_sol.yml` | `use_dicom: false` → `true` for `candid_ptx` | Fix `FileNotFoundError` in CANDID-PTX classification loader |
| 2 | `util/dataset_locations.py` *(new)* | Extracted shared YAML loader from `datasets_medical.py` | Allow `datasets/coco.py` to use the same path logic without circular import |
| 2 | `datasets_medical.py` | Replaced inline helpers with import from `util.dataset_locations` | Code deduplication |
| 3 | `datasets/coco.py` | Route all 6 localization dataset paths through `_dataset_location()` | Make paths portable across clusters via YAML |
| 3 | `config/dataset_locations_asu_sol.yml` | Added `localization:` section for all 6 datasets | ASU SOL scratch filesystem path definitions |
| 4 | `engine.py` | Added DN-aware `features_cons` slice after EMA forward | Fix `RuntimeError: size mismatch` when DN and EMA are both enabled |
| 5 | `main_Consolidated.py` | Wrapped bounding box inference passes in `torch.no_grad()` | Fix DDP `RuntimeError: Expected to mark a variable ready only once` by preventing DDP hook registration for non-backpropagated parameters |
## Change 5 — DDP Parameter Ready Twice Error in Segmentation Training

### Problem

During segmentation training with `DistributedDataParallel` (DDP), the training crashed with:

```
RuntimeError: Expected to mark a variable ready only once. This error is caused by one of the following reasons: 1) Use of a module parameter outside the `forward` function. Please make sure model parameters are not shared across multiple concurrent forward-backward passes.
...
Parameter at index 747 with name backbone.0.segmentation_FPN.conv_fusion.1.weight has been marked as ready twice.
```

This occurred during the backward pass in `train_one_epoch_SEGMENTATION_SharedLocSeg` inside `main_Consolidated.py`.

### Root cause

In `train_one_epoch_SEGMENTATION_SharedLocSeg`, the script runs a detection forward pass to generate pseudo-label bounding boxes for `roi_align`.
```python
if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    model.task_DetHead = head_number
    model.eval()
    outputs, _, _ = model(samples)
```
Because `model(samples)` was not wrapped in `with torch.no_grad():`, PyTorch tracked the computational graph, and DDP registered synchronization hooks for the involved parameters. However, `outputs` is only used to compute `.tolist()` bounding boxes and is not backpropagated through. Later, when `model.module.backbone[0].extra_features_seg_sharedLocSeg(...)` is explicitly called for segmentation and its loss is backpropagated, PyTorch computes gradients for the same parameters. The active DDP hooks from the first unused forward pass fire, causing PyTorch to incorrectly assume that multiple backward passes (or reentrant ones) are improperly marking the variables ready twice.

### Fix

**File:** `main_Consolidated.py`

Wrapped the detection forward pass block inside `train_one_epoch_SEGMENTATION_SharedLocSeg` with `with torch.no_grad():` for both the main `model` and `model_ema`.

```python
if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    model.task_DetHead = head_number
    model.eval()
    with torch.no_grad():
        outputs, _, _ = model(samples) # outputs = [24, 900, 4]
```

This ensures PyTorch skips building a computational graph and does not trigger DDP's autograd hooks for the bounding box generation step, resolving the error and improving memory efficiency.

---
