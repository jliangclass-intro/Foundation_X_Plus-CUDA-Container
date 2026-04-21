# Foundation X+ Architecture

## Overview

Foundation X+ is a multi-task medical image foundation model that simultaneously performs **classification**, **localization (object detection)**, and **segmentation** on chest radiographs across multiple datasets. It is built on a modified DINO (Detection with Transformers) architecture with a Swin Transformer backbone.

---

## 1. Model Architecture

### 1.1 Active Model: `dino_F6_consolidated`

**File:** `models/dino/dino_F6_consolidated.py`  
**Registered as:** `modelname='dino'` in `models/dino/__init__.py`

**Forward signature:**
```python
outputs, hs_temp, hs_enc = model(samples, targets=None)
# outputs  — dict: pred_logits, pred_boxes, aux_outputs, dn_meta, ...
# hs_temp  — decoder hidden states (shape: [num_dec_layers, B, num_queries, d_model])
# hs_enc   — encoder hidden states
```

### 1.2 Backbone: Swin Transformer

- Default: `SwinL` (Large) or `SwinB` (Base) configured via `--backbonemodel`
- Multi-scale feature maps extracted at 4 levels → fed into deformable attention encoder

### 1.3 Task Heads

| Component | Name in code | Purpose |
|-----------|-------------|---------|
| Localization encoder | `transformer.encoder` | Deformable DETR encoder, 4-scale features |
| Localization decoder | `transformer.decoder` | Deformable DETR decoder with multi-head task queries |
| Content queries | `transformer.tgt_embed[i]` | Per-task learnable queries (one set per detection task) |
| BBox embed | `transformer.bbox_embed[i]` | Per-task bounding-box regression heads |
| Class embed | `transformer.class_embed[i]` | Per-task classification heads |
| Classification heads | `backbone[0].classification_heads[i]` | Multi-label CLS heads on backbone features |
| Segmentation PPN/FPN | `backbone[0].segmentation_PPN/FPN` | Pyramid feature decoder for segmentation |
| Segmentation heads | `backbone[0].segmentation_heads[i]` | Per-task sigmoid mask predictors |

### 1.4 Denoising (DN) — CDN / DINO-DN

When `use_dn=True` (training mode), `prepare_for_cdn` prepends `pad_size` noised GT queries to the decoder input.  
**Important:** this happens *only* when `self.training=True`. The EMA teacher model runs in `eval()` mode and receives no DN tokens.

```
Student (train):   [DN_queries | content_queries]  → decoder → hs_temp.shape[-2] = pad_size + num_queries
EMA teacher (eval): [content_queries only]          → decoder → hs_temp.shape[-2] = num_queries
```

---

## 2. Data Loading

### 2.1 Dataset Path Configuration

**File:** `util/dataset_locations.py`

Central YAML loader used by both classification/segmentation (`datasets_medical.py`) and localization (`datasets/coco.py`). Avoids circular imports.

```python
from util.dataset_locations import _dataset_location

# Looks up nested keys with dot-style fallback, returns default if missing.
path = _dataset_location("classification", "candid_ptx", "images_dicom", default=None)
```

YAML is selected via environment variable (default: `config/dataset_locations_asu_sol.yml`):
```bash
export FOUNDATION_X_DATASET_LOCATIONS_YML=/my/custom/paths.yml
```

**ASU SOL cluster YAML:** `config/dataset_locations_asu_sol.yml`  
**DFS cluster YAML:** `config/dataset_locations_dfs.yml`

### 2.2 YAML Structure

```yaml
classification:
  <dataset_key>:
    images: /path/...
    use_dicom: true/false          # for CANDID-PTX only
    images_dicom: /path/...        # used when use_dicom=true
    splits:
      train: ...
      val: ...
      test: ...
      train_json: ...              # localization split JSONs also stored here
      test_json: ...

localization:
  <dataset_key>:
    images: /path/...              # or images_train / images_test / images_val
    splits:
      train_json: ...
      val_json: ...
      test_json: ...

segmentation:
  <dataset_key>:
    images: /path/...
    ...
```

### 2.3 Classification Datasets (`datasets_medical.py`)

| Dataset | Key | Notes |
|---------|-----|-------|
| NIH ChestXray14 | `nih_chestxray14` | 14-class multilabel |
| CheXpert | `chexpert` | 14-class multilabel, test diseases subset |
| VinDR-CXR | `vindr_cxr` | 14-class multilabel |
| NIH Shenzhen | `shenzhen` | TB binary |
| MIMIC-CXR | `mimic_cxr` | 14-class multilabel |
| TBX11K | `tbx11k` | TB agnostic binary |
| NODE21 | `node21` | Nodule binary |
| ChestX-Det | `chestx_det` | 13-class |
| RSNA Pneumonia | `rsna_pneumonia` | 3-class multi-class (softmax) |
| SIIM-ACR PTX | `siim_acr_ptx` | Binary |
| CANDID-PTX | `candid_ptx` | Binary; reads DICOM when `use_dicom=true` |

**CANDID-PTX special path logic:**
```python
use_dicom = _dataset_location("classification", "candid_ptx", "use_dicom", default=False)
images_key = "images_dicom" if use_dicom else "images"
root_dir = _dataset_location("classification", "candid_ptx", images_key, default=FALLBACK)
```

### 2.4 Localization Datasets (`datasets/coco.py`)

All six medical detection datasets have their `PATHS` dict entries resolved through `_dataset_location("localization", ...)` with hardcoded `DATASETS_CONFIG` values as fallback.

| Dataset tag | YAML key | Split dirs |
|-------------|----------|------------|
| `tbx11k_*` | `localization.tbx11k` | single `images` dir |
| `node21_*` | `localization.node21` | single `images` dir |
| `candid_ptx_*` | `localization.candid_ptx` | single `images` dir |
| `chestxdet_*` | `localization.chestx_det` | `images_train` / `images_test` |
| `rsnapneumonia_*` | `localization.rsna_pneumonia` | single `images` dir |
| `siimacr_*` | `localization.siim_acr_ptx` | `images_train` / `images_val` / `images_test` |

### 2.5 Segmentation Datasets

Loaded via `datasets_medical.py` using the `segmentation` YAML section. Segmentation loader calls `model.backbone[0].extra_features_seg(img, head_n=head_number)`.

### 2.6 `dataloader_return()` Factory

**File:** `datasets_medical.py`

Central factory that reads `args.taskcomponent` and `args.dataset_file` and returns the appropriate combination of DataLoaders. For `foundation_x3_pretraining` / `foundation6Ark6_datasets`, it returns loaders for all 11 datasets simultaneously (5 CLS-only + 6 CLS+LOC+SEG).

---

## 3. Training Pipeline

### 3.1 Entry Point: `main()` in `main_Consolidated.py`

1. Parse args → load config → init distributed
2. Build model (`build_model_main(args)`)
3. Optionally build EMA teacher (`copy.deepcopy(model)`)
4. Build optimizer (per-component learning rates)
5. Load weights / resume checkpoint
6. Loop `for epoch in range(start_epoch, total_epochs):`

### 3.2 Cyclic Multi-Task Training

Controlled by `--cyclictask` flag (e.g. `nihchestxray14CLS_node21CLS_node21LOC_candidptxCLS_candidptxLOC_candidptxSEG`).

```python
ACTIVE_TASKS = [
    ("CHEXPERTCLS", 0), ("NIHCHESTXRAY14CLS", 1), ...,
    ("CANDIDPTXSEG", 11), ..., ("SIIMACRSEG", 19),
]
active_heads = [head for tag, head in ACTIVE_TASKS if tag in cyclictask.upper()]
```

Per epoch, the training loop:
1. Iterates over active classification tasks → calls `train_CLASSIFICATION()`
2. Iterates over active localization tasks → calls `train_one_epoch()`
3. Iterates over active segmentation tasks → calls `train_one_epoch_SEGMENTATION()` or `train_one_epoch_SEGMENTATION_SharedLocSeg()`

### 3.3 `train_one_epoch()` — Localization (engine.py)

```
for samples, targets in data_loader:
    outputs, features_cons, features_Encons = model(samples, targets)  # student forward
    loss_dict = criterion(outputs, targets)  # Hungarian matching + DN loss
    losses = sum(weighted losses)

    if model_ema:
        # teacher forward (no_grad, eval mode → no DN queries)
        _, features_cons_ema, features_Encons_ema = model_ema(samples)

        # DN-aware slice: strip pad_size DN tokens from student features
        dn_meta = outputs.get('dn_meta')
        if dn_meta and dn_meta.get('pad_size', 0) > 0:
            features_cons = features_cons[:, :, pad_size:, :]

        loss_cons   = MSE(features_cons, features_cons_ema)
        loss_cons_2 = MSE(features_Encons, features_Encons_ema)
        losses = (1-coff)*losses + coff*((loss_cons + loss_cons_2)/2)

    # L2 regulariser on localization encoder weights
    losses += 0.0001 * l2_regularizer(transformer.encoder weights)

    losses.backward()
    optimizer.step()

if epoch-based EMA:
    ema_update_teacher(model, model_ema, momentum=0.80)
```

### 3.4 EMA Teacher-Student

- `model_ema = copy.deepcopy(model)` at init
- Momentum: **0.80** (AdamW); **0.90** (SGD)
- Update: `param_k = m * param_k + (1-m) * param_q`
- EMA always in `.eval()` during training
- EMA requires-grad: False (no gradient flows through teacher)

### 3.5 Segmentation Training

**`train_one_epoch_SEGMENTATION()`**
- Calls `model.backbone[0].extra_features_seg(img, head_n=head_number)`
- Loss: Dice coefficient loss (`torch_dice_coef_loss`)
- EMA consistency: `MSE(features_backbone, features_backbone_ema)` + `MSE(features_cons, features_cons_ema)`
- Combined: `loss = (1-coff)*dice_loss + coff*((cons1 + cons2)/2)`

**`train_one_epoch_SEGMENTATION_SharedLocSeg()`**
- Extra step: runs localization forward pass to get bounding boxes
- Uses RoI-Align to focus segmentation on detected regions

### 3.6 Optimizers (per-component learning rates)

| Component | Optimizer group | Typical LR |
|-----------|-----------------|------------|
| Backbone (Swin) | `param_groups[0]` | `lr_backbone` (1e-5) |
| Segmentation PPN/FPN/Heads | `param_groups[1]` | `lr_segmentor` |
| Localization Encoder | `param_groups[2]` | `lr_locEnc` |
| Localization Decoder + Queries + Embeds | `param_groups[3+]` | `lr_locDec` |

LR scheduler: `StepLR(drop=15)` by default; `OneCycleLR` or `MultiStepLR` optional.

### 3.7 Freeze / Unfreeze Utilities (`main_Consolidated.py`)

| Function | Effect |
|----------|--------|
| `Freeze_Backbone_and_Localization_Encoder(model)` | Freeze backbone + loc. encoder |
| `Freeze_Backbone_unFreezeClassifierHeads(model)` | Only train classification heads |
| `Freeze_Backbone_SegmentationDecoder(model)` | Only train segmentation heads |
| `Freeze_Backbone(model)` / `Freeze_Localization_Encoder(model)` | Modular freezing |
| `unFreeze_*` variants | Reverse of above |

---

## 4. Evaluation Pipeline

### 4.1 Classification Evaluation

**Function:** `evaluateClsSepFunc()` / `evaluate_CLASSIFICATION()` / `test_CLASSIFICATION()`

```
y_test, p_test = test_CLASSIFICATION(dataset, loader, model, head_number, args)
AUC per class  = roc_auc_score(y_test[:, i], p_test[:, i])
mean AUC       = np.array(individual_results).mean()
```

Metrics written to:
- `export_csvFile.csv` — `[Epoch, Dataset, Task-Train, Model, Task-Test, AUC, -, -, -, -]`
- `resultsTEST.txt` — human-readable

**Student vs Teacher:** both models evaluated every epoch; teacher (EMA) typically outperforms student in later epochs.

### 4.2 Localization Evaluation

**Function:** `evaluateLocSepFunc()` calls `evaluate()` from `engine.py`

Uses COCO evaluation (IoU-based mAP):
- **mAP@40** (`value[1]`)
- **mAP@50** (`value[2]`)
- **mAP@50:95** (`value[0]`)

Written to `export_csvFile.csv` and `resultsTEST.txt`.

### 4.3 Segmentation Evaluation

**Function:** `evaluateSegSepFunc()` calls `test_SEGMENTATION()`

```
test_y, test_p, _ = test_SEGMENTATION(model, loader, head_number)
dice_score        = 100.0 * dice_score(test_p, test_y)
mean_dice_score   = 100.0 * mean_dice_coef(test_y > 0.5, test_p > 0.5)
```

Written to `export_csvFile.csv`.  
For multi-channel ChestX-Det: per-class Dice computed and averaged.

### 4.4 Output Files

| File | Contents |
|------|---------|
| `export_csvFile.csv` | All results: epoch, dataset, model (student/teacher), task, metric values |
| `export_csvFile_TRAIN.csv` | Training losses per epoch |
| `resultsTEST.txt` | Raw COCO stats dict, human-readable |
| `Logs/log.txt` | Full training log |
| `checkpoint.pth` | Latest checkpoint (model + optimizer + epoch) |
| `checkpoint_best_regular.pth` | Best student mAP checkpoint |
| `seed_val.txt` | Random seed used |

---

## 5. Container / Execution

### 5.1 Apptainer (Singularity) Container

```bash
apptainer exec --nv \
  --bind /scratch/$USER:/scratch/$USER \
  --bind /data/jliang12:/data/jliang12 \
  apptainer-cuda.sif \
  bash scripts-apptainer/run_apptainer_v108.sh
```

Container definition: `apptainer-cuda.def`  
Pre-built image: `apptainer-cuda.sif`

### 5.2 SLURM Launch

Typical launch via `scripts/run_IntegratedModel_Foundation6_ClsLocSeg_v106.sh`:
```bash
srun --gpus=4 --nodes=1 torchrun --nproc_per_node=4 \
  main_Consolidated.py \
  --taskcomponent foundation_x3_pretraining \
  --dataset_file foundation6Ark6_datasets \
  --cyclictask nihchestxray14CLS_chexpertCLS_... \
  --modelname dino \
  --config_file config/DINO/DINO_4scale_swinBASE.py \
  --modelEMA True_Epoch \
  ...
```

### 5.3 Distributed Training

Uses PyTorch DDP (`DistributedDataParallel`). Model accessed via:
```python
model.module.backbone[0]   # inside DDP wrapper
model.backbone[0]          # single-GPU
```

---

## 6. Configuration

### 6.1 Model Config (`config/DINO/`)

| File | Backbone | Purpose |
|------|----------|---------|
| `DINO_4scale_swinBASE.py` | Swin-B | Standard base model |
| `DINO_4scale_swinLARGE.py` | Swin-L 224 | Large model |
| `DINO_4scale_swinLARGE384.py` | Swin-L 384 | Large model, high resolution |
| `DINO_4scale_convnext.py` | ConvNeXt | Alternative backbone |

### 6.2 Key Hyperparameters

| Param | Default | Description |
|-------|---------|-------------|
| `--num_queries` | 900 | Number of learnable detection queries |
| `--imgsize` | 448 | Input image size |
| `--batch_size` | 24 | Per-GPU batch size |
| `--lr_backbone` | 1e-5 | Backbone LR |
| `--lr_locDec` | 1e-4 | Localization decoder LR |
| `--modelEMA` | `True_Epoch` | EMA momentum=0.80 |
| `--use_dn` | True | Enable denoising queries |
