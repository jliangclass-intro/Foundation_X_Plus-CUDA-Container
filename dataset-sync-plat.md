# Dataset Sync Plan (scratch-first, no /data mount)

## Goal
Run Foundation_X+ without mounting `/data/jliang12` by ensuring required inputs exist under:

- `/scratch/scratch/sejong/class-dataset/`

## Evidence from scratch-only run
Command was executed with `MOUNT_DATA=false` and failed at model load.

Missing file observed at runtime:

- `/data/jliang12/dongaoma/Ark_models/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar`

Error:

- `FileNotFoundError: [Errno 2] No such file or directory`

## Required sync targets to remove /data dependency

### 1) Backbone checkpoint used by run_apptainer_v107.sh
- Source: `/data/jliang12/dongaoma/Ark_models/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar`
- Target (scratch-first): `/scratch/scratch/sejong/class-dataset/model-checkpoints/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar`

### 2) Annotation/split files currently referenced from /data in config
The following are still configured under `/data/jliang12/...` in `config/config_datasets.py` and should be mirrored under `/scratch/scratch/sejong/class-dataset/`:

- NIH localization JSONs (`/data/jliang12/shared/dataset/NIH_Localization/...`)
- VinDr CXR localization JSONs (`/data/jliang12/nuislam/FoundationX_localization_bbox_annotation_collections/...`)
- Node21, ChestX-Det, RSNA pneumonia, SIIM-ACR, CANDID-PTX split JSON/TXT files (`/data/jliang12/nuislam/data_files_splits/...`)
- RSNA-PE lists (`/data/jliang12/nuislam/data_files_splits/rsna_pe_cls/...`)
- INSPECT CSV lists (`/data/jliang12/nuislam/CT_INSPECT_26gb/...`)
- Shared CT dataset annotation files under `/data/jliang12/shared/dataset/...`

## Suggested copy commands

```bash
mkdir -p /scratch/sejong/class-dataset/models
cp -av /data/jliang12/dongaoma/Ark_models/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar \
  /scratch/sejong/class-dataset/models/

mkdir -p /scratch/sejong/class-dataset/external-annotations
rsync -auv /data/jliang12/nuislam/data_files_splits/ \
  /scratch/sejong/class-dataset/external-annotations/data_files_splits/

rsync -auv /data/jliang12/shared/dataset/NIH_Localization/ \
  /scratch/sejong/class-dataset/external-annotations/NIH_Localization/
```

## Next wiring after sync
1. Update `scripts/run_apptainer_v107.sh` backbone checkpoint path to scratch target.
2. Update `config/config_datasets.py` paths from `/data/jliang12/...` to `/scratch/scratch/sejong/class-dataset/...`.
3. Re-run with `MOUNT_DATA=false` until one-iteration debug run succeeds.
