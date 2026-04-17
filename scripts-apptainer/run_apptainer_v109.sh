#!/bin/bash

#SBATCH --job-name=foundationx_v108
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH --gres=gpu:a100:2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

## all of them
# cyclictask=chexpertCLS_nihchestxray14CLS_vindrcxrCLS_nihshenzenCLS_mimic2CLS_tbx11kCLS_node21CLS_candidptxCLS_rsnapneumoniaCLS_chestxdetCLS_siimacrCLS_tbx11kLOC_node21LOC_candidptxLOC_rsnapneumoniaLOC_chestxdetLOC_siimacrLOC_candidptxSEG_chestxdetSEG_siimacrSEG

## v108 configuration trains on NIH ChestX-ray14 classification, Node21 classification, Node21 localization, CANDID-PTX classification, CANDID-PTX localization and CANDID-PTX segmentation.
# Reference: https://github.com/jlianglab/Foundation_X/tree/main/Foundation_X%2B
# cyclictask=nihchestxray14CLS_node21CLS_node21LOC_candidptxCLS_candidptxLOC_candidptxSEG

## v109 uses only classification - ChexPert, Xray14 and VINDR, like ARK.
cyclictask=chexpertCLS_nihchestxray14CLS_vindrcxrCLS

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# This script launches the training process inside an Apptainer container.
# It is based on scripts/run_IntegratedModel_Foundation6_ClsLocSeg_v107.sh

# Configuration from the original script
CONFIGFILE=config/DINO/DINO_4scale_swinBASE.py
LOGFILE=${LOGFILE:-${SCRATCH:-/scratch/$USER}/FoundationX/v109/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run104_Ark6F6_ClsLocSeg_b24_AdamW_LockReleaseAll_RCons_1LocDec_TESTrun}

# backbone_dir=/data/jliang12/dongaoma/Ark_models/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar
backbone_dir=/scratch/sejong/class-dataset/models/Ark_models/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar

BACKBONEMODEL=Swin-B # Swin-T, Swin-B, Swin-L
IMGSIZE=224 # 448

# coco_path=/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/
coco_path=/scratch/sejong/class-dataset/VinDr-CXR/

DATASETFILE=foundation6Ark6_datasets

### 1e-1 = 0.1
### 1e-2 = 0.01
### 1e-3 = 0.001
### 1e-4 = 0.0001
### 1e-5 = 0.00001

## ADAMW
lr_backbone=1e-5
lr_locEnc=1e-4
lr_locDec=1e-4
lr_segmentor=1e-4

BATCHSIZE=${BATCHSIZE:-24}
num_workers=${num_workers:-12}
INIT=${INIT:-ark}
total_epochs=${total_epochs:-2000}
opt=${opt:-adamw} # sgd adamw
EMAMODE=${EMAMODE:-True_Epoch}

# Default to SOL dataset locations. To use DFS paths instead, run with:
# DATASET_LOCATIONS_YML=config/dataset_locations_dfs.yml ./scripts/run_apptainer_v107.sh
DATASET_LOCATIONS_YML=${DATASET_LOCATIONS_YML:-config/dataset_locations_asu_sol.yml}

# Lightweight sanity run mode:
# DEBUG_RUN=true ./scripts/run_apptainer_v107.sh
# - enables --debug
# - limits to 1 epoch
# - uses small batch and workers
# - disables EMA deepcopy to reduce GPU memory pressure
DEBUG_RUN=${DEBUG_RUN:-false}
EXTRA_ARGS=()
if [[ "$DEBUG_RUN" == "true" ]]; then
	total_epochs=1
	BATCHSIZE=${BATCHSIZE:-2}
	num_workers=${num_workers:-0}
	EMAMODE=None
	EXTRA_ARGS+=(--debug)
fi

# Mount policy:
# - default: bind /scratch (not just /scratch/sejong — binding a subdirectory when the
#   parent doesn't exist in the SIF overlay causes unreliable mounts in worker processes)
# - set MOUNT_DATA=true to also bind /data/jliang12
MOUNT_DATA=${MOUNT_DATA:-false}
DEFAULT_BIND_DIRS="/scratch"
if [[ "$MOUNT_DATA" == "true" ]]; then
	DEFAULT_BIND_DIRS="/scratch /data/jliang12"
fi

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

# torchrun workers per node (priority: explicit override -> Slurm allocation -> visible devices)
if [[ -n "${NPROC_PER_NODE:-}" ]]; then
	TORCHRUN_NPROC_PER_NODE="$NPROC_PER_NODE"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
	TORCHRUN_NPROC_PER_NODE="$(echo "$SLURM_GPUS_ON_NODE" | grep -Eo '[0-9]+' | head -n 1)"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
	TORCHRUN_NPROC_PER_NODE="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')"
elif command -v nvidia-smi >/dev/null 2>&1; then
	TORCHRUN_NPROC_PER_NODE="$(nvidia-smi -L | wc -l | awk '{print $1}')"
else
	TORCHRUN_NPROC_PER_NODE=1
fi

if [[ -z "${TORCHRUN_NPROC_PER_NODE}" || "${TORCHRUN_NPROC_PER_NODE}" -lt 1 ]]; then
	TORCHRUN_NPROC_PER_NODE=1
fi

echo "torchrun nproc_per_node=${TORCHRUN_NPROC_PER_NODE}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
	echo "Running outside SLURM. This mode assumes GPU resources are already available (interactive session)."
else
	LOGFILE="${LOGFILE}_job${SLURM_JOB_ID}"
	echo "Running under SLURM job ${SLURM_JOB_ID}"
	echo "SLURM output directory: ${LOGFILE}"
fi

# Execute the python script within the Apptainer container
# The --nv flag enables NVIDIA GPU support
echo "Launching training script inside Apptainer..."
DEFAULT_BIND_DIRS="$DEFAULT_BIND_DIRS" ./cuda-apptainer.sh exec env FOUNDATION_X_DATASET_LOCATIONS_YML="$DATASET_LOCATIONS_YML" \
	torchrun --standalone --nnodes=1 --nproc_per_node "$TORCHRUN_NPROC_PER_NODE" \
	main_Consolidated.py --taskcomponent foundation_x5_pretraining \
	--train --numClasses 1 --dataset_file $DATASETFILE --classification_dataset $DATASETFILE --num_workers $num_workers \
	--coco_path $coco_path --weight-decay 0.0001 \
	--output_dir $LOGFILE -c $CONFIGFILE --imgsize $IMGSIZE --backbonemodel $BACKBONEMODEL --init $INIT \
	--total_epochs $total_epochs --batch_size $BATCHSIZE --opt $opt \
	--finetune_ignore label_enc.weight class_embed \
	--backbone_dir $backbone_dir --lr_backbone $lr_backbone --lr_locEnc $lr_locEnc --lr_locDec $lr_locDec  --lr_segmentor $lr_segmentor \
	--cyclictask $cyclictask --modelEMA $EMAMODE --lockrelease --saveAllModel \
	${EXTRA_ARGS[@]} \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0

echo "Execution finished."
