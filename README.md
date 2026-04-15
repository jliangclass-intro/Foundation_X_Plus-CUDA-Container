![](https://github.com/jlianglab/Foundation_X/blob/main/Foundation_X%2B/Figures/FoundationX%2B_Logo.png)
<!-- # Integrating Classification, Localization, and Segmentation through Lock-Release Pretraining Strategy for Chest X-ray Analysis -->

## CUDA Apptainer Container Version

This repository has been converted into a CUDA-enabled Apptainer container workflow so training can be run in a more reproducible environment on shared GPU systems.

## Student Quick Start

For students, the recommended workflow is:

1. Build the Apptainer image:
```bash
./cuda-apptainer.sh build
```
2. Verify that the image was created successfully:
```bash
ls -lh ./apptainer-cuda.sif
```
3. For an interactive session (GPU already allocated), start training inside the container:
```bash
./scripts/run_apptainer_v107.sh
```

If `apptainer-cuda.sif` is missing, do not start training. Re-run the build step and make sure `apptainer` is available on the system.

## Notes for Students

- The container runner automatically enables NVIDIA GPU support with `--nv`.
- The default bind mounts include `/scratch`, and you can also mount `/data/jliang12` by running with `MOUNT_DATA=true ./scripts/run_apptainer_v107.sh`.
- The default dataset configuration is `config/dataset_locations_asu_sol.yml`, which is intended for the ASU SOL-style scratch layout.
- `./scripts/run_apptainer_v107.sh` is `sbatch`-ready with SOL defaults (`--gres=gpu:a100:2`, `--time=7-00:00:00`). For PHXRC, submit with `--gres=gpu:v100:<count>` to override GPU type.
- If you need different dataset paths, override them with `DATASET_LOCATIONS_YML=config/dataset_locations_dfs.yml ./scripts/run_apptainer_v107.sh`.
- For a quick sanity check before a long run, use `DEBUG_RUN=true ./scripts/run_apptainer_v107.sh`.
- Make sure your datasets and pretrained backbone checkpoint are available at the paths referenced by the run script before launching a full job.
- If the build is slow or fails for temporary space reasons, set `SCRATCH` so Apptainer can use a writable cache and temp directory.

## SLURM Submission

Use `sbatch` for scheduled training jobs. The script contains these defaults:
- 2x A100 GPUs (`--gres=gpu:a100:2`)
- 7-day walltime (`--time=7-00:00:00`)
- public partition/QOS (`-p public`, `-q public`)

### ASU SOL (A100)

Use the script defaults:
```bash
sbatch ./scripts/run_apptainer_v107.sh
```
Override GPU resource to A100 at submit time (from 2 GPUs to 4 GPUs):
```bash
sbatch --gres=gpu:a100:4 ./scripts/run_apptainer_v107.sh
```

Example with variable overrides:
```bash
sbatch --export=ALL,DEBUG_RUN=true,BATCHSIZE=8 ./scripts/run_apptainer_v107.sh
```

### ASU PHXRC (V100)

Override GPU resource to V100 at submit time (from 2 GPUs to 4 GPUs):
```bash
sbatch --gres=gpu:v100:4 ./scripts/run_apptainer_v107.sh
```

Example with additional runtime overrides:
```bash
sbatch --gres=gpu:v100:4 --export=ALL,DEBUG_RUN=true,BATCHSIZE=8 ./scripts/run_apptainer_v107.sh
```

You can change the V100 count as needed, for example:
```bash
sbatch --gres=gpu:v100:1 ./scripts/run_apptainer_v107.sh
```

Monitor the queue:
```bash
squeue --me
```

Check logs:
```bash
ls -lh foundationx_v107_*.out foundationx_v107_*.err
```

Directly running `./scripts/run_apptainer_v107.sh` is intended for interactive sessions where GPU resources have already been allocated (for example, an interactive SLURM allocation).

## Recommended Training Workflow

1. Confirm that `apptainer` is installed and your node has access to an NVIDIA GPU.
2. Build `apptainer-cuda.sif` once in the repository root.
3. Verify the `.sif` file exists before every training run.
4. Run `./scripts/run_apptainer_v107.sh` from the repository root.
5. Use `DEBUG_RUN=true` first if you are launching this code for the first time on a new machine.

Foundation X+ is an efficient end-to-end framework for multi-task medical imaging that integrates classification, localization, and segmentation using diverse expert-level annotations across datasets. It leverages Cyclic Training, Lock-Release Pretraining, and a Student-Teacher learning paradigm to ensure balanced learning, strong generalization, and reduced overfitting. Foundation X+ further introduces Region-Guided ROI Alignment for improved localization-aware feature learning, and adopts a single shared localization decoder with lightweight task-specific heads, significantly reducing model size while maintaining performance.

## Data Splits and Bounding Box Annotations
- Data splits and generated COCO-format localization bouding box annotation files can be downloaded through this [Google Form](https://forms.gle/wdiq3s6SNvsd6nn78). <br/>
- The [config.py](https://github.com/jlianglab/Foundation_X/blob/main/Foundation_X%2B/config/config_datasets.py) file defines the directory structure for all datasets and specifies the task head indices associated with each task.
  
## Pre-trained models
- You can download the pretrained models through this [Google Form](https://docs.google.com/forms/d/e/1FAIpQLScK8x0awPHDWGWnN-pU64nTBWaK0nTKHyl2OcxPGp4rOCsSHg/viewform?usp=publish-editor).

## Loading pre-trained Foundation X+ checkpoints and extracting features
We provide a utility script `load_weights.py` to initialize a Swin-B backbone using our pretrained Foundation X+ checkpoints. The model only loads the encoder weights from the checkpoint, and supports an optional projection layer.

 #### Example: Load model and extract features
 ```python
from load_weights import build_model

# Path to the pretrained Foundation X+ checkpoint
pretrained_weights = "path/to/weights/ckpt.pth"

# Initialize the model
foundationx_model = build_model(
    pretrained_weights,
    num_classes=0,
    projector_features=256,     # Optional: dimensionality of projection layer
    use_mlp=True
)

foundationx_model.eval()  # Set model to evaluation mode

# extract features from input batch (e.g., [B, 3, 224, 224])
with torch.no_grad():
    features = foundationx_model.forward_features(input_tensor)
```

## Setting-up the multiscaledeformableattention package
- Please follow the steps described in [DINO GitHub Repo](https://github.com/IDEA-Research/DINO) to install the package "multiscaledeformableattention".

## Pretraining Foundation X+ Instructions
**Script:**
- Use the script [scripts/run_IntegratedModel_Foundation6_ClsLocSeg_v107.sh](https://github.com/jlianglab/Foundation_X/blob/main/Foundation_X%2B/scripts/run_IntegratedModel_Foundation6_ClsLocSeg_v107.sh) to start pretraining **Foundation X+ model**. <br/>
- This script launches pretraining across all 11 datasets and 20 tasks.
- Update the LOGFILE parameter to specify where pretrained models will be saved.
- Use the `--debug` flag to run a quick check and identify potential issues before full pretraining.
- Use `--resume $RESUME` to load pretrained weights and continue training from a specified checkpoint.
- Follow the `foundation_x5_pretraining` taskcomponent in [main_Consolidated.py](Foundation_X+/main_Consolidated.py) to see how Foundation X+ is trained. You can also create your own custom task component to experiment with new datasets.

**Flexible Dataset Picking:**
- The updated code provides flexibility to select which datasets and tasks to use for pretraining Foundation X+.
- To include all datasets and tasks, use the following parameter in the script:
```
cyclictask=chexpertCLS_nihchestxray14CLS_vindrcxrCLS_nihshenzenCLS_mimic2CLS_tbx11kCLS_node21CLS_candidptxCLS_rsnapneumoniaCLS_chestxdetCLS_siimacrCLS_tbx11kLOC_node21LOC_candidptxLOC_rsnapneumoniaLOC_chestxdetLOC_siimacrLOC_candidptxSEG_chestxdetSEG_siimacrSEG
```
- You can select any combination of datasets and tasks by concatenating them with `_`. For example:
```
cyclictask=nihchestxray14CLS_node21CLS_node21LOC_candidptxSEG
```
  This configuration trains on NIH ChestX-ray14 classification, Node21 classification, Node21 localization, and CANDID-PTX segmentation. <br/>
- Below is a list of available dataset–task combinations:  <br/>

| Classification (CLS) | Localization (LOC) | Segmentation (SEG) |
|---------------------|--------------------|--------------------|
| chexpertCLS          | tbx11kLOC          | candidptxSEG   |
| nihchestxray14CLS    | node21LOC          | chestxdetSEG   |
| vindrcxrCLS          | candidptxLOC       | siimacrSEG     |
| nihshenzenCLS        | rsnapneumoniaLOC   |                |
| mimic2CLS            | chestxdetLOC       |                |
| tbx11kCLS            | siimacrLOC         |                |
| node21CLS            |                    |                |
| candidptxCLS         |                    |                |
| rsnapneumoniaCLS     |                    |                |
| chestxdetCLS         |                    |                |
| siimacrCLS           |                    |                |

## Foundation X+ Pretraining Results
<p align="left">
<img src="https://github.com/jlianglab/Foundation_X/blob/main/Foundation_X%2B/Figures/PretrainingResult_FX%2B.png" width=75% height=75%>
</p>

## Acnkowledgement
This research was partially supported by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant, as well as by the NIH under Award Number R01HL128785. The authors are solely responsible for the content, which does not necessarily reflect the official views of the NIH. This work also utilized GPUs provided by ASU Research Computing (SOL), Bridges-2 at the Pittsburgh Supercomputing Center (allocated under BCS190015), and Anvil at Purdue University (allocated under MED220025). These resources are supported by the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, funded by the National Science Foundation under grants #2138259, #2138286, #2138307, #2137603, and #2138296. We also extend our gratitude to Anirudh Kaniyar Narayana Iyengar for his contributions to collecting localization data, preparing bounding boxes in COCO format, and developing some of the data loaders. Finally, the content of this paper is covered by patents pending.


## Contact
For any questions, feel free to reach out: <br/>
Email: nuislam (at) asu.edu
