# AASIST3 — Experimental KAN-Based Audio Anti-Spoofing

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Task](https://img.shields.io/badge/Task-Speech%20Anti--Spoofing-orange)
![Dataset](https://img.shields.io/badge/Dataset-ASVspoof%202019%20LA-blue)

**AASIST3** is an experimental PyTorch project exploring Kolmogorov–Arnold Network (KAN) components in a graph-based audio anti-spoofing model inspired by AASIST.

The model is designed to classify an audio recording as either **bonafide** (genuine speech) or **spoof** (spoofed speech). It operates on raw audio waveforms and is intended to be trained and evaluated using the **ASVspoof 2019 Logical Access (LA)** dataset.

> **Status:** The model implementation and basic synthetic-input smoke tests are in place. Real-dataset training and evaluation are still pending. No model-performance results are reported in this README; this section will be updated after training.

---

## Contents

- [Project Overview](#project-overview)
- [Why KANs?](#why-kans)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Run a Smoke Test](#run-a-smoke-test)
- [Training](#training)
- [Configuration](#configuration)
- [Evaluation and Results](#evaluation-and-results)
- [Current Status and Roadmap](#current-status-and-roadmap)
- [Acknowledgements](#acknowledgements)

## Project Overview

Speech anti-spoofing aims to distinguish genuine speech from audio created or manipulated to imitate a real speaker. It is an important component of reliable automatic speaker verification systems.

This project explores a graph-based architecture with KAN-related components to learn patterns that may help distinguish bonafide and spoofed speech. It draws inspiration from **AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)** while experimenting with KAN-based graph-attention, graph-pooling and classification components.

The implementation is a research prototype, not a claim of an exact reproduction of the original AASIST model or its published results.

### Project goals

- Build an experimental KAN-based architecture for speech anti-spoofing.
- Explore spatial and temporal graph-based processing of audio representations.
- Establish a reproducible training and development-evaluation workflow.
- Evaluate the trained model on the intended dataset and document the findings.
- Identify possible architectural improvements through future experiments and ablations.

## Why KANs?

Conventional neural-network layers commonly combine learned linear transformations with fixed activation functions. Kolmogorov–Arnold Networks (KANs) instead represent learnable univariate functions on connections; implementations such as `efficient-kan` use spline-based functions.

In this project, KAN components are explored within graph-attention layers, graph-pooling scorers and the output classifier. The motivation is to investigate whether these learnable nonlinear transformations are useful for audio anti-spoofing. Their effectiveness remains a research question and must be established through experiments.

Implementation dependency: [`efficient-kan`](https://github.com/Blealtan/efficient-kan).

## Architecture

The current model processes a fixed-length raw waveform and combines temporal and spatial graph representations.

```text
Raw waveform
(16 kHz, 64,600 samples; approximately 4.04 seconds)
        |
        v
Pre-emphasis (x[n] - 0.97 * x[n-1])
        |
        v
Fixed SincConv front end
70 non-trainable band-pass filters, mel-initialised
        |
        v
abs -> 2-D MaxPool (3x3) -> BatchNorm2d -> SELU
        |
        v
Residual encoder: 6 conv blocks (24 channels out)
        |
        +-----------------------------+
        |                             |
        v                             v
max over frequency + pos. emb.   max over time + pos. emb.
Temporal graph (~29 nodes, 24-d)  Spatial graph (23 nodes, 24-d)
        |                             |
        v                             v
KAN graph attention              KAN graph attention
        |                             |
        v                             v
KAN graph pooling                KAN graph pooling
(keep 40%)                       (keep 50%)
        |                             |
        +-------------+---------------+
                      |
                      v
             Learnable master node
                      |
                      v
       Four parallel heterogeneous graph branches
       KAN-HS-GAL -> KAN-GraphPool -> KAN-HS-GAL
                      |
                      v
   Residual aggregation (averaged) across branches
                      |
                      v
             Dropout and statistics pooling
                      |
                      v
                  LayerNorm
                      |
                      v
         KAN classifier (grid range [-3, 3])
                      |
                      v
              Two output logits
                [spoof, bonafide]
```

### Main components

- **Pre-emphasis and SincConv front end:** a high-frequency boost followed by a fixed mel-initialised band-pass filter bank, as in the AASIST3 paper.
- **Residual encoder:** six convolutional blocks that turn the 2-D filter-bank output into a 24-channel feature map; its frequency axis becomes the spatial graph and its time axis the temporal graph.
- **KAN-GAL:** graph-attention processing using KAN-based projections.
- **KAN-HS-GAL:** heterogeneous graph-attention processing across temporal and spatial nodes with a shared master node.
- **KAN-GraphPool:** selects nodes using learned scoring functions.
- **Classifier:** produces two logits corresponding to spoof and bonafide classes.

The active implementation is in [`MODELDEFINITION.py`](MODELDEFINITION.py), and its configuration lives in `DEFAULT_D_ARGS` so training, evaluation and tests share one set of settings. Both graphs are small (about 23 spatial and 29 temporal nodes), so dense pairwise attention is cheap.

Known differences from the paper: the HS-GAL layers keep a 24-dimensional node width (`gat_dims[1]` is not used), the KAN layers use the `efficient-kan` defaults, and the readout averages the branch outputs and adds a LayerNorm before the classifier for numerical stability.

These architectural details describe the implementation, not verified evidence of model effectiveness.

## Dataset

The training pipeline is intended for the **ASVspoof 2019 Logical Access (LA)** dataset. The dataset is not included in this repository; obtain it from the [official ASVspoof resources](https://www.asvspoof.org/) or an authorised source and follow the applicable access and usage terms.

### Expected directory layout

The data utilities and training script expect the training audio, development audio and protocol files to be available in a structure equivalent to:

```text
LA/
├── ASVspoof2019_LA_train/
│   └── flac/
├── ASVspoof2019_LA_dev/
│   └── flac/
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    └── ASVspoof2019.LA.cm.dev.trl.txt
```

The exact root directory can vary, but the expected subdirectory and protocol filenames must be preserved. The data pipeline expects audio sampled at **16 kHz** and prepares clips of 64,600 samples for model input. Check the implementation in [`DATAUTLIS.py`](DATAUTLIS.py) if your dataset layout differs.

### Download or provide the dataset

The repository includes a download helper:

```bash
python DATASET_Download.py
```

The script is intended to download and unpack the LA dataset under `./LA`. If you are using Kaggle, you can instead attach the dataset as notebook input; the training script searches common Kaggle and local locations. If automatic discovery does not find your dataset, set `DATASET_ROOT` to the correct root directory.

## Repository Structure

| File | Purpose |
|---|---|
| [`TRAIN.py`](TRAIN.py) | Training loop, development evaluation, progress logging and checkpoint saving. |
| [`MODELDEFINITION.py`](MODELDEFINITION.py) | Active model definition, including the SincConv front end, graph components and classifier. |
| [`DATAUTLIS.py`](DATAUTLIS.py) | Protocol parsing, audio preprocessing and dataset utilities. |
| [`DATASET_Download.py`](DATASET_Download.py) | Dataset download and extraction helper. |
| [`EVAL.py`](EVAL.py) | Scores the LA eval (or dev) split with a checkpoint and reports EER and ROC-AUC. |
| [`SMOKE_TEST.py`](SMOKE_TEST.py) | Basic forward/backward and output-validity checks using synthetic inputs. |
| [`requirements.txt`](requirements.txt) | Python package dependencies. |
| [`.github/workflows/model-smoke-test.yml`](.github/workflows/model-smoke-test.yml) | GitHub Actions workflow for automated smoke testing. |
| `KAN_GRAPH_MODULES.py`, `SINCCONVENCODER.py` | Additional modular files; the active training path is centred on `MODELDEFINITION.py`. |
| `MODELS.py` | Compatibility wrapper around the canonical model definition. |

## Setup

Python 3 and the packages listed in `requirements.txt` are required. A CUDA-compatible GPU is recommended for training; actual memory and runtime requirements need to be confirmed on the target hardware.

### 1. Clone the repository

```bash
git clone https://github.com/gladiator456/aasist3.git
cd aasist3
```

### 2. Create and activate a virtual environment (recommended)

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / Kaggle**

```bash
python -m venv .venv
source .venv/bin/activate
```

On a managed Kaggle notebook, you may use the notebook's existing Python environment instead.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, ensure that the installed PyTorch build is compatible with the CUDA environment available on the machine.

## Run a Smoke Test

Before attempting a full training run, execute:

```bash
python SMOKE_TEST.py
```

The smoke test checks basic model execution, backward propagation, output shape and numerical validity using synthetic inputs.

**Important:** passing this test only confirms that the tested model path executes. It does not verify dataset loading, full real-data training, GPU memory requirements, convergence or detection quality. Those checks require a run with the actual dataset and target hardware.

## Training

1. Make sure the dataset and protocol files are available.
2. Check that the dataset root is detected correctly.
3. Run a short trial before starting a long training job.
4. Inspect training logs, development evaluation output, memory use and saved checkpoints.
5. If the trial is stable, proceed to a longer run.

### Basic command

```bash
python TRAIN.py
```

By default, the training script is configured for **25 epochs** and a **batch size of 8**. These settings may need to be reduced depending on the available GPU memory and observed runtime. Checkpoints are written to `./checkpoints` by default; `CHECKPOINT_DIR` can be used to change the output location.

### Resuming an interrupted run

`last.pt` is written after every epoch (model, optimizer, scheduler, scaler, best dev EER). If a session is cut off, rerun with the same settings plus `RESUME_FROM=auto` (or a path to a checkpoint):

```bash
RESUME_FROM=auto python TRAIN.py
```

See [`FULL_TRAINING_GUIDE.md`](FULL_TRAINING_GUIDE.md) for the Kaggle workflow.

### Short trial run

For a small initial check on Linux or Kaggle:

```bash
MAX_TRAIN_SAMPLES=2000 MAX_DEV_SAMPLES=1000 NUM_EPOCHS=2 BATCH_SIZE=2 LOG_EVERY=50 python TRAIN.py
```

On Windows PowerShell, set the environment variables before launching Python:

```powershell
$env:MAX_TRAIN_SAMPLES = "2000"
$env:MAX_DEV_SAMPLES = "1000"
$env:NUM_EPOCHS = "2"
$env:BATCH_SIZE = "2"
$env:LOG_EVERY = "50"
python TRAIN.py
```

The subset limits are intended for quick debugging, not for reporting final research results. The first real-data run may still reveal dataset-path, dependency, audio-decoding or memory issues.

## Configuration

The training script supports environment-based configuration. Relevant options include:

| Variable | Purpose |
|---|---|
| `DATASET_ROOT` | Explicit dataset root directory. |
| `NUM_EPOCHS` | Number of training epochs. |
| `BATCH_SIZE` | Number of examples per batch. |
| `MAX_TRAIN_SAMPLES` | Optional limit on training examples for a quick trial. |
| `MAX_DEV_SAMPLES` | Optional limit on development examples for a quick trial. |
| `LOG_EVERY` | Training progress logging interval. |
| `CHECKPOINT_DIR` | Directory used to save checkpoints. |
| `USE_AMP` | Opt-in setting for automatic mixed precision where supported. |
| `BASE_LR` | Peak learning rate (default `1e-4`). |
| `OPTIMIZER` | `adam` (default, as in the paper) or `adamw`. |
| `WARMUP_STEPS` | Linear LR warm-up length in optimizer steps before cosine decay (default `500`). |
| `RESUME_FROM` | Restore full training state. `auto` uses `CHECKPOINT_DIR/last.pt` if present; or give a checkpoint path. |
| `INIT_FROM` | Load model weights only (fresh optimizer and schedule). Not combinable with `RESUME_FROM`. |
| `NUM_WORKERS` | DataLoader worker processes (default `2`). |
| `SEED` | Random seed (default `42`). |

Consult [`TRAIN.py`](TRAIN.py) for current defaults and implementation details. Start with a stable baseline before enabling optional mixed precision or changing multiple training settings at once.

## Evaluation and Results

The training code includes development-set evaluation. Once real-data training is complete, this section should document the evaluation protocol, checkpoint-selection procedure, dataset split and relevant anti-spoofing metrics.

`TRAIN.py` selects `best.pt` by dev EER, so the dev EER it prints is slightly optimistic. Report the number from the held-out eval split instead:

```bash
DATASET_ROOT=/path/to/LA CHECKPOINT=checkpoints_full/best.pt python EVAL.py
```

`EVAL.py` reports EER and ROC-AUC on the first ~4 s of each clip; min t-DCF is not computed because it needs the official ASV score files.

### Training history

| Run | Model | Setup | Best dev EER | Notes |
|---|---|---|---|---|
| 1 | Earlier encoder-less model (commit `21d1c66`) | 12 epochs, batch 8, AdamW, lr 1e-4, Kaggle T4 | 14.57% (epoch 10); dev ROC-AUC peaked at 0.928 | Learned but plateaued from epoch 7. No eval-split result. |

Run 1 omitted the paper's residual encoder, pre-emphasis and positional embeddings, which is the likely reason for the plateau. The current model adds them. No result for the current model is reported yet.

## Current Status and Roadmap

- [x] Experimental model implementation.
- [x] Audio preprocessing and dataset utilities.
- [x] Training and development-evaluation code.
- [x] Synthetic-input forward/backward smoke test.
- [x] Automated smoke-test workflow.
- [ ] Verify the full data pipeline using the real ASVspoof 2019 LA dataset.
- [ ] Complete a real-data trial run and confirm checkpoint saving.
- [ ] Complete the planned training run.
- [ ] Record evaluation settings and measured results.
- [ ] Compare against an appropriate baseline and perform ablation studies.
- [ ] Consider a small command-line interface for scoring audio files.

## Acknowledgements

- **AASIST:** Jung et al., *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks*. Parts of the data utilities are derived from the AASIST codebase; consult the source files and their notices for attribution and licence details.
- **KAN:** Liu et al., *KAN: Kolmogorov-Arnold Networks*.
- **efficient-kan:** [Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan).
- **ASVspoof 2019 organisers:** for the dataset and challenge resources.

Please review the licences and attribution requirements for all reused code, dependencies and dataset resources before redistribution.

## Author

**Anuj Saxena**  
B.Tech, Chemical Engineering — IIT Jammu
