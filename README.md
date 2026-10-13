# AASIST3: KAN-Enhanced Audio Deepfake Detection

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Task](https://img.shields.io/badge/Task-Speech%20Anti--Spoofing-orange)
![Dataset](https://img.shields.io/badge/Dataset-ASVspoof%202019%20LA-blue)

A PyTorch model that classifies speech as **bonafide** (real human voice) or **spoof** (synthetic or converted voice). It starts from the AASIST graph-attention architecture for anti-spoofing and replaces its dense `Linear` layers with **Kolmogorov-Arnold Network (KAN)** layers, which learn their activation functions as splines instead of using fixed ones.

The model works directly on the raw waveform, so no spectrogram or hand-crafted feature extraction is needed. It is trained on the [ASVspoof 2019](https://www.asvspoof.org/) Logical Access (LA) data, and the training script is set up to run on Kaggle GPUs.

## Why KANs here

A standard layer computes `W·x` and then applies a fixed activation. A KAN layer instead puts a learnable univariate function (a B-spline) on every edge. The hypothesis behind this project is that these learnable non-linearities can capture the subtle artifacts that synthetic speech leaves behind, with an architecture that stays compact. Layers come from [`efficient-kan`](https://github.com/Blealtan/efficient-kan).

## Architecture

```
 raw waveform (16 kHz, 64,600 samples ≈ 4.04 s)
        │
        ▼  Parametric SincConv, 70 learnable band-pass filters (mel-initialised)
        ▼  MaxPool1d → BatchNorm1d → SELU
        │
        ├──────────────── temporal axis ────────┬──────── spatial axis ────────┐
        ▼  Linear → 24-dim nodes                ▼  Linear → 24-dim nodes       │
        ▼  KAN graph attention (KAN-GAL)        ▼  KAN-GAL                     │
        ▼  KAN-GraphPool (keep 40%)             ▼  KAN-GraphPool (keep 50%)    │
        └──────────────────────┬────────────────────────────────────────────────┘
                               │   + one learnable "stack" master node
                               ▼
        4 parallel branches, each:  KAN-HS-GAL → KAN-GraphPool → KAN-HS-GAL
                               │
                               ▼  residual element-wise sum across branches
                               ▼  Dropout 0.2
        statistics pooling: max |·| and mean over nodes (temporal, spatial) + max |·| of master node
                               ▼  Dropout 0.5 → concat (5 × 24 = 120)
                               ▼
        KAN classifier → 2 logits [spoof, bonafide]
```

- **KAN-GAL:** graph attention layer whose projections are KANs. It is applied to the temporal and spatial node sets separately.
- **KAN-HS-GAL:** heterogeneous stacked graph attention. It processes temporal and spatial nodes together with a shared master node.
- **KAN-GraphPool:** keeps the top-k nodes, where the node scores come from a KAN.
- **Where KANs are used:** inside the graph-attention layers, the pooling scorers and the output classifier. The two input projections into the graph (`temporal_proj`, `spatial_proj`) are still ordinary `nn.Linear` layers.

The implementation is in [`MODELDEFINITION.py`](MODELDEFINITION.py).

## Training setup

| Setting | Value |
| --- | --- |
| Data | ASVspoof 2019 LA, train split for training and dev split for validation |
| Input | 64,600 samples per clip: random crop for training, deterministic crop or tiling for dev |
| Loss | Cross-entropy with class weights `[0.1, 0.9]` for `[spoof, bonafide]` to offset the spoof-heavy label distribution |
| Optimizer | Adam, learning rate 1e-4, weight decay 1e-4 |
| Schedule | Cosine annealing from 1e-4 down to 5e-6, stepped per batch |
| Batch size / epochs | 32 / 25 |
| Stability | Mixed precision (AMP) on CUDA, gradient clipping at 5.0 |
| Checkpoints | `last.pt` every epoch, `best.pt` when dev accuracy improves |

## Repository layout

| File | Role |
| --- | --- |
| [`TRAIN.py`](TRAIN.py) | Training and validation loop; finds the dataset automatically on Kaggle |
| [`MODELDEFINITION.py`](MODELDEFINITION.py) | The complete model: SincConv front-end, KAN graph layers, classifier |
| [`DATAUTLIS.py`](DATAUTLIS.py) | Protocol-file parsing, padding and cropping, `Dataset` classes |
| [`DATASET_Download.py`](DATASET_Download.py) | Downloads the ASVspoof 2019 LA archive from the University of Edinburgh DataShare |
| `KAN_GRAPH_MODULES.py`, `SINCCONVENCODER.py`, `MODELS.py` | Earlier modular drafts of the same components. `TRAIN.py` does not use them, and `MODELS.py` still imports from a `core/` package that is not in this repo |

## Getting started

```bash
git clone https://github.com/gladiator456/aasist3.git
cd aasist3
pip install -r requirements.txt efficient-kan
python DATASET_Download.py        # fetches and unzips the LA dataset into ./LA
python TRAIN.py
```

On Kaggle, add the ASVspoof 2019 LA dataset as an input instead of downloading it; `TRAIN.py` searches `/kaggle/input` for it. Turn on a GPU for reasonable training times. `TRAIN.py` writes checkpoints to `/kaggle/working/checkpoints`, so when running elsewhere change `CHECKPOINT_DIR` near the top of the file.

### Minimal inference

```python
import torch
from MODELDEFINITION import Model

d_args = {
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.4, 0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}
model = Model(d_args)
model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
model.eval()

wav = torch.randn(1, 64600)              # replace with a real 16 kHz clip, cropped or tiled to 64,600 samples
with torch.no_grad():
    _, logits = model(wav)
print("bonafide" if logits.argmax(1).item() == 1 else "spoof")   # class 1 = bonafide
```

## Status and roadmap

The model, data pipeline and training loop are implemented, and validation currently reports **dev-set accuracy**. Still to do:

- Evaluate with the standard anti-spoofing metrics, **EER** and **min t-DCF**, on the dev and eval sets.
- Publish the trained weights and the final numbers from a full 25-epoch run.
- Run an ablation against the dense-layer baseline to measure what the KAN layers actually contribute.
- Add a small command-line script for scoring audio files.

## Acknowledgments

- AASIST: Jung et al., *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks* (NAVER, MIT licence). Parts of the data utilities derive from their code.
- KAN: Liu et al., *KAN: Kolmogorov-Arnold Networks*, and the [`efficient-kan`](https://github.com/Blealtan/efficient-kan) implementation.
- ASVspoof 2019 challenge organisers, for the dataset.

## Author

Anuj Saxena, B.Tech Chemical Engineering, IIT Jammu.
