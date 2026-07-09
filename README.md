<p align="center">
  <h1 align="center">🛡️ AASIST3 — KAN-Powered Audio Deepfake Detector</h1>
  <p align="center">
    <em>State-of-the-Art Kolmogorov-Arnold Network (KAN) for Real-Time Audio Anti-Spoofing</em>
  </p>
  <p align="center">
    <a href="#-quick-start"><img src="https://img.shields.io/badge/Quick_Start-▶-brightgreen?style=for-the-badge" alt="Quick Start"></a>
    <a href="#-training"><img src="https://img.shields.io/badge/Training-Kaggle_GPU-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Training"></a>
    <a href="#-model-architecture"><img src="https://img.shields.io/badge/Core-KAN--HS--GAL-orange?style=for-the-badge" alt="Architecture"></a>
    <a href="#-results"><img src="https://img.shields.io/badge/Condition-Closed/SincConv-red?style=for-the-badge" alt="Condition"></a>
  </p>
</p>

---
#🛡️ AASIST3 — KAN-Powered Audio Deepfake Detector
A production-ready, highly optimized implementation of AASIST3 (Audio Anti-Spoofing using Kolmogorov-Arnold Graph Attention Networks). This next-generation model eliminates traditional linear layers and heavy CNN encoders, replacing them with learnable non-linear activations (KANs). Trained on the benchmark ASVspoof 2019 LA dataset, this repository provides multi-module scripts optimized for Kaggle GPU environments and scalable inference.

Mathematical Blueprint: Replaces standard matrix multiplication (W * x) with learnable univariate B-spline functions on edges, providing superior non-linear modeling of synthetic phase distortions and AI voice artifacts.

✨ Key Upgrades (AASIST vs AASIST3)
Feature Extraction: Heavy 2D CNN Encoder (6 Residual Blocks) in AASIST2 -> Streamlined Equation 32 Pipeline (Direct Front-End to Graphs) in AASIST3

Node Transformations: Standard Linear Projections (nn.Linear) in AASIST2 -> Spline-Based KAN Layers (Learnable non-linear activations) in AASIST3

Attention Mechanism: Traditional Softmax Graph Attention (GAT) in AASIST2 -> KAN-GAL & KAN-HS-GAL Blocks (Cross-domain matrix fusion) in AASIST3

Downsampling Engine: Score-based Linear Projection Top-K Pooling in AASIST2 -> KAN-GraphPool (Non-linear node significance mapping) in AASIST3

Final Classification: Linear Readout Layer in AASIST2 -> Native KAN Logit Layer in AASIST3

#🏗️ Model Architecture & Math Flows
The entire framework operates on an advanced end-to-end tensor pipeline mapped directly from structural research specifications:

Raw Audio Waveform -> Parametric SincConv (Learnable Bands) -> MaxPool1D + BatchNorm1D + SELU [Equation 32 Pre-Encoder] -> Axis Split (Temporal Graph & Spatial Graph via KAN-GAL) -> Expand Learnable Stack Node (S) -> 4 Parallel Cross-Attention Branches (KAN-HS-GAL Block 1 + KAN-GraphPool -> KAN-HS-GAL Block 2 + Tensor Stacking) -> Residual Stacking Element-Wise Sum [Equations 37, 38, 39] -> Statistical Pooling & Dropout 0.5 -> KAN Output Classifier (2 Logits)

#📁 Project Structure
Unlike script-heavy notebooks, this pipeline is engineered as clean, modularized python scripts:

aasist3_project/

SINC_CONV_ENCODER.py (Learnable parametric SincConv & Eq 32 implementation)

KAN_GRAPH_MODULES.py (KAN-GAL, KAN-HS-GAL, and KAN-GraphPool layers)

MODELDEFINITION.py (Core Model class combining splits & 4 parallel branches)

DATAUTLIS.py (FLAC parsing, protocol generation & dynamic padding loaders)

train.py (Main PyTorch Training Engine with Cosine Schedulers & AMP)

README.md (This documentation file)

#⚙️ Installation
Before executing on a local environment or cloud container, make sure the dependencies are fetched:

pip install torch torchaudio soundfile tqdm efficient-kan

#🚀 Kaggle Training Quick Start
Since this repository is fully modularized and optimized for Kaggle Cloud GPUs (Tesla T4 x2 or P100), you can train it without copy-pasting code walls into messy notebooks.

Step 1: Open a Kaggle Notebook & Toggle Internet Access
Go to Kaggle, create a new notebook.

In the right-side options panel, ensure Accelerator is set to GPU T4 x2.

Crucial: Turn Internet ON under the notebook options panel (requires phone verification on Kaggle).

Step 2: Clone & Deploy Dependencies
Create the first code cell in your notebook and run it to pull the latest production architecture:

!rm -rf aasist3_project
!git clone https://github.com/gladiator456/aasist3.git aasist3_project
%cd aasist3_project
!pip install -q efficient-kan soundfile tqdm

Step 3: Trigger the Training Engine
Create a second code cell to fire up the native PyTorch execution loop:

!python train.py

#📊 Readout & Vector Concat Logic
The final categorization layer relies on full spatial and temporal context concatenation. The graph representations extracted across all branches are compressed using node-wise parameters and mapped to the flat hidden embedding vector L:

L = CONCAT(H_max_t, H_mean_t, H_max_s, H_mean_s, S_max_f)

After L, a native non-linear Kolmogorov-Arnold Network block maps the 320-dimensional spectro-temporal context to target binary logits [Real, Fake] dynamically.



Built by Anuj Saxena — Detecting deepfakes via Kolmogorov-Arnold Networks, one waveform at a time.
