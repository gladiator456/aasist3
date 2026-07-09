<p align="center">
  <h1 align="center">🛡️ AASIST3 — KAN-Powered Audio Deepfake Detector</h1>
  <p align="center">
    <em>Next-Generation Kolmogorov-Arnold Graph Networks for Real-Time Audio Anti-Spoofing</em>
  </p>
  <p align="center">
    <a href="#-quick-start"><img src="https://img.shields.io/badge/Quick_Start-▶-brightgreen?style=for-the-badge" alt="Quick Start"></a>
    <a href="#-training"><img src="https://img.shields.io/badge/Training-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Training"></a>
    <a href="#-core-math"><img src="https://img.shields.io/badge/Architecture-KAN--HS--GAL-orange?style=for-the-badge" alt="Core Math"></a>
    <a href="#-results"><img src="https://img.shields.io/badge/Condition-Closed--SincConv-red?style=for-the-badge" alt="Condition"></a>
  </p>
</p>

---

A production-ready, highly optimized implementation of **AASIST3** (Audio Anti-Spoofing using Kolmogorov-Arnold Graph Attention Networks), trained on the [ASVspoof 2019 LA](https://www.asvspoof.org/) dataset. This next-generation model eliminates traditional linear layers and heavy CNN encoders, replacing them with learnable non-linear activations (**KANs**). This repository provides multi-module scripts optimized for Kaggle GPU environments and a robust cloud pipeline that classifies audio files as **bonafide (real human speech)** or **spoof (AI-generated / deepfake / synthetic)**.

> **Mathematical Blueprint:** Replaces standard matrix multiplication ($W \cdot x$) with learnable univariate B-spline functions on edges, providing superior non-linear modeling of synthetic phase distortions and AI voice artifacts.
> Developed by  Anuj Saxena. 

---

## 📑 Table of Contents

- [✨ Highlights](#-highlights)
- [🏗️ Model Architecture](#️-model-architecture)
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation](#️-installation)
- [🚀 Quick Start](#-quick-start)
- [🎯 Training](#-training)
- [📊 Results](#-results)
- [🔧 Inference API](#-inference-api)
- [📝 Configuration Reference](#-configuration-reference)
- [🤝 Acknowledgments](#-acknowledgments)
- [📄 License](#-license)

---

## ✨ Highlights

| Feature | Details |
|---|---|
| **KAN-Powered Integration** | Replaces standard `nn.Linear` with **Spline-Based KAN Layers** for supreme non-linear mapping |
| **Streamlined Pre-Encoder** | Implements the **Equation 32 Pipeline** bypassing traditional bulky 2D CNN feature blocks |
| **Advanced Graph Fusion** | Utilizes **KAN-GAL** and **KAN-HS-GAL** blocks for multi-domain cross-attention tracking |
| **Non-Linear Downsampling** | Implements **KAN-GraphPool** to map node significance dynamically via learnable splines |
| **GPU + CPU Support** | CUDA-accelerated distributed training with dual-GPU (T4 x2) scaling capabilities |
| **Kaggle-Optimized Training** | Clean cloud deployment notebook with AMP, TF32, and automated data loaders |
| **Production-Ready** | Seamless script-based execution (`train.py`) built natively on PyTorch frameworks |

---

## 🏗️ Model Architecture

The framework operates on an advanced end-to-end tensor pipeline mapped directly from structural research specifications, handling raw waveforms natively through a split spatial-temporal topology.

```
              ┌─────────────────────────────────────────┐
              │          Raw Audio Waveform             │
              │        (16kHz, mono, ~4.04 sec)         │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
              ┌─────────────────────────────────────────┐
              │   Parametric SincConv (70 Learnable)    │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼  [Equation 32 Pre-Encoder]
              ┌─────────────────────────────────────────┐
              │    MaxPool1D ➔ BatchNorm1D ➔ SELU       │
              └────────────┬─────────────────────┬──────┘
                           │                     │
                 [Temporal Axis Split]   [Spatial Axis Split]
                           │                     │
                           ▼                     ▼
              ┌────────────────────┐ ┌────────────────────┐
              │   KAN-GAL (Eq 33)  │ │   KAN-GAL (Eq 34)  │
              │   Temporal Graph   │ │   Spatial Graph    │
              └────────────┬───────┘ └───────────┬────────┘
                           │                     │
                           └──────────┬──────────┘
                                      │  + Expand Learnable Stack Node (S)
                                      ▼
              ┌─────────────────────────────────────────┐
              │  4 Parallel Cross-Attention Branches   │
              │  (KAN-HS-GAL Block 1 ➔ KAN-GraphPool)   │
              │  (KAN-HS-GAL Block 2 ➔ Tensor Stacking) │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼  [Equations 37, 38, 39]
              ┌─────────────────────────────────────────┐
              │     Residual Stacking Element-Wise Sum  │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
              ┌─────────────────────────────────────────┐
              │    Statistical Pooling & Dropout 0.5    │
              │  L = CONCAT(Hmax_t, Hmean_t, ... S_max) │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
              ┌─────────────────────────────────────────┐
              │     KAN Output Classifier (2 Logits)    │
              │             [REAL / FAKE]               │
              └────────────────────────────────────────
```
