# DRCFNet: Dynamic Relational Context Fusion Network 🚀

> **Streaming-Resilient Multimodal Emotion Recognition via Neuro-Symbolic Disentanglement**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This is the official repository for **DRCFNet**, a novel architecture designed to perform real-time, low-latency Multimodal Emotion Recognition (MER) in unconstrained edge environments (e.g., telehealth, virtual assistants). 

Current Information Systems fail catastrophically when subjected to real-world sensor noise (e.g., camera blur, microphone static). DRCFNet resolves the long-standing trade-off between predictive accuracy and real-time execution resilience, achieving state-of-the-art results without relying on massive, latency-inducing Large Language Models.

---

## 🌟 Key Innovations

1. **Streaming Governance Layer (Zero-Latency Fallback):**
   Actively monitors the model's predictive entropy. If the live webcam or microphone degrades (causing high uncertainty), the model instantly routes data to pre-trained, noise-free **Meta ImageBind** proxies without dropping a single frame.
   
2. **Dynamic Gated Disentanglement:**
   Mathematically separates raw multimodal streams into **Modality-Specific Representations (MSR)** (which quarantine noise and specific nuances) and **Shared Semantic Representations (SSR)** (which capture the core, universal meaning of the conversation).

3. **Neuro-Symbolic Graph Reasoning:**
   Instead of using computationally heavy LLMs to understand complex reasoning (like sarcasm), we fuse the neural features with a symbolic **Knowledge Graph (ConceptNet)**. A logical hinge loss penalizes the network during training if it predicts emotions that contradict basic human logic.

4. **Ultra-Low Latency:**
   Thanks to the *Lite-MRU* memory cell, our model completely avoids full-sequence transformer buffering, operating under a strict **15.8 ms inference latency budget**.

---

## 📊 Experimental Results

DRCFNet was rigorously evaluated against state-of-the-art baselines on two gold-standard datasets. Under standard testing, our model achieves highly competitive accuracy. Under severe environmental noise injections, DRCFNet only drops by a mere 0.5% in performance, whereas competing models collapse entirely.

| Dataset      | Binary Accuracy (Acc-2) | F1-Score | Mean Absolute Error (MAE) |
|--------------|-------------------------|----------|---------------------------|
| **CMU-MOSI** | **84.2%**               | **84.5%**| 0.750                     |
| **CMU-MOSEI**| **85.8%**               | **85.6%**| 0.580                     |

---

## 🛠️ Repository Structure

```text
├── src/                    # Source code for DRCFNet
│   ├── data/               # Data loaders and dataset preprocessing
│   ├── models/             # PyTorch model definitions (CRE, GCF, GNN)
│   ├── utils/              # Helper functions, training loops, and loss definitions
│   └── train.py            # Main training script
├── Paper/                  # The compiled final academic manuscript and LaTeX source
│   ├── figures/            # High-resolution architectural diagrams and plots
│   ├── main.tex            # Single-column manuscript
│   └── DRCFNet_24_06_26.pdf# Final compiled PDF
├── requirements.txt        # Python dependencies
└── README.md               # You are here
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/ranbeer06052009/DRCFNet.git
cd DRCFNet
pip install -r requirements.txt
```

### 2. Dataset Preparation
Ensure you have downloaded the pre-aligned CMU-MOSI and CMU-MOSEI features (CMU Multimodal SDK). Place them in the `data/` directory.

### 3. Training the Model
To train DRCFNet from scratch using the default hyperparameter configurations (as detailed in the paper):
```bash
python src/train.py --dataset mosi --batch_size 32 --epochs 50 --lr 1e-4 --dropout 0.2
```

---

## 📜 Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{drcfnet2026,
  title={Streaming-Resilient Multimodal Emotion Recognition via Neuro-Symbolic Disentanglement},
  author={Ranbeer Singh and Soumyojit Mukhopadhyay},
  journal={Information Systems Frontiers},
  year={2026}
}
```

## 📄 License
This project is licensed under the MIT License.
