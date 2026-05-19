# Final Walkthrough: MEA Implementation & DRCFNet Research Visualizations

Welcome to the complete guide for your newly updated repository! This document is designed for newcomers and collaborators to seamlessly understand and utilize the complex architectural upgrades implemented in the `DRCFNet-Research` project.

This guide is broken into two main parts:
1. **Running the MEA Architecture:** How to train and evaluate the Modality-Exclusive and Modality-Agnostic baseline model.
2. **Generating DRCFNet Research Visuals:** How to utilize the new visualization pipeline to generate professional graphs and tables for your research paper using *real inference data*.

---

## Part 1: Running the MEA Architecture

We have fully implemented the architecture from the paper *"Multimodal fusion approach for learning modality-Exclusive and modality-Agnostic representations"*. Due to its complex double-discriminator and HSIC disparity constraints, it requires a dedicated training loop.

### Step 1.1: Initialize the MEA Model
The `MEA` model is designed as a drop-in replacement for your existing models. It accepts the exact same input dimensions `(vision, audio, text)`.

```python
import torch
from models.mea import MEA
from training.train_mea import train_mea_loop, mea_criterion

# Initialize the MEA Model
model = MEA(
    dim_v=35,    # Vision dimension
    dim_a=74,    # Audio dimension
    dim_t=300,   # Text dimension
    d=128,       # Hidden dimension
    n_heads=8    # Number of attention heads
).cuda()
```

### Step 1.2: The MEA Training Loop
The MEA model cannot be trained using standard Cross-Entropy loss alone. It relies on a carefully balanced combination of:
1. **Task Loss**: Standard prediction loss.
2. **HSIC Loss**: Disparity constraint to keep exclusive representations separated.
3. **Adversarial Modality Loss**: To bridge domain gaps between heterogenous modalities.
4. **Adversarial Importance Loss**: To align agnostic representations.

We have handled all of this inside `train_mea_loop` and `mea_criterion`.

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Start the training loop
best_mea_model, history = train_mea_loop(
    model=model,
    train_loader=train_loader,   # Your existing PyTorch DataLoader
    valid_loader=valid_loader,   # Your existing PyTorch DataLoader
    optimizer=optimizer,
    epochs=50,
    device='cuda',
    alpha=2e-2,  # HSIC disparity tradeoff
    beta=3e-2    # Adversarial tradeoff
)
```
*Note: This training loop will output the validation Accuracy and F1 scores natively at the end of each epoch, allowing you to directly compare `MEA` against `DRCFNet`.*

---

## Part 2: Generating DRCFNet Research Visuals

To ensure your research paper builds trust, we've integrated automated code to output high-quality, publication-ready visuals (Attention Heatmaps, Modality Pair Weights, Hyperparameter Graphs, and Ablation Tables) directly from your *real* inference data.

### Step 2.1: Hooking into the `train()` Loop
We modified your existing `src/training/train.py` script. The `train()` and `test()` functions now accept a `generate_visuals` argument.

When you pass `generate_visuals=True`, the pipeline will:
1. Hook into the `test` phase and extract the exact modality `pair_weights` ($\psi_m$) used to classify the very first batch of data.
2. Automatically plot and save these weights over the epochs.
3. Output the final Training/Validation Loss & Accuracy Curves when training finishes.

```python
from training.train import train

# Run your normal DRCFNet training, but enable visual generation
model, history = train(
    model=drcfnet_model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=50,
    generate_visuals=True  # <--- Turn this ON!
)
```
**Output:** Check the newly created `visuals_output/` folder in your project root! You will find files like `pair_weights_epoch_1.png` and `training_curves_final.png`.

### Step 2.2: Generating Hyperparameter Graphs
In your research paper, you need to show how tuning hyperparameters (like number of layers, or the $\gamma$ NeuroSymbolic weight) affects your F1 score.

Instead of doing this manually, use the `plot_hyperparameter_effect` tool from your Jupyter notebooks after running a loop:

```python
from evaluation.visualizations import plot_hyperparameter_effect

# Let's say you ran 5 training loops with different gamma weights:
gamma_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
real_f1_scores = [83.2, 84.6, 84.3, 83.1, 82.7, 82.5] # From your history dicts

plot_hyperparameter_effect(
    param_values=gamma_values, 
    f1_scores=real_f1_scores, 
    param_name=r"NeuroSymbolic Loss Weight (\gamma)", 
    dataset_name="MOSI Performance",
    save_path="visuals_output/hyperparam_gamma_effect.png"
)
```

### Step 2.3: Generating Ablation Tables
When performing your Ablation Study (removing the MSR/SSR split, removing Graph Fusion, etc.), organize your final metrics into a dictionary and let `generate_ablation_table` format it flawlessly for your paper.

```python
from evaluation.visualizations import generate_ablation_table

# Your real test results after running your ablated models
ablation_results = {
    "DRCFNet (Full Model)": {"Acc_7": 42.5, "F1": 84.6},
    "w/o Gated MSR/SSR Split": {"Acc_7": 41.2, "F1": 83.2},
    "w/o Bimodal GCF": {"Acc_7": 40.8, "F1": 82.5},
    "w/o Dual-Graph Neural Fusion": {"Acc_7": 41.0, "F1": 82.9}
}

generate_ablation_table(
    results_dict=ablation_results, 
    dataset_name="MOSI", 
    save_path="visuals_output/ablation_study_mosi.csv"
)
```

### Summary
All tools are self-contained within `src/evaluation/visualizations.py`. By keeping `generate_visuals=True` inside your training loops and utilizing the provided graph generation scripts, your repository will passively document itself with research-grade assets as you experiment!
