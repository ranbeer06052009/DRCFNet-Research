# Final Walkthrough: MEA Implementation & DRCFNet Research Visualizations

Welcome to the complete guide for your newly updated repository! This document provides a **step-by-step run process** for newcomers and collaborators to seamlessly test the new architectures and generate professional graphs.

---

## Part 1: How to Run the MEA Architecture

We have fully implemented the architecture from the paper *"Multimodal fusion approach for learning modality-Exclusive and modality-Agnostic representations"*. 

**Before you run:** Ensure you have your standard dataset (MOSI or MOSEI) loaded using your existing data loaders. 

### Step 1: Open Your Training Notebook or Script
Open the Jupyter Notebook or Python file where you usually run your training loop (e.g., inside your `notebooks/` folder).

### Step 2: Make Code Modifications Before Running
You need to swap out your existing `DRCFNet` imports and training loop with the new MEA components. Replace your existing initialization code with the following snippet:

```python
import torch
import torch.optim as optim
from src.models.mea import MEA
from src.training.train_mea import train_mea_loop

# 1. Initialize the MEA Model
model = MEA(
    dim_v=35,    # Vision dimension (Verify this matches your dataset)
    dim_a=74,    # Audio dimension
    dim_t=300,   # Text dimension
    d=128,       # Hidden dimension
    n_heads=8    # Number of attention heads
).cuda()

# 2. Setup Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

### Step 3: Run the Training Process
Execute the training loop block by running the `train_mea_loop` function. This custom loop automatically handles the HSIC Disparity Loss and the Double-Discriminator Adversarial losses.

```python
# 3. Execute the MEA training loop
best_mea_model, history = train_mea_loop(
    model=model,
    train_loader=train_loader,   # Pass your existing train PyTorch DataLoader
    valid_loader=valid_loader,   # Pass your existing validation PyTorch DataLoader
    optimizer=optimizer,
    epochs=50,
    device='cuda',
    alpha=2e-2,  # HSIC disparity tradeoff
    beta=3e-2    # Adversarial tradeoff
)
```
**Execution:** Run the notebook cell or execute the python script. It will print Accuracy and F1 scores per epoch natively so you can compare them with your `DRCFNet` logs.

---

## Part 2: How to Generate DRCFNet Research Visuals

To build trust in your research paper, we've integrated automated code to output high-quality visuals directly from your *real* inference data. 

**Before you run:** Ensure you are using the updated `src/models/drcfnet.py` and `src/training/train.py` files.

### Workflow A: Generating Automatic Attention Weights and Training Curves

**Step 1: Open your standard DRCFNet Training Notebook/Script**
Open the file where you currently train your novel `DRCFNet` model.

**Step 2: Modify the Training Call**
Locate the line where you call the `train()` function from `src/training/train.py`. You only need to add one single flag: `generate_visuals=True`.

```python
from src.training.train import train

# Run your normal DRCFNet training, but enable visual generation
model, history = train(
    model=drcfnet_model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=50,
    generate_visuals=True  # <--- Turn this ON before running!
)
```

**Step 3: Run the Code**
Run your training script. 
*   **What happens?** As it evaluates the validation set, it will automatically save bar charts of the Modality Pair Weights ($\psi_m$) for a sample in the batch. Once all epochs are finished, it generates your F1/Loss curves.
*   **Where to find results?** Look inside the newly created `visuals_output/` folder in your project root.

---

### Workflow B: Generating Hyperparameter Graphs and Ablation Tables

For generating specific ablation tables or testing hyperparameter sensitivity (like varying $\gamma$), you must run your models across multiple configurations and collect the results.

**Step 1: Create a new Notebook or Script (e.g., `run_ablations.py`)**

**Step 2: Insert the Visualizer Code**
Copy and paste the following snippet to use the `visualizations.py` tools:

```python
from src.evaluation.visualizations import plot_hyperparameter_effect, generate_ablation_table

# 1. Hyperparameter Graphs
# Let's say you ran 5 training loops with different gamma weights.
# Replace these lists with the real metrics you recorded:
gamma_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
real_f1_scores = [83.2, 84.6, 84.3, 83.1, 82.7, 82.5] 

plot_hyperparameter_effect(
    param_values=gamma_values, 
    f1_scores=real_f1_scores, 
    param_name=r"NeuroSymbolic Loss Weight (\gamma)", 
    dataset_name="MOSI Performance",
    save_path="visuals_output/hyperparam_gamma_effect.png"
)

# 2. Ablation Tables
# Replace these values with the real metrics you got after testing ablated models
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

**Step 3: Run the Code**
Run this block of code. The formatted CSV tables and hyperparameter graphs will instantly generate inside the `visuals_output/` folder, ready to be dropped into your LaTeX or Word manuscript!
