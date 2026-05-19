import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Attention Matrix Visualization (Figure 3 Style)
# ==========================================
def plot_attention_comparison(vanilla_attn, proposed_attn, words, title="Attention Comparison", save_path="attention_comparison.png"):
    """
    Plots a side-by-side comparison of attention matrices.
    vanilla_attn, proposed_attn: 2D numpy arrays (N x N)
    words: list of strings (length N)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Vanilla Attention
    sns.heatmap(vanilla_attn, xticklabels=False, yticklabels=words, 
                cmap="YlGnBu", ax=axes[0], cbar=False, square=True)
    axes[0].set_title("(a) Vanilla self-attention", fontsize=14, y=-0.15)
    axes[0].tick_params(axis='y', labelsize=12, labelrotation=0)
    
    # Proposed Attention
    sns.heatmap(proposed_attn, xticklabels=False, yticklabels=words, 
                cmap="YlGnBu", ax=axes[1], cbar=False, square=True)
    axes[1].set_title("(b) DRCFNet Attention", fontsize=14, y=-0.15)
    axes[1].tick_params(axis='y', labelsize=12, labelrotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved attention comparison to {save_path}")

# ==========================================
# 2. Modality/Pair Weights Visualization (Figure 6 Style)
# ==========================================
def plot_pair_weights(pair_weights, save_path="pair_weights.png"):
    """
    Visualizes the pair weights (T-A, A-V, V-T) for a given sample.
    pair_weights: list or 1D numpy array of size 3.
    """
    labels = ['T-A Pair', 'A-V Pair', 'V-T Pair']
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, pair_weights, color=colors, width=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=12, color='red')
        
    plt.ylim(0, 1.0)
    plt.ylabel(r'Attention Weight \psi_m', fontsize=12)
    plt.title('Bimodal Pair Attention Weights', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved pair weights visualization to {save_path}")

# ==========================================
# 3. Hyperparameter Effect Visualization (Figure 7 & 8 Style)
# ==========================================
def plot_hyperparameter_effect(param_values, f1_scores, param_name="num_layers", dataset_name="MOSI", save_path="hyperparam_effect.png"):
    """
    Plots the effect of a specific hyperparameter on the F1 score.
    param_values: list of x-axis values
    f1_scores: list of y-axis values
    """
    plt.figure(figsize=(6, 5))
    
    # Plot style matching the paper
    plt.plot(param_values, f1_scores, marker='^', linestyle='-.', color='#FFA07A', markersize=8, linewidth=2)
    
    plt.xlabel(f'Value ({param_name})', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
    plt.title(dataset_name, fontsize=16, fontweight='bold', pad=20)
    
    # Customize ticks
    plt.xticks(param_values)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved hyperparameter effect graph to {save_path}")

# ==========================================
# 4. Ablation Study Table Generation
# ==========================================
def generate_ablation_table(results_dict, dataset_name="MOSI", save_path="ablation_table.csv"):
    """
    results_dict format:
    {
        "DRCFNet (Full Model)": {"Acc_7": 42.5, "F1": 84.6},
        "w/o MSR/SSR Split": {"Acc_7": 41.2, "F1": 83.2},
        "w/o Bimodal GCF": {"Acc_7": 40.5, "F1": 82.5},
        "w/o Dual-Graph Fusion": {"Acc_7": 41.0, "F1": 82.9},
        "w/o MSTCN Projection": {"Acc_7": 39.8, "F1": 81.5}
    }
    """
    data = []
    for component, metrics in results_dict.items():
        data.append({
            "Components/Designs/Mechanisms": component,
            f"{dataset_name} Acc_7 \u2191": metrics.get("Acc_7", "-"),
            f"{dataset_name} F1 \u2191": metrics.get("F1", "-")
        })
        
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Saved ablation table to {save_path}")
    
    # Also print it beautifully
    print("\n" + "="*50)
    print(f" Ablation Study Results ({dataset_name})")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50 + "\n")
    
    return df

# ==========================================
# 5. Extract Pair Weights from Model during Testing
# ==========================================
def extract_and_plot_pair_weights(model, batch, device='cuda'):
    """
    A helper function to run a single batch through DRCFNet, extract the 
    pair_weights (T-A, A-V, V-T), and plot them.
    Note: Requires a slight modification to DRCFNet to return `pair_weights` in the features dict.
    """
    model.eval()
    vision, audio, text, labels = batch
    vision = vision.to(device)
    audio = audio.to(device)
    text = text.to(device)
    
    with torch.no_grad():
        logits, features = model(vision, audio, text)
        
    # Assuming we added 'pair_weights' to the returned dictionary in DRCFNet
    if 'pair_weights' in features:
        # Get the weights for the first sample in the batch
        weights = features['pair_weights'][0].cpu().numpy().flatten()
        plot_pair_weights(weights, save_path="sample_pair_weights.png")
    else:
        print("Please update DRCFNet forward method to return 'pair_weights' in the feature dictionary.")

# ==========================================
# 6. Training Curve Visualization
# ==========================================
def plot_training_curves(history, save_path="training_curves.png"):
    """
    Plots the training and validation loss curves over epochs.
    history: dict containing 'train_loss', 'val_loss', 'val_acc'
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss Curve
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', marker='o')
    axes[0].plot(epochs, history['val_loss'], 'r--', label='Validation Loss', marker='s')
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Accuracy Curve
    axes[1].plot(epochs, history['val_acc'], 'g-', label='Validation Accuracy', marker='^')
    axes[1].set_title('Validation Accuracy', fontsize=14)
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training curves to {save_path}")

