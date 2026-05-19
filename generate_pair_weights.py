import os
import matplotlib.pyplot as plt

os.makedirs('outputs', exist_ok=True)

# Set up academic plotting style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

# Data matching the user's request (V-T exactly 74.3, others scaled around 11 and 14)
pairs = ['T-A', 'A-V', 'V-T']
weights = [11.5, 14.2, 74.3]  # Sum exactly to 100

colors = ['#4682B4', '#CD5C5C', '#2E8B57']  # SteelBlue, IndianRed, SeaGreen

fig, ax = plt.subplots(figsize=(8, 6))

# Create the bar plot
bars = ax.bar(pairs, weights, color=colors, edgecolor='black', linewidth=1.5, width=0.5, alpha=0.9)

# Add percentage labels exactly on top of the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.1f}%", 
            ha='center', va='bottom', fontweight='bold', fontsize=14)

# Formatting
ax.set_ylabel('Modality Pair Contribution (%)', fontweight='bold')
ax.set_title('Final Learned Pair Weights (MOSI Dataset)', fontweight='bold', pad=15)
ax.set_ylim(0, 90)  # Room for the text labels on top of the 75% bar
ax.grid(axis='y', linestyle='--', alpha=0.7, color='gray')

# Make the axes lines look distinct
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params(width=2, length=6)

plt.tight_layout()
plt.savefig('outputs/pair_weights_mosi.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated outputs/pair_weights_mosi.png")
