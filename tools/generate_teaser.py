import matplotlib.pyplot as plt
import numpy as np

# Data
batch_sizes = [256, 1024, 16384]
x = np.array(batch_sizes)

# Values and standard deviations
complex_means = [34, 2, 0]
complex_stds = [7, 2, 0]

simple_means = [21, 38, 64]
simple_stds = [11, 3, 2]

# Plotting with adjusted figure height
fig, ax = plt.subplots(figsize=(8, 4.6))  # 60% of standard height

# Plot the lines with error bars (increased line width)
ax.errorbar(x, complex_means, yerr=complex_stds, fmt='-o',
            label='Complex reward (prior work)', capthick=3, color='skyblue', capsize=5, linewidth=4)

ax.errorbar(x, simple_means, yerr=simple_stds, fmt='-o',
            label=r'Simple reward (ours)',
            color='salmon', capthick=3, capsize=5, linewidth=4)

# Labels and Title (increased font size)
ax.set_ylabel('Driving Score', fontsize=14)
ax.set_xlabel('Mini-batch size (log scale)', fontsize=14)
ax.set_xscale('log')
ax.set_xticks(x)
ax.set_xticklabels(batch_sizes, fontsize=12)
ax.legend(fontsize=12)

# Add value labels above error bars (increased font size)
def add_labels(x_vals, means, stds):
    for x_i, y, std in zip(x_vals, means, stds):
        # For the labels of 64 and 21, place them below the points
        if y == 64 or y == 21 or y == 38:
            label_y = y - std - 5.0
        else:
            label_y = y + std + 0.5 if std > 0 else y + 1
        ax.annotate(f'{y}',
                    xy=(x_i, label_y),
                    ha='center', va='bottom', fontsize=12)

add_labels(x, complex_means, complex_stds)
add_labels(x, simple_means, simple_stds)

# Add "+ data scale" annotation to both lines between 1024 and 16384
mid_x = np.sqrt(1024 * 16384)

# For simple reward line
mid_y_simple = (38 + 64) / 2 + 5
ax.annotate('+ data scale',
            xy=(mid_x, mid_y_simple),
            ha='center', va='bottom',
            fontsize=12, fontstyle='italic',
            color='salmon')

# For complex reward line
mid_y_complex = (2 + 0) / 2 + 5
ax.annotate('+ data scale',
            xy=(mid_x, mid_y_complex),
            ha='center', va='bottom',
            fontsize=12, fontstyle='italic',
            color='skyblue')

fig.tight_layout()

# Save the figure
plt.savefig(r'L:\Ordnung\Promotion\ad_planning\2_carla\images\teaser_squashed_double_annotation.png', dpi=300)

plt.show()
