import matplotlib.pyplot as plt

# Data
diag_size = [0, 1, 2, 4, 8, 16, 32]
diag_time_us = [17960.447, 15100.096, 14348.96, 14057.6, 13437.344, 12158.24, 10193.92]
diag_cossim = [0.778, 0.782, 0.784, 0.786, 0.792, 0.801, 0.809]

sink_size = [0, 1, 2, 4, 8, 16, 32]
sink_time_us = [17960.447, 7344.928, 24788.639, 24495.199, 24270.336, 23857.023, 24483.2]
sink_cossim = [0.778, 0.781, 0.782, 0.786, 0.792, 0.809, 0.834]

# Convert x to percentage
# x_percent = [x / 32 * 100 for x in diag_size]

# Start plotting
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot Time (left Y-axis)
color_diag = 'tab:blue'
color_sink = 'tab:green'
ax1.set_xlabel('High-bit (%)')
ax1.set_ylabel('Time (us)', color='black')
l1 = ax1.plot(diag_size, diag_time_us, marker='o', linestyle='-', color=color_diag, label='Diag Time')
l2 = ax1.plot(diag_size, sink_time_us, marker='s', linestyle='--', color=color_sink, label='Sink Time')
ax1.tick_params(axis='y', labelcolor='black')

# Plot CosSim (right Y-axis)
ax2 = ax1.twinx()
ax2.set_ylabel('Cosine Similarity', color='black')
l3 = ax2.plot(diag_size, diag_cossim, marker='o', linestyle='-', color=color_diag, alpha=0.5, label='Diag CosSim')
l4 = ax2.plot(diag_size, sink_cossim, marker='s', linestyle='--', color=color_sink, alpha=0.5, label='Sink CosSim')
ax2.tick_params(axis='y', labelcolor='black')

# Legend
lines = l1 + l2 + l3 + l4
labels = [line.get_label() for line in lines]
plt.legend(lines, labels, loc='center right')

plt.ylim(0.7, 0.9)

# Grid and layout
ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.title('Inference Time and Cosine Similarity vs High-bit Size')
plt.tight_layout()
# plt.show()

plt.savefig('saved_figs/tile_size_ablation/tile_size_vs_time_and_cossim.png')