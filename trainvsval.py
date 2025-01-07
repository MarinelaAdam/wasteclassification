import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for a modern look
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("bright6")

# Read the CSV file
df = pd.read_csv('Models\model_90_85/results.csv')

# Create figure and subplots with a modern aspect ratio
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.set_facecolor('#f0f0f0')

# Customize the plot style
plot_style = {
    'linewidth': 2.5,
    'markersize': 6,
    'alpha': 0.8
}

# Plot Loss
train_loss = df[df['phase'] == 'train']['loss']
val_loss = df[df['phase'] == 'val']['loss']
epochs = df[df['phase'] == 'train']['epoch']

line1, = ax1.plot(epochs, train_loss, marker='o', label='Training Loss', **plot_style)
line2, = ax1.plot(epochs, val_loss, marker='s', label='Validation Loss', **plot_style)

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.tick_params(axis='both', which='major', labelsize=10)

# Plot Accuracy
train_acc = df[df['phase'] == 'train']['accuracy']
val_acc = df[df['phase'] == 'val']['accuracy']

line3, = ax2.plot(epochs, train_acc, marker='o', label='Training Accuracy', **plot_style)
line4, = ax2.plot(epochs, val_acc, marker='s', label='Validation Accuracy', **plot_style)

ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.tick_params(axis='both', which='major', labelsize=10)

# Format accuracy as percentage
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

# Adjust layout and display
plt.tight_layout(pad=3.0)

# Save the plot with high DPI for better quality
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
plt.show()