import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors

def plot_metrics(csv_paths, model_names=None):

    if model_names is None:
        model_names = [os.path.splitext(os.path.basename(path))[0] for path in csv_paths]

    metrics = ['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']
    metric_titles = {
        'train_loss': 'Train Loss',
        'train_accuracy': 'Train Accuracy',
        'val_loss': 'Validation Loss',
        'val_accuracy': 'Validation Accuracy'
    }

    data = {metric: [] for metric in metrics}

    for path, name in zip(csv_paths, model_names):
        df = pd.read_csv(path)
        for metric in metrics:
            data[metric].append((name, df['epoch'], df[metric]))

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    for i, metric in enumerate(metrics):
        ax = axs[i]
        for model_name, epochs, values in data[metric]:
            ax.plot(epochs, values, label=model_name)
        ax.set_title(metric_titles[metric])
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_titles[metric])
        ax.legend()
        ax.grid(True)

        if 'loss' in metric:
            all_loss_values = [v for _, _, v in data[metric]]
            min_loss = min(min(v) for v in all_loss_values)
            max_loss = max(max(v) for v in all_loss_values)
            ax.set_ylim([min_loss - 0.05, max_loss + 0.05])

    plt.tight_layout()
    plt.show()

def plot_metrics_sepa(csv_paths, model_names=None):

    if model_names is None:
        model_names = [os.path.splitext(os.path.basename(path))[0] for path in csv_paths]

    base_colors = plt.cm.get_cmap('tab10', len(model_names))  # Use a colormap

    loss_data = {'train': [], 'val': []}
    accuracy_data = {'train': [], 'val': []}

    for path, name in zip(csv_paths, model_names):
        df = pd.read_csv(path)
        loss_data['train'].append((name, df['epoch'], df['train_loss']))
        loss_data['val'].append((name, df['epoch'], df['val_loss']))
        accuracy_data['train'].append((name, df['epoch'], df['train_accuracy']))
        accuracy_data['val'].append((name, df['epoch'], df['val_accuracy']))

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    ax = axs[0]
    for i, model_name in enumerate(model_names):
        base_color = base_colors(i)
        for data_type in ['train', 'val']:
            for name, epochs, values in loss_data[data_type]:
                if name == model_name:
                    color = mcolors.to_rgba(base_color)
                    if data_type == 'val':
                        color = tuple([c * 0.8 for c in color[:3]] + [color[3]])
                    ax.plot(epochs, values, label=f"{name} ({data_type})", color=color)

    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True)

    all_loss_values = [v for _, _, v in loss_data['train']] + [v for _, _, v in loss_data['val']]
    min_loss = min(min(v) for v in all_loss_values)
    max_loss = max(max(v) for v in all_loss_values)
    ax.set_ylim([min_loss - 0.05, max_loss + 0.05])

    ax = axs[1]
    for i, model_name in enumerate(model_names):
        base_color = base_colors(i)
        for data_type in ['train', 'val']:
            for name, epochs, values in accuracy_data[data_type]:
                if name == model_name:
                    color = mcolors.to_rgba(base_color)
                    if data_type == 'val':
                        color = tuple([c * 0.8 for c in color[:3]] + [color[3]])
                    ax.plot(epochs, values, label=f"{name} ({data_type})", color=color)

    ax.set_title('Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True)

    plt.tight_layout()
    plt.show()

csv_paths = [
    '../Models/mobilnetv2_90/results.csv',
    '../Models/mobilnetv3_93_16/results.csv',
    '../Models/model_90_85/results.csv',
    '../Models/resnet152_90_61/results.csv',
    '../Models/resnext_91_5/results.csv',
    '../Models/vgg16_94_16/results.csv',
]
model_names = [
    'MobileNetV2',
    'MobileNetV3',
    'MobileNetV4',
    'WasteClassifer',
    'ResNet152',
    'ResNext101',
    'VGG16'
]

# plot_metrics(csv_paths, model_names)
plot_metrics_sepa(csv_paths, model_names)