import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD, RAdam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import torchvision.models as models
import copy
import os
from tqdm import tqdm
import torch.nn as nn

# Constants
DATA_DIR = "WasteClassificationBig"
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 5
MODEL_PATH = "model.pth"
RESULTS_CSV = "results.csv"

class EarlyStopping:
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_acc = None
        self.early_stop = False

    def __call__(self, val_acc, model):
        if self.best_acc is None:
            self.best_acc = val_acc
        elif val_acc <= self.best_acc:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_acc = val_acc
            self.counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            if self.verbose:
                print(f'Validation accuracy increased. Saving model ...')

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, dataset_sizes, num_epochs=NUM_EPOCHS):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    results = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_epoch_loss = running_loss / dataset_sizes['train']
        train_epoch_acc = running_corrects.double() / dataset_sizes['train']

        print(f'Train Loss: {train_epoch_loss:.4f} Acc: {train_epoch_acc:.4f}')

        # Validation Phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1} - Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = running_loss / dataset_sizes['val']
        val_epoch_acc = running_corrects.double() / dataset_sizes['val']

        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')

        scheduler.step(val_epoch_acc)
        early_stopping(val_epoch_acc, model)

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        results.append({
            'epoch': epoch + 1,
            'train_loss': train_epoch_loss,
            'train_accuracy': train_epoch_acc.item(),
            'val_loss': val_epoch_loss,
            'val_accuracy': val_epoch_acc.item()
        })

        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)

    return model

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), val_transforms)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    # For VGGs
    # model = models.vgg16(weights="DEFAULT")
    # num_ftrs = model.classifier[6].in_features
    # model.classifier[6] = nn.Linear(num_ftrs, 2)

    # For ResNexts
    # model = models.resnext101_32x8d(weights='DEFAULT')
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)

    # For ResNets
    model = models.resnet152(weights = 'ResNet152_Weights.IMAGENET1K_V2')
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )

    model = model.to(device)

    # for VGGs and ResNext
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # for ResNets
    # for param in model.fc.parameters():
    #     param.requires_grad = True

    # Class Weights
    all_train_labels = [label for _, label in train_dataset]
    num_class_0 = all_train_labels.count(0)
    num_class_1 = all_train_labels.count(1)

    total_samples = num_class_0 + num_class_1
    weight_class_0 = total_samples / (2 * num_class_0)
    weight_class_1 = total_samples / (2 * num_class_1)

    class_weights = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float).to(device)
    print("Class Weights:", class_weights)

    # Loss, Optimizer, Scheduler, Early Stopping
    criterion = CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    # for ResNets
    # optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max',factor=0.5, patience=3)

    early_stopping = EarlyStopping(verbose=True)

    model = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, dataset_sizes)