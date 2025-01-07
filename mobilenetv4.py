import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "Datasets\WasteClassificationBig"
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
VAL_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 4
MODEL_SAVE_PATH = "best_model.pth"
CSV_LOG_FILE = "results.csv"

# Data Augmentation and Normalization
train_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
])

test_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
])

def load_data():
    train_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    test_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=test_transform)

    train_size = int((1 - VAL_SPLIT) * len(train_data))
    val_size = len(train_data) - train_size
    train_subset, val_subset = random_split(train_data, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

def create_model():
    model_name = 'mobilenetv4_hybrid_large.e600_r384_in1k'
    model = timm.create_model(model_name, pretrained=True)
    
    # Get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)

    classifier = model.get_classifier()
    if isinstance(classifier, nn.Linear):
        num_features = classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1)
        )
    elif isinstance(classifier, nn.Sequential):
        if isinstance(classifier[-1], nn.Linear):
            num_features = classifier[-1].in_features
            model.classifier = nn.Sequential(
                *list(classifier.children())[:-1],
                nn.Dropout(0.5),
                nn.Linear(num_features, 1)
            )
        else:
            model.classifier.add_module("my_dropout", nn.Dropout(0.5))
            model.classifier.add_module("my_linear", nn.Linear(num_features, 1))
    else:
        raise ValueError("Unsupported classifier type.")

    return model.to(device), data_config

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = torch.sigmoid(outputs).round().detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss / len(loader), accuracy

def validate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            preds = torch.sigmoid(outputs).round().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss / len(loader), accuracy

def test(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs).round().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['Organic', 'Recyclable'])
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

def save_metrics_to_csv(metrics):
    with open(CSV_LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy'])
        writer.writerows(metrics)

def main():
    train_loader, val_loader, test_loader = load_data()
    model, data_config = create_model()

    global train_transform, test_transform
    train_transform = transforms.Compose([
        transforms.Resize(data_config['input_size'][1:]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(data_config['input_size'][1:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
    ])

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.2)

    best_val_accuracy = 0
    patience_counter = 0
    metrics = []

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = validate(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        metrics.append([epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy])
        scheduler.step(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    save_metrics_to_csv(metrics)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test(model, test_loader)

if __name__ == "__main__":
    main()