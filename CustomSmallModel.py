import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "Datasets\WasteClassification"
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
VAL_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 4
MODEL_SAVE_PATH = "best_model.pth"
CSV_LOG_FILE = "results.csv"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
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
    model = models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.IMAGENET1K_V2')
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(nn.Linear(in_features, 1))
    return model.to(device)

# for V3 large
# def create_model():
#     model = models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.IMAGENET1K_V2')
#     in_features = model.classifier[0].in_features
#     model.classifier = nn.Sequential(
#         nn.Linear(in_features, 128),
#         nn.ReLU(),
#         nn.Dropout(0.4),
#         nn.Linear(128, 1)
#     )
#     return model.to(device)

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
    model = create_model()

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

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    test(model, test_loader)

if __name__ == "__main__":
    main()
