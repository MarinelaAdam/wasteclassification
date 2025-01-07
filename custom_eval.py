import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from arhitecture import WasteClassifier
import torchvision.models as models
import torch.nn as nn

BATCH_SIZE = 16
MODEL_PATH = r"Models\resnet152_90_61\model.pth"

def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f'Accuracy on test images: {accuracy * 100:.2f}%')
    print(f'Precision on test images: {precision * 100:.2f}%')
    print(f'Recall on test images: {recall * 100:.2f}%')
    print(f'F1-score on test images: {f1 * 100:.2f}%')

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Organic", "Recyclable"],
                yticklabels=["Organic", "Recyclable"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_dataset = datasets.ImageFolder(r"Datasets\WasteClassification\test", test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # # for ResNext
    # model = models.resnext101_32x8d()
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)

    # for VGG16s
    # model = models.vgg16(weights="DEFAULT")
    # num_ftrs = model.classifier[6].in_features
    # model.classifier[6] = nn.Linear(num_ftrs, 2)

    # for ResNet
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

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))
    model = model.to(device)
    evaluate_model(model, test_loader, device)