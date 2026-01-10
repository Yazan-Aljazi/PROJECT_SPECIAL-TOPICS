import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


train_dir = "Data/train_variant_balanced"   
test_dir  = "Data/test"

batch_size = 32
num_epochs = 10
learning_rate = 1e-4
img_size = 64

train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
test_dataset  = datasets.ImageFolder(root=test_dir,  transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes
print("Train classes:", class_names)
print("Number of train images:", len(train_dataset))
print("Number of test images:", len(test_dataset))

#  نموذج CNN (ResNet18)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)  # cats, dogs
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Accuracy 

def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total > 0 else 0
    return acc

def evaluate_with_metrics(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")

    return cm, precision, recall, f1


best_test_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_acc = evaluate_accuracy(model, train_loader, device)
    test_acc  = evaluate_accuracy(model, test_loader, device)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {epoch_loss:.4f} "
          f"Train Acc: {train_acc:.4f} "
          f"Test Acc: {test_acc:.4f}")

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        os.makedirs("../Outputs/classifier", exist_ok=True)
        torch.save(model.state_dict(),
                   "../Outputs/classifier/cnn_variant_balanced_best.pth")

print("Best Test Accuracy (Variant Balanced):", best_test_acc)

cm, precision, recall, f1 = evaluate_with_metrics(model, test_loader, device)
print("Confusion Matrix (Variant Balanced):\n", cm)
print(f"Precision (Variant): {precision:.4f}")
print(f"Recall (Variant):    {recall:.4f}")
print(f"F1-score (Variant):  {f1:.4f}")
