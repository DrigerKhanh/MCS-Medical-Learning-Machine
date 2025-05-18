import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tabulate import tabulate
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


import os

from torchvision.models import AlexNet_Weights

# Thiết bị sử dụng: GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))

# Tham số
batch_size = 32
num_classes = 5  # Sửa theo số lớp thực tế
num_epochs = 10
train_dir = "./dataset/Train"
val_dir = "./dataset/Validation"
test_dir = "./dataset/Test"

# Tiền xử lý ảnh
# Augmentation
def training_transform():
    transform = transforms.Compose([
        transforms.Resize(256),                                                     # Resize cạnh ngắn nhất về 256 pixel
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),                                                      # Chuyển đổi thành Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Chuẩn hóa theo ImageNet
    ])
    return transform

def val_and_test_transform():
    # Cấu hình preprocessing cho ảnh
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Kích thước input của AlexNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform

# Tải dataset
train_dataset = datasets.ImageFolder(train_dir, transform=training_transform())
val_dataset = datasets.ImageFolder(val_dir, transform=val_and_test_transform())
test_dataset = datasets.ImageFolder(test_dir, transform=val_and_test_transform())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load mô hình AlexNet
model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
model.classifier[6] = nn.Linear(4096, num_classes)  # Thay lớp output
model = model.to(device)

# Hàm mất mát và tối ưu
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Hàm đánh giá
def evaluate_model(model, dataloader, return_only_accuracy=False):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(all_labels, all_preds)

    if return_only_accuracy:
        return accuracy

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print("\nPrediction result")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")

    return accuracy, precision, recall, f1, all_labels, all_preds

# Huấn luyện mô hình
# Hàm train mô hình
def train_model(model, train_loader, val_loader, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_losses = []  # <-- Thêm dòng này để lưu loss

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)  # Trung bình loss mỗi batch
        train_losses.append(epoch_loss)  # <-- Ghi lại loss của epoch

        train_acc = 100 * running_corrects.double() / total
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}%")

    # Vẽ biểu đồ loss
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Training Loss')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Huấn luyện mô hình
train_model(model, train_loader, val_loader, num_epochs=num_epochs)

# Đánh giá trên validation set
print("\n🔎 Validation Evaluation:")
val_accuracy, val_precision, val_recall, val_f1, _, _ = evaluate_model(model, val_loader)
print(f"Validation Accuracy: {val_accuracy:.2f}%")

# Đánh giá trên test set
print("\n🧪 Test Evaluation:")
test_accuracy, test_precision, test_recall, test_f1, y_true, y_pred = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.2f}%")

# # Đánh giá trên tập test
# model.eval()
# y_test = []
# y_pred = []
# y_score = []
#
# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         y_test.extend(labels.cpu().numpy())
#         y_pred.extend(predicted.cpu().numpy())
#         y_score.extend(torch.softmax(outputs, dim=1).cpu().numpy())


# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
class_names = ['DYK', 'KOC', 'MEP', 'PAB', 'SFI']  # Tên lớp theo thứ tự trong tập dữ liệu

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)

plt.title("Confusion Matrix for Cervical Cancer Classification")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# Classification report
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True,zero_division=0)

headers = ["Class", "Precision (%)", "Recall (%)", "F1-score (%)", "Support"]
report_table = []

for label in class_names:
    row = report_dict[label]
    report_table.append([
        label,
        f"{row['precision'] * 100:.2f}",
        f"{row['recall'] * 100:.2f}",
        f"{row['f1-score'] * 100:.2f}",
        int(row['support'])
    ])

# Thêm macro/micro/weighted avg nếu bạn muốn
for avg in ['macro avg', 'weighted avg']:
    row = report_dict[avg]
    report_table.append([
        avg,
        f"{row['precision'] * 100:.2f}",
        f"{row['recall'] * 100:.2f}",
        f"{row['f1-score'] * 100:.2f}",
        int(row['support'])
    ])

# In bảng
print("\nClassification Report:")
print(tabulate(report_table, headers=headers, tablefmt="grid"))

# ROC Curve cho từng class
def plot_roc(model, dataloader, num_classes):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(labels)

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()



    binary_labels = label_binarize(all_labels, classes=list(range(num_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(binary_labels[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

plot_roc(model, test_loader, num_classes)

# Save
torch.save(model.state_dict(), 'alexnet_model_weights.pth')
