import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
import matplotlib.pyplot as plt
import numpy as np

# ===================== Config =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dir = "./dataset/Training"
validation_dir = "./dataset/Validation"
test_dir = "./dataset/Test"

num_classes = 2
batch_size = 32
epochs = 10
learning_rate = 1e-4

# ===================== Data Transform =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
validation_dataset = datasets.ImageFolder(validation_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names = train_dataset.classes

# ===================== Load VGG16 & Fine-tune =====================
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# Đóng băng tất cả layers
for param in vgg16.parameters():
    param.requires_grad = False

# Cho phép fine-tune classifier[3] (fc6) và classifier[6] (fc7)
for i, layer in enumerate(vgg16.classifier):
    if i in [3, 6]:
        for param in layer.parameters():
            param.requires_grad = True

# Thay lớp cuối cùng phù hợp với số lớp
vgg16.classifier[6] = nn.Linear(4096, num_classes)
vgg16 = vgg16.to(device)

# ===================== Loss & Optimizer =====================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, vgg16.parameters()), lr=learning_rate)

# ===================== Training =====================
train_losses = []
vgg16.train()

for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = vgg16(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# ===================== Plot Loss =====================
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs+1), train_losses, marker='o')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()

# ===================== Evaluation =====================
vgg16.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = vgg16(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Cat', 'Predicted Dog'], yticklabels=['Actual Cat', 'Actual Dog'])
plt.title('Confusion Matrix - VGG16')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("\nClassification Report:")
# Tính các chỉ số đánh giá
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(classification_report(y_true, y_pred, target_names=class_names))
# Ve classification report
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
values = [accuracy, precision, recall, f1]
colors = ['skyblue', 'sandybrown', 'lightgreen', 'lightcoral']

# Vẽ biểu đồ cột
plt.figure(figsize=(6, 5))
bars = plt.bar(metrics, values, color=colors)

# Hiển thị giá trị trên đầu mỗi cột
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')

plt.ylim(0, 1.1)  # Đặt giới hạn trục Y
plt.title("VGG16 Classification Report Summary")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()