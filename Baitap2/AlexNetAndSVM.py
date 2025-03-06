import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load Pretrained AlexNet & Modify
alexnet = models.alexnet(pretrained=True)
for param in alexnet.features.parameters():
    param.requires_grad = False  # Freeze convolutional layers

# Thay thế hai lớp fully connected cuối cùng
alexnet.classifier[6] = torch.nn.Linear(4096, 2)  # 2 classes (dog & cat)
alexnet = alexnet.to('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Tiền xử lý ảnh với Augmentation
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # Lật ngang ngẫu nhiên
    transforms.RandomRotation(15),  # Xoay nhẹ ảnh
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Điều chỉnh màu sắc
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Cắt ngẫu nhiên
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 3. Dataset tùy chỉnh
class CustomDataset(Dataset):
    def __init__(self, root_dir, label):
        self.root_dir = root_dir
        self.label = label
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = data_transform(image)
        return image, self.label


# Load dataset
cat_dataset = CustomDataset("./dataset/cat/", label=0)
dog_dataset = CustomDataset("./dataset/dog/", label=1)
dataset = cat_dataset + dog_dataset  # Gộp dữ liệu lại


# Chia dataset thành hai bộ tách biệt
def split_dataset(dataset, train_ratio, val_ratio, test_ratio):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])


# Bộ 1: Train (70%), Validation (10%), Test (20%)
train_set1, val_set1, test_set1 = split_dataset(dataset, 0.7, 0.1, 0.2)
# Bộ 2: Train (60%), Validation (20%), Test (20%)
train_set2, val_set2, test_set2 = split_dataset(dataset, 0.6, 0.2, 0.2)

# Tạo DataLoader
train_loader1 = DataLoader(train_set1, batch_size=32, shuffle=True)
val_loader1 = DataLoader(val_set1, batch_size=32, shuffle=False)
test_loader1 = DataLoader(test_set1, batch_size=32, shuffle=False)

train_loader2 = DataLoader(train_set2, batch_size=32, shuffle=True)
val_loader2 = DataLoader(val_set2, batch_size=32, shuffle=False)
test_loader2 = DataLoader(test_set2, batch_size=32, shuffle=False)

# 4. Cấu hình huấn luyện
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(alexnet.classifier[6].parameters(), lr=1e-5)


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    train_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

    # Vẽ biểu đồ loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()




# Đánh giá mô hình
def evaluate_model(model, test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Vẽ confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Huấn luyện mô hình trên cả hai bộ
train_model(alexnet, train_loader1, val_loader1, criterion, optimizer, epochs=5)
evaluate_model(alexnet, test_loader1)

train_model(alexnet, train_loader2, val_loader2, criterion, optimizer, epochs=5)
evaluate_model(alexnet, test_loader2)


evaluate_model(alexnet, test_loader1)
evaluate_model(alexnet, test_loader2)
