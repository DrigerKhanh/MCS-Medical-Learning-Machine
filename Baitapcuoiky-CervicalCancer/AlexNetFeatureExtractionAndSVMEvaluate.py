import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from torchvision import datasets
from torch.utils.data import DataLoader

train_dir = "./dataset/Train"
test_dir = "./dataset/Test"

# Kiểm tra nếu có GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load AlexNet pre-trained
# weights=models.AlexNet_Weights.IMAGENET1K_V1: sử dụng trọng số đã được huấn luyện trên ImageNet.
# IMAGENET1K_V1 :mô hình được huấn luyện trên ImageNet-1K (1.3 triệu ảnh, 1000 lớp phân loại).
alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

# Đóng băng các trọng số của các lớp convolutional (feature extractor)
for param in alexnet.features.parameters():
    param.requires_grad = False

# Thay thế classifier cuối cùng để phù hợp với bài toán 4 lớp
alexnet.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(9216, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 4)
)
alexnet = alexnet.to(device)

# Cấu hình preprocessing cho ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Kích thước input của AlexNet
    transforms.ToTensor(),
])

# Load dataset
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Cấu hình Loss và Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(alexnet.classifier.parameters(), lr=0.0001)

# Training Loop
num_epochs = 10
loss_history = []  # Save loss for each epoch
for epoch in range(num_epochs):
    alexnet.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = alexnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    loss_history.append(running_loss / len(train_loader))
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

print("Training Completed")

# Ve loss function
plt.plot(range(1, num_epochs + 1), loss_history, marker='o', label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

# Đánh giá trên tập test
alexnet.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = alexnet(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Train SVM on extracted features
# Tạo feature extractor từ AlexNet

# Chuyển model sang chế độ đánh giá (evaluation mode)
alexnet.eval()

# Trích xuất chỉ phần feature extractor của AlexNet
feature_extractor = alexnet.features  # Trích xuất phần features chuẩn
feature_extractor = nn.Sequential(
    alexnet.features,
    nn.AdaptiveAvgPool2d((6, 6)),
    nn.Flatten()
)
feature_extractor = feature_extractor.to(device)

# Hàm trích xuất đặc trưng từ tập dữ liệu
def extract_features(dataloader, model):
    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            # Đảm bảo shape đầu ra luôn là (B, N)
            if outputs.dim() > 2:
                outputs = outputs.view(outputs.size(0), -1)

            features_list.append(outputs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    return np.vstack(features_list), np.hstack(labels_list)

# Trích xuất đặc trưng cho tập train, validation & test
X_train, y_train = extract_features(train_loader, feature_extractor)
X_test, y_test = extract_features(test_loader, feature_extractor)

# Kiểm tra kích thước đầu ra
print("\nTrain SVM on extracted features")
print("X_train shape:", X_train.shape)  # Ví dụ: (400, 9216)
print("X_test shape:", X_test.shape)    # Ví dụ: (200, 9216)

# Huấn luyện mô hình SVM trên feature đã trích xuất
# Tạo mô hình SVM với tiền xử lý StandardScaler
svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))

# Huấn luyện mô hình
svm_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = svm_model.predict(X_test)

# Đánh giá SVM
# Chọn kernel cho SVM
kernel_choice = 'rbf'  # Chọn : 'linear', 'rbf', 'poly', 'sigmoid'

# Tính các chỉ số đánh giá
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro',zero_division=0)
recall = recall_score(y_test, y_pred, average='macro',zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro',zero_division=0)

# In kết quả đánh giá
print("\nPrediction result")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-score: {f1 * 100:.2f}%")

# Confussion matrix
cm = confusion_matrix(y_test, y_pred)
class_names = ['HSIEL', 'LSIEL', 'NFIM', 'SCC']  # Tên lớp theo thứ tự trong tập dữ liệu

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)

plt.title("Confusion Matrix for Cervical Cancer Classification")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# Tạo classification report dạng dict với tên lớp
report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True,zero_division=0)

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


# # Classification - Method 2
# # Tên lớp theo thứ tự class_to_idx trong ImageFolder
# class_names = test_dataset.classes  # Hoặc tự định nghĩa nếu biết chắc thứ tự
#
# # In classification report từng lớp
# print("\nDetailed Classification Report per class:")
# print(classification_report(y_test, y_pred, target_names=class_names, digits=2, zero_division=0))

# Binarize y_test để tính AUC cho từng lớp (one-vs-rest)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])

# Dự đoán xác suất cho từng lớp
y_pred_proba = svm_model.predict_proba(X_test)  # shape: (n_samples, n_classes)

# Tính AUC cho mỗi lớp và vẽ ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(4):  # 4 lớp
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])

# Vẽ tất cả các đường ROC cho từng lớp
plt.figure(figsize=(8, 6))
colors = ['blue', 'orange', 'green', 'red']
for i in range(4):
    plt.plot(fpr[i], tpr[i], color=colors[i], label=f'Class {i} (AUC = {roc_auc[i]:.4f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (One-vs-Rest)")
plt.legend()
plt.grid(True)
plt.show()