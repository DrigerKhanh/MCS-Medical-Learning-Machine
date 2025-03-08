import numpy as np
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision import datasets
from torch.utils.data import DataLoader

train_dir = "./dataset/Training"
validation_dir = "./dataset/Validation"
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

# Thay thế classifier cuối cùng để phù hợp với bài toán 2 lớp (Cats vs Dogs)
alexnet.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(9216, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 1024),
    nn.ReLU(),
    nn.Linear(1024, 2)  # Output 2 lớp (Cats vs Dogs)
)

# Cấu hình preprocessing cho ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Kích thước input của AlexNet
    transforms.ToTensor(),
])

# Load dataset
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
validation_dataset = datasets.ImageFolder(root=validation_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Cấu hình Loss và Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(alexnet.classifier.parameters(), lr=0.0001)

# Training Loop
num_epochs = 5
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

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

print("Training hoàn tất!")

# Đánh giá trên tập validation
alexnet.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in validation_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = alexnet(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

validation_accuracy = 100 * correct / total
print(f"Validation Accuracy: {validation_accuracy:.2f}%")

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
feature_extractor = nn.Sequential(*list(alexnet.children())[:-1])
feature_extractor = feature_extractor.to(device)

# Hàm trích xuất đặc trưng từ tập dữ liệu
def extract_features(dataloader, model):
    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = model(images).squeeze()  # Bỏ chiều không cần thiết
            features = features.view(features.shape[0], -1)  # Flatten
            features_list.append(features.cpu().numpy())  # Đưa về CPU để tránh lỗi
            labels_list.append(labels.cpu().numpy())

    return np.vstack(features_list), np.hstack(labels_list)

# Trích xuất đặc trưng cho tập train, validation & test
X_train, y_train = extract_features(train_loader, feature_extractor)
X_validation, y_validation = extract_features(validation_loader, feature_extractor)
X_test, y_test = extract_features(test_loader, feature_extractor)

# Kiểm tra kích thước đầu ra
print("\nTrain SVM on extracted features")
print("X_train shape:", X_train.shape)  # Ví dụ: (400, 9216)
print("X_validation shape:", X_validation.shape)
print("X_test shape:", X_test.shape)    # Ví dụ: (200, 9216)

# Huấn luyện mô hình SVM trên feature đã trích xuất
# Tạo mô hình SVM với tiền xử lý StandardScaler
svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))

# Huấn luyện mô hình
svm_model.fit(X_train, y_train)

# Dự đoán trên tập validation
y_validation_pred = svm_model.predict(X_validation)

# Dự đoán trên tập kiểm tra
y_pred = svm_model.predict(X_test)

# Đánh giá SVM
# Chọn kernel cho SVM
kernel_choice = 'sigmoid'  # Chọn : 'linear', 'rbf', 'poly', 'sigmoid'

# Khởi tạo mô hình SVM với tiền xử lý StandardScaler
svm_model = make_pipeline(StandardScaler(), SVC(kernel=kernel_choice, probability=True))

# Huấn luyện mô hình
svm_model.fit(X_train, y_train)

# Dự đoán nhãn
y_pred = svm_model.predict(X_test)

# Tính các chỉ số đánh giá
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# In kết quả đánh giá
print("\nPrediction result")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-score: {f1 * 100:.2f}%")

# Confussion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(tabulate(cm, headers=["Predicted 0", "Predicted 1"], tablefmt="grid"))

# Classification Report
report = classification_report(y_test, y_pred, output_dict=True)

# Chỉ lấy các mục có dạng dictionary (bỏ qua "accuracy" vì nó là float)
report_table = [[label] + list(metrics.values()) for label, metrics in report.items() if isinstance(metrics, dict)]
headers = ["Class", "Precision", "Recall", "F1-score", "Support"]

# Tạo classification report dưới dạng dictionary
report = classification_report(y_test, y_pred, output_dict=True)

# Chuyển report thành bảng với giá trị hiển thị dưới dạng %
headers = ["Class", "Precision (%)", "Recall (%)", "F1-score (%)", "Support"]
report_table = []

for label, metrics in report.items():
    if isinstance(metrics, dict):  # Lọc ra các dòng có số liệu
        report_table.append([
            label,
            f"{metrics['precision'] * 100:.2f}",
            f"{metrics['recall'] * 100:.2f}",
            f"{metrics['f1-score'] * 100:.2f}",
            int(metrics['support'])  # Support giữ nguyên số nguyên
        ])

# Hiển thị bảng Classification Report
print("\nClassification Report:")
print(tabulate(report_table, headers=headers, tablefmt="grid"))

# Dự đoán xác suất cho AUC-ROC
if kernel_choice in ['linear', 'sigmoid']:
    y_pred_proba = svm_model.decision_function(X_test)
else:
    y_pred_proba = svm_model.predict_proba(X_test)[:, 1]

# Tính AUC-ROC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC Score: {auc_score:.4f}")

# Vẽ đường ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

