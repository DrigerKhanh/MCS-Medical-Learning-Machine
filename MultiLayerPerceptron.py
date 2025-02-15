# Import các thư viện cần thiết
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Tải bộ dữ liệu Breast Cancer từ sklearn
from sklearn.datasets import load_breast_cancer

# Tải bộ dữ liệu
data = load_breast_cancer()
X = data.data
y = data.target

# Chia bộ dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình Multi-Layer Perceptron (MLP)
model = MLPClassifier(hidden_layer_sizes=(30, 30), max_iter=1000, random_state=42)  # Bạn có thể thay đổi tham số như số lượng lớp ẩn
model.fit(X_train, y_train)

# Dự đoán nhãn từ bộ dữ liệu kiểm tra
y_pred = model.predict(X_test)

# Tính toán các chỉ số đánh giá
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Tính toán confusion matrix
cm = confusion_matrix(y_test, y_pred)

# In kết quả
print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
