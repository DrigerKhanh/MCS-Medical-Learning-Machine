import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 1. Load bộ dữ liệu breast cancer
file_path = 'dataset/breast-cancer.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Chuyển đổi cột 'diagnosis' từ M/B thành 1/0
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Xác định X và y
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# 2. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)  # 60% train
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)  # 20% validation, 20% test

# 3. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 4. Khởi tạo và huấn luyện mô hình SVM với kernel RBF
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# 5. Dự đoán trên tập xác thực (validation)
y_val_pred = svm_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Dự đoán trên tập Test
y_test_pred = svm_model.predict(X_test)

# 6. Tính các chỉ số đánh giá
test_accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

# Tính toán confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

# In kết quả
print(f"Accuracy: {test_accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

print("Confusion Matrix:")
print(conf_matrix)


# 7. Vẽ Confusion Matrix
plt.figure(figsize=(7, 5))
conf_matrix = confusion_matrix(y_test, y_test_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()