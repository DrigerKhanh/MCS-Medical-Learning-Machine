import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 1. Load bộ dữ liệu breast cancer
file_path = 'dataset/breast-cancer.csv'  # Replace with your file path
data = pd.read_csv(file_path)
data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})

X = data.drop(columns=['diagnosis'])  # Features
y = data['diagnosis']  # Target variable

# 2. Chia dữ liệu thành tập huấn luyện và kiểm thử
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)  # 60% train
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp) # 20% validation, 20% test

# 3. Khởi tạo và huấn luyện mô hình Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Đánh gia mô hình trên tập xác thực (validation)
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# 4. Dự đoán trên tập kiểm thử
y_test_pred = model.predict(X_test)

# 5. Tính các chỉ số đánh giá
test_accuracy = accuracy_score(y_test, y_test_pred)
conf_matrix = confusion_matrix(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

# 6. Hiển thị kết quả
print(f'Accuracy: {test_accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Benign', 'Malignant']))
print(f"\nConfusion Matrix:" )
print(confusion_matrix(y_test, y_test_pred))

# 7. Vẽ Confusion Matrix
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
