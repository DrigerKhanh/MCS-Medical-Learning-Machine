import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

# Step 1: Load the dataset
file_path = 'dataset/breast-cancer.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Step 2: Data preprocessing
# Assuming the dataset has features (X1, X2, ..., Xn) and a target column 'target'
# 'target' = 1 for heart disease, 0 for no heart disease
data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
X = data.drop(columns=['diagnosis'])  # Features
y = data['diagnosis']  # Target variable

# Check for missing values
if data.isnull().sum().sum() > 0:
    print("Dataset contains missing values. Handling them by imputation.")
    X.fillna(X.mean(), inplace=True)

# Step 3: Splitting the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)  # 60% train
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)  # 20% validation, 20% test

# Step 4: Training the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model on the validation set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Step 6: Evaluate the model on the test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
conf_matrix = confusion_matrix(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

# Vẽ biểu đồ các evaluation metrics
metrics = {
    'Accuracy': test_accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}


print("\n--- Evaluation on Test Set ---")
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# # Tạo biểu đồ thanh (bar plot)
# plt.figure(figsize=(8, 6))
# sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="viridis")

# Thêm tiêu đề và nhãn cho đồ thị
# plt.title('Evaluation Metrics of MLP Classifier', fontsize=14)
# plt.xlabel('Metrics', fontsize=12)
# plt.ylabel('Scores', fontsize=12)
# plt.ylim(0, 1)  # Đảm bảo rằng các giá trị trên trục y nằm trong khoảng từ 0 đến 1

# 7. Vẽ Confusion Matrix
plt.figure(figsize=(7, 5))
conf_matrix = confusion_matrix(y_test, y_test_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
# Hiển thị đồ thị
# plt.show()