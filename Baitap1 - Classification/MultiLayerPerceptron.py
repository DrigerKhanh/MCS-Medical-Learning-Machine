# Import các thư viện cần thiết
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Tải bộ dữ liệu Breast Cancer từ sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Tải bộ dữ liệu
# data = load_breast_cancer()
file_path = 'dataset/breast-cancer.csv'  # Replace with your file path
data = pd.read_csv(file_path)
data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
X = data.drop(columns=['diagnosis'])  # Features
y = data['diagnosis']  # Target variable

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)  # 60% train
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)  # 20% validation, 20% test

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Hyperparameter tuning for hidden_layer_sizes
param_grid = {'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)]}
model = MLPClassifier(hidden_layer_sizes=(64, 32, 32),max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Train best model
# best_mlp = grid_search.best_estimator_
# best_mlp.fit(X_train, y_train)

# Evaluate the model on the validation set
# y_pred = best_mlp.predict(X_test)
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Evaluate the model on the test set
y_test_pred = model.predict(X_test)


# Compute evaluation metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

# Print metrics
print("\n--- Evaluation on Test Set ---")
# print(f'Best hidden_layer_sizes: {grid_search.best_params_}')
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - MLP')
plt.show()

tn, fp, fn, tp = conf_matrix.ravel()
print(f'True Positives (TP): {tp}')
print(f'True Negatives (TN): {tn}')
print(f'False Positives (FP): {fp}')
print(f'False Negatives (FN): {fn}')
