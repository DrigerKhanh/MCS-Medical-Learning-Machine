import numpy as np
import matplotlib.pyplot as plt

# Dữ liệu
ages = np.array([25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                 45, 46, 47, 48, 49, 50, 51, 52, 53, 54])
blood_pressures = np.array([118, 119, 121, 122, 123, 124, 125, 126, 127, 128,
                            129, 130, 130, 131, 132, 133, 134, 135, 136, 137,
                            138, 139, 139, 140, 141, 142, 143, 144, 145, 146])

# Normalize data
ages_normalized = (ages - ages.mean()) / ages.std()
n = len(ages)

# Hàm dự đoán
def predict(x, b, w):
    return b + w * x

# Hàm tính chi phí
def compute_cost(b, w, x, y):
    total_cost = np.sum((predict(x, b, w) - y) ** 2) / (2 * n)
    return total_cost

# Thuật toán Gradient Descent
def gradient_descent(x, y, b_init, w_init, learning_rate, iterations):
    b = b_init
    w = w_init
    cost_history = []

    for i in range(iterations):
        # Tính d/db và d/dw
        y_pred = predict(x, b, w)
        b_gradient = np.sum(y_pred - y) / n
        w_gradient = np.sum((y_pred - y) * x) / n

        # Cập nhật tham số
        b = b - learning_rate * b_gradient
        w = w - learning_rate * w_gradient

        # Lưu lịch sử chi phí
        current_cost = compute_cost(b, w, x, y)
        cost_history.append(current_cost)

    return b, w, cost_history

# Khởi tạo tham số
b_init = 0
w_init = 0
learning_rate = 0.01
iterations = 1000

# Thực thi Gradient Descent
b, w, cost_history = gradient_descent(ages_normalized, blood_pressures, b_init, w_init, learning_rate, iterations)

print(f"Optimized parameters: b = {b}, w = {w}")

# Trực quan hóa kết quả
plt.scatter(ages, blood_pressures, color='red', label='Dữ liệu thực tế')
plt.plot(ages, predict(ages_normalized, b, w), color='blue', label='Đường hồi quy')
plt.xlabel('Tuổi (năm)')
plt.ylabel('Huyết Áp (mmHg)')
plt.legend()
plt.show()

# Plot cost history
plt.plot(range(iterations), cost_history, color='blue')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost function over iterations')
plt.show()