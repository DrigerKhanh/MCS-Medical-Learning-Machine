import os
import cv2

# Thư mục chứa ảnh và annotation
image_dir = "dataset/images"
label_dir = "dataset/labels"

# Duyệt qua từng file .txt
for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue

    # Tên ảnh tương ứng
    image_file = os.path.splitext(label_file)[0] + ".jpg"  # hoặc .png
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, label_file)

    if not os.path.exists(image_path):
        print(f"Không tìm thấy ảnh: {image_path}")
        continue

    # Đọc ảnh để lấy kích thước
    img = cv2.imread(image_path)
    h_img, w_img = img.shape[:2]

    # Đọc file label
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_c, y_c, w, h = map(float, line.strip().split())

            # Chuyển từ tọa độ chuẩn hóa sang pixel
            x_center = x_c * w_img
            y_center = y_c * h_img
            width = w * w_img
            height = h * h_img

            x = x_center - width / 2
            y = y_center - height / 2

            print(f"Image: {image_file} | Class: {int(class_id)} | x: {x:.2f}, y: {y:.2f}, w: {width:.2f}, h: {height:.2f}")
