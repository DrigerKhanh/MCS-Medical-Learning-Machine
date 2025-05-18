import random

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_dir = "./dataset/Train"
val_dir = "./dataset/Validation"
test_dir = "./dataset/Test"

# Transform the training set with augmentation and resizing

# Augmentation
def training_transform():
    transform = transforms.Compose([
        transforms.Resize(256),                                                     # Resize cạnh ngắn nhất về 256 pixel
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),                                                      # Chuyển đổi thành Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Chuẩn hóa theo ImageNet
    ])
    return transform

def val_and_test_transform():
    # Cấu hình preprocessing cho ảnh
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Kích thước input của AlexNet
        transforms.ToTensor(),
    ])
    return transform

# Hàm bỏ chuẩn hóa và hiển thị
def im_convert(tensor):
    image = tensor.numpy().transpose((1, 2, 0))
    image = np.clip(image, 0, 1)
    return image

def verify_augmentation():
    train_transform = training_transform()
    # Load Dataset 1 lần (Không dùng 2 DataLoader riêng)
    train_dataset = datasets.ImageFolder(root=train_dir)

    # Dictionary để lưu chỉ số ảnh theo class
    class_to_indices = {i: [] for i in range(len(train_dataset.classes))}

    # Duyệt toàn bộ dataset để phân loại index theo class
    for idx, (_, label) in enumerate(train_dataset):
        class_to_indices[label].append(idx)

    # Chọn ngẫu nhiên 2 ảnh từ mỗi class
    selected_indices = []
    for class_id, indices in class_to_indices.items():
        selected = random.sample(indices, 2)  # lấy ngẫu nhiên 2 ảnh
        selected_indices.extend(selected)

    # selected_indices bây giờ chứa 8 ảnh (2 ảnh cho mỗi class)

    # Chỉ Load ảnh gốc (Không Augmentation)
    original_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Lấy cùng ảnh, một cái gốc, một cái Augmentation
    original_images = [original_transform(train_dataset[i][0]) for i in selected_indices]
    augmented_images = [train_transform(train_dataset[i][0]) for i in selected_indices]
    labels = [train_dataset[i][1] for i in selected_indices]

    fig, axes = plt.subplots(2, 10, figsize=(15, 6))
    # fig, axes = plt.subplots(2, 4, figsize=(8, 6))

    original_label_map = {
        0: "Ori dyk",
        1: "Ori koc",
        2: "Ori mep",
        3: "Ori pab",
        4: "Ori sfi",
    }

    augmented_label_map = {
        0: "Aug dyk",
        1: "Aug koc",
        2: "Aug mep",
        3: "Aug pab",
        4: "Aug sfi",
    }

    for i in range(10):
        # Ảnh gốc
        ax = axes[0, i]
        ax.imshow(im_convert(original_images[i]))
        ax.axis("off")
        ax.set_title(original_label_map.get(labels[i], "Unknown"))

        # Ảnh Augmentation
        ax = axes[1, i]
        img = augmented_images[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(augmented_label_map.get(labels[i], "Unknown"))

    plt.suptitle("Original and Augmentation Images", fontsize=14)
    plt.figtext(0.5, 0.01,
                "cervix_dyk: Điểm bất thường trong nhân tế bào   |   "
                "cervix_koc: Keratinizing/Others Cells - Tế bào sừng hóa/Các tế bào khác\n"
                "cervix_mep: Metaplastic Epithelium — tế bào biểu mô chuyển sản   |   "
                "cervix_pab: Papillomavirus Associated Biopsy — Sinh thiết liên quan đến Papillomavirus HPV\n"
                "cervix_sfi: Squamous Intraepithelial Lesion - Tổn thương biểu mô vảy",
                ha="center", fontsize=14, wrap=True
                )
    plt.show()


if __name__ == '__main__':
    verify_augmentation()
    # Code cho vui chu ko lam gi ca
    # Load tập test
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_and_test_transform())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Kiểm tra số lượng ảnh trong tập test
    print(f"Số lượng ảnh trong tập test: {len(test_dataset)}")

