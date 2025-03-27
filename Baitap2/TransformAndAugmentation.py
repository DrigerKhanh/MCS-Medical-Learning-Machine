import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_dir = "./dataset/Training"
validation_dir = "./dataset/Validation"
test_dir = "./dataset/Test"

# Transform the training set with augmentation and resizing
# Chỉ Load ảnh gốc (Không Augmentation)
original_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Augmentation
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Dataset 1 lần (Không dùng 2 DataLoader riêng)
train_dataset = datasets.ImageFolder(root=train_dir)

# Lấy cùng ảnh, một cái gốc, một cái Augmentation
indices = torch.randperm(len(train_dataset))[:6]  # Chọn ngẫu nhiên 4 ảnh
original_images = [original_transform(train_dataset[i][0]) for i in indices]
augmented_images = [train_transform(train_dataset[i][0]) for i in indices]
labels = [train_dataset[i][1] for i in indices]

# Hàm bỏ chuẩn hóa và hiển thị
def im_convert(tensor):
    image = tensor.numpy().transpose((1, 2, 0))
    image = np.clip(image, 0, 1)
    return image

fig, axes = plt.subplots(2, 6, figsize=(12, 6))
# fig, axes = plt.subplots(2, 4, figsize=(8, 6))

for i in range(6):
    # Ảnh gốc
    ax = axes[0, i]
    ax.imshow(im_convert(original_images[i]))
    ax.axis("off")
    ax.set_title("Original Dog" if labels[i] == 1 else "Original Cat")

    # Ảnh Augmentation
    ax = axes[1, i]
    img = augmented_images[i].numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Augumented Dog" if labels[i] == 1 else "Augumented Cat")

plt.suptitle("before & after Augmentation", fontsize=14)
plt.show()


# 2. Transform cho tập Validation (Không Augmentation)
validation_transform = transforms.Compose([
    transforms.Resize(256),        # Resize cạnh ngắn nhất về 256 pixel
    transforms.CenterCrop(224),    # Cắt vùng trung tâm kích thước 224x224
    transforms.ToTensor(),         # Chuyển đổi thành Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa theo ImageNet
])

# Load tập validation
validation_dataset = datasets.ImageFolder(root=test_dir, transform=validation_transform)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

# Kiểm tra số lượng ảnh trong tập validation
print(f"Số lượng ảnh trong tập validation: {len(validation_dataset)}")


# 3. Transform cho tập test (Không Augmentation)
test_transform = transforms.Compose([
    transforms.Resize(256),        # Resize cạnh ngắn nhất về 256 pixel
    transforms.CenterCrop(224),    # Cắt vùng trung tâm kích thước 224x224
    transforms.ToTensor(),         # Chuyển đổi thành Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa theo ImageNet
])

# Load tập test
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Kiểm tra số lượng ảnh trong tập test
print(f"Số lượng ảnh trong tập test: {len(test_dataset)}")