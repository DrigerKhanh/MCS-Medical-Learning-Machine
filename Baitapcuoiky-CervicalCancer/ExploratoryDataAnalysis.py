import os
from PIL import Image

train_dir = "./dataset/Train"
test_dir = "./dataset/Test"

print("Tồn tại Train folder:", os.path.exists(train_dir))
print("Tồn tại Test folder:", os.path.exists(test_dir))


# Hàm lấy kích thước ảnh
def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size  # (width, height)


# Kiểm tra số ảnh và kích thước ảnh trong mỗi thư mục con trong tập Train
if os.path.exists(train_dir):
    high_squamous_intra_epithelial_lesion_dir = os.path.join(train_dir, "High-squamous-intra-epithelial-lesion")
    low_squamous_intra_epithelial_lesion_dir = os.path.join(train_dir, "Low-squamous-intra-epithelial-lesion")
    negative_for_Intraepithelial_malignancy_dir = os.path.join(train_dir, "Negative-for-Intraepithelial-malignancy")
    squamous_cell_carcinoma_dir = os.path.join(train_dir, "Squamous-cell-carcinoma")

    high_squamous_intra_epithelial_lesion_images = os.listdir(
        high_squamous_intra_epithelial_lesion_dir) if os.path.exists(high_squamous_intra_epithelial_lesion_dir) else []
    low_squamous_intra_epithelial_lesion_images = os.listdir(
        low_squamous_intra_epithelial_lesion_dir) if os.path.exists(low_squamous_intra_epithelial_lesion_dir) else []
    negative_for_Intraepithelial_malignancy_images = os.listdir(
        negative_for_Intraepithelial_malignancy_dir) if os.path.exists(
        negative_for_Intraepithelial_malignancy_dir) else []
    squamous_cell_carcinoma_images = os.listdir(squamous_cell_carcinoma_dir) if os.path.exists(
        squamous_cell_carcinoma_dir) else []

    print('Số ảnh trong Train:')
    print(f" - high_squamous_intra_epithelial_lesion: {len(high_squamous_intra_epithelial_lesion_images)}")
    print(f" - low_squamous_intra_epithelial_lesion: {len(low_squamous_intra_epithelial_lesion_images)}")
    print(f" - negative_for_Intraepithelial_malignancy: {len(negative_for_Intraepithelial_malignancy_images)}")
    print(f" - squamous_cell_carcinoma: {len(squamous_cell_carcinoma_images)}")

    # Lấy kích thước của ảnh đầu tiên trong mỗi class
    if high_squamous_intra_epithelial_lesion_images:
        high_squamous_intra_epithelial_lesion_sample = os.path.join(high_squamous_intra_epithelial_lesion_dir,
                                                                    high_squamous_intra_epithelial_lesion_images[0])
        print(
            f"Kích thước ảnh High_squamous_intra_epithelial_lesion mẫu trong Train: {get_image_size(high_squamous_intra_epithelial_lesion_sample)}")

    if low_squamous_intra_epithelial_lesion_images:
        low_squamous_intra_epithelial_lesion_sample = os.path.join(low_squamous_intra_epithelial_lesion_dir,
                                                                   low_squamous_intra_epithelial_lesion_images[0])
        print(f"Kích thước ảnh Low_squamous_intra_epithelial_lesion mẫu trong Train: {get_image_size(low_squamous_intra_epithelial_lesion_sample)}")

    if negative_for_Intraepithelial_malignancy_images:
        negative_for_Intraepithelial_malignancy_sample = os.path.join(negative_for_Intraepithelial_malignancy_dir,
                                                                      negative_for_Intraepithelial_malignancy_images[0])
        print(f"Kích thước ảnh Negative_for_Intraepithelial_malignancy mẫu trong Train: {get_image_size(negative_for_Intraepithelial_malignancy_sample)}")

    if squamous_cell_carcinoma_images:
        squamous_cell_carcinoma_sample = os.path.join(squamous_cell_carcinoma_dir, squamous_cell_carcinoma_images[0])
        print(f"Kích thước ảnh Squamous_cell_carcinoma mẫu trong Train: {get_image_size(squamous_cell_carcinoma_sample)}")
