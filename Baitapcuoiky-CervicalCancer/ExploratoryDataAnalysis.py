import os
from PIL import Image

train_dir = "./dataset/Train"
val_dir = "./dataset/Validation"
test_dir = "./dataset/Test"

print("Tồn tại Train folder:", os.path.exists(train_dir))
print("Tồn tại Validation folder:", os.path.exists(val_dir))
print("Tồn tại Test folder:", os.path.exists(test_dir))


# Hàm lấy kích thước ảnh
def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size  # (width, height)


# Kiểm tra số ảnh và kích thước ảnh trong mỗi thư mục con trong tập Train
if os.path.exists(train_dir):
    cervix_dyk_dir = os.path.join(train_dir, "cervix_dyk")
    cervix_koc_dir = os.path.join(train_dir, "cervix_koc")
    cervix_mep_dir = os.path.join(train_dir, "cervix_mep")
    cervix_pab_dir = os.path.join(train_dir, "cervix_pab")
    cervix_sfi_dir = os.path.join(train_dir, "cervix_sfi")

    cervix_dyk_images = os.listdir(cervix_dyk_dir) if os.path.exists(cervix_dyk_dir) else []
    cervix_koc_images = os.listdir(cervix_koc_dir) if os.path.exists(cervix_koc_dir) else []
    cervix_mep_images = os.listdir(cervix_mep_dir) if os.path.exists(cervix_mep_dir) else []
    cervix_pab_images = os.listdir(cervix_pab_dir) if os.path.exists(cervix_pab_dir) else []
    cervix_sfi_images = os.listdir(cervix_sfi_dir) if os.path.exists(cervix_sfi_dir) else []

    print('Số ảnh trong Train:')
    print(f" - cervix_dyk - Điểm bất thường trong nhân tế bào: {len(cervix_dyk_images)}")
    print(f" - cervix_koc - Keratinizing/Others Cells - Tế bào sừng hóa/Các tế bào khác: {len(cervix_koc_images)}")
    print(f" - cervix_mep - Metaplastic Epithelium — tế bào biểu mô chuyển sản: {len(cervix_mep_images)}")
    print(f" - cervix_pab - Papillomavirus Associated Biopsy — Sinh thiết liên quan đến Papillomavirus HPV: {len(cervix_pab_images)}")
    print(f" - cervix_sfi - Squamous Intraepithelial Lesion - Tổn thương biểu mô vảy: {len(cervix_sfi_images)}")

    # Lấy kích thước của ảnh đầu tiên trong mỗi class
    if cervix_dyk_images:
        cervix_dyk_sample = os.path.join(cervix_dyk_dir,cervix_dyk_images[0])
        print(f"Kích thước ảnh cervix_dyk mẫu trong Train: {get_image_size(cervix_dyk_sample)}")

    if cervix_koc_images:
        cervix_koc_sample = os.path.join(cervix_koc_dir,cervix_koc_images[0])
        print(f"Kích thước ảnh cervix_koc mẫu trong Train: {get_image_size(cervix_koc_sample)}")

    if cervix_mep_images:
        cervix_mep_sample = os.path.join(cervix_mep_dir,cervix_mep_images[0])
        print(f"Kích thước ảnh cervix_mep mẫu trong Train: {get_image_size(cervix_mep_sample)}")

    if cervix_pab_images:
        cervix_pab_sample = os.path.join(cervix_pab_dir, cervix_pab_images[0])
        print(f"Kích thước ảnh Squamous_cell_carcinoma mẫu trong Train: {get_image_size(cervix_pab_sample)}")

    if cervix_sfi_images:
        cervix_sfi_sample = os.path.join(cervix_sfi_dir, cervix_sfi_images[0])
        print(f"Kích thước ảnh cervix_sfi mẫu trong Train: {get_image_size(cervix_sfi_sample)}")
