import os
from PIL import Image

train_dir = "./dataset/Training"
validation_dir = "./dataset/Validation"
test_dir = "./dataset/Test"

print("Tồn tại Train folder:", os.path.exists(train_dir))
print("Tồn tại Validation folder:", os.path.exists(validation_dir))
print("Tồn tại Test folder:", os.path.exists(test_dir))

# Hàm lấy kích thước ảnh
def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size  # (width, height)

# Kiểm tra số ảnh và kích thước ảnh trong mỗi thư mục con (cats, dogs) trong tập Train
if os.path.exists(train_dir):
    cats_dir = os.path.join(train_dir, "cats")
    dogs_dir = os.path.join(train_dir, "dogs")

    cat_images = os.listdir(cats_dir) if os.path.exists(cats_dir) else []
    dog_images = os.listdir(dogs_dir) if os.path.exists(dogs_dir) else []

    print('Số ảnh trong Train:')
    print(f" - Cats: {len(cat_images)}")
    print(f" - Dogs: {len(dog_images)}")

    # Lấy kích thước của ảnh đầu tiên trong mỗi class
    if cat_images:
        cat_sample = os.path.join(cats_dir, cat_images[0])
        print(f"Kích thước ảnh Cat mẫu trong Train: {get_image_size(cat_sample)}")

    if dog_images:
        dog_sample = os.path.join(dogs_dir, dog_images[0])
        print(f"Kích thước ảnh Dog mẫu trong Train: {get_image_size(dog_sample)}")

# Kiểm tra số ảnh và kích thước ảnh trong mỗi thư mục con (cats, dogs) trong tập Validation
if os.path.exists(validation_dir):
    cats_dir_validation = os.path.join(validation_dir, "cats")
    dogs_dir_validation = os.path.join(validation_dir, "dogs")

    cat_images_test = os.listdir(cats_dir_validation) if os.path.exists(cats_dir_validation) else []
    dog_images_test = os.listdir(dogs_dir_validation) if os.path.exists(dogs_dir_validation) else []

    print('Số ảnh trong Test:')
    print(f" - Cats: {len(cat_images_test)}")
    print(f" - Dogs: {len(dog_images_test)}")

    # Lấy kích thước của ảnh đầu tiên trong mỗi class
    if cat_images_test:
        cat_sample_test = os.path.join(cats_dir_validation, cat_images_test[0])
        print(f"Kích thước ảnh Cat mẫu trong Test: {get_image_size(cat_sample_test)}")

    if dog_images_test:
        dog_sample_test = os.path.join(dogs_dir_validation, dog_images_test[0])
        print(f"Kích thước ảnh Dog mẫu trong Test: {get_image_size(dog_sample_test)}")

# Kiểm tra số ảnh và kích thước ảnh trong mỗi thư mục con (cats, dogs) trong tập Test
if os.path.exists(test_dir):
    cats_dir_test = os.path.join(test_dir, "cats")
    dogs_dir_test = os.path.join(test_dir, "dogs")

    cat_images_test = os.listdir(cats_dir_test) if os.path.exists(cats_dir_test) else []
    dog_images_test = os.listdir(dogs_dir_test) if os.path.exists(dogs_dir_test) else []

    print('Số ảnh trong Test:')
    print(f" - Cats: {len(cat_images_test)}")
    print(f" - Dogs: {len(dog_images_test)}")

    # Lấy kích thước của ảnh đầu tiên trong mỗi class
    if cat_images_test:
        cat_sample_test = os.path.join(cats_dir_test, cat_images_test[0])
        print(f"Kích thước ảnh Cat mẫu trong Test: {get_image_size(cat_sample_test)}")

    if dog_images_test:
        dog_sample_test = os.path.join(dogs_dir_test, dog_images_test[0])
        print(f"Kích thước ảnh Dog mẫu trong Test: {get_image_size(dog_sample_test)}")