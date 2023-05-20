import os
import shutil

IMAGES_FOLDER_PATH = 'images'
TRAIN_DIR_PATH = 'dataset/train'
TEST_DIR_PATH = 'dataset/test'

def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)    

make_directory(TRAIN_DIR_PATH)
make_directory(TEST_DIR_PATH)

img_dirs = os.listdir(IMAGES_FOLDER_PATH)
print("Directory Names In Images FOLDER", img_dirs)

for img_dir_name in img_dirs:
    make_directory(os.path.join(TRAIN_DIR_PATH, img_dir_name))
    make_directory(os.path.join(TEST_DIR_PATH, img_dir_name))
    
    img_dir_path = os.path.join(IMAGES_FOLDER_PATH, img_dir_name)
    print("Image Directory Path-: ",img_dir_path)
    
    img_name_list = os.listdir(img_dir_path)
    total_images = len(img_name_list)
    print("Total Images-:", total_images)
    
    train_images_length = int(total_images * 0.8) # First 80% images for training
    print("Train Images Length-:", train_images_length)
    
    test_images_length = total_images - train_images_length # 20% images for testing
    print("Test Images Length-:", test_images_length)
    
    train_img_names = img_name_list[:train_images_length]
    # print("Train Image Names-:", train_img_names)

    test_img_names = img_name_list[train_images_length:]
    # print("Test Image Names-:", test_img_names)
    
    for img_name in train_img_names:
        img_path = os.path.join(img_dir_path, img_name)
        print("Copying Image Path To Train Directory-:", img_path)
        shutil.copy(img_path, os.path.join(TRAIN_DIR_PATH, img_dir_name))
        
    for img_name in test_img_names:
        img_path = os.path.join(img_dir_path, img_name)
        print("Copying Image Path To Test Directory-:", img_path)
        shutil.copy(img_path, os.path.join(TEST_DIR_PATH, img_dir_name))
        