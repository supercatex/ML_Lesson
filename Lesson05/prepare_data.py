import os
import shutil


original_dataset_dir = r"../../datasets/dogs-vs-cats/train"

base_dir = r"../../datasets/dogs-vs-cats/cats_and_dogs_small"
if not os.path.isdir(base_dir): os.mkdir(base_dir)

train_dir = os.path.join(base_dir, "train")
if not os.path.isdir(train_dir): os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, "validation")
if not os.path.isdir(validation_dir): os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, "test")
if not os.path.isdir(test_dir): os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, "cats")
if not os.path.isdir(train_cats_dir): os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, "dogs")
if not os.path.isdir(train_dogs_dir): os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, "cats")
if not os.path.isdir(validation_cats_dir): os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, "dogs")
if not os.path.isdir(validation_dogs_dir): os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, "cats")
if not os.path.isdir(test_cats_dir): os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, "dogs")
if not os.path.isdir(test_dogs_dir): os.mkdir(test_dogs_dir)

if __name__ == "__main__":
    fnames = ["cat.{}.jpg".format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ["cat.{}.jpg".format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ["cat.{}.jpg".format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ["dog.{}.jpg".format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ["dog.{}.jpg".format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ["dog.{}.jpg".format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

    print("訓練用的貓圖片張數：", len(os.listdir(train_cats_dir)))
    print("訓練用的貓圖片張數：", len(os.listdir(train_dogs_dir)))
    print("驗證用的貓圖片張數：", len(os.listdir(validation_cats_dir)))
    print("驗證用的狗圖片張數：", len(os.listdir(validation_dogs_dir)))
    print("測試用的貓圖片張數：", len(os.listdir(test_cats_dir)))
    print("測試用的狗圖片張數：", len(os.listdir(test_dogs_dir)))
