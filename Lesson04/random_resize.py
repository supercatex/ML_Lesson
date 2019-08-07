import os
import sys
import cv2
import numpy as np


def generate_csv(
        dir_root="../../dataset",
        dir_img="img",
        csv="data.csv",
        new_csv="new_data.csv"
):
    dir_output_img = os.path.join(dir_root, dir_img)
    path_output_csv = os.path.join(dir_root, csv)
    path_output_new_csv = os.path.join(dir_root, new_csv)

    if not os.path.exists(dir_output_img):
        raise ("Image file not found->", dir_output_img)

    print("Reading csv file...")
    f = open(path_output_csv, "r")
    lines = f.readlines()
    f.close()

    print("Create csv file...")
    f = open(path_output_new_csv, "w")
    f.write("img,width,height,x1,y1,x2,y2,class\n")

    for i, line in enumerate(lines):
        sys.stdout.write("\rReading image file...%d/%d" % (i + 1, len(lines)))
        sys.stdout.flush()

        temp = line.split(",")
        name = temp[0].strip()
        if name == "img":
            continue
        w = int(temp[1].strip())
        h = int(temp[2].strip())
        x1 = int(temp[3].strip())
        y1 = int(temp[4].strip())
        x2 = int(temp[5].strip())
        y2 = int(temp[6].strip())
        label = int(temp[7])

        path = os.path.join(dir_output_img, name)
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(path, image)

        f.write("%s,%d,%d,%d,%d,%d,%d,%d\n" % (
            name,
            image.shape[1],
            image.shape[0],
            0,
            0,
            image.shape[1],
            image.shape[0],
            label
        ))
        del image
    f.close()


def random_resize(
        dir_root="../../dataset",
        dir_img="img",
        csv="data.csv",
        new_csv="new_data.csv"
):
    dir_output_img = os.path.join(dir_root, dir_img)
    path_output_csv = os.path.join(dir_root, csv)
    path_output_new_csv = os.path.join(dir_root, new_csv)

    if not os.path.exists(path_output_csv):
        raise("CSV file not found->", path_output_csv)

    print("Reading csv file...")
    f = open(path_output_csv, "r")
    lines = f.readlines()
    f.close()

    print("Create csv file...")
    f = open(path_output_new_csv, "w")
    f.write("img,width,height,x1,y1,x2,y2,class\n")

    for i, line in enumerate(lines):
        sys.stdout.write("\rReading image file...%d/%d" % (i + 1, len(lines)))
        sys.stdout.flush()

        temp = line.split(",")
        name = temp[0].strip()
        if name == "img":
            continue
        w = int(temp[1].strip())
        h = int(temp[2].strip())
        x1 = int(temp[3].strip())
        y1 = int(temp[4].strip())
        x2 = int(temp[5].strip())
        y2 = int(temp[6].strip())
        label = int(temp[7])

        rs = np.random.randint(20, 100)

        path = os.path.join(dir_output_img, name)
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (rs, rs))
        cv2.imwrite(path, image)

        f.write("%s,%d,%d,%d,%d,%d,%d,%d\n" % (
            name,
            image.shape[1],
            image.shape[0],
            0,
            0,
            image.shape[1],
            image.shape[0],
            label
        ))

        del image
    f.close()


if __name__ == "__main__":
    # random_resize()
    generate_csv()
