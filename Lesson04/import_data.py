import os
import cv2
import numpy as np


def import_data(
        dir_root="dataset",
        dir_img="img",
        csv="data.csv",
        image_size=(100, 100),
        limit=-1
):
    dir_output_img = os.path.join(dir_root, dir_img)
    path_output_csv = os.path.join(dir_root, csv)

    if not os.path.exists(path_output_csv):
        raise("CSV file not found->", path_output_csv)

    print("Reading csv file...")
    f = open(path_output_csv, "r")
    lines = f.readlines()
    f.close()

    if limit > 0:
        lines = np.random.choice(lines, limit)

    print("Reading image file...", len(lines))
    samples = []
    labels = []
    for line in lines:
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
        image = image[y1:y2, x1:x2]
        image = cv2.resize(image, image_size)
        if len(image.shape) == 2:
            image = np.reshape(image, (image.shape[0], image.shape[1], 1))
        elif len(image.shape) == 4:
            image = image[0:3]

        samples.append(image)
        labels.append(label)

    data = list(zip(samples, labels))
    np.random.shuffle(data)
    samples, labels = zip(*data)

    return np.array(samples), np.array(labels)


if __name__ == "__main__":
    _X, _y = import_data()
    print(_X.shape, _y.shape)
