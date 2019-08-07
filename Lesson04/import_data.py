import os
import cv2
import sys
import numpy as np


def remapping(labels):
    for i in range(len(labels)):
        # idx = labels[i]
        # if 1 <= idx <= 39 or idx in [77]:
        #     labels[i] = 0
        # elif 40 <= idx <= 82 and idx not in [54, 77]:
        #     labels[i] = 1
        # elif 83 <= idx <= 99 or idx in [54, 77, 119]:
        #     labels[i] = 2
        # elif 100 <= idx <= 120 and idx not in [119]:
        #     labels[i] = 3
        # else:
        #     labels[i] = 2
        labels[i] = labels[i] - 1
    return labels


def import_data(
        dir_root="../../dataset",
        dir_img="img",
        csv="data.csv",
        image_size=(100, 100),
        limit=10
):
    dir_output_img = os.path.join(dir_root, dir_img)
    path_output_csv = os.path.join(dir_root, csv)

    if not os.path.exists(path_output_csv):
        raise("CSV file not found->", path_output_csv)

    print("Reading csv file...")
    f = open(path_output_csv, "r")
    lines = f.readlines()
    f.close()

    new_lines = lines
    if limit > 0:
        new_lines = np.random.choice(lines, limit)

    samples = []
    labels = []
    for i, line in enumerate(new_lines):
        sys.stdout.write("\rReading image file...%d/%d" % (i + 1, len(new_lines)))
        sys.stdout.flush()

        temp = line.split(",")
        name = temp[0].strip()
        if name == "img" or len(temp) == 0:
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
        image = image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
        if image is None or image.shape[0] == 0 or image.shape[1] == 0:
            continue
        image = cv2.resize(image, image_size)
        if len(image.shape) == 2:
            image = np.reshape(image, (image.shape[0], image.shape[1], 1))
        elif len(image.shape) == 3:
            if image.shape[2] == 4:
                image = image[:, :, :3]

        samples.append(image)
        labels.append(label)
    print()

    data = list(zip(samples, labels))
    np.random.shuffle(data)
    samples, labels = zip(*data)

    return np.array(samples), np.array(labels)


if __name__ == "__main__":
    _X, _y = import_data(
        dir_root="MacauAI_TrainingSet_2",
        dir_img="img",
        csv="training.csv",
        image_size=(30, 30),
        limit=-1
    )
    print(_X.shape, _y.shape)
