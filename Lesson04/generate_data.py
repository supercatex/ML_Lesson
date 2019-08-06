import os
import cv2
import uuid
import numpy as np


_DIR_ROOT = "elements"
_DIR_IMAGES = os.path.join(_DIR_ROOT, "images")
_DIR_BACKGROUND = os.path.join(_DIR_ROOT, "background_images")

_DIR_OUTPUT = "dataset"
_DIR_OUTPUT_IMG = os.path.join(_DIR_OUTPUT, "img")
_PATH_OUTPUT_CSV = os.path.join(_DIR_OUTPUT, "data.csv")

_NUM_OF_SAMPLES = 100000
_IMAGE_SIZE = (100, 100)


def add_noise(img):
    h, w, c = img.shape
    if c != 4:
        raise Exception("Only PNG format supported!")

    dst = img.copy()

    tmp = dst[:, :, 0:3]
    hsv = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)
    p1 = np.random.randint(255 - 35, 255 + 35) / 255
    hsv[:, :, 1] = np.array(hsv[:, :, 1] * p1, dtype=np.uint8)
    p2 = np.random.randint(255 - 100, 255) / 255
    hsv[:, :, 2] = np.array(hsv[:, :, 2] * p2, dtype=np.uint8)
    tmp = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    dst[:, :, 0:3] = tmp

    return dst


def generate_image(img, bg):
    h, w, c = img.shape
    if c != 4:
        raise Exception("Only PNG format supported!")

    pts1 = np.float32([[0, 0], [h, 0], [0, w], [h, w]])
    pts2 = np.float32([
        [np.random.randint(0, h / 4), np.random.randint(0, w / 4)],
        [h - np.random.randint(0, h / 4), np.random.randint(0, w / 4)],
        [np.random.randint(0, h / 4), w - np.random.randint(0, w / 4)],
        [h - np.random.randint(0, h / 4), w - np.random.randint(0, w / 4)]
    ])

    m = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, m, (w, h))
    tmp = dst.copy()

    rh = np.random.randint(0, bg.shape[0] - h)
    rw = np.random.randint(0, bg.shape[1] - w)
    bg = bg[rh:rh+h, rw:rw+w, :]

    r = bg
    for i in range(c - 1):
        tmp[:, :, i] = cv2.bitwise_and(r[:, :, i], 255 - dst[:, :, c - 1])
        dst[:, :, i] = cv2.bitwise_and(dst[:, :, i], dst[:, :, c - 1])
    dst += tmp

    return dst


def generate_data():
    global _DIR_ROOT, _DIR_IMAGES, _DIR_BACKGROUND, \
           _DIR_OUTPUT, _PATH_OUTPUT_CSV, _DIR_OUTPUT_IMG, \
           _NUM_OF_SAMPLES, _IMAGE_SIZE

    if not os.path.exists(_DIR_ROOT):
        raise("ROOT directory not found->", _DIR_ROOT)

    if not os.path.exists(_DIR_IMAGES):
        raise("Image directory not found->", _DIR_IMAGES)

    if not os.path.exists(_DIR_BACKGROUND):
        raise("Background image directory not found->", _DIR_BACKGROUND)

    f_in_dir_image = os.listdir(_DIR_IMAGES)
    if len(f_in_dir_image) == 0:
        raise("No image in the image directory->", _DIR_IMAGES)

    f_in_dir_background = os.listdir(_DIR_BACKGROUND)
    if len(f_in_dir_background) == 0:
        raise("No image in the background image directory->", _DIR_BACKGROUND)

    if not os.path.exists(_DIR_OUTPUT):
        os.makedirs(_DIR_OUTPUT)

    if not os.path.exists(_DIR_OUTPUT_IMG):
        os.makedirs(_DIR_OUTPUT_IMG)

    f = open(_PATH_OUTPUT_CSV, "w")
    f.write("img,width,height,x1,y1,x2,y2,class\n")

    for i in range(_NUM_OF_SAMPLES):
        name1 = np.random.choice(f_in_dir_image, 1)[0]
        file1 = os.path.join(_DIR_IMAGES, name1)
        image1 = cv2.imread(file1, cv2.IMREAD_UNCHANGED)

        name2 = np.random.choice(f_in_dir_background, 1)[0]
        file2 = os.path.join(_DIR_BACKGROUND, name2)
        image2 = cv2.imread(file2, cv2.IMREAD_UNCHANGED)

        try:
            image1 = add_noise(image1)
            image1 = generate_image(image1, image2)
            image1 = cv2.resize(image1, _IMAGE_SIZE)

            name = str(uuid.uuid4()) + ".jpg"
            path = os.path.join(_DIR_OUTPUT_IMG, name)
            cv2.imwrite(path, image1)
        except Exception as e:
            i -= 1
            del image1, image2
            continue

        del image1, image2

        label = int(name1.split(".")[0][1:])
        f.write("%s,%d,%d,%d,%d,%d,%d,%d\n" % (
            name,
            _IMAGE_SIZE[1],
            _IMAGE_SIZE[0],
            0,
            0,
            _IMAGE_SIZE[1],
            _IMAGE_SIZE[0],
            label
        ))

    f.close()
    print("GENERATE FINISHED!")


if __name__ == "__main__":
    generate_data()
