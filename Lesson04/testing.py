import cv2
import numpy as np
from import_data import import_data
from import_data import remapping


def testing(model, samples, labels, debug=False):
    samples = samples.copy()
    labels = labels.copy()

    samples = samples.astype(np.float32)
    samples /= 255
    labels = remapping(labels)
    print(samples.shape, labels.shape)

    n = 0
    c = 0
    for i, sample in enumerate(samples):
        y = model.predict(np.array([sample]))
        idx = np.argmax(y[0])
        if idx == labels[i]:
            c += 1
        else:
            if debug:
                print(idx + 1)
                cv2.imshow("image", sample)
                cv2.waitKey(0)
        n += 1

    return n, c
