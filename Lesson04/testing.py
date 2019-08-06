import numpy as np
from import_data import import_data


def testing(model, dir_root="MacauAI_TrainingSet_1", dir_img="img", csv="training.csv"):
    samples, labels = import_data(
        dir_root=dir_root,
        dir_img="img",
        csv="training.csv",
        image_size=(100, 100),
        limit=-1
    )
    samples = samples.astype(np.float32)
    samples /= 255
    labels -= 1
    print(samples.shape, labels.shape)

    n = 0
    c = 0
    for i, sample in enumerate(samples):
        y = model.predict(np.array([sample]))
        idx = np.argmax(y[0])
        if idx == labels[i]:
            c += 1
        n += 1

    return n, c
