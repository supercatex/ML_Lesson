import os
import numpy as np
from keras.utils import np_utils
from keras.models import load_model
import matplotlib.pyplot as plt
from import_data import import_data
from import_data import remapping
from create_model import create_model
from testing import testing
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

_model_name = "models/m3_6.h5"

_need_to_train = False
_num_of_classes = 120
_image_size = 30

if os.path.exists(_model_name):
    _model = load_model(_model_name)
else:
    _need_to_train = True
    _model = create_model((_image_size, _image_size, 3), _num_of_classes)
_model.summary()

_test_X, _test_y = import_data(
    dir_root="MacauAI_TrainingSet_2",
    dir_img="img",
    csv="training.csv",
    image_size=(_image_size, _image_size),
    limit=-1
)

if _need_to_train:
    _lines = None
    _X, _y = import_data(
        dir_root="../../dataset",
        dir_img="img",
        csv="data.csv",
        image_size=(_image_size, _image_size),
        limit=5000
    )

    _X2, _y2 = import_data(
        dir_root="MacauAI_TrainingSet_3",
        dir_img="img",
        csv="training.csv",
        image_size=(_image_size, _image_size),
        limit=-1
    )

    _X = np.concatenate((_X, _X2))
    _y = np.concatenate((_y, _y2))
    data = list(zip(_X, _y))

    for i in range(10000):
        print("ROUND:", i + 1)

        np.random.shuffle(data)
        _samples, _labels = zip(*data)
        _samples = np.array(_samples)
        _labels = np.array(_labels)

        _samples = _samples.astype(np.float32)
        _samples /= 255
        _labels = remapping(_labels)
        _labels = np_utils.to_categorical(_labels, _num_of_classes)

        _history = _model.fit(
            x=_samples,
            y=_labels,
            validation_split=0.1,
            epochs=10,
            batch_size=128,
            verbose=2
        )

        if i % 1 == 0:
            n, c = testing(_model, _test_X, _test_y)
            p = round(c / n, 4)
            print(n, c, p)
            _model.save(_model_name[:-3] + "_" + str(i) + "_" + str(round(p * 10000)) + ".h5")

        # plt.plot(_history.history['acc'])
        # plt.plot(_history.history['val_acc'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()

    _model.save(_model_name)

n, c = testing(_model, _test_X, _test_y, False)
print(n, c, c / n)

n, c = testing(_model, _test_X, _test_y, True)
print(n, c, c / n)
