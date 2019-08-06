import os
import numpy as np
from keras.utils import np_utils
from keras.models import load_model
import matplotlib.pyplot as plt
from import_data import import_data
from create_model import create_model
from testing import testing


_model_name = "models/m_1_1000_10.h5"

if os.path.exists(_model_name):
    _model = load_model(_model_name)
else:
    _model = create_model((100, 100, 3), 120)
    _model.summary()

    for i in range(1):
        _samples, _labels = import_data(
            dir_root="dataset",
            dir_img="img",
            csv="data.csv",
            image_size=(100, 100),
            limit=1000
        )

        _samples = _samples.astype(np.float32)
        _samples /= 255
        _labels -= 1
        _labels = np_utils.to_categorical(_labels, 120)

        _history = _model.fit(
            x=_samples,
            y=_labels,
            validation_split=0.1,
            epochs=10,
            batch_size=128,
            verbose=1
        )

        _model.save(_model_name)

        # plt.plot(_history.history['acc'])
        # plt.plot(_history.history['val_acc'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()

        n, c = testing(_model)
        print(n, c)

n, c = testing(_model)
print(n, c, c / n)
