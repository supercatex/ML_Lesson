# https://www.kaggle.com/scolianni/mnistasjpg

import cv2
import numpy as np
import matplotlib.pyplot as plt
from Step_1_import_data import import_data
from Step_2_normalize_data import normalize_data
from Step_3_create_model import  create_model
from Step_4_training import training
from Step_5_save_model import save_model
from Step_6_load_model import load_model
from Step_7_predict import predict
import os


model, labels = load_model("model1")
if model is None:
    X, y, labels = import_data("./mnistasjpg/trainingSet", (28, 28), 0)
    print(X.shape, y.shape, labels)

    X = X / 255
    X, y = normalize_data(X, y)
    print(X.shape, y.shape, labels)
    print(y[0])

    model = create_model(X.shape[1:], len(y[0]))
    model.summary()

    model, history = training(X, y, model, epochs=50, batch_size=128)
    save_model(model, labels, "model")

    print(history.history.keys())

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


testing_data = os.listdir("./mnistasjpg/testSet")
for f in testing_data:
    image = cv2.imread(os.path.join("./mnistasjpg/testSet", f), cv2.IMREAD_UNCHANGED)

    X = np.array([image])
    X = X / 255
    X = normalize_data(X)
    y = predict(X, model)
    print(np.argmax(y), y)

    cv2.imshow("image", image)
    k = cv2.waitKey(0)
    if k == ord('q'):
        break
