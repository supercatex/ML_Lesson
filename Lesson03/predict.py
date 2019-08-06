import cv2 as cv
import numpy as np
from Step_2_normalize_data import normalize_data
from Step_6_load_model import load_model
from Step_7_predict import  predict


frame = np.zeros((400, 400, 1))
model, labels = load_model("model")


def do_predict():
    global frame, model, labels
    image = frame
    image = cv.resize(image, (28, 28))
    image = np.reshape(image, (28, 28, 1))
    cv.imwrite('temp.jpg', image)

    X = np.array([image])
    X = X / 255
    X = normalize_data(X)
    p = np.argmax(predict(X, model)[0])
    print(p)
    cv.waitKey(1000)
    frame = np.zeros((400, 400, 1))


def mouse_callback(event, x, y, flags, param):
    global frame
    if flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(frame, (x, y), 15, (255, 255, 255), -1)


while True:
    cv.namedWindow("frame")
    cv.setMouseCallback("frame", mouse_callback)
    cv.imshow('frame', frame)

    k = cv.waitKey(1)
    if k in [ord('q'), 27]:
        break

    if k in [32]:
        do_predict()
