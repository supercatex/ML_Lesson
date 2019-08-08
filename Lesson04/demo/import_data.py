import os
import sys
import cv2
import numpy as np
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Dropout, Flatten


_csv_path = "../dataset/training.csv"

_file = open(_csv_path, "r")
_lines = _file.readlines()
_file.close()

_X = []
_y = []
_s = [0] * 120
print("Begin")
for _i, _line in enumerate(_lines[1:]):
    sys.stdout.write("\r" + str(_i) + "/" + str(len(_lines[1:])))
    sys.stdout.flush()

    _t = _line.split(",")
    _name = _t[0]
    _w = int(_t[1])
    _h = int(_t[2])
    _x1 = int(_t[3])
    _y1 = int(_t[4])
    _x2 = int(_t[5])
    _y2 = int(_t[6])
    _class = int(_t[7])

    if _x1 < 0: _x1 = 0
    if _y1 < 0: _y1 = 0
    if _x2 < 0: _x2 = 0
    if _y2 < 0: _y2 = 0

    if _x1 > _x2:
        _z = _x1
        _x1 = _x2
        _x2 = _z

    if _y1 > _y2:
        _z = _y1
        _y1 = _y2
        _y2 = _z

    _path = os.path.join("../dataset/img", _name)
    _img = cv2.imread(_path, cv2.IMREAD_COLOR)
    _sign = _img[_y1:_y2, _x1:_x2]
    _sign = cv2.resize(_sign, (50, 50))

    _X.append(_sign)
    _y.append(_class)
    _s[_class - 1] += 1
print()

_X = np.array(_X)
_y = np.array(_y)

_X = _X.astype(np.float32)
_X /= 255

_y -= 1
_y = np_utils.to_categorical(_y, 120)


for _i, _n in enumerate(_s):
    print(_i + 1, ":", _n)
_s = np.max(_s) - _s

# _data_path = "../../../dataset/data.csv"
# _file = open(_data_path, "r")
# _lines = _file.readlines()
# _file.close()
#
# _csv_path = "../dataset/training.csv"
# _file = open(_csv_path, "a")
#
# for _i, _line in enumerate(_lines[1:]):
#     sys.stdout.write("\r" + str(_i) + "/" + str(len(_lines[1:])))
#     sys.stdout.flush()
#
#     _t = _line.split(",")
#     _name = _t[0]
#     _w = int(_t[1])
#     _h = int(_t[2])
#     _x1 = int(_t[3])
#     _y1 = int(_t[4])
#     _x2 = int(_t[5])
#     _y2 = int(_t[6])
#     _class = int(_t[7])
#
#     if _s[_class - 1] == 0:
#         continue
#
#     _path = os.path.join("../../../dataset/img", _name)
#     _img = cv2.imread(_path, cv2.IMREAD_COLOR)
#     cv2.imwrite(os.path.join("../dataset/img", _name), _img)
#     _file.write("%s,%d,%d,%d,%d,%d,%d,%d\n" % (
#         _name,
#         _w,
#         _h,
#         _x1,
#         _y1,
#         _x2,
#         _y2,
#         _class
#     ))
#     _s[_class - 1] -= 1
# print()
# _file.close()
print("FINISHED")
