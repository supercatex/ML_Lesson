import os
import cv2
import numpy as np
import pandas as pd
from eval import predict


_ROOT = "../MacauAI_TrainingSet_1/"
_CSV = os.path.join(_ROOT, "training.csv")

data = pd.read_csv(_CSV, sep=",")
y = predict(data, _ROOT, "./upload",)

n = 0
c = 0
for i, row in data.iterrows():
    print(y[i], row["class"])
    if y[i] == row["class"]:
        c += 1
    n += 1
print(n, c, c / n)
