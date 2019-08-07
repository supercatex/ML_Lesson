# [Description]
# In this sample evaluation script, we evaluate the model being 
# trained in Session 03 which does not use the "cropped rectangle" 
# from the labeled data.

# [FREE TO UPDATE] import your modules
import os, cv2, re, random
import numpy as np
from keras.models import load_model
from PIL import Image
import pandas

# [DO NOT CHANGE] Each record of "image_df" contains 5 attributes: img, x1, y1, x2, y2
def predict(image_df, testpath, submitpath):
	# [DO NOT CHANGE] load the CSV file for evaluation
	workpath=submitpath.rsplit('/',1)[0]+'/'
	
	# [FREE TO UPDATE] load your model
	model = load_model(workpath+'sign.h5')

	# [DO NOT CHANGE] loop all images for evaluation
	labels = []
	for index, row in image_df.iterrows():
		# [DO NOT CHANGE] get the image path, imagepath
		imagepath = testpath+'img/'+ row["img"]

		x1 = row["x1"]
		y1 = row["y1"]
		x2 = row["x2"]
		y2 = row["y2"]
		x1 = max(x1, 0)
		y1 = max(y1, 0)
		x2 = max(x2, 0)
		y2 = max(y2, 0)

		img = cv2.imread(imagepath, cv2.IMREAD_COLOR)
		img = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
		size = 25
		# [FREE TO UPDATE - begin] The prediction being made by your logic and / or model(s)
		x = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
		x = x.astype(np.float32)
		x /= 255
		# x = x.reshape(1, size, size, 3)
		y = model.predict(np.array([x]))

		labels.append(np.argmax(y[0]) + 1)
		# [FREE TO UPDATE - end] The prediction being made by your logic and / or model(s)

	# [DO NOT CHANGE] return a list of labels
	print(labels)

	return labels
