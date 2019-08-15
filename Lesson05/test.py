from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import models


img_path = "../../datasets/dogs-vs-cats/cats_and_dogs_small/test/cats/cat.1700.jpg"
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255
print(img_tensor.shape)

plt.imshow(img_tensor[0])
plt.show()

model = load_model("cats_and_dogs_small_2.h5")
model.summary()

layer_outputs = [layer.output for layer in model.layers[:8]]
for op in layer_outputs:
    print(op)
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)
print(len(activations))

for i in range(len(activations)):
    plt.matshow(activations[0][0, :, :, i], cmap="viridis")
    plt.show()
