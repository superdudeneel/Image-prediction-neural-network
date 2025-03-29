import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
print(train_images.shape)
train_images = train_images/255.0
test_images = test_images / 255.0
class_names = ['airplane','automobile' ,'bird','cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]




model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10))
model.compile(optimizer = 'adam',
              metrics = ['accuracy'],
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)),


model.fit(train_images, train_labels, epochs = 10)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print(test_acc)


predictions = model.predict(test_images)
print(class_names[np.argmax(predictions[8])])
plt.imshow(test_images[8]) # you can change the index here
plt.colorbar()

plt.show()

