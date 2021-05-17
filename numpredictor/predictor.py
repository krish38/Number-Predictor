import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train / 255
X_train = np.around(X_train)
X_train = X_train.astype(int)
# for i in range(28):
#     print(X_train[0][i])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1)
model.save("numpredictor/m.model")

# ind=2
# print(model.predict(np.expand_dims(X_train[ind],0)))
# plt.imshow(X_train[ind], cmap=plt.cm.binary)
# plt.show()