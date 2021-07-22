import tensorflow as tf
from tensorflow import keras
from keras import callbacks
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train / 255
X_train = np.around(X_train)
X_train = X_train.astype(int)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = callbacks.EarlyStopping(
    min_delta = 0.001,
    patience = 5,
    restore_best_weights = True
)

model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    epochs = 15,
    callbacks = [early_stopping]
)
model.save("Number-Predictor/numpredictor/m.model")
