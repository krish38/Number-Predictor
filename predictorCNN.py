import tensorflow as tf
from tensorflow import keras
from keras import callbacks
import matplotlib.pyplot as plt
import numpy as np

# Setting up training set
(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train / 255
X_train = np.around(X_train)
X_train = X_train.astype(int)

# Defining Model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
    keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.BatchNormalization(),
    
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'),
    keras.layers.Dense(10, activation='softmax')
])

# Compiling Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Defining Early Stopping
early_stopping = callbacks.EarlyStopping(
    monitor="val_acc",
    min_delta = 0.001,
    patience = 5,
    restore_best_weights = True
)

# Training Model
model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    epochs = 15,
    callbacks = [early_stopping]
)

# Saving Model
model.save("m1.model")