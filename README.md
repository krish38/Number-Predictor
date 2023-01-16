## What it is
A 28x28 grid canvas is created, onto which the user can draw a digit (0-9). The Convolution Neural Network that has been trained in predictorCNN.py is used to predict what number the user has drawn.

## How it works
A multi-layered convolutional neural network is trained on thousands of data samples from a keras dataset of handrawn digits.
![Handrawn Digit](https://machinelearningmastery.com/wp-content/uploads/2019/02/Plot-of-a-Subset-of-Images-from-the-MNIST-Dataset.png)

After training for up to 15 epochs, it has an accuracy of 98%. It is then used to predict what number the user has drawn into a 28x28 grid. The neural network can be trained more, however validation accuracy does not increase, causing overfitting.
