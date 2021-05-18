## What it is
A 28x28 grid canvas appears, and the user can draw a digit (0-9) onto it. The neural network that was trained before, is used to predict what number the user has drawn.

## How it works
A 4 layer neural network is trained on thousands of data from a keras dataset of handrawn digits. 
![Handrawn Digit](https://machinelearningmastery.com/wp-content/uploads/2019/02/Plot-of-a-Subset-of-Images-from-the-MNIST-Dataset.png)

After training for 5 epochs, it has an accuracy of 98%. It is then used to predict what number the user has drawn into a 28x28 grid. The neural network can be trained more, however accuracy does not increase significantly, and can causes overfitting.
