# LeNet-MNIST
This repositary contains a simple example to get in touch with Tensorflow and a toy example which trains with different models a classificator with MNIST dataset.

## Simple Network
In the file `simple_network.py` you will find an easy example on how to make a neural network with three fully-connected layers. Using a random input of 0 and 1, the script trains the model and provides with the accuracy of the system.

## Train MNIST
In the file `train_mnist.py` you will find an example of how to train and classify the MNIST dataset. Within the code you will find two different architectures. The first one which consist on three fully-connected layers and the second one which consist on the LeNet architecture. To use the latter set: 
```
lenet_model = 1
```

