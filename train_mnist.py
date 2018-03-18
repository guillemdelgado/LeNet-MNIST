import tensorflow as tf
import numpy as np
import os
import keras.utils
from keras.layers import Input, Dense, concatenate, add, Dropout, Flatten, AveragePooling2D, MaxPooling2D, Reshape
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import utils_mnist as umnist
from lenet_model import get_model_lenet

def get_model(experiment_name, imgsize, optimizer='adadelta'):

    # create new model
    model = Sequential()
    model.add(Flatten(input_shape=(imgsize, imgsize, 1)))
    model.add(Dense(1024,activation='relu', name='fc1'))
    model.add(Dense(512, activation='relu', name='fc2'))
    model.add(Dense(10, activation='softmax', name='fc3'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #plot_model(model, to_file=experiment_name + '.png', show_shapes=True, show_layer_names=True)

    return model

lenet = 0
img_size = 32
batch_size = 32
number_of_epoch = 30
experiment_name = 'fully_3'
WEIGHTS_FNAME = './models/' + experiment_name + '_weights.h5'

print('Loading MNIST dataset...')
mnist = umnist.load_mnist_32x32(True)
print('Loaded!')


print('Preparing net...')

if lenet == 1:
    optimizer = SGD(lr=0.01)
    model = get_model_lenet(experiment_name, img_size, optimizer)
    print('Training the dataset with LeNet...')
else:
    optimizer = Adam(lr=1e-5)
    model = get_model(experiment_name, img_size, optimizer)
    print('Training the dataset with 3 fully connected layers...')
model.summary()

train_datagen = ImageDataGenerator()
train_datagen.fit(mnist.train.images)

test_datagen = ImageDataGenerator()
test_datagen.fit(mnist.test.images)

validation_datagen = ImageDataGenerator()
validation_datagen.fit(mnist.validation.images)



#Callbacks definition:
checkpoint = ModelCheckpoint(WEIGHTS_FNAME, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
tb = TensorBoard(log_dir='./logs/'+experiment_name+'/', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False,
            write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=0, mode='auto')

history = model.fit_generator(train_datagen.flow(mnist.train.images, keras.utils.to_categorical(mnist.train.labels), batch_size=batch_size),
                              steps_per_epoch = 1881 // batch_size,
                              epochs=number_of_epoch,
                              validation_data=validation_datagen.flow(mnist.validation.images, keras.utils.to_categorical(mnist.validation.labels), batch_size=batch_size),
                              callbacks=[checkpoint, tb, reduce_lr, early_stopping])

model.load_weights(WEIGHTS_FNAME)

result = model.evaluate_generator(test_datagen.flow(mnist.test.images, keras.utils.to_categorical(mnist.test.labels)))
print(result)