import os
import keras.utils
from keras.layers import Input, Dense, concatenate, add, Dropout, Flatten, AveragePooling2D, MaxPooling2D, Reshape, Conv2D
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import utils_mnist as umnist

def get_model_lenet(experiment_name, imgsize, optimizer):

    # create new model
    model = Sequential()
    model.add(Conv2D(20,(5,5), activation='relu', border_mode='same',input_shape=(imgsize, imgsize, 1)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(50, (5, 5), activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(500,activation='relu', name='fc1'))
    model.add(Dense(84, activation='relu', name='fc2'))
    model.add(Dense(10, activation='softmax', name='fc3'))
    #plot_model(model, to_file=experiment_name + '.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
