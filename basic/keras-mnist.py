import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
import os,sys
import argparse
import tensorflow as tf
import cv2 as cv
from os import path

# Model / data parameters
def set_device():
    #if which_gpu == -1:
        #print("setting CPU********************")
        #my_devices = tf.config.list_physical_devices(device_type='CPU')
        #tf.config.set_visible_devices([], 'GPU')
        #tf.config.set_visible_devices(devices= my_devices, device_type='CPU')

    gpu_devices = tf.config.list_physical_devices('GPU')
    if (gpu_devices[1:]):
        print(gpu_devices)
        tf.config.set_visible_devices(gpu_devices[0], 'GPU')
    else:
        my_devices = tf.config.list_physical_devices(device_type='CPU')
        tf.config.set_visible_devices(devices= my_devices, device_type='CPU')

def parse_arguments():
        args_parser = argparse.ArgumentParser(description="training parameters")
        args_parser.add_argument('-n', "--epochs", default=3, action='store',type=float, dest='epochs',help="number of epochs")
        args_parser.add_argument('-b', "--batch_size", default=16, action='store', dest='batch_size',type=float,help="Batch Size")
        args = args_parser.parse_args()
        return args


def predict(batch_size=16):
    path="/mnt/data/mnist/mnist-model"
    model = load_model(path)
    model.summary()
    #model.load_weights(path)
    file = input("enter file path: ")
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    image = cv.resize(image, (28,28))
    image = 255-image          #inverts image. Always gets read inverted.

    #plt.imshow(image.reshape(28, 28),cmap='Greys')
    #plt.show()
    pred = model.predict(image.reshape(1, 28, 28, 1), batch_size=1)
    print(file, "is ",pred.argmax(),np.argmax(pred))
    return


def main():
    set_device()
    if path.exists("/mnt/data/mnist/mnist-model"):
        print("###########################\nTrained model already exists at path /mnt/data/mnist/\n Predicting from pretrained model\n#######################")
        predict()
        return
    args = parse_arguments()
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    print(f"#############################\nTraining the model for {epochs} epochs and batchsize {batch_size}\n##############################")

    num_classes = 10
    input_shape = (28, 28, 1)
    
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
    )

    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    model.save('/mnt/data/mnist/mnist-model')
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    predict()
if __name__ == '__main__':
    main()
