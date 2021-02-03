from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from os import path
from model import get_model
import os,sys
import argparse
import numpy as np
import cv2 as cv


def parse_arguments():
        args_parser = argparse.ArgumentParser(description="training parameters")
        args_parser.add_argument('-n', "--epochs", default=3, action='store',type=float, dest='epochs',help="number of epochs")
        args_parser.add_argument('-b', "--batch_size", default=16, action='store', dest='batch_size',type=float,help="Batch Size")
        args = args_parser.parse_args()
        return args

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

    return
def predict(batch_size=16):
    """if (not sys.argv[1:]):
        path="/mnt/data/Data/mnist/model.h5"
        print("######################\n\n No model path specified. Switching to default model /mnt/data/Data/mnist/model.h5\n\n######################")
    else:
        if os.path.exists(sys.argv[1]):
            path = sys.argv[1]
        else:
            print("######################\n\n No such model exists. Switching to default model /mnt/data/Data/mnist/model.h5\n\n######################")
            path="/mnt/data/Data/mnist/model.h5"
    """
    path="/mnt/data/Data/mnist/model.h5"
    model = get_model()
    model.summary()
    model.load_weights(path)
    file = input("enter file path: ")
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    image = cv.resize(image, (28,28))
    image = 255-image          #inverts image. Always gets read inverted.

    #plt.imshow(image.reshape(28, 28),cmap='Greys')
    #plt.show()
    pred = model.predict(image.reshape(1, 28, 28, 1), batch_size=1)
    print(file, "is ",pred.argmax(),np.argmax(pred))
    return

def main(epochs=3, batch_size=16):
    args = parse_arguments()
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    set_device()
    if path.exists("/mnt/data/Data/mnist/model.h5"):
        print("###########################\nTrained model already exists at path /mnt/data/Data/mnist/model.h5\n Predicting from pretrained model\n#######################")
        predict()
        return
    print(f"training the model for {epochs} epochs and batchsize {batch_size}")
    train_data_generator = ImageDataGenerator(
        rescale=1. / 255
    )
    train_generator = train_data_generator.flow_from_directory(
        directory='/mnt/data/Data/mnist/train/',
        target_size=(28, 28),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_data_generator = ImageDataGenerator(
        rescale=1. / 255
    )
    validation_generator = validation_data_generator.flow_from_directory(
        directory='/mnt/data/Data/mnist/test/',
        target_size=(28, 28),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical'
    )

    model = get_model()
    model.summary()
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=60000 // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=10000 // batch_size)
    model.save_weights('/mnt/data/Data/mnist/model.h5')
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    predict()

if __name__ == '__main__':
    main()
