import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
import os,sys
import argparse
import cv2 as cv

# Model / data parameters
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
    path="/mnt/data/Data/mnist/mnist-model"
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
batch_size = 128
epochs = 3

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.save('/mnt/data/Data/mnist/mnist-model')
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
predict()
