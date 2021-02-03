from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import sys,os
import cv2 as cv

from model import get_model


def main(batch_size=16):
    if (not sys.argv[1:]):
        path="/mnt/data/Data/mnist/model.h5"
        print("######################\n\n No model path specified. Switching to default model /mnt/data/Data/mnist/model.h5\n\n######################")
    else:
        if os.path.exists(sys.argv[1]):
            path = sys.argv[1]
        else:
            print("######################\n\n No such model exists. Switching to default model /mnt/data/Data/mnist/model.h5\n\n######################")
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

if __name__ == '__main__':
    main()
