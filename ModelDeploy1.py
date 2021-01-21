import cv2
import numpy as np
import scipy
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt


def predict(name):

    # Charger le Model
    model = keras.models.load_model("model1.h5")
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    class_names = ['T-shirt/top', 'Pantalon', 'Pullover', 'Robe', 'Manteau',
                   'Sandale', 'Chemise', 'Sneaker', 'Sac', 'Bottine']


    def testimg(img1) :
        Bgblack(img1)
        img = scipy.misc.imread(img1, mode='L')
        img = scipy.misc.imresize(img1, (28, 28))
        plt.figure()
        plt.imshow(img)
        plt.show()
        img=(np.expand_dims(img, 0))
        prediction_single = probability_model.predict(img)
        shape = img.shape
        prediction = np.argmax(prediction_single[0])
        return shape, prediction

    def Bgblack(pathToImage):
        img = cv2.imread(pathToImage)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        img[thresh == 255]=0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erosion = cv2.erode(img, kernel, interations=1)
        cv2.imwrite(pathToImage, erosion)


