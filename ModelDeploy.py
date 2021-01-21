import numpy as np
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

    image = Image.open(name).convert('L')
    pic = image.resize((28, 28))
    picPlot = plt.imshow(pic)
    plt.show()

    imgArray = np.asarray(pic, dtype='uint8')
    # print(imgArray)
    resizedArray = (np.expand_dims(imgArray, 0))
    prediction = probability_model.predict(resizedArray)
    print(prediction)
    print("Predict label ->", np.argmax(prediction[0]))
    print("Predict Image ->", class_names[np.argmax(prediction[0])])
    return class_names[np.argmax(prediction[0])]
