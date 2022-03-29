import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("test_model/")

img = tf.io.read_file("data/boat/00406d1b3.jpg")

x = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32)

print(model(x))