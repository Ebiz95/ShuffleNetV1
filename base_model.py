import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def base_model(args):
    model = keras.Sequential(
                [
                    keras.Input(shape=(args.img_height, args.img_width, 3)),
                    layers.Conv2D(filters=24, kernel_size=3, strides=2, activation='relu', padding='same'),
                    layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
                    layers.BatchNormalization(),
                    layers.Conv2D(filters=144, kernel_size=3, strides=2, activation='relu', padding='same'),
                    layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
                    layers.BatchNormalization(),
                    layers.GlobalAveragePooling2D(),
                    layers.Dense(512, activation="relu"),
                    layers.BatchNormalization(),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(args.num_classes),
                ]
            )

    return model