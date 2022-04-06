import argparse
import os

from numpy import float32

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def get_args():
    parser = argparse.ArgumentParser("ShuffleNetV1")

    parser.add_argument('--img-height', type=int, default=768)
    parser.add_argument('--img-width', type=int, default=768)
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--precision', type='str', default='float32')

    parser.add_argument('--save-path', type=str, default='./saved_models_tflite', help='directory in which the model will be exported to')

    args = parser.parse_args()
    return args

def make_model(args):
    print("Creating the model!")

    model = keras.Sequential(
        [
            keras.Input(shape=(args.img_height, args.img_width, args.num_channels)),
            layers.Conv2D(filters=24, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
            layers.Conv2D(filters=144, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(args.num_classes, activation='softmax'),
        ]
    )

    print("Model done!")
    print(model.summary())

    return model

def export(args, model):
    print("Converting the model!")
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if args.precision == 'float16':
        precision = tf.float16
    elif args.precision == 'int8':
        precision = tf.int8
    else:
        precision = tf.float32
    converter.target_spec.supported_types = [precision]
    tflite_model = converter.convert()
    
    print("Saving the model!")
    # Save the model.
    filename = f'tflite-{args.img_height}-{args.img_width}-{args.num_channels}-{args.precision}'
    with open(f"{args.save_path}/{filename}.tflite", 'wb') as f:
        f.write(tflite_model)

def main():
    args = get_args()
    
    print(f"Img height: {args.img_height}")
    print(f"Img width: {args.img_width}")
    print(f"Num channels: {args.num_channels}")
    print(f"Precision: {args.precision}")

    model = make_model(args)
    export(args, model)

if __name__ == '__main__':
    main()