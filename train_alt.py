import os
import argparse
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def get_args():
    parser = argparse.ArgumentParser("ShuffleNetV1")

    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='total epochs')
    parser.add_argument('--save-dir', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--save-interval', type=int, default=10, help='save interval')

    parser.add_argument('--img-height', type=int, default=768, help='image height')
    parser.add_argument('--img-width', type=int, default=768, help='image width')
    parser.add_argument('--num-classes', type=int, default=2, help='number of classes')

    parser.add_argument('--data-dir', type=str, default='data/', help='path to training dataset')
    args = parser.parse_args()
    return args

def prepare_dataset(args):
    ds_train = keras.preprocessing.image_dataset_from_directory(
        args.data_dir,
        labels="inferred",
        label_mode="int",
        class_names=['boats', 'no_boats'],
        color_mode='rgb',
        batch_size=args.batch_size,
        image_size=(args.img_height, args.img_width),  # reshape if not in this size
        shuffle=True,
        seed=123,
        validation_split=0.15,
        subset="training",
    )

    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        args.data_dir,
        labels="inferred",
        label_mode="int",  # categorical, binary
        class_names=['boats', 'no_boats'],
        color_mode='rgb',
        batch_size=args.batch_size,
        image_size=(args.img_height, args.img_width),  # reshape if not in this size
        shuffle=True,
        seed=123,
        validation_split=0.15,
        subset="validation",
    )

    return ds_train, ds_validation

def main():
    args = get_args()
    print(f"num classes: {args.num_classes}")
    print(f"batch size: {args.batch_size}")

    ds_train, ds_val = prepare_dataset(args)


    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        AUTOTUNE = tf.data.AUTOTUNE

        ds_train = ds_train.cache().prefetch(buffer_size=AUTOTUNE)
        ds_val = ds_val.cache().prefetch(buffer_size=AUTOTUNE)
        print("Init model")
        # model = ShuffleNet(groups=args.groups, num_classes=args.num_classes)

        model = keras.Sequential(
            [
                keras.Input(shape=(args.img_height, args.img_width, 3)),
                layers.Conv2D(filters=24, kernel_size=3, strides=2, activation='relu', padding='same'),
                layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
                layers.Conv2D(filters=144, kernel_size=3, strides=2, activation='relu', padding='same'),
                layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
                layers.GlobalAveragePooling2D(),
                layers.Dense(512, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(args.num_classes),
            ]
        )

        print("Init model done")

        print("Model compiling...")
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
            metrics=["accuracy"],
        )
        print("Model compiling done")

        print(model.summary())

        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H:%M")

        # cp_callback = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=checkpoint_path,
        #     verbose=1,
        #     save_weights_only=True,
        #     save_freq=args.save_interval * args.batch_size # Number of images / batch_size
        # )

        model.fit(ds_train, validation_data=ds_val, epochs=args.epochs, verbose=1)
        model.save_weights(f"{args.save_dir}/model_weights/{dt_string}/")

        model_path = f"{args.save_dir}/models/{dt_string}/"
        model.save(model_path)

        print("Init converter")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_types = [tf.float32]
        print("Init converter done")
        print("Converting model")
        tflite_model = converter.convert()
        print("Converting model done")

        tflite_model_path = f"{args.save_dir}/models-tflite/{dt_string}"
        if not os.path.exists(tflite_model_path):
            os.makedirs(tflite_model_path)
            print(f"'{tflite_model_path}' directory is created!")

        with open(f"{tflite_model_path}/model.tflite", 'wb') as f:
            f.write(tflite_model)

if __name__ == "__main__":
    main()