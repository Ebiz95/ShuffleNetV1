import argparse
import os
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from base_model import base_model
from prepare_dataset import prepare_dataset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def get_args():
    parser = argparse.ArgumentParser("ShuffleNetV1")

    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=2, help='total epochs')
    parser.add_argument('--save-dir', type=str, default='./models', help='path for saving trained models')

    parser.add_argument('--img-height', type=int, default=768, help='image height')
    parser.add_argument('--img-width', type=int, default=768, help='image width')
    parser.add_argument('--val-split', type=float, default=0.2, help='Fraction of training dataset to be used for validation')
    parser.add_argument('--num-classes', type=int, default=2, help='number of classes')

    parser.add_argument('--data-dir', type=str, default='data/', help='path to data directory')
    args = parser.parse_args()
    return args



def main():
    args = get_args()
    print(f"num classes: {args.num_classes}")
    print(f"batch size: {args.batch_size}")

    ds_train, ds_val, ds_test = prepare_dataset(args)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        print("Init model")
        # model = ShuffleNet(groups=args.groups, num_classes=args.num_classes)

        model = base_model(args)

        print("Init model done")

        print("Model compiling...")
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=[
                keras.losses.SparseCategoricalCrossentropy(),
                # keras.losses.BinaryCrossentropy(),
            ],
            metrics=[
                "accuracy", 
                # "binary_crossentropy",
            ],
        )
        print("Model compiling done")

        print(model.summary())

        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H:%M")

        model.fit(ds_train, validation_data=ds_val, epochs=args.epochs, verbose=1)
        
        print("Evaluating the model on the test dataset...")
        model.evaluate(ds_test)

        print("Saving weights...")
        model.save_weights(f"{args.save_dir}/model_weights/{dt_string}/")

        print("Saving the model...")
        model_path = f"{args.save_dir}/models/{dt_string}/"
        model.save(model_path)

if __name__ == "__main__":
    main()
