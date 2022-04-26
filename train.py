import argparse
import os
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from model import ShuffleNet
from prepare_dataset import prepare_dataset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def get_args():
    parser = argparse.ArgumentParser("ShuffleNetV1")

    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='total epochs')
    parser.add_argument('--val-split', type=float, default=0.2, help='Fraction of training dataset to be used for validation')


    parser.add_argument('--img-height', type=int, default=768, help='image height')
    parser.add_argument('--img-width', type=int, default=768, help='image width')
    parser.add_argument('--num-classes', type=int, default=2, help='number of classes')
    parser.add_argument('--groups', type=int, default=3, help='groups number')

    parser.add_argument('--save-dir', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--data-dir', type=str, default='data/', help='path to training dataset')

    parser.add_argument('--weights-path', type=str, default=None, help='path to saved model weights')
    parser.add_argument('--model-dir', type=str, default=None, help='path to saved model')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    print(f"groups: {args.groups}")
    print(f"num classes: {args.num_classes}")
    print(f"batch size: {args.batch_size}")


    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        ds_train, ds_val, ds_test = prepare_dataset(args)
        print("Init model")
        model = ShuffleNet(groups=args.groups, num_classes=args.num_classes)
        print("Init model done")

        if not args.weights_path is None:
            # model.built = True
            model.load_weights(args.weights_path)

        print("Model compiling...")
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
            metrics=[keras.metrics.SparseCategoricalAccuracy(), "accuracy"],
        )
        print("Model compiling done")

        print(model.model((args.img_height, args.img_width, 3)).summary())


        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H:%M")

        model.fit(ds_train, validation_data=ds_val, epochs=args.epochs, verbose=1)
        
        print("Evaluating the model on the test dataset...")
        model.evaluate(ds_test)

        print("Saving weights")
        model.save_weights(f"{args.save_dir}/model_weights/{dt_string}/")

        print("Saving model")
        model_path = f"{args.save_dir}/models/{dt_string}/"
        model.save(model_path)

if __name__ == "__main__":
    main()
