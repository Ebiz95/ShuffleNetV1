import os
import argparse
import tensorflow as tf
from tensorflow import keras
from model import ShuffleNet

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def get_args():
    parser = argparse.ArgumentParser("ShuffleNetV1")

    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='total epochs')
    parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--save-interval', type=int, default=10, help='save interval')

    parser.add_argument('--img-height', type=int, default=768, help='image height')
    parser.add_argument('--img-width', type=int, default=768, help='image width')
    parser.add_argument('--num-classes', type=int, default=2, help='number of classes')
    parser.add_argument('--groups', type=int, default=3, help='groups number')

    parser.add_argument('--data-dir', type=str, default='data/', help='path to training dataset')

    args = parser.parse_args()
    return args

def prepare_dataset(args):
    ds_train = keras.preprocessing.image_dataset_from_directory(
        args.data_dir,
        labels="inferred",
        label_mode="int",
        class_names=['boat', 'no_boat'],
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
        class_names=['boat', 'no_boat'],
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
    print(f"groups: {args.groups}")
    print(f"num classes: {args.num_classes}")
    print(f"batch size: {args.batch_size}")

    ds_train, _ = prepare_dataset(args)

    print("Init model")
    model = ShuffleNet(groups=args.groups, num_classes=args.num_classes)
    print("Init model done")

    print("Model compiling")
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
        metrics=["accuracy"],
    )
    print("Model compiling done")

    print(model.model((args.img_height, args.img_width, 3)).summary())

    # config = model.get_config() # Returns pretty much every information about your model
    # print(config["layers"][0]["config"]["batch_input_shape"]) # returns a tuple of width, height and channels
    
    model.fit(ds_train, epochs=1, verbose=1)


if __name__ == "__main__":
    main()