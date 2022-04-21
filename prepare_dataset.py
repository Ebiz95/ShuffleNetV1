import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def prepare_dataset(args):
    ds_train = keras.preprocessing.image_dataset_from_directory(
        f"{args.data_dir}/train",
        labels="inferred",
        label_mode="int",
        class_names=['boats', 'no_boats'],
        color_mode='rgb',
        batch_size=args.batch_size,
        image_size=(args.img_height, args.img_width),  # reshape if not in this size
        shuffle=True,
        seed=123,
        validation_split=args.val_split,
        subset="training",
    )

    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        f"{args.data_dir}/train",
        labels="inferred",
        label_mode="int",  # categorical, binary
        class_names=['boats', 'no_boats'], 
        color_mode='rgb',
        batch_size=args.batch_size,
        image_size=(args.img_height, args.img_width),  # reshape if not in this size
        shuffle=True,
        seed=123,
        validation_split=args.val_split,
        subset="validation",
    )

    ds_test = tf.keras.preprocessing.image_dataset_from_directory(
        f"{args.data_dir}/test",
        labels="inferred",
        label_mode="int",  # categorical, binary
        class_names=['boats', 'no_boats'],
        color_mode='rgb',
        batch_size=args.batch_size,
        image_size=(args.img_height, args.img_width),  # reshape if not in this size
    )

    return ds_train, ds_validation, ds_test