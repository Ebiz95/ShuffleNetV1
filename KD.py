import argparse
import os
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from base_model import base_model
from distiller import Distiller
from prepare_dataset import prepare_dataset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def get_args():
    parser = argparse.ArgumentParser("ShuffleNetV1")

    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='total epochs')
    parser.add_argument('--save-dir', type=str, default='./models', help='path for saving trained models')

    parser.add_argument('--img-height', type=int, default=768, help='image height')
    parser.add_argument('--img-width', type=int, default=768, help='image width')
    parser.add_argument('--num-classes', type=int, default=2, help='number of classes')
    parser.add_argument('--val-split', type=float, default=0.2, help='Fraction of training dataset to be used for validation')

    parser.add_argument('--data-dir', type=str, default='data/', help='path to training dataset')

    # parser.add_argument('--weights-path', type=str, default=None, help='path to saved teacher model weights')
    parser.add_argument('--teacher-model-dir', type=str, default=None, help='path to saved teacher model')
    parser.add_argument('--student-model-dir', type=str, default=None, help='path to saved teacher model')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    if args.teacher_model_dir is None:
        print("A directory to the teacher model is required.")
        return
    
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    print("Loading teacher model...")
    teacher = keras.models.load_model(args.teacher_model_dir)
    teacher._name = 'teacher'
    print("Loading teacher model done!")

    if not args.student_model_dir is None:
        # model.built = True
        print("Loading the student model...")
        student = keras.models.load_model(args.student_model_dir)
        print("Loading the student model done!")
    else:
        print("Initializing student model...")
        student = base_model(args)
        print("Initializing student model done!")
    student._name = 'student'

    ds_train, ds_val, ds_test = prepare_dataset(args)

    # Initialize and compile distiller
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(),],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )

    # Distill teacher to student
    distiller.fit(ds_train, validation_data=ds_val, epochs=args.epochs, verbose=1)

    # Evaluate student on test dataset
    distiller.evaluate(ds_test)

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M")

    print("Saving weights...")
    student.save_weights(f"{args.save_dir}/model_weights/{dt_string}/")

    print("Saving the model...")
    model_path = f"{args.save_dir}/models/{dt_string}/"
    student.save(model_path)

if __name__ == '__main__':
    main()