import os
import argparse
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def get_args():
    parser = argparse.ArgumentParser("ShuffleNetV1")

    parser.add_argument('--model-path', type=str, default='./saved_models', help='path to saved model')
    parser.add_argument('--save-path', type=str, default='./saved_models_tflite', help='directory in which the model will be exported to')

    args = parser.parse_args()
    return args

def export(args):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(args.model_path) # path to the SavedModel directory
    tflite_model = converter.convert()
    
    # Save the model.
    filename = args.model_path.split("/")[-1]
    with open(f"{args.save_path}/{filename}.tflite", 'wb') as f:
        f.write(tflite_model)

if __name__ == "__main__":
    args = get_args()
    export(args)