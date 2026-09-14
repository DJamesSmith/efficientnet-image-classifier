# Single-image inference script.

import sys
from decorators import log_execution_time


import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

import config


def load_and_preprocess(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=config.IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

@log_execution_time
def predict(image_path):
    model = tf.keras.models.load_model(config.MODEL_PATH)
    img_array = load_and_preprocess(image_path)
    prediction = model.predict(img_array, verbose=0)[0][0]

    label = config.CLASS_NAMES[1] if prediction > 0.5 else config.CLASS_NAMES[0]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    print(f"Prediction: {label} (confidence: {confidence:.2%})")
    return label, confidence


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    predict(sys.argv[1])