# Model definition: EfficientNetB0-based CNN for binary cat/dog classification.

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
import config


"""
Builds and compiles an EfficientNetB0-backed binary classifier.
Returns:
    model: compiled tf.keras.Model
    base_model: reference to the EfficientNetB0 backbone (needed later to selectively unfreeze layers for fine-tuning)
"""
def build_model():
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=config.IMAGE_SIZE + (3,),
        pooling="avg")
    base_model.trainable = config.BASE_TRAINABLE

    inputs = layers.Input(shape=config.IMAGE_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="efficientnet_cat_dog")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"])
    return model, base_model


def unfreeze_for_fine_tuning(model, base_model):
    """
    Unfreezes the top layers of the backbone (from config.FINE_TUNE_AT_LAYER onward) and re-compiles
    with a smaller learning rate, for a fine-tuning phase run after the classification head has already been trained.
    """
    base_model.trainable = True
    for layer in base_model.layers[:config.FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.FINE_TUNE_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"])
    return model