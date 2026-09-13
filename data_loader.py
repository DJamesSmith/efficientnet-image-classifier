"""
Data loading and preprocessing utilities.
Expects a directory structure:
    DATA_DIR/
        Cat/
            *.jpg
        Dog/
            *.jpg

Builds efficient tf.data.Dataset pipelines, applying the EfficientNet-specific
preprocessing function and light augmentation on the training split.
"""

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

import config


# Applies augmentation (optional), EfficientNet preprocessing, and performance tuning (shuffle/prefetch) to a raw image dataset.
def _prepare(ds, augment=False, shuffle=False):
    if augment:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(lambda x, y: (preprocess_input(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000, seed=config.SEED)

    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


# Builds train, validation, and test tf.data.Dataset objects from config.DATA_DIR.
# Returns: (train_ds, val_ds, test_ds)
def get_datasets():
    train_raw = tf.keras.utils.image_dataset_from_directory(
        config.DATA_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=config.CLASS_NAMES,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        subset="training",
        seed=config.SEED)

    val_test_raw = tf.keras.utils.image_dataset_from_directory(
        config.DATA_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=config.CLASS_NAMES,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        subset="validation",
        seed=config.SEED)

    # Split the held-out "validation" portion further into val + test
    val_batches = tf.data.experimental.cardinality(val_test_raw).numpy()
    test_batches = int(val_batches * (config.TEST_SPLIT / config.VALIDATION_SPLIT))

    test_raw = val_test_raw.take(test_batches)
    val_raw = val_test_raw.skip(test_batches)

    train_ds = _prepare(train_raw, augment=True, shuffle=True)
    val_ds = _prepare(val_raw)
    test_ds = _prepare(test_raw)

    return train_ds, val_ds, test_ds
