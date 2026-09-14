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

------------------------------- Cleaning the dataset -------------------------------
Scans the dataset directory and removes any image file that can't be fully decoded, or that isn't convertible to standard RGB (e.g. truncated JPEGs,
zero-byte files, or files with a .jpg extension that are actually a different format/mode -- such as grayscale+alpha PNGs, which have 2
channels and make TensorFlow's decoder crash mid-training with: "Number of channels inherent in the image must be 1, 3 or 4, was 2"

There is a known issue with the classic Kaggle "Dogs vs Cats" dataset.
Hence clean_dataset() is used to clean the dataset before training, any time you add/replace images.
"""

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import config
import os
from PIL import Image, UnidentifiedImageError


def clean_dataset(data_dir=config.DATA_DIR):
    checked = 0
    removed = []

    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Skipping missing directory: {class_dir}")
            continue

        for filename in os.listdir(class_dir):
            filepath = os.path.join(class_dir, filename)
            if not os.path.isfile(filepath):
                continue  # skip things like .DS_Store subfolders, etc.

            checked += 1
            try:
                # Pass 1: verify() checks the file isn't truncated/malformed
                with Image.open(filepath) as img:
                    img.verify()

                # Pass 2: verify() leaves the file object unusable for further
                # decoding, so reopen and force a full RGB decode -- this is
                # what actually catches unsupported channel counts (e.g. the
                # 2-channel grayscale+alpha files that crash tf.data).
                with Image.open(filepath) as img:
                    img.convert("RGB")

            except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
                print(f"Removing unusable file: {filepath}  ({e})")
                os.remove(filepath)
                removed.append(filepath)

    print(f"\nChecked {checked} files, removed {len(removed)} unusable file(s).")


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