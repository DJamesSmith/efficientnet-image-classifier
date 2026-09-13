# Miscellaneous helper functions: reproducibility and plotting.

import os
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import config


# Sets random seeds across libraries for reproducible runs.
def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Creates the model/output directories if they don't already exist.
def ensure_dirs():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


# Plots training/validation accuracy and loss curves and saves to disk.
def plot_history(history, fine_tune_history=None, save_path=config.HISTORY_PLOT_PATH):
    acc = list(history.history["accuracy"])
    val_acc = list(history.history["val_accuracy"])
    loss = list(history.history["loss"])
    val_loss = list(history.history["val_loss"])

    if fine_tune_history is not None:
        acc += fine_tune_history.history["accuracy"]
        val_acc += fine_tune_history.history["val_accuracy"]
        loss += fine_tune_history.history["loss"]
        val_loss += fine_tune_history.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.legend(loc="lower right")
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.legend(loc="upper right")
    plt.title("Loss")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training history plot to {save_path}")