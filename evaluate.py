# Evaluation script: loads a trained model and reports metrics on the test set, including a classification report and a confusion matrix plot.

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import config
import utils
from data_loader import get_datasets


if __name__ == "__main__":
    utils.ensure_dirs()

    print(f"Loading model from {config.MODEL_PATH}")
    model = tf.keras.models.load_model(config.MODEL_PATH)

    _, _, test_ds = get_datasets()

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten().tolist())
        y_pred.extend((preds.flatten() > 0.5).astype(int).tolist())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(config.CLASS_NAMES))
    plt.xticks(tick_marks, config.CLASS_NAMES)
    plt.yticks(tick_marks, config.CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH)
    plt.close()
    print(f"Saved confusion matrix to {config.CONFUSION_MATRIX_PATH}")


# Run command:
# python evaluate.py