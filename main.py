import argparse
from predict import predict
from decorators import log_execution_time

# Training & Evaluation imports
import utils
import config
from data_loader import get_datasets

# Training imports
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_model, unfreeze_for_fine_tuning

# Evaluation imports
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt




# Training script for the EfficientNetB0 cat/dog classifier.
# Trains the classification head first (backbone frozen), then unfreezes the top of the backbone for a short fine-tuning phase.
@log_execution_time
def train():
    utils.set_seed()
    utils.ensure_dirs()

    print("Loading datasets...")
    train_ds, val_ds, test_ds = get_datasets()

    print("Building model...")
    model, base_model = build_model()
    model.summary()

    callbacks = [
            EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True, monitor="val_loss"),
            ModelCheckpoint(config.MODEL_PATH, save_best_only=True, monitor="val_loss")
        ]

    print("\n--- Phase 1: training classification head ---")
    history = model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS, callbacks=callbacks)

    print("\n--- Phase 2: fine-tuning top layers of EfficientNetB0 ---")
    model = unfreeze_for_fine_tuning(model, base_model)
    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.FINE_TUNE_EPOCHS,
        callbacks=callbacks)

    utils.plot_history(history, fine_tune_history)

    print("\nEvaluating on held-out test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    model.save(config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")



# Evaluation script: loads a trained model and reports metrics on the test set, including a classification report and a confusion matrix plot.
@log_execution_time
def evaluate():
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cat vs Dog EfficientNet Classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train the model")
    subparsers.add_parser("evaluate", help="Evaluate the model on the test set")

    predict_parser = subparsers.add_parser("predict", help="Predict a single image")
    predict_parser.add_argument("image_path", type=str, help="Path to the image file")

    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == "evaluate":
        evaluate()
    elif args.command == "predict":
        predict(args.image_path)