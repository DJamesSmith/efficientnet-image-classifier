# Training script for the EfficientNetB0 cat/dog classifier.
# Trains the classification head first (backbone frozen), then unfreezes the top of the backbone for a short fine-tuning phase.

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import config
import utils
from data_loader import get_datasets
from model import build_model, unfreeze_for_fine_tuning


if __name__ == "__main__":
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



# Run command:
# python train.py