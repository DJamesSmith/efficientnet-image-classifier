# Configuration file for the Cat vs Dog Classification project. Centralizes all hyperparameters and paths so nothing is hardcoded elsewhere in the codebase.

import os

# -------------------------------------- Paths --------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Point this at the folder that contains the "Cat" and "Dog" sub-folders after downloading/extracting the Kaggle dataset:
# https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

# Expected structure:
#   data/PetImages/Cat/*.jpg
#   data/PetImages/Dog/*.jpg
DATA_DIR = os.path.join(BASE_DIR, "data", "PetImages")

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "efficientnet_cat_dog.keras")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
HISTORY_PLOT_PATH = os.path.join(OUTPUT_DIR, "training_history.png")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

# --------------------------------- Data Parameters ---------------------------------

IMAGE_SIZE = (224, 224)      # EfficientNetB0's native input size
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2       # fraction of data held out from training
TEST_SPLIT = 0.1             # carved out of the validation split above
SEED = 42
CLASS_NAMES = ["Cat", "Dog"]

# -------------------- Model / training parameters -----------------------------------

NUM_CLASSES = 1                # binary classification -> single sigmoid unit
BASE_TRAINABLE = False         # freeze EfficientNet backbone initially
DROPOUT_RATE = 0.3
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
EPOCHS = 10
FINE_TUNE_EPOCHS = 5
FINE_TUNE_AT_LAYER = 100       # unfreeze from this layer onward when fine-tuning

EARLY_STOPPING_PATIENCE = 3