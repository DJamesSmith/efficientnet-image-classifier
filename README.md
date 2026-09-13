# Cat vs Dog Classification — EfficientNetB0

A small, modular project that fine-tunes an **EfficientNetB0** CNN
(pretrained on ImageNet) to classify images as **Cat** or **Dog**.

Dataset: [Dog and Cat Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

## Project structure

```
cat_dog_classifier/
├── config.py          # all paths & hyperparameters in one place
├── data_loader.py      # tf.data pipeline: loading, augmentation, preprocessing
├── model.py            # EfficientNetB0 model definition + fine-tuning helper
├── utils.py             # seeding, directory setup, plotting
├── train.py             # trains head, then fine-tunes backbone, saves model
├── evaluate.py         # loads saved model, prints report + confusion matrix
├── predict.py           # single-image inference
├── main.py              # CLI entry point (train / evaluate / predict)
├── requirements.txt
├── data/                # put the dataset here (see below)
├── saved_models/       # trained model is written here
└── outputs/              # plots (training curves, confusion matrix) go here
```

Each file has a single responsibility, so you can change the data pipeline,
model architecture, or training loop independently.

## 1. Environment setup

Python 3.9–3.11 recommended (matches current TensorFlow support).

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Libraries used** (all in `requirements.txt`):
- `tensorflow` — model (Keras EfficientNetB0), data pipeline (`tf.data`), training
- `numpy` — array handling
- `matplotlib` — plotting training curves / confusion matrix
- `scikit-learn` — classification report & confusion matrix
- `pillow` — image loading (dependency of `tf.keras.utils.load_img`)

A GPU (with matching CUDA/cuDNN) is optional but strongly recommended for
reasonable training times.

## 2. Get the data

1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset
   (you'll need a free Kaggle account, or the `kaggle` CLI with an API token).

   ```bash
   pip install kaggle
   kaggle datasets download -d bhavikjikadara/dog-and-cat-classification-dataset
   unzip dog-and-cat-classification-dataset.zip -d data/
   ```

2. Arrange it so that `config.DATA_DIR` (`data/PetImages` by default) contains
   two sub-folders, one per class:

   ```
   data/PetImages/
       Cat/
           0.jpg
           1.jpg
           ...
       Dog/
           0.jpg
           1.jpg
           ...
   ```

   If the extracted dataset uses different folder names, either rename them
   to `Cat` / `Dog`, or update `config.DATA_DIR` and `config.CLASS_NAMES`
   accordingly.

   Note: this dataset (derived from the classic Kaggle "Dogs vs. Cats" set)
   is known to contain a handful of corrupt/zero-byte images. `tf.data`'s
   `image_dataset_from_directory` will error on those — if that happens,
   remove the offending files (their paths appear in the error message).

## 3. Run it

Train (head training + fine-tuning, saves model + plots):
```bash
python main.py train
```

Evaluate the saved model on the held-out test split:
```bash
python main.py evaluate
```

Predict a single image:
```bash
python main.py predict path/to/some_image.jpg
```

Each command can also be run as its own script directly, e.g. `python train.py`.

## How it works, briefly

- **`data_loader.py`** builds train/validation/test `tf.data.Dataset`s from the
  folder structure, resizes images to 224×224 (EfficientNetB0's native size),
  applies light augmentation (flip/rotate/zoom) on the training set only, and
  runs EfficientNet's own `preprocess_input`.
- **`model.py`** loads `EfficientNetB0` with ImageNet weights, `include_top=False`,
  and adds a small classification head (Dropout → Dense(128, relu) → Dropout →
  Dense(1, sigmoid)) for binary classification.
- **`train.py`** trains the head with the backbone frozen, then unfreezes the
  top layers of the backbone (from `config.FINE_TUNE_AT_LAYER` onward) and
  continues training at a lower learning rate — standard transfer-learning
  practice.
- **`evaluate.py`** reports precision/recall/F1 and a confusion matrix on a
  held-out test split.
- **`predict.py`** runs inference on a single arbitrary image file.

All tunable values (image size, batch size, learning rates, epochs, split
ratios, fine-tune layer cutoff, etc.) live in `config.py`.
