# Entry point for the project. Provides a simple CLI to run training, evaluation, or single-image prediction.


import argparse

from train import train
from evaluate import evaluate
from predict import predict


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


# Run commands:
# python main.py train
# python main.py evaluate
# python main.py predict path/to/image.jpg