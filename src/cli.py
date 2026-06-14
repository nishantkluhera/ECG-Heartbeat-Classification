"""Command-line interface for the ECG Heartbeat Classifier.

Usage:
    python main.py train [--epochs N]
    python main.py evaluate
    python main.py predict [--sample-index N]
"""
import argparse
import logging

import numpy as np

from src.config import EPOCHS, TEST_CSV, N_FEATURES, CLASS_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _cmd_train(args):
    from src.train import train_model
    train_model(epochs=args.epochs)


def _cmd_evaluate(args):
    from src.evaluate import evaluate_model
    evaluate_model()


def _cmd_predict(args):
    import os
    import pandas as pd
    from src.predict import predict_heartbeat, predict_proba

    if os.path.exists(TEST_CSV):
        df = pd.read_csv(TEST_CSV, header=None)
        idx = args.sample_index % len(df)
        signal = df.iloc[idx, :N_FEATURES].values.astype(np.float32)
        actual = CLASS_NAMES[int(df.iloc[idx, N_FEATURES])]
        print(f"\nUsing test sample #{idx} (actual class: {actual})")
    else:
        signal = np.random.rand(N_FEATURES).astype(np.float32)
        actual = None
        print("\nTest CSV not found - using a random segment.")

    name, conf = predict_heartbeat(signal)
    if name is None:
        print("Prediction failed (is the model trained?).")
        return
    probs = predict_proba(signal)
    print(f"Predicted: {name}  (confidence {conf:.2%})")
    if actual is not None:
        print(f"Actual:    {actual}  {'[correct]' if actual == name else '[wrong]'}")
    print("Class probabilities:")
    for cname, p in zip(CLASS_NAMES, probs):
        print(f"  {cname:<18} {p:.2%}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecg-classify",
        description="Train, evaluate and run inference for the ECG heartbeat CNN.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the CNN on data/mitbih_train.csv")
    p_train.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate the trained model on the test set")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_pred = sub.add_parser("predict", help="Classify a single heartbeat from the test set")
    p_pred.add_argument("--sample-index", type=int, default=0,
                        help="Row index in mitbih_test.csv to classify.")
    p_pred.set_defaults(func=_cmd_predict)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
