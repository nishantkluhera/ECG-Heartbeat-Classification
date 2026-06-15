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


def _cmd_diag_train(args):
    from src.diagnostic.train import train_diag
    train_diag(lead_config=args.leads, epochs=args.epochs)


def _cmd_diag_eval(args):
    from src.diagnostic.evaluate import evaluate_diag
    evaluate_diag(lead_config=args.leads)


def _cmd_diagnose_image(args):
    from src.diagnose import diagnose_image, DISCLAIMER
    result = diagnose_image(args.image, lead_config=args.leads, with_saliency=False)
    print(f"\n{'='*60}\n  ⚠️  {DISCLAIMER}\n{'='*60}")
    if result.get("warning"):
        print(f"\nNote: {result['warning']}")
    print(f"\nCalibrated digitization: {result['digitization']['calibrated']}")
    print("\nDiagnostic estimate (single-lead, screening only):")
    for f in result["findings"]:
        flag = " <== flagged" if f["flagged"] else ""
        print(f"  {f['name']:<26} {f['probability']:.1%}{flag}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecg-classify",
        description="Train, evaluate and run inference for the ECG models "
                    "(MIT-BIH beat classifier + PTB-XL diagnostic model).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- MIT-BIH beat classifier --- #
    p_train = sub.add_parser("train", help="Train the beat CNN on data/mitbih_train.csv")
    p_train.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate the beat model on the test set")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_pred = sub.add_parser("predict", help="Classify a single heartbeat from the test set")
    p_pred.add_argument("--sample-index", type=int, default=0,
                        help="Row index in mitbih_test.csv to classify.")
    p_pred.set_defaults(func=_cmd_predict)

    # --- PTB-XL diagnostic model --- #
    p_dtrain = sub.add_parser("diagnose-train", help="Train the PTB-XL diagnostic model")
    p_dtrain.add_argument("--leads", choices=["12lead", "lead2"], default="12lead")
    p_dtrain.add_argument("--epochs", type=int, default=EPOCHS)
    p_dtrain.set_defaults(func=_cmd_diag_train)

    p_deval = sub.add_parser("diagnose-eval", help="Evaluate the PTB-XL diagnostic model")
    p_deval.add_argument("--leads", choices=["12lead", "lead2"], default="12lead")
    p_deval.set_defaults(func=_cmd_diag_eval)

    p_dimg = sub.add_parser("diagnose-image",
                            help="Digitize a single-lead ECG strip image and diagnose it")
    p_dimg.add_argument("--image", required=True, help="Path to an ECG strip image.")
    p_dimg.add_argument("--leads", choices=["lead2"], default="lead2")
    p_dimg.set_defaults(func=_cmd_diagnose_image)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
