import argparse
import os

import numpy as np
from sklearn.linear_model import LogisticRegressionCV

from src.alignment import build_aligner
from src.utils import setup_seed

LANS = ["en", "de", "fr", "ja"]
ALIGN_LANS = ["en", "de", "fr", "ja"]


def main(args):
    print(args)
    setup_seed(args.seed)

    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    # Build aligner.
    source_lan_emb = {
        lan: np.load(os.path.join(args.source_lan_dir, args.model_name, f"{lan}.emb.npy"))
        for lan in ALIGN_LANS
    }
    if args.layer is not None:
        source_lan_emb = {lan: emb[args.layer] for lan, emb in source_lan_emb.items()}
    aligner = build_aligner(args.align_method, source_lan_emb)

    # Load training data.
    X = np.load(os.path.join(train_dir, args.model_name, f"{args.train_lan}.emb.npy"))
    if args.layer is not None:
        X = X[args.layer]
    X = aligner(X, args.train_lan)
    y = np.load(os.path.join(train_dir, f"{args.train_lan}_label.npy"))
    rand_idxs = np.random.permutation(len(y))
    X, y = X[rand_idxs], y[rand_idxs]

    # Train.
    clf = LogisticRegressionCV(
        cv=args.n_folds,
        random_state=args.seed,
        verbose=0,
        max_iter=50000,
        n_jobs=8
    )
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std
    clf.fit(X, y)

    # Load test data.
    test_data = {
        lan: (
            aligner(
                np.load(os.path.join(test_dir, args.model_name, f"{lan}.emb.npy"))
                if args.layer is None
                else np.load(os.path.join(test_dir, args.model_name, f"{lan}.emb.npy"))[args.layer],
                lan
            ),
            np.load(os.path.join(test_dir, f"{lan}_label.npy")),
        ) for lan in LANS
    }
    for lan, (X_test, y_test) in test_data.items():
        test_data[lan] = ((X_test - mean) / std, y_test)

    # Evaluate.
    for lan, (X_test, y_test) in test_data.items():
        score = clf.score(X_test, y_test)
        print(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_lan", type=str, default="en")
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased_512")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--source_lan_dir", type=str, default=".")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--align_method", type=str, default="none")
    args = parser.parse_args()

    main(args)
