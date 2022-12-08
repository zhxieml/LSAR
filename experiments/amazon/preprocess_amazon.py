import argparse
import os

import numpy as np
import pandas as pd

LANS = ["en", "de", "fr", "jp"]
renamed_func = lambda lan: "ja" if lan == "jp" else lan


def main(args):
    for lan in LANS:
        with open(os.path.join(args.input, f"{lan}_train.tsv"), "r") as f:
            lines = f.readlines()
            lines = [line.strip().split("\t") for line in lines]
            assert all(len(line) == 2 for line in lines)
            train_text = [line[1] for line in lines]
            train_label = [int(line[0]) for line in lines]

        with open(os.path.join(args.input, f"{lan}_test.tsv"), "r") as f:
            lines = f.readlines()
            lines = [line.strip().split("\t") for line in lines]
            assert all(len(line) == 2 for line in lines)
            test_text = [line[1] for line in lines]
            test_label = [int(line[0]) for line in lines]

        with open(os.path.join(args.input, f"{lan}_unlabled.tsv"), "r") as f:
            lines = f.readlines()
            unlabeled_text = [line.strip() for line in lines]

        train_folder = os.path.join(args.output, "train")
        test_folder = os.path.join(args.output, "test")
        unlabeled_folder = os.path.join(args.output, "unlabeled")

        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        if not os.path.exists(unlabeled_folder):
            os.makedirs(unlabeled_folder)

        with open(os.path.join(args.output, "train", f"{renamed_func(lan)}.txt"), "w") as f:
            f.write("\n".join(train_text))
        np.save(os.path.join(args.output, "train", f"{renamed_func(lan)}_label.npy"), np.array(train_label))

        with open(os.path.join(args.output, "test", f"{renamed_func(lan)}.txt"), "w") as f:
            f.write("\n".join(test_text))
        np.save(os.path.join(args.output, "test", f"{renamed_func(lan)}_label.npy"), np.array(test_label))

        with open(os.path.join(args.output, "unlabeled", f"{renamed_func(lan)}.txt"), "w") as f:
            f.write("\n".join(unlabeled_text))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()

    main(args)
