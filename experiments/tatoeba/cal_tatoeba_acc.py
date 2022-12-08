import argparse
import os

import numpy as np

LANS = [
    "af", "ar", "bg", "bn", "de", "el",
    "es", "et", "eu", "fa", "fi", "fr",
    "he", "hi", "hu", "id", "it", "ja",
    "jv", "ka", "kk", "ko", "ml", "mr",
    "nl", "pt", "ru", "sw", "ta", "te",
    "th", "tl", "tr", "ur", "vi", "zh"
]


def main(args):
    # print(args)
    accs = {}
    corrects = {}
    nums = {}

    for lan in LANS:
        predicition_file = os.path.join(args.output_dir, f"test_{lan}_predictions.txt")
        correct = 0.0

        with open(predicition_file, "r") as f:
            lines = list(f.readlines())

        for line_idx, line in enumerate(lines):
            if line_idx == int(line.strip()[1:-1]):
                correct += 1

        num = len(lines)
        acc = correct / num
        accs[lan] = acc
        corrects[lan] = correct
        nums[lan] = num

    print(np.mean(list(accs.values())))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)