"""This script is adapted from https://github.com/ziyi-yang/LIR/blob/master/lir.ipynb"""
import argparse
from collections import defaultdict
import os

import tensorflow_datasets as tfds
import tensorflow_text as text  # Import TF.text ops
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

LANS = ["ar", "de", "el", "en", "es", "hi", "ru", "th", "tr", "vi", "zh"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="download/wiki")
    args = parser.parse_args()

    pbar = tqdm(LANS)

    # Define arguments.
    sents_limit = 10000
    lan_list = LANS
    lan_list[lan_list.index("zh")] = "zh-cn"
    pbar = tqdm(lan_list)

    text_wiki = defaultdict(list)
    for lan in pbar:
        dataset = tfds.load(f"wiki40b/{lan}")["train"]
        dataset = tfds.as_numpy(dataset)
        cnt = 0

        for example in dataset:
            for text_sub in example["text"].decode("utf-8").split("_START_PARAGRAPH_"):
                if "_START_" not in text_sub:
                    text_list = text_sub.split("_NEWLINE_")
                    text_list = [a_.replace("\n", "") for a_ in text_list if len(a_) >= 10]
                    if text_list:
                        text_wiki[lan].extend(text_list)
                        cnt += len(text_list)
            if cnt >= sents_limit:
                break

    for lan, text in tqdm(text_wiki.items(), desc="writing sentences"):
        with open(os.path.join(args.output, f"{lan}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(text) + "\n")
