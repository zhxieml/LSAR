import argparse
import os

from datasets import load_dataset
from tqdm import tqdm

SENT_LIMIT = 10000
LANS = [
    "en",
    "af", "ar", "bg", "bn", "de", "el",
    "es", "et", "eu", "fa", "fi", "fr",
    "he", "hi", "hu", "id", "it", "ja",
    "jv", "ka", "kk", "ko", "ml", "mr",
    "nl", "pt", "ru", "sw", "ta", "te",
    "th", "tl", "tr", "ur", "vi", "zh"
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="download/oscar")
    args = parser.parse_args()

    pbar = tqdm(LANS)

    for lan in pbar:
        pbar.set_description(lan)
        dataset = load_dataset('oscar', f"unshuffled_deduplicated_{lan}", split='train', streaming=True)
        shuffled_dataset = dataset.shuffle(seed=42, buffer_size=100_000)
        data_iter = iter(shuffled_dataset)
        sentences = []

        while len(sentences) < SENT_LIMIT:
            pbar.set_description(f"{lan} (# = {len(sentences)})")
            try:
                sent = next(data_iter)["text"]
                sentences.extend(sent.strip().split("\n"))
            except:
                break

        with open(os.path.join(args.output, f"{lan}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sentences))
