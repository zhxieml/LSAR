import argparse
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def extract_embeddings(data_dir, output_dir, model, lans):
    batch_size = 8

    for lan in tqdm(lans, desc="collecting oscar"):
        text_file = os.path.join(
            data_dir, '{}.txt'.format(lan))
        emb_file = os.path.join(
            output_dir, '{}.emb.npy'.format(lan))

        if os.path.exists(emb_file):
            print(f"{emb_file} already exists")
            continue

        with open(text_file, 'r') as f:
            texts = list(f.readlines())

        embs = []
        for start_idx in range(0, len(texts), batch_size):
            emb = model.encode(texts[start_idx:start_idx + batch_size])
            embs.append(emb)

        embs = np.concatenate(embs)
        np.save(emb_file, embs)

def extract_embeddings_parallel(data_dir, output_dir, model, lans):
    batch_size = 8

    for lan in tqdm(lans, desc="collecting parallel"):
        src_text_file = os.path.join(data_dir, '{}-en.{}'.format(lan, lan))
        tgt_text_file = os.path.join(data_dir, '{}-en.en'.format(lan))
        src_emb_file = os.path.join(output_dir, '{}-en.emb.{}'.format(lan, lan))
        tgt_emb_file = os.path.join(output_dir, '{}-en.emb.en'.format(lan))

        if os.path.exists(src_emb_file):
            print(f"{src_emb_file} already exists")
            continue

        # Extract source embeddings.
        with open(src_text_file, 'r') as f:
            src_texts = list(f.readlines())

        src_embs = []
        for start_idx in range(0, len(src_texts), batch_size):
            src_emb = model.encode(src_texts[start_idx:start_idx + batch_size])
            src_embs.append(src_emb)

        src_embs = np.concatenate(src_embs)
        np.save(src_emb_file, src_embs)

        # Extract target embeddings.
        with open(tgt_text_file, 'r') as f:
            tgt_texts = list(f.readlines())

        tgt_embs = []
        for start_idx in range(0, len(tgt_texts), batch_size):
            tgt_emb = model.encode(tgt_texts[start_idx:start_idx + batch_size])
            tgt_embs.append(tgt_emb)

        tgt_embs = np.concatenate(tgt_embs)
        np.save(tgt_emb_file, tgt_embs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="download/lareqa")
    parser.add_argument("-o", "--output", type=str, default="data/lareqa")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--lans", type=str, required=True)
    args = parser.parse_args()

    model = SentenceTransformer('sentence-transformers/LaBSE')

    if args.parallel:
        extract_embeddings_parallel(
            args.input, args.output,
            model, args.lans.split(" ")
        )
    else:
        extract_embeddings(
            args.input, args.output,
            model, args.lans.split(" ")
        )
