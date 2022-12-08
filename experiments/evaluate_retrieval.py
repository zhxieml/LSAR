# coding=utf-8
# Copyright The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate pretrained or fine-tuned models on retrieval tasks."""

import argparse
import logging
import os
import time

import numpy as np
import torch

from src.alignment import build_aligner
from src.utils_extract import (
    MODEL_CLASSES, ALL_MODELS
)
from src.utils_retrieve import similarity_search


def main(args):
    logging.basicConfig(handlers=[logging.FileHandler(os.path.join(args.output_dir, args.log_file)), logging.StreamHandler()],
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logging.info("Input args: %r" % args)

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    source_lans = [
        "en",
        "af", "ar", "bg", "bn", "de", "el",
        "es", "et", "eu", "fa", "fi", "fr",
        "he", "hi", "hu", "id", "it", "ja",
        "jv", "ka", "kk", "ko", "ml", "mr",
        "nl", "pt", "ru", "sw", "ta", "te",
        "th", "tl", "tr", "ur", "vi", "zh"
    ]
    if args.align_method != "none":
        assert args.source_dir is not None, "Please specify the source directory for alignment."

        lan_emb = {
            lan: np.load(
                os.path.join(
                    args.source_dir,
                    args.model_name_or_path,
                    f"{lan}.emb.npy"
                )
            )
            for lan in source_lans
        }
        lan_emb = {lan: emb[args.specific_layer] if len(emb.shape) == 3 else emb for lan, emb in lan_emb.items()}
    else:
        lan_emb = None

    start = time.time()
    aligner = build_aligner(args.align_method, lan_emb)
    print(f"It took {time.time() - start}s to build aligner.")

    for lan in args.lans.split(" "):
        src_emb_file = os.path.join(
            args.data_dir, '{}-en.emb.{}.npy'.format(lan, lan))
        tgt_emb_file = os.path.join(
            args.data_dir, '{}-en.emb.en.npy'.format(lan))

        all_src_embeds = np.load(src_emb_file)
        all_tgt_embeds = np.load(tgt_emb_file)

        x, y = all_src_embeds, all_tgt_embeds
        if len(x.shape) == 3 and len(y.shape) == 3:
            x, y = x[args.specific_layer], y[args.specific_layer]
        x, y = aligner(x, lan), aligner(y, "en")
        x, y = x.astype(np.float32), y.astype(np.float32)
        predictions = similarity_search(
            x, y, args.embed_size, normalize=(args.dist == 'cosine'))
        with open(os.path.join(args.output_dir, f'test_{lan}_predictions.txt'), 'w') as fout:
            for p in predictions:
                fout.write(str(p) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_size', type=int, default=768,
                        help='Dimensions of output embeddings')

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the input files for the task.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " +
        ", ".join(ALL_MODELS),
    )
    parser.add_argument("--lans", type=str, required=True, help="source languages separated by ','.")
    parser.add_argument("--dist", type=str, default='cosine')
    parser.add_argument("--align_method", type=str, default="none")

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory",
    )
    parser.add_argument("--log_file", default="train",
                        type=str, help="log file")

    # Other parameters
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--specific_layer", type=int,
                        default=7, help="use specific layer")
    parser.add_argument("--source_dir", type=str, default=None,
                        help="source monolingual corpora")
    args = parser.parse_args()

    main(args)
