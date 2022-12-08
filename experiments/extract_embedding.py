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

import numpy as np
import torch

from src.utils_extract import (
    MODEL_CLASSES, ALL_MODELS,
    extract_embeddings
)

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

    lan = args.lan
    text_file = os.path.join(
        args.data_dir, '{}.txt'.format(lan))
    tok_file = os.path.join(
        args.output_dir, '{}.tok'.format(lan))
    emb_file = os.path.join(
        args.output_dir, '{}.emb.npy'.format(lan))

    if os.path.exists(emb_file):
        all_embeds = np.load(emb_file)
    else:
        all_embeds = extract_embeddings(
            args, text_file, tok_file, None, lang=lan,
            pool_type=args.pool_type, with_pairs=args.with_pairs
        )
        np.save(emb_file, all_embeds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embedding preparation')
    parser.add_argument('--encoding', default='utf-8',
                        help='character encoding for input/output')
    parser.add_argument('--src_file', default=None, help='src file')
    parser.add_argument('--tgt_file', default=None, help='tgt file')
    parser.add_argument('--gold', default=None,
                        help='File name of gold alignments')
    parser.add_argument('--threshold', type=float, default=-1,
                        help='Threshold (used with --output)')
    parser.add_argument('--embed_size', type=int, default=768,
                        help='Dimensions of output embeddings')
    parser.add_argument('--pool_type', type=str, default='mean',
                        help='pooling over work embeddings')
    parser.add_argument('--with_pairs', action='store_true')

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the input files for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " +
        ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list"
    )
    parser.add_argument("--lan", type=str,
                        default="en", help="source language.")
    parser.add_argument("--batch_size", type=int,
                        default=100, help="batch size.")
    parser.add_argument("--tgt_text_file", type=str,
                        default=None, help="tgt_text_file.")
    parser.add_argument("--src_text_file", type=str,
                        default=None, help="src_text_file.")
    parser.add_argument("--tgt_embed_file", type=str,
                        default=None, help="tgt_embed_file")
    parser.add_argument("--src_embed_file", type=str,
                        default=None, help="src_embed_file")
    parser.add_argument("--tgt_tok_file", type=str,
                        default=None, help="tgt_tok_file")
    parser.add_argument("--src_tok_file", type=str,
                        default=None, help="src_tok_file")
    parser.add_argument("--tgt_id_file", type=str,
                        default=None, help="tgt_id_file")
    parser.add_argument("--src_id_file", type=str,
                        default=None, help="src_id_file")
    parser.add_argument("--num_layers", type=int,
                        default=12, help="num layers")
    parser.add_argument("--candidate_prefix", type=str, default="candidates")
    parser.add_argument("--pool_skip_special_token", action="store_true")
    parser.add_argument("--dist", type=str, default='cosine')
    parser.add_argument("--use_shift_embeds", action="store_true")
    parser.add_argument("--extract_embeds", action="store_true")
    parser.add_argument("--mine_bitext", action="store_true")
    parser.add_argument("--predict_dir", type=str,
                        default=None, help="prediction folder")

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
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=92,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--unify", action="store_true", help="unify sentences"
    )
    parser.add_argument("--split", type=str, default='training',
                        help='split of the bucc dataset')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--concate_layers",
                        action="store_true", help="concate_layers")
    args = parser.parse_args()

    main(args)
