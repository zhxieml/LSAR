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
import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from transformers import (
    BertConfig, BertModel, BertTokenizer, XLMConfig, XLMModel,
    XLMRobertaTokenizer, XLMTokenizer
)

from src.bert import BertForRetrieval
from src.xlm_roberta import XLMRobertaConfig, XLMRobertaForRetrieval, XLMRobertaModel

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys())
     for conf in (BertConfig, XLMConfig, XLMRobertaConfig)),
    ()
)
MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "xlm": (XLMConfig, XLMModel, XLMTokenizer),
    "xlmr": (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer),
    "bert-retrieval": (BertConfig, BertForRetrieval, BertTokenizer),
    "xlmr-retrieval":
        (XLMRobertaConfig, XLMRobertaForRetrieval, XLMRobertaTokenizer),
}
LOGGER = logging.getLogger(__name__)

def load_embeddings(embed_file, num_sentences=None):
    embeds = np.load(embed_file)
    return embeds


def prepare_batch(sentences, tokenizer, model_type, device="cuda", max_length=512, lang='en', langid=None, use_local_max_length=True, pool_skip_special_token=False):
    pad_token = tokenizer.pad_token
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token

    pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    pad_token_segment_id = 0

    batch_input_ids = []
    batch_token_type_ids = []
    batch_attention_mask = []
    batch_size = len(sentences)
    batch_pool_mask = []

    local_max_length = min(max([len(s) for s in sentences]) + 2, max_length)
    if use_local_max_length:
        max_length = local_max_length

    for sent in sentences:

        if len(sent) > max_length - 2:
            sent = sent[: (max_length - 2)]
        input_ids = tokenizer.convert_tokens_to_ids(
            [cls_token] + sent + [sep_token])

        padding_length = max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        pool_mask = [0] + [1] * (len(input_ids) - 2) + \
            [0] * (padding_length + 1)
        input_ids = input_ids + ([pad_token_id] * padding_length)

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_pool_mask.append(pool_mask)

    input_ids = torch.LongTensor(batch_input_ids).to(device)
    attention_mask = torch.LongTensor(batch_attention_mask).to(device)

    if pool_skip_special_token:
        pool_mask = torch.LongTensor(batch_pool_mask).to(device)
    else:
        pool_mask = attention_mask

    if model_type == "xlm":
        langs = torch.LongTensor(
            [[langid] * max_length for _ in range(len(sentences))]).to(device)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "langs": langs}, pool_mask
    elif model_type == 'bert' or model_type == 'xlmr':
        token_type_ids = torch.LongTensor(
            [[0] * max_length for _ in range(len(sentences))]).to(device)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}, pool_mask
    elif model_type in ('bert-retrieval', 'xlmr-retrieval'):
        token_type_ids = torch.LongTensor(
            [[0] * max_length for _ in range(len(sentences))]).to(device)
        return {"q_input_ids": input_ids, "q_attention_mask": attention_mask, "q_token_type_ids": token_type_ids}, pool_mask


def tokenize_text(text_file, tok_file, tokenizer, lang=None, with_pairs=False):
    if os.path.exists(tok_file):
        tok_sentences = [l.strip().split(' ') for l in open(tok_file)]
        LOGGER.info(' -- loading from existing tok_file {}'.format(tok_file))
        return tok_sentences

    tok_sentences = []
    sents = [l.strip() for l in open(text_file)]
    with open(tok_file, 'w') as writer:
        for sent in tqdm(sents, desc='tokenize'):
            if with_pairs:
                sub_sents = sent.split('\t')
            else:
                sub_sents = [sent]

            if isinstance(tokenizer, XLMTokenizer):
                tok_sent = sum([tokenizer.tokenize(s, lang=lang) + [tokenizer.sep_token] for s in sub_sents], [])[:-1]
            else:
                tok_sent = sum([tokenizer.tokenize(s) + [tokenizer.sep_token] for s in sub_sents], [])[:-1]
            tok_sentences.append(tok_sent)
            writer.write(' '.join(tok_sent) + '\n')
    LOGGER.info(' -- save tokenized sentences to {}'.format(tok_file))

    LOGGER.info('============ First 5 tokenized sentences ===============')
    for i, tok_sentence in enumerate(tok_sentences[:5]):
        LOGGER.info('S{}: {}'.format(i, ' '.join(tok_sentence)))
    LOGGER.info('==================================')
    return tok_sentences


def load_model(args, lang, output_hidden_states=None):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    if output_hidden_states is not None:
        config.output_hidden_states = output_hidden_states
    langid = config.lang2id.get(
        lang, config.lang2id["en"]) if args.model_type == 'xlm' else 0
    LOGGER.info("langid={}, lang={}".format(langid, lang))
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path, do_lower_case=args.do_lower_case)
    LOGGER.info("tokenizer.pad_token={}, pad_token_id={}".format(
        tokenizer.pad_token, tokenizer.pad_token_id))
    if args.init_checkpoint:
        model = model_class.from_pretrained(
            args.init_checkpoint, config=config, cache_dir=args.init_checkpoint)
    else:
        model = model_class.from_pretrained(
            args.model_name_or_path, config=config)
    model.to(args.device)
    model.eval()
    return config, model, tokenizer, langid


def extract_embeddings(args, text_file, tok_file, embed_file, lang='en', pool_type='mean', with_pairs=False):
    num_embeds = args.num_layers
    all_embed_files = ["{}_{}.npy".format(
        embed_file, i) for i in range(num_embeds)]
    if all(os.path.exists(f) for f in all_embed_files):
        LOGGER.info('loading files from {}'.format(all_embed_files))
        return [load_embeddings(f) for f in all_embed_files]

    config, model, tokenizer, langid = load_model(args, lang,
                                                  output_hidden_states=True)

    sent_toks = tokenize_text(text_file, tok_file, tokenizer, lang, with_pairs=with_pairs)
    max_length = max([len(s) for s in sent_toks])
    LOGGER.info('max length of tokenized text = {}'.format(max_length))

    batch_size = args.batch_size
    num_batch = int(np.ceil(len(sent_toks) * 1.0 / batch_size))
    num_sents = len(sent_toks)

    all_embeds = [np.zeros(shape=(num_sents, args.embed_size),
                           dtype=np.float32) for _ in range(num_embeds)]
    for i in tqdm(range(num_batch), desc='Batch'):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, num_sents)
        batch, pool_mask = prepare_batch(sent_toks[start_index: end_index],
                                         tokenizer,
                                         args.model_type,
                                         args.device,
                                         args.max_seq_length,
                                         lang=lang,
                                         langid=langid,
                                         pool_skip_special_token=args.pool_skip_special_token)

        with torch.no_grad():
            outputs = model(**batch)

            if args.model_type == 'bert' or args.model_type == 'xlmr':
                last_layer_outputs, first_token_outputs, all_layer_outputs = outputs
            elif args.model_type == 'xlm':
                last_layer_outputs, all_layer_outputs = outputs
                # first element of the last layer
                first_token_outputs = last_layer_outputs[:, 0]

            # get the pool embedding
            if pool_type == 'cls':
                all_batch_embeds = cls_pool_embedding(
                    all_layer_outputs[-args.num_layers:])
            else:
                all_batch_embeds = []
                all_layer_outputs = all_layer_outputs[-args.num_layers:]
                all_batch_embeds.extend(
                    mean_pool_embedding(all_layer_outputs, pool_mask))

        for embeds, batch_embeds in zip(all_embeds, all_batch_embeds):
            embeds[start_index: end_index] = batch_embeds.cpu(
            ).numpy().astype(np.float32)
        del last_layer_outputs, first_token_outputs, all_layer_outputs
        torch.cuda.empty_cache()

    if embed_file is not None:
        for file, embeds in zip(all_embed_files, all_embeds):
            LOGGER.info('save embed {} to file {}'.format(embeds.shape, file))
            np.save(file, embeds)
    return all_embeds

def mean_pool_embedding(all_layer_outputs, masks):
    """
      Args:
        embeds: list of torch.FloatTensor, (B, L, D)
        masks: torch.FloatTensor, (B, L)
      Return:
        sent_emb: list of torch.FloatTensor, (B, D)
    """
    sent_embeds = []
    for embeds in all_layer_outputs:
        embeds = (embeds * masks.unsqueeze(2).float()).sum(dim=1) / \
            masks.sum(dim=1).view(-1, 1).float()
        sent_embeds.append(embeds)
    return sent_embeds


def cls_pool_embedding(all_layer_outputs):
    sent_embeds = []
    for embeds in all_layer_outputs:
        embeds = embeds[:, 0, :]
        sent_embeds.append(embeds)
    return sent_embeds


def concate_embedding(all_embeds, last_k):
    if last_k == 1:
        return all_embeds[-1]
    else:
        embeds = np.hstack(all_embeds[-last_k:])  # (B,D)
        return embeds