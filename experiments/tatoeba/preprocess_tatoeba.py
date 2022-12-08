# coding=utf-8
# Copyright 2020 Google and DeepMind.
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
from __future__ import absolute_import, division, print_function

import argparse
from transformers import BertTokenizer, XLMTokenizer, XLMRobertaTokenizer
import os
from collections import defaultdict
import csv
import random
import os
import shutil
import json


TOKENIZERS = {
    'bert': BertTokenizer,
    'xlm': XLMTokenizer,
    'xlmr': XLMRobertaTokenizer,
}


def tatoeba_preprocess(args):
    lang3_dict = {
        'afr': 'af', 'ara': 'ar', 'bul': 'bg', 'ben': 'bn',
        'deu': 'de', 'ell': 'el', 'spa': 'es', 'est': 'et',
        'eus': 'eu', 'pes': 'fa', 'fin': 'fi', 'fra': 'fr',
        'heb': 'he', 'hin': 'hi', 'hun': 'hu', 'ind': 'id',
        'ita': 'it', 'jpn': 'ja', 'jav': 'jv', 'kat': 'ka',
        'kaz': 'kk', 'kor': 'ko', 'mal': 'ml', 'mar': 'mr',
        'nld': 'nl', 'por': 'pt', 'rus': 'ru', 'swh': 'sw',
        'tam': 'ta', 'tel': 'te', 'tha': 'th', 'tgl': 'tl',
        'tur': 'tr', 'urd': 'ur', 'vie': 'vi', 'cmn': 'zh',
        'eng': 'en', 'aze': 'az', 'lit': 'lt', 'pol': 'pl',
        'ukr': 'uk', 'ron': 'ro'
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for sl3, sl2 in lang3_dict.items():
        if sl3 != 'eng':
            src_file = f'{args.data_dir}/tatoeba.{sl3}-eng.{sl3}'
            tgt_file = f'{args.data_dir}/tatoeba.{sl3}-eng.eng'
            src_out = f'{args.output_dir}/{sl2}-en.{sl2}'
            tgt_out = f'{args.output_dir}/{sl2}-en.en'
            shutil.copy(src_file, src_out)
            tgts = [l.strip() for l in open(tgt_file)]
            idx = range(len(tgts))
            data = zip(tgts, idx)
            with open(tgt_out, 'w') as ftgt:
                for t, _ in data:
                    ftgt.write(f'{t}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output data dir where any processed files will be written to.")
    parser.add_argument("--task", default="panx", type=str, required=True,
                        help="The task name")
    parser.add_argument("--model_name_or_path", default="bert-base-multilingual-cased", type=str,
                        help="The pre-trained model")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="model type")
    parser.add_argument("--max_len", default=512, type=int,
                        help="the maximum length of sentences")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="whether to do lower case")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="cache directory")
    parser.add_argument("--languages", default="en", type=str,
                        help="process language")
    parser.add_argument("--remove_last_token", action='store_true',
                        help="whether to remove the last token")
    parser.add_argument("--remove_test_label", action='store_true',
                        help="whether to remove test set label")
    args = parser.parse_args()

    if args.task == 'tatoeba':
        tatoeba_preprocess(args)
    else:
        raise NotImplementedError
