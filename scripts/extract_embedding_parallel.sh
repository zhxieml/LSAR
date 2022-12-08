#!/bin/bash
# Copyright 2020 Google, DeepMind and Alibaba inc.
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

REPO=$PWD
DATA_DIR=$1
OUT_DIR=$2
LANS=$3
MODEL=${4:-bert-base-multilingual-cased}

TL='en'
MAXL=512
LC=""
NLAYER=12
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
  DIM=768
elif [ $MODEL == "xlm-mlm-100-1280" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
  DIM=1280
elif [ $MODEL == "xlm-roberta-large" ]; then
  MODEL_TYPE="xlmr"
  DIM=1024
  NLAYER=24
fi

OUT=$OUT_DIR/${MODEL}/
mkdir -p $OUT
for SL in $LANS; do
  echo "Extracting $SL with $MODEL"
  python experiments/extract_embedding_parallel.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --embed_size $DIM \
    --batch_size 64 \
    --lan $SL \
    --data_dir $DATA_DIR \
    --max_seq_length $MAXL \
    --output_dir $OUT \
    --log_file embed-cosine \
    --num_layers $NLAYER \
    --dist cosine $LC
done
