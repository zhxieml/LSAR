#!/bin/bash
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
REPO=$PWD
DATA_DIR=$1
OUT_DIR=$2
SOURCE_DIR=$3
LANS=$4
MODEL=${5:-bert-base-multilingual-cased}
LAYER=${6:-7}
ALIGN_METHOD=${7:-"none"}

TASK='tatoeba'
TL='en'
MAXL=512
LC=""
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
elif [ $MODEL == "labse" ]; then
  MODEL_TYPE="labse"
  DIM=768
fi

OUT=$OUT_DIR/${MODEL}_$LAYER.ALIGN$ALIGN_METHOD/
mkdir -p $OUT
python experiments/evaluate_retrieval.py \
  --model_name_or_path $MODEL \
  --embed_size $DIM \
  --lans "$LANS" \
  --data_dir $DATA_DIR/${MODEL} \
  --output_dir $OUT \
  --log_file embed-cosine \
  --dist cosine \
  --specific_layer $LAYER \
  --align_method $ALIGN_METHOD \
  --source_dir $SOURCE_DIR
