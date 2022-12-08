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
DIR=$REPO/download/
mkdir -p $DIR

base_dir=$DIR/tatoeba-tmp/
wget https://github.com/facebookresearch/LASER/archive/main.zip
unzip -qq -o main.zip -d $base_dir/
mv $base_dir/LASER-main/data/tatoeba/v1/* $base_dir/
python experiments/tatoeba/preprocess_tatoeba.py \
  --data_dir $base_dir \
  --output_dir $DIR/tatoeba \
  --task tatoeba
rm -rf $base_dir main.zip
