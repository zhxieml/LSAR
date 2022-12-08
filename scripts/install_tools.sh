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

set -eux  # for easier debugging

REPO=$PWD
LIB=$REPO/src/third_party
mkdir -p $LIB

# install conda env
conda create --name xtreme --file conda-env.txt
source activate
conda activate xtreme

# install latest transformer
cd $LIB
git clone https://github.com/huggingface/transformers
cd transformers
git checkout cefd51c50cc08be8146c1151544495968ce8f2ad
pip install .
cd $LIB

pip install seqeval
pip install tensorboardx
pip install matplotlib

# install XLM tokenizer
pip install sacremoses
pip install pythainlp
pip install jieba

git clone https://github.com/neubig/kytea.git && cd kytea
autoreconf -i
./configure --prefix=${CONDA_PREFIX}
make && make install
pip install kytea
mkdir -p ~/local/share/kytea
ln -s $LIB/kytea/data/model.bin ~/local/share/kytea/model.bin

# additional packages for source corpora
pip install datasets==1.18.4
pip install tensorflow-datasets==4.0.1

# additional packages for lareqa
pip install tensorflow-gpu==2.3.0
pip install tensorflow-text==2.3.0
pip install tensorflow-hub==0.12.0
