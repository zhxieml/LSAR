REPO=$PWD
DIR=$REPO/download/
mkdir -p $DIR
mkdir -p $DIR/wiki

python experiments/wiki/download_wiki.py -o $DIR/wiki
