REPO=$PWD
DIR=$REPO/download/
mkdir -p $DIR
mkdir -p $DIR/oscar

python experiments/oscar/download_oscar.py -o $DIR/oscar
