REPO=$PWD
DIR=$REPO/download/
mkdir -p $DIR

git clone https://github.com/google-research-datasets/lareqa.git $DIR/lareqa
cd $DIR/lareqa
git checkout 354d67d1cd066854cdfbf5ad0e6105528976141d
rm -rf .git/