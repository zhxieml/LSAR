REPO=$PWD
DIR=$REPO/download/
mkdir -p $DIR/amazon

wget \
    https://automl-mm-bench.s3.amazonaws.com/multilingual-datasets/amazon_review_sentiment_cross_lingual.zip
unzip -o amazon_review_sentiment_cross_lingual.zip -d $DIR
mv $DIR/amazon_review_sentiment_cross_lingual $DIR/amazon_tmp
python experiments/amazon/preprocess_amazon.py -i $DIR/amazon_tmp -o $DIR/amazon
rm -rf $DIR/amazon_tmp amazon_review_sentiment_cross_lingual.zip
rm -rf $DIR/amazon/unlabeled  # unused
