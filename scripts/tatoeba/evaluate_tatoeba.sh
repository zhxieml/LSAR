REPO=$PWD
DATA_DIR=$REPO/data/tatoeba
DIR=$REPO/res/tatoeba
SOURCE_DIR=$REPO/data/oscar
mkdir -p $DIR

lans="af ar bg bn de el es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr nl pt ru sw ta te th tl tr ur vi zh"
align_methods="none demean lir+1 lsar+36"

for model in bert-base-multilingual-cased xlm-mlm-100-1280 xlm-roberta-large labse; do
    if [[ "$model" == "xlm-roberta-large" ]]; then
        layer=10
    else
        layer=7  # meaningless for LaBSE
    fi

    for align_method in $align_methods; do
        bash scripts/evaluate_retrieval.sh \
            $DATA_DIR \
            $DIR \
            $SOURCE_DIR \
            "$lans" \
            $model \
            $layer \
            $align_method
    done
done
