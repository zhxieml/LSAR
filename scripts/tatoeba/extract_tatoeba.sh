REPO=$PWD
DATA_DIR=$REPO/download/
DIR=$REPO/data/
mkdir -p $DIR
mkdir -p $DIR/tatoeba

lans="af ar bg bn de el es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr nl pt ru sw ta te th tl tr ur vi zh"
for model in labse; do
    if [[ "$model" == "labse" ]]; then
        mkdir -p $DIR/tatoeba/labse
        python experiments/extract_embedding_labse.py \
            -i $DATA_DIR/tatoeba \
            -o $DIR/tatoeba/labse \
            --lans "$lans" \
            --parallel
    else
        bash scripts/extract_embedding_parallel.sh \
        $DATA_DIR/tatoeba \
        $DIR/tatoeba \
        "$lans" \
        $model
    fi
done
