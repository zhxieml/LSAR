REPO=$PWD
DATA_DIR=$REPO/download/
DIR=$REPO/data/
mkdir -p $DIR/oscar

lans="en af ar bg bn de el es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr nl pt ru sw ta te th tl tr ur vi zh"
for model in bert-base-multilingual-cased xlm-mlm-100-1280 xlm-roberta-large labse X_X En_En; do
    if [[ "$model" == "labse" ]]; then
        mkdir -p $DIR/oscar/labse
        python experiments/extract_embedding_labse.py \
            -i $DATA_DIR/oscar \
            -o $DIR/oscar/labse \
            --lans "$lans"
    elif [ "$model" == "X_X" ] || [ "$model" == "En_En" ]; then
        mkdir -p $DIR/oscar/$model
        python experiments/lareqa/extract_embedding_lareqa.py \
            -i $DATA_DIR \
            -o $DIR \
            --model_name $model \
            --extract_dataset oscar
    else
        bash scripts/extract_embedding.sh \
            $DATA_DIR/oscar \
            $DIR/oscar \
            "$lans" \
            $model
    fi
done
