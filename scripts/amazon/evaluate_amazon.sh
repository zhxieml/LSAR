REPO=$PWD
DATA_DIR=$REPO/data/amazon
DIR=$REPO/res/amazon
SOURCE_DIR=$REPO/data/oscar
mkdir -p $DIR

align_methods="none demean lir+1 lsar+1 lsar+2 lsar+3"

for model in bert-base-multilingual-cased xlm-mlm-100-1280 xlm-roberta-large labse; do
    if [[ "$model" == "xlm-roberta-large" ]]; then
        layer_arg="--layer 10"
    elif [[ "$model" == "labse" ]]; then
        layer_arg=""
    else
        layer_arg="--layer 7"
    fi

    for align_method in $align_methods; do
        python experiments/amazon/evaluate_amazon.py \
            --model_name $model \
            --align_method $align_method \
            --train_lan "en" \
            --data_dir $DATA_DIR \
            --source_lan_dir $SOURCE_DIR \
            $layer_arg \
        > $DIR/${model}_${align_method}.log
    done
done