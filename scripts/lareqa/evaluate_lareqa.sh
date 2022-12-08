REPO=$PWD
DATA_DIR=$REPO/data/lareqa
DIR=$REPO/res/lareqa
SOURCE_DIR=$REPO/data/oscar
mkdir -p $DIR

for dataset in mlqa-r xquad-r; do
    if [[ "$dataset" == "mlqa-r" ]]; then
        align_methods="none demean lir+1 lsar+7"
    else
        align_methods="none demean lir+1 lsar+11"
    fi

    for model in X_X En_En; do
        for align_method in $align_methods; do
            python experiments/lareqa/evaluate_lareqa.py \
                --align_method $align_method \
                --data_path $DATA_DIR \
                --source_path $SOURCE_DIR \
                --dataset_name $dataset \
                --model_name $model \
            > $DIR/${dataset}_${model}_${align_method}.log
        done
    done
done
