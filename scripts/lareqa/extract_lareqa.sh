REPO=$PWD
DIR=$REPO/data/lareqa
mkdir -p $DIR

for model_name in X_X En_En; do
    mkdir -p $DIR/$model_name
    echo $model_name

    for dataset_name in xquad-r mlqa-r; do
        mkdir -p $DIR/$model_name/$dataset_name
        python experiments/lareqa/extract_embedding_lareqa.py \
            --model_name $model_name \
            --extract_dataset $dataset_name
    done
done
