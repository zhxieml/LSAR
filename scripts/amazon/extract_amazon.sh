REPO=$PWD
DATA_DIR=$REPO/download/amazon
DIR=$REPO/data/amazon
mkdir -p $DIR

lans="en de fr ja"
for model in bert-base-multilingual-cased xlm-mlm-100-1280 xlm-roberta-large labse; do
    for split in train test; do
        if [[ "$model" == "labse" ]]; then
            mkdir -p $DIR/$split/labse
            python experiments/extract_embedding_labse.py \
                -i $DATA_DIR/$split \
                -o $DIR/$split/labse \
                --lans "$lans"
        else
            bash scripts/extract_embedding.sh \
                $DATA_DIR/$split \
                $DIR/$split \
                "$lans" \
                $model
        fi
    done
done

for split in train test; do
    cp $DATA_DIR/$split/*_label.npy $DIR/$split
done
