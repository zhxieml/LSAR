import argparse
from collections import defaultdict
import json
import os

import numpy as np
import tensorflow_datasets as tfds
import tensorflow_text as text  # Import TF.text ops
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

DATASET_TO_LANS = {
    "xquad-r": ["ar", "de", "el", "en", "es", "hi", "ru", "th", "tr", "vi", "zh"],
    "mlqa-r": ["ar", "de", "en", "es", "hi", "vi", "zh"],
    "oscar": ["ar", "de", "el", "en", "es", "hi", "ru", "th", "tr", "vi", "zh"]
}
BATCH_SIZE = 128


def extract_source(input_path, output_path, dataset_name, model_name):
    input_path = os.path.join(input_path, dataset_name)
    output_path = os.path.join(output_path, dataset_name, model_name)
    lan_list = DATASET_TO_LANS[dataset_name]

    loaded = hub.load(f"https://tfhub.dev/google/LAReQA/mBERT_{model_name}/1")
    question_encoder = loaded.signatures["query_encoder"]

    for lan in lan_list:
        text_file = os.path.join(input_path, '{}.txt'.format(lan))

        with open(text_file, 'r') as f:
            texts = list(f.readlines())

        embs = []
        batch_size = 8
        for start_idx in range(0, len(texts), batch_size):
            embs.append(question_encoder(input=tf.constant(texts[start_idx:start_idx + batch_size]))["outputs"].numpy())

        embs = np.concatenate(embs)
        np.save(os.path.join(output_path, f"{lan}.npy"), embs)

def extract_lareqa(input_path, output_path, dataset_name, model_name):
    input_path = os.path.join(input_path, dataset_name)
    output_path = os.path.join(output_path, model_name)
    lan_list = DATASET_TO_LANS[dataset_name]

    # Load LAReQA encoder.
    loaded = hub.load(f"https://tfhub.dev/google/LAReQA/mBERT_{model_name}/1")
    question_encoder = loaded.signatures["query_encoder"]
    response_encoder = loaded.signatures["response_encoder"]

    questions = {}
    for lan in lan_list:
        questions[lan] = defaultdict(list)
    candidates = defaultdict(list)

    # Memorize the corresponding index of each language in "candidates".
    index = {}

    start_ = 0
    candidates["sentences"] = []
    candidates["context"] = []
    candidates["qid"] = []

    for lan in tqdm(lan_list):
        questions[lan]["qid"] = []
        questions[lan]["question"] = []
        f = open(os.path.join(input_path, f"{lan}.json"))
        data = json.load(f)["data"]
        for entry in data:
            for para in entry["paragraphs"]:
                n_sents = len(para["sentences"])
                context = [para["context"]]*n_sents
                qid_list = [qs["id"] for qs in para["qas"]]
                q_list = [qs["question"] for qs in para["qas"]]
                sent_qid_list = [set() for _ in range(n_sents)]
                for qs in para["qas"]:
                    a_start = qs["answers"][0]["answer_start"]
                    for i in range(n_sents):
                        if a_start >= para["sentence_breaks"][i][0] and a_start <= para["sentence_breaks"][i][1]:
                            sent_qid_list[i].add(qs["id"])
                            break
                candidates["sentences"].extend(para["sentences"])
                candidates["context"].extend(context)
                candidates["qid"].extend(sent_qid_list)
                questions[lan]["qid"].extend(qid_list)
                questions[lan]["question"].extend(q_list)
        index[lan] = [start_, len(candidates["sentences"])]
        start_ = len(candidates["sentences"])
        f.close()

    # Encoder dataset.
    for lan in lan_list:
        for k, v in questions[lan].items():
            if k != "qid":
                questions[lan][k] = np.asarray(v)

    for k, v in candidates.items():
        if k != "qid":
            candidates[k] = np.asarray(v)

    for lan in tqdm(lan_list):
        q_lan = questions[lan]["question"]
        for i in range(0, len(q_lan), BATCH_SIZE):
            embeds_ = question_encoder(
                input=tf.constant(q_lan[i:i+BATCH_SIZE]))["outputs"]
            questions[lan]["embeds"].append(embeds_.numpy())

    for i in tqdm(range(0, len(candidates["sentences"]), BATCH_SIZE)):
        embeds_ = response_encoder(
            input=tf.constant(candidates["sentences"][i:i+BATCH_SIZE]),
            context=tf.constant(candidates["context"][i:i+BATCH_SIZE]))["outputs"]
        candidates["embeds"].append(embeds_.numpy())

    for lan in lan_list:
        questions[lan]["embeds"] = np.concatenate(questions[lan]["embeds"])
    candidates["embeds"] = np.concatenate(candidates["embeds"])

    # Save embeddings.
    model_path = os.path.join(output_path, dataset_name)
    np.save(os.path.join(model_path, "candidates.npy"), candidates)
    np.save(os.path.join(model_path, "questions.npy"), questions)
    np.save(os.path.join(model_path, "index.npy"), index)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="download/lareqa")
    parser.add_argument("-o", "--output", type=str, default="data/lareqa")
    parser.add_argument(
        "--model_name", type=str, default="X_X",
        choices=["X_X", "En_En"]
    )
    parser.add_argument(
        "--extract_dataset", type=str, default="xquad-r",
        choices=["xquad-r", "mlqa-r", "oscar"]
    )
    args = parser.parse_args()

    if args.extract_dataset in ["xquad-r", "mlqa-r"]:
        extract_lareqa(args.input, args.output, args.extract_dataset, args.model_name)
    else:
        extract_source(args.input, args.output, args.extract_dataset, args.model_name)
