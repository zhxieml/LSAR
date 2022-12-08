import argparse
from collections import defaultdict
import copy
import json
import os

import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text  # Import TF.text ops
import tensorflow as tf
import tensorflow_hub as hub

from src.alignment import build_aligner

DATASET_TO_LANS = {
    "xquad-r": ["ar", "de", "el", "en", "es", "hi", "ru", "th", "tr", "vi", "zh"],
    "mlqa-r": ["ar", "de", "en", "es", "hi", "vi", "zh"]
}


def eval_lareqa(q_dict, cand_dict, languages):
    """
    Compute the averaged mAP across all languages.
    """
    score = 0
    cnt = 0
    result = {}
    for lan in tqdm(languages):
        cnt_lan = 0
        score_lan = 0
        dots = np.matmul(q_dict[lan]["embeds"], np.transpose(cand_dict["embeds"]))
        n_q = len(q_dict[lan]["qid"])
        scores_lan = []
        for i in range(n_q):
            qid = q_dict[lan]["qid"][i]
            labels = [1 if qid in id_set else 0 for id_set in cand_dict["qid"]]
            tmp = average_precision_score(labels, dots[i, :])
            score += tmp
            score_lan += tmp
            scores_lan.append(tmp)
            cnt += 1
            cnt_lan += 1
        result[lan] = score_lan/cnt_lan
    result["all"] = score/cnt
    return result

def main(args):
    print(args)

    # Prepare embeddings.
    lan_list = DATASET_TO_LANS[args.dataset_name]
    dataset_path = os.path.join(args.data_path, args.model_name, args.dataset_name)
    source_path = os.path.join(args.source_path, args.model_name)
    candidates = np.load(os.path.join(dataset_path, "candidates.npy"), allow_pickle=True).item()
    questions = np.load(os.path.join(dataset_path, "questions.npy"), allow_pickle=True).item()
    index = np.load(os.path.join(dataset_path, "index.npy"), allow_pickle=True).item()
    lan_emb = {lan: np.load(os.path.join(source_path, f"{lan}.npy")) for lan in lan_list}

    # Choose an alignment method.
    align = build_aligner(args.align_method, lan_emb)

    # Evaluate.
    candidates_aligned = copy.deepcopy(candidates)
    questions_aligned = copy.deepcopy(questions)
    candidates_aligned["embeds"] = []
    for lan in lan_list:
        questions_aligned[lan]["embeds"] = []

    for lan, (s_, e_) in tqdm(index.items()):  # FIXME: be careful
        candidates_aligned["embeds"].append(align(candidates["embeds"][s_:e_, :], lan))

    for lan in tqdm(lan_list):
        questions_aligned[lan]["embeds"].append(align(questions[lan]["embeds"], lan))

    candidates_aligned["embeds"] = np.concatenate(candidates_aligned["embeds"])
    for lan in lan_list:
        questions_aligned[lan]["embeds"] = np.concatenate(questions_aligned[lan]["embeds"])

    results_aligned = eval_lareqa(questions_aligned, candidates_aligned, lan_list)
    for lan, result in results_aligned.items():
        print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Experiment on LAReQA.")
    parser.add_argument("--align_method", type=str, default="none")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True, choices=["xquad-r", "mlqa-r"])
    parser.add_argument("--model_name", type=str, required=True, choices=["En_En", "X_X"])
    args = parser.parse_args()

    main(args)
