import os

import numpy as np
from sklearn.linear_model import SGDClassifier

from ._lir import lir
from ._low_rank import lsar


def build_aligner(align_method, lan_emb):
    if align_method == "none":
        align = lambda embed, _: embed
    elif align_method == "demean":
        lan_mean_emb = {lan: np.mean(emb, axis=0) for lan, emb in lan_emb.items()}
        align = lambda embed, lan: embed - lan_mean_emb[lan]
    elif align_method.startswith("lsar+"):
        rank = align_method.split("+")[-1]
        rank = rank.split("/")
        if len(rank) == 1:
            n_removed = rank = int(rank[0])
        elif len(rank) == 2:
            n_removed, rank = int(rank[0]), int(rank[1])
        else:
            raise ValueError(f"Invalid align_method: {align_method}")
        lan_mean_emb = {lan: np.mean(emb, axis=0) for lan, emb in lan_emb.items()}
        _, _, Ws, _ = lsar(np.stack(list(lan_mean_emb.values())).T, rank, returns_all=True)
        align = lambda embed, lan: lir(embed, Ws, r=n_removed)
    elif align_method.startswith("lir+"):
        rank = int(align_method.split("+")[-1])
        lan_svd = {lan: np.linalg.svd(np.transpose(emb), full_matrices=False) for lan, emb in lan_emb.items()}
        lan_info = {lan: u[:, :20] for lan, (u, diag, vh) in lan_svd.items()}
        align = lambda embed, lan: lir(embed, lan_info[lan], r=rank)
    else:
        raise NotImplementedError

    return align