import numpy as np


def lir(embed, c, r = 1):
    # Remove the language information from embeddings.
    # r is how many components to remove.
    c /= np.linalg.norm(c, axis=0, keepdims=True)
    proj = np.matmul(embed, c[:, :r])
    return embed - np.matmul(proj, np.transpose(c[:, :r]))
