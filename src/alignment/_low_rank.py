import numpy as np


def lsar(W, k, returns_all=False):
    # W: d x D
    d, D = W.shape

    wc = W @ np.ones(D) / D
    u, s, vh = np.linalg.svd(W - wc.reshape(-1, 1) @ np.ones((1, D)))
    Ws, Gamma  = u[:, :k], vh.T[:, :k] @ np.diag(s[:k])
    best_fit_W = wc.reshape(-1, 1) @ np.ones((1, D)) + Ws @ Gamma.T

    wc_new = np.linalg.pinv(best_fit_W).T @ np.ones(D)
    wc_new /= (wc_new ** 2).sum()
    prod = best_fit_W - wc_new.reshape(-1, 1) @ np.ones((1, D))
    print(np.linalg.norm(W - wc_new.reshape(-1, 1) @ np.ones((1, D)) - prod, axis=0))

    if returns_all:
        u, s, vh = np.linalg.svd(prod)
        Ws_new, Gamma_new = u[:, :k], vh.T[:, :k] @ np.diag(s[:k])

        return wc_new, prod, Ws_new, Gamma_new

    return wc_new, prod
