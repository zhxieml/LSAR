import random


def setup_seed(seed):
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except:
        pass

    try:
        import torch
        torch.manual_seed(seed)
    except:
        pass
