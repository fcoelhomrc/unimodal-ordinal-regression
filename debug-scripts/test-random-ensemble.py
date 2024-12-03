import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

import torch

from ensembles.models import RandomEnsemble

rng = np.random.default_rng(42)

plt.figure()

model = RandomEnsemble()

x = np.array([[
    [0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0],
    [0, 0, 0, 0, 1, 2, 3, 4, 5, 3, 1, 0, 0, 0, 0, 0],
]], dtype=np.float32)

x = x.repeat(3, axis=0)
print(x.shape)

batch_size, n_models, n_classes = x.shape

x /= x.sum(axis=-1, keepdims=True)

x = torch.from_numpy(x)
_, labels = model(x)

cls = [i for i in range(1, n_classes + 1)]

print(labels.shape)
print(labels)