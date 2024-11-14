import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

import torch

from ensembles.models import NaiveWassersteinEnsemble

rng = np.random.default_rng(42)

plt.figure()

model = NaiveWassersteinEnsemble()

x = np.array([[
    [0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0],
    [0, 0, 0, 0, 1, 2, 3, 4, 5, 3, 1, 0, 0, 0, 0, 0],
    # [3, 1, 0, 1, 2, 3, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
]], dtype=np.float32)

batch_size, n_models, n_classes = x.shape

x /= x.sum(axis=-1, keepdims=True)

x = torch.from_numpy(x)
proba, ypred = model(x)

cls = [i for i in range(1, n_classes + 1)]
for i, xx in enumerate(x[0]):
    plt.plot(cls, xx, label=f"$y_{i}$", alpha=0.5)
plt.plot(cls, proba[0], color="k", label=r"$y_{\text{W}}$", alpha=0.9)
plt.xlabel("Classes")
plt.ylabel("Probability")
plt.title("Naive Wasserstein Ensemble")
plt.grid()
plt.legend()
plt.show()

