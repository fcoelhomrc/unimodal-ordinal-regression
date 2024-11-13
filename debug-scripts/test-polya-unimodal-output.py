import torch
from src.losses import *

N = 2**3
K_classes = 12
K_outputs = 2

ps = torch.rand(size=(N, 1))
rs = torch.randint(0, int(1.5*K_classes), size=(N, 1))
dummy = torch.stack((ps, rs), dim=1)

dummy_gt = torch.randint(0, K_classes, size=(N,))

polya = PolyaUnimodal(K_classes)

print(f"fake model output: {dummy}")
print(f"fake labels: {dummy_gt}")
print(f"N: {N}, outputs: {K_outputs}, classes: {K_classes}")

nll = polya(dummy, dummy_gt)
logits = polya.to_log_proba(dummy)
probas = polya.to_proba(dummy)

print(f"nll: {nll.shape}")
print(f"logits: {logits.shape}")
print(f"probas: {probas.shape}")


print(f"nll: {nll}")
print(f"logits: {logits}")
print(f"probas: {probas}")

print(f"is probas normalized?: {probas.sum(dim=1)}")

import numpy as np
import matplotlib.pyplot as plt
classes = torch.arange(1, K_classes + 1)

plt.figure()
ax = plt.gca()
for i in range(N):
    ax.plot(classes, probas[i], alpha=0.5)
    plt.title("Polya model")
    plt.xlabel("Classes")
    plt.ylabel("Probability")
plt.grid(axis="both")
plt.ylim(0, 1.)
plt.show()