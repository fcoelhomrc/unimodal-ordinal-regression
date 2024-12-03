import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from ensembles.models import MajorityVotingEnsemble, RandomEnsemble, AverageEnsemble
from ensembles.utils import EnsembleDataset


plt.figure()
losses = [
    "BinomialUnimodal_CE", "PoissonUnimodal", "CrossEntropy", "UnimodalNet", "ORD_ACL", "VS_SL"
]

results = {loss: [] for loss in losses}
results["Ensemble"] = []
revs = [1, 2, 3, 4]
model = MajorityVotingEnsemble()

eval_dataset = "FOCUSPATH"

compute_single = True
for rev in revs:
    # Compute for one at a time
    for loss in losses:
        if not compute_single:
            break
        ds = EnsembleDataset(dataset=eval_dataset, loss=[loss], rep=rev)
        batch_size = 32
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        total_acc = 0
        for x, y in dl:
            probas, ypred = model(x)
            acc = (ypred == y).sum()
            total_acc += acc.item()
        total_acc /= len(ds)
        print(f"{loss} -> {total_acc:3f}")
        results[loss].append(total_acc)

    # Compute with all at same time
    ds = EnsembleDataset(dataset=eval_dataset, loss=losses, rep=rev)
    batch_size = 32
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    total_acc = 0
    for x, y in dl:
        probas, ypred = model(x)
        acc = (ypred == y).sum()
        total_acc += acc.item()


    total_acc /= len(ds)
    # print(f"Ensemble -> {total_acc:3f}")
    results["Ensemble"].append(total_acc)

for key, value in results.items():
    if len(value) > 0:
        value = np.array(value)
        print(f"{key} -> {value.mean():.2f} ± {2 * value.std():.2f}")